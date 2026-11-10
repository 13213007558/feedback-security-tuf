# Copyright 2026, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""A composable mirror-aware fetcher.

``MirrorFetcher`` wraps another :class:`~tuf.ngclient.fetcher.FetcherInterface`
implementation and serves the same interface. The set of mirrors is fixed at
construction time from server-side configuration: the authority (scheme and
host) of any URL handed to the fetcher is discarded and replaced by the
configured mirror prefixes, so content encountered in metadata cannot widen
the set of reachable hosts.

For one logical file, the fetcher tries mirrors in order and may retry the
same logical file against another mirror after a connection failure, a
timeout, or an HTTP status that is explicitly allowed for retry. The total
number of attempts and the cumulative number of downloaded bytes are bounded.
Each attempt independently enforces the caller's ``max_length``: a partial
response from a failed attempt is discarded and never concatenated with the
next attempt.

Trust verification stays in the client layer (for example
:class:`~tuf.ngclient.updater.Updater`): bytes that download successfully but
fail signature, version or digest checks are returned to that layer and are
not hidden behind mirror rotation.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING
from urllib import parse

import urllib3

import tuf
from tuf.api import exceptions
from tuf.ngclient._internal.proxy import ProxyEnvironment
from tuf.ngclient.fetcher import FetcherInterface

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)

#: Failure category for a connection that could not be established.
FAIL_CONNECTION = "connection"
#: Failure category for a timeout while connecting or reading.
FAIL_TIMEOUT = "timeout"
#: Failure category for an HTTP error status.
FAIL_HTTP_STATUS = "http_status"
#: Failure category for an attempt that exceeded a length limit.
FAIL_LENGTH = "length"
#: Failure category for a redirect to a different origin.
FAIL_CROSS_ORIGIN_REDIRECT = "cross_origin_redirect"
#: Failure category for an unparseable or disallowed request.
FAIL_INVALID_REQUEST = "invalid_request"
#: Failure category for any other download failure.
FAIL_DOWNLOAD = "download"

_MAX_REDIRECT_HOPS = 5
_READ_CHUNK_SIZE = 1_048_576


@dataclass(frozen=True)
class AttemptRecord:
    """One fetch attempt against one mirror.

    The record deliberately contains only non-sensitive diagnostic data:
    mirror identifier, the requested logical file and a failure category.
    Authorization headers and full URLs are never recorded.

    Attributes:
        mirror: Identifier of the mirror that was contacted.
        logical_file: Path-relative name of the requested logical file.
        failure: ``None`` for a successful attempt, otherwise one of the
            ``FAIL_*`` category constants.
        status_code: HTTP status code for an ``FAIL_HTTP_STATUS`` failure,
            otherwise ``None``.
    """

    mirror: str
    logical_file: str
    failure: str | None
    status_code: int | None = None


@dataclass(frozen=True)
class Mirror:
    """A statically configured repository mirror.

    Attributes:
        url: Mirror base URL. It must use HTTP/HTTPS, must not contain a
            query string or userinfo and is normalized to end in a slash.
        identifier: Short label used in attempt records. Defaults to the
            mirror host and port.
    """

    url: str
    identifier: str | None = None


class CrossOriginRedirectError(exceptions.DownloadError):
    """Raised when a response redirects outside the mirror's allowed origin."""


class _ByteBudgetExceededError(exceptions.DownloadLengthMismatchError):
    """Raised internally when the cumulative byte budget is exhausted."""


class _OriginBoundFetcher(FetcherInterface):
    """HTTP fetcher bound to one mirror origin.

    Redirects are handled manually: same-origin redirects are followed, a
    redirect to another origin is rejected unless the mirror explicitly
    allows cross-origin redirects. This keeps requests inside the configured
    mirror prefix even when the server answers with a redirect.
    """

    def __init__(
        self,
        mirror: Mirror,
        *,
        socket_timeout: int,
        chunk_size: int,
        allow_cross_origin_redirects: bool,
        app_user_agent: str | None,
    ) -> None:
        parsed = parse.urlparse(mirror.url)
        self._scheme = parsed.scheme
        self._host = parsed.hostname or ""
        self._port = parsed.port
        self._allow_cross_origin = allow_cross_origin_redirects
        self.socket_timeout = socket_timeout
        self.chunk_size = chunk_size

        ua = f"python-tuf/{tuf.__version__}"
        if app_user_agent is not None:
            ua = f"{app_user_agent} {ua}"
        self._proxy_env = ProxyEnvironment(headers={"User-Agent": ua})

    def _fetch(self, url: str) -> Iterator[bytes]:
        current_url = url
        for _ in range(_MAX_REDIRECT_HOPS + 1):
            try:
                response = self._proxy_env.request(
                    "GET",
                    current_url,
                    preload_content=False,
                    redirect=False,
                    timeout=urllib3.Timeout(self.socket_timeout),
                )
            except urllib3.exceptions.MaxRetryError as e:
                if isinstance(e.reason, urllib3.exceptions.TimeoutError):
                    raise exceptions.SlowRetrievalError from e
                raise

            if response.status < 300:
                return self._chunks(response)

            if response.status not in (301, 302, 303, 307, 308):
                response.close()
                raise exceptions.DownloadHTTPError(
                    f"HTTP error occurred with status {response.status}",
                    response.status,
                )

            location = response.get_redirect_location()
            response.close()
            current_url = self._check_redirect(current_url, location)

        raise CrossOriginRedirectError(
            f"Redirect limit of {_MAX_REDIRECT_HOPS} hops exceeded"
        )

    def _check_redirect(self, request_url: str, location: str | None) -> str:
        if location is None:
            raise exceptions.DownloadError(
                "Redirect response did not contain a Location header"
            )

        target = parse.urlparse(parse.urljoin(request_url, location))
        if target.scheme not in ("http", "https"):
            raise CrossOriginRedirectError(
                f"Redirect to disallowed scheme {target.scheme!r}"
            )

        same_origin = (
            target.scheme == self._scheme
            and (target.hostname or "") == self._host
            and target.port == self._port
        )
        if not same_origin and not self._allow_cross_origin:
            raise CrossOriginRedirectError(
                "Cross-origin redirects are not allowed"
            )

        return target.geturl()

    def _chunks(
        self, response: urllib3.response.BaseHTTPResponse
    ) -> Iterator[bytes]:
        try:
            yield from response.stream(self.chunk_size)
        except urllib3.exceptions.TimeoutError as e:
            raise exceptions.SlowRetrievalError from e
        except urllib3.exceptions.MaxRetryError as e:
            if isinstance(e.reason, urllib3.exceptions.TimeoutError):
                raise exceptions.SlowRetrievalError from e
            raise
        finally:
            response.release_conn()


class MirrorFetcher(FetcherInterface):
    """Fetcher that retries logical files across a fixed set of mirrors.

    The mirror list is part of the client configuration. Requests only carry
    a path-relative logical file: the authority of the URL passed to
    :meth:`fetch` is ignored and replaced by each configured mirror. Path
    traversal sequences and encoded separators are rejected so requests cannot
    escape a mirror prefix.

    Args:
        mirrors: Non-empty sequence of :class:`Mirror` instances.
        inner_fetcher: Optional fetcher used for every actual request. When
            not given, an origin-bound urllib3 fetcher with redirect
            containment is created per mirror.
        max_attempts: Maximum number of attempts per logical file.
        retry_on_status: HTTP status codes that allow trying another mirror.
        total_byte_budget: Cumulative byte limit across all attempts of one
            logical file. ``None`` disables the cumulative limit.
        allow_cross_origin_redirects: When ``False`` (default), a redirect to
            a different origin aborts the attempt with
            :class:`CrossOriginRedirectError`.
        socket_timeout: Timeout in seconds for the default inner fetchers.
        chunk_size: Chunk size in bytes for the default inner fetchers.
        app_user_agent: Application user agent for the default inner
            fetchers.
    """

    def __init__(
        self,
        mirrors: Sequence[Mirror],
        *,
        inner_fetcher: FetcherInterface | None = None,
        max_attempts: int = 4,
        retry_on_status: frozenset[int] = frozenset({404, 500, 502, 503}),
        total_byte_budget: int | None = None,
        allow_cross_origin_redirects: bool = False,
        socket_timeout: int = 30,
        chunk_size: int = 400_000,
        app_user_agent: str | None = None,
    ) -> None:
        if not mirrors:
            raise ValueError("at least one mirror must be configured")
        if max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if total_byte_budget is not None and total_byte_budget < 0:
            raise ValueError("total_byte_budget must not be negative")

        self._mirrors: list[Mirror] = []
        self._mirror_ids: list[str] = []
        identifiers: set[str] = set()
        for mirror in mirrors:
            normalized, identifier = self._validate_mirror(mirror)
            if identifier in identifiers:
                raise ValueError(f"duplicate mirror identifier {identifier!r}")
            identifiers.add(identifier)
            self._mirrors.append(normalized)
            self._mirror_ids.append(identifier)

        self._inner = inner_fetcher
        self._inner_fetchers: dict[str, FetcherInterface] = {}
        self._max_attempts = max_attempts
        self._retry_on_status = retry_on_status
        self._total_byte_budget = total_byte_budget
        self._allow_cross_origin = allow_cross_origin_redirects
        self._socket_timeout = socket_timeout
        self._chunk_size = chunk_size
        self._app_user_agent = app_user_agent
        self._attempt_history: list[AttemptRecord] = []

    @property
    def mirrors(self) -> tuple[Mirror, ...]:
        """Configured mirrors in rotation order."""
        return tuple(self._mirrors)

    @property
    def attempt_history(self) -> tuple[AttemptRecord, ...]:
        """Attempt records of the most recent fetch call."""
        return tuple(self._attempt_history)

    @staticmethod
    def _validate_mirror(mirror: Mirror) -> tuple[Mirror, str]:
        parsed = parse.urlparse(mirror.url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"mirror {mirror.url!r} must use http or https")
        if not parsed.hostname:
            raise ValueError(f"mirror {mirror.url!r} has no host")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(f"mirror {mirror.url!r} must not contain userinfo")
        if parsed.query:
            raise ValueError(
                f"mirror {mirror.url!r} must not contain a query string"
            )

        base = parsed._replace(fragment="").geturl()
        if not base.endswith("/"):
            base = f"{base}/"

        identifier = mirror.identifier
        if identifier is None:
            identifier = parsed.hostname
            if parsed.port is not None:
                identifier = f"{identifier}:{parsed.port}"
        return Mirror(url=base, identifier=identifier), identifier

    def _get_inner_fetcher(self, index: int) -> FetcherInterface:
        if self._inner is not None:
            return self._inner

        identifier = self._mirror_ids[index]
        fetcher = self._inner_fetchers.get(identifier)
        if fetcher is None:
            fetcher = _OriginBoundFetcher(
                self._mirrors[index],
                socket_timeout=self._socket_timeout,
                chunk_size=self._chunk_size,
                allow_cross_origin_redirects=self._allow_cross_origin,
                app_user_agent=self._app_user_agent,
            )
            self._inner_fetchers[identifier] = fetcher
        return fetcher

    def _resolve_request(self, url: str) -> tuple[str, str]:
        """Build a contained mirror URL and the logical file name.

        Only the path and query of ``url`` are used. Any authority is
        discarded so remote content cannot introduce new hosts. Path
        segments that resolve to ``.`` or ``..`` (including percent-encoded
        forms) are rejected before joining with the mirror prefix.
        """
        parsed = parse.urlparse(url)
        if parsed.scheme and parsed.scheme not in ("http", "https"):
            raise exceptions.DownloadError(
                f"URL scheme {parsed.scheme!r} is not allowed"
            )
        # Absolute http(s) URLs contribute only path and query: the authority
        # is silently discarded so remote content cannot widen the set of
        # reachable hosts beyond the configured mirrors.

        path = parsed.path
        if "\\" in path:
            raise exceptions.DownloadError(
                f"backslash in path {path!r} is not allowed"
            )

        segments = path.split("/")
        for segment in segments:
            decoded = parse.unquote(segment)
            if decoded in (".", "..") or "/" in decoded or "\\" in decoded:
                raise exceptions.DownloadError(
                    f"path {path!r} escapes the mirror prefix"
                )

        logical_file = path.lstrip("/")
        return path, logical_file

    def _build_url(self, mirror: Mirror, path: str, query: str) -> str:
        base = parse.urlparse(mirror.url)
        joined_path = base.path.rstrip("/") + "/" + path.lstrip("/")

        # Defense in depth: the normalized joined path must still start with
        # the configured mirror prefix.
        if not joined_path.startswith(base.path):
            raise exceptions.DownloadError(
                f"resolved path {joined_path!r} escapes mirror prefix"
            )

        resolved = parse.SplitResult(
            base.scheme,
            base.netloc,
            joined_path,
            query,
            "",
        )
        return resolved.geturl()

    def _categorize(self, exc: Exception) -> tuple[str, int | None, bool]:
        """Return failure category, status and whether retry is allowed."""
        if isinstance(exc, CrossOriginRedirectError):
            return FAIL_CROSS_ORIGIN_REDIRECT, None, False
        # _ByteBudgetExceededError is a DownloadLengthMismatchError subclass.
        if isinstance(exc, exceptions.DownloadLengthMismatchError):
            return FAIL_LENGTH, None, False
        if isinstance(exc, exceptions.DownloadHTTPError):
            retryable = exc.status_code in self._retry_on_status
            return FAIL_HTTP_STATUS, exc.status_code, retryable
        if isinstance(exc, exceptions.SlowRetrievalError):
            return FAIL_TIMEOUT, None, True
        if isinstance(exc, exceptions.DownloadError):
            return FAIL_DOWNLOAD, None, True
        return FAIL_CONNECTION, None, True

    def _download_to_file(self, url: str, max_length: int) -> IO[bytes]:
        try:
            path, logical_file = self._resolve_request(url)
            query = parse.urlparse(url).query
        except exceptions.DownloadError as exc:
            self._attempt_history = [
                AttemptRecord("", url, FAIL_INVALID_REQUEST)
            ]
            raise exc

        self._attempt_history = []
        bytes_total = 0
        last_error: Exception | None = None

        for attempt_number in range(self._max_attempts):
            mirror_index = attempt_number % len(self._mirrors)
            mirror = self._mirrors[mirror_index]
            identifier = self._mirror_ids[mirror_index]
            target_url = self._build_url(mirror, path, query)
            fetcher = self._get_inner_fetcher(mirror_index)

            temp_file = tempfile.TemporaryFile()
            attempt_bytes = 0
            try:
                chunks = fetcher.fetch(target_url)
                for chunk in chunks:
                    attempt_bytes += len(chunk)
                    if attempt_bytes > max_length:
                        raise exceptions.DownloadLengthMismatchError(
                            f"Downloaded {attempt_bytes} bytes exceeding"
                            f" the maximum allowed length of {max_length}"
                        )
                    if (
                        self._total_byte_budget is not None
                        and bytes_total + attempt_bytes
                        > self._total_byte_budget
                    ):
                        raise _ByteBudgetExceededError(
                            "Cumulative download byte budget exhausted"
                        )
                    temp_file.write(chunk)

                temp_file.flush()
                temp_file.seek(0)
                self._attempt_history.append(
                    AttemptRecord(identifier, logical_file, None)
                )
                logger.debug(
                    "Fetched %s from mirror %s after %d attempt(s)",
                    logical_file,
                    identifier,
                    attempt_number + 1,
                )
                return temp_file
            except Exception as exc:  # noqa: BLE001 - categorized below
                with contextlib.suppress(Exception):
                    temp_file.close()
                category, status, retryable = self._categorize(exc)
                self._attempt_history.append(
                    AttemptRecord(identifier, logical_file, category, status)
                )
                bytes_total += attempt_bytes
                last_error = exc
                logger.debug(
                    "Mirror %s failed for %s (%s)",
                    identifier,
                    logical_file,
                    category,
                )
                if not retryable:
                    break

        # Every attempt failed. last_error is always set here because each
        # iteration either returned successfully or recorded an exception.
        if isinstance(last_error, exceptions.DownloadError):
            raise last_error
        raise exceptions.DownloadError(
            f"Failed to download {logical_file} from all configured mirrors"
        ) from last_error

    def _fetch(self, url: str) -> Iterator[bytes]:
        max_length = (
            self._total_byte_budget
            if self._total_byte_budget is not None
            else 2**63
        )
        temp_file = self._download_to_file(url, max_length)
        try:
            while True:
                chunk = temp_file.read(_READ_CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
        finally:
            temp_file.close()

    @contextlib.contextmanager
    def download_file(self, url: str, max_length: int) -> Iterator[IO[bytes]]:
        """Download ``url`` trying mirrors according to the retry policy.

        Every attempt enforces ``max_length`` independently: partial data
        from a failed attempt is discarded, so response fragments are never
        concatenated across mirrors.

        Raises:
            exceptions.DownloadError: All attempts failed, or the request was
                not contained in the mirror prefixes.
            exceptions.DownloadHTTPError: Exhausted on an HTTP error status.
            exceptions.DownloadLengthMismatchError: An attempt exceeded
                ``max_length`` or the cumulative byte budget.
        """
        temp_file = self._download_to_file(url, max_length)
        try:
            yield temp_file
        finally:
            temp_file.close()
