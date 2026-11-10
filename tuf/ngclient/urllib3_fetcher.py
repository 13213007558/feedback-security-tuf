# Copyright 2021, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Provides an implementation of ``FetcherInterface`` using the urllib3 HTTP
library.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

# Imports
import urllib3

import tuf
from tuf.api import exceptions
from tuf.ngclient._internal.proxy import ProxyEnvironment
from tuf.ngclient.fetcher import FetcherInterface, RangeResponse

if TYPE_CHECKING:
    from collections.abc import Iterator

# Globals
logger = logging.getLogger(__name__)

# Content-Range header for a single byte range: "bytes START-END/TOTAL"
# where TOTAL may be "*"
_CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")


# Classes
class Urllib3Fetcher(FetcherInterface):
    """An implementation of ``FetcherInterface`` based on the urllib3 library.

    Attributes:
        socket_timeout: Timeout in seconds, used for both initial connection
            delay and the maximum delay between bytes received.
        chunk_size: Chunk size in bytes used when downloading.
    """

    def __init__(
        self,
        socket_timeout: int = 30,
        chunk_size: int = 400000,
        app_user_agent: str | None = None,
    ) -> None:
        # Default settings
        self.socket_timeout: int = socket_timeout  # seconds
        self.chunk_size: int = chunk_size  # bytes

        # Create User-Agent.
        ua = f"python-tuf/{tuf.__version__}"
        if app_user_agent is not None:
            ua = f"{app_user_agent} {ua}"

        self._proxy_env = ProxyEnvironment(headers={"User-Agent": ua})

    def _fetch(self, url: str) -> Iterator[bytes]:
        """Fetch the contents of HTTP/HTTPS url from a remote server.

        Args:
            url: URL string that represents a file location.

        Raises:
            exceptions.SlowRetrievalError: Timeout occurs while receiving
                data.
            exceptions.DownloadHTTPError: HTTP error code is received.

        Returns:
            Bytes iterator
        """

        # Defer downloading the response body with preload_content=False.
        # Always set the timeout. This timeout value is interpreted by
        # urllib3 as:
        #  - connect timeout (max delay before first byte is received)
        #  - read (gap) timeout (max delay between bytes received)
        try:
            response = self._proxy_env.request(
                "GET",
                url,
                preload_content=False,
                timeout=urllib3.Timeout(self.socket_timeout),
            )
        except urllib3.exceptions.MaxRetryError as e:
            if isinstance(e.reason, urllib3.exceptions.TimeoutError):
                raise exceptions.SlowRetrievalError from e
            # Any other reason (e.g. TLS or connection error): let the original
            # error propagate instead of falling through to an unbound response.
            raise

        if response.status >= 400:
            response.close()
            raise exceptions.DownloadHTTPError(
                f"HTTP error occurred with status {response.status}",
                response.status,
            )

        return self._chunks(response)

    def _chunks(
        self, response: urllib3.response.BaseHTTPResponse
    ) -> Iterator[bytes]:
        """A generator function to be returned by fetch.

        This way the caller of fetch can differentiate between connection
        and actual data download.
        """

        try:
            yield from response.stream(self.chunk_size)
        except urllib3.exceptions.TimeoutError as e:
            # Raised directly when the gap timeout expires mid-stream:
            # ReadTimeoutError is a TimeoutError but not a MaxRetryError.
            raise exceptions.SlowRetrievalError from e
        except urllib3.exceptions.MaxRetryError as e:
            if isinstance(e.reason, urllib3.exceptions.TimeoutError):
                raise exceptions.SlowRetrievalError from e
            # Any other reason: propagate rather than ending the stream
            # silently, which would look like a complete download.
            raise

        finally:
            response.release_conn()

    def _fetch_range(
        self, url: str, start: int, end: int, etag: str | None = None
    ) -> RangeResponse:
        """Fetch the bytes in range [start, end] of url from a remote server.

        Args:
            url: URL string that represents a file location.
            start: First byte position to fetch (0-based, inclusive).
            end: Last byte position to fetch (inclusive).
            etag: Optional ETag to make the request conditional (If-Match).

        Raises:
            exceptions.SlowRetrievalError: Timeout occurs while receiving
                data.
            exceptions.DownloadHTTPError: HTTP error code is received
                (e.g. 412 or 416).
            exceptions.DownloadError: The 206 response lacks a valid
                Content-Range header.

        Returns:
            ``RangeResponse`` with a 206 partial or 200 full response.
        """
        headers = {"Range": f"bytes={start}-{end}"}
        if etag is not None:
            headers["If-Match"] = etag

        try:
            response = self._proxy_env.request(
                "GET",
                url,
                headers=headers,
                preload_content=False,
                timeout=urllib3.Timeout(self.socket_timeout),
            )
        except urllib3.exceptions.MaxRetryError as e:
            if isinstance(e.reason, urllib3.exceptions.TimeoutError):
                raise exceptions.SlowRetrievalError from e
            raise

        if response.status >= 400:
            status = response.status
            response.close()
            raise exceptions.DownloadHTTPError(
                f"HTTP error occurred with status {status}", status
            )

        if response.status not in (200, 206):
            status = response.status
            response.close()
            raise exceptions.DownloadError(
                f"Unexpected status {status} for range request on {url}"
            )

        result = RangeResponse(
            status_code=response.status,
            chunks=self._chunks(response),
            etag=response.headers.get("ETag"),
        )
        if response.status == 200:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                result.total_length = int(content_length)
            return result

        # 206 partial content: parse "Content-Range: bytes START-END/TOTAL"
        content_range = response.headers.get("Content-Range")
        match = (
            None
            if content_range is None
            else _CONTENT_RANGE_RE.fullmatch(content_range.strip())
        )
        if match is None:
            response.close()
            raise exceptions.DownloadError(
                f"Missing or invalid Content-Range header "
                f"{content_range!r} for {url}"
            )
        result.range_start = int(match.group(1))
        result.range_end = int(match.group(2))
        if match.group(3) != "*":
            result.total_length = int(match.group(3))
        return result
