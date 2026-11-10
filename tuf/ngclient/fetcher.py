# Copyright 2021, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Provides an interface for network IO abstraction."""

# Imports
from __future__ import annotations

import abc
import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO

from tuf.api import exceptions

logger = logging.getLogger(__name__)


@dataclass
class RangeResponse:
    """Result of an HTTP range request, see ``FetcherInterface.fetch_range()``.

    Attributes:
        status_code: HTTP status code: 200 (full response: the server
            ignored the range request) or 206 (partial content).
        chunks: Iterator over the response body bytes.
        range_start: First byte position of the returned range (206 only).
        range_end: Last byte position (inclusive) of the range (206 only).
        total_length: Total size of the remote resource, if the server
            reported one (Content-Range total or Content-Length).
        etag: ETag header value if the server provided one. The ETag may
            be used to make later range requests conditional: it is a
            transport-level optimization only and never replaces length
            and hash verification of the downloaded file.
    """

    status_code: int
    chunks: Iterator[bytes]
    range_start: int | None = None
    range_end: int | None = None
    total_length: int | None = None
    etag: str | None = None


# Classes
class FetcherInterface(metaclass=abc.ABCMeta):
    """Defines an interface for abstract network download.

    By providing a concrete implementation of the abstract interface,
    users of the framework can plug-in their preferred/customized
    network stack.

    Implementations of FetcherInterface only need to implement ``_fetch()``.
    The public API of the class is already implemented.
    """

    @abc.abstractmethod
    def _fetch(self, url: str) -> Iterator[bytes]:
        """Fetch the contents of HTTP/HTTPS ``url`` from a remote server.

        Implementations must raise ``DownloadHTTPError`` if they receive
        an HTTP error code.

        Implementations may raise any errors but the ones that are not
        ``DownloadErrors`` will be wrapped in a ``DownloadError`` by
        ``fetch()``.

        Args:
            url: URL string that represents a file location.

        Raises:
            exceptions.DownloadHTTPError: HTTP error code was received.

        Returns:
            Bytes iterator
        """
        raise NotImplementedError  # pragma: no cover

    def fetch(self, url: str) -> Iterator[bytes]:
        """Fetch the contents of HTTP/HTTPS ``url`` from a remote server.

        Args:
            url: URL string that represents a file location.

        Raises:
            exceptions.DownloadError: An error occurred during download.
            exceptions.DownloadHTTPError: An HTTP error code was received.

        Returns:
            Bytes iterator
        """
        # Ensure that fetch() only raises DownloadErrors, regardless of the
        # fetcher implementation
        try:
            return self._fetch(url)
        except exceptions.DownloadError as e:
            raise e
        except Exception as e:
            raise exceptions.DownloadError(f"Failed to download {url}") from e

    def _fetch_range(
        self, url: str, start: int, end: int, etag: str | None = None
    ) -> RangeResponse:
        """Fetch the bytes in range [``start``, ``end``] of ``url``.

        Implementations of this optional method must parse the response
        status and headers into a ``RangeResponse``: ``fetch_range()``
        validates the result against the requested range. Implementations
        must raise ``DownloadHTTPError`` if they receive an HTTP error
        code (e.g. 412 or 416).

        The default implementation raises ``NotImplementedError``: callers
        are expected to fall back to a full download with ``fetch()``.

        Args:
            url: URL string that represents a file location.
            start: First byte position to fetch (0-based, inclusive).
            end: Last byte position to fetch (inclusive).
            etag: Optional ETag to make the request conditional (If-Match).

        Raises:
            NotImplementedError: Range requests are not supported.
            exceptions.DownloadHTTPError: HTTP error code was received.

        Returns:
            ``RangeResponse`` with a 206 partial or 200 full response.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support range requests"
        )

    def fetch_range(
        self, url: str, start: int, end: int, etag: str | None = None
    ) -> RangeResponse:
        """Fetch the bytes in range [``start``, ``end``] of ``url``.

        A 206 partial response is validated against the requested range:
        the returned range must start at ``start``, must not exceed
        [``start``, ``end``] and the reported total length must cover the
        returned range. A 200 full response (server ignored the range
        request) is returned as is: callers are expected to restart the
        download from scratch.

        Args:
            url: URL string that represents a file location.
            start: First byte position to fetch (0-based, inclusive).
            end: Last byte position to fetch (inclusive).
            etag: Optional ETag to make the request conditional (If-Match).

        Raises:
            NotImplementedError: The fetcher does not support ranges.
            exceptions.DownloadError: An error occurred during download or
                the response did not match the requested range.
            exceptions.DownloadHTTPError: An HTTP error code was received.

        Returns:
            Validated ``RangeResponse``.
        """
        try:
            response = self._fetch_range(url, start, end, etag)
        except NotImplementedError:
            raise
        except exceptions.DownloadError as e:
            raise e
        except Exception as e:
            raise exceptions.DownloadError(f"Failed to download {url}") from e

        if response.status_code == 200:
            # Server ignored the Range request: full response
            return response

        if response.status_code != 206:
            raise exceptions.DownloadError(
                f"Unexpected status {response.status_code} "
                f"for range request on {url}"
            )

        # Validate the partial response against the requested range
        if response.range_start != start:
            raise exceptions.DownloadError(
                f"Range response for {url} starts at "
                f"{response.range_start}, expected {start}"
            )
        if (
            response.range_end is None
            or response.range_end < start
            or response.range_end > end
        ):
            raise exceptions.DownloadError(
                f"Range response for {url} ends at {response.range_end}, "
                f"expected a position in [{start}, {end}]"
            )
        if (
            response.total_length is not None
            and response.total_length <= response.range_end
        ):
            raise exceptions.DownloadError(
                f"Range response for {url} reports total length "
                f"{response.total_length} that does not cover the range"
            )
        return response

    @contextmanager
    def download_file(self, url: str, max_length: int) -> Iterator[IO]:
        """Download file from given ``url``.

        It is recommended to use ``download_file()`` within a ``with``
        block to guarantee that allocated file resources will always
        be released even if download fails.

        Args:
            url: URL string that represents the location of the file.
            max_length: Upper bound of file size in bytes.

        Raises:
            exceptions.DownloadError: An error occurred during download.
            exceptions.DownloadLengthMismatchError: Downloaded bytes exceed
                ``max_length``.
            exceptions.DownloadHTTPError: An HTTP error code was received.

        Yields:
            ``TemporaryFile`` object that points to the contents of ``url``.
        """
        logger.debug("Downloading: %s", url)

        number_of_bytes_received = 0

        with tempfile.TemporaryFile() as temp_file:
            chunks = self.fetch(url)
            for chunk in chunks:
                number_of_bytes_received += len(chunk)
                if number_of_bytes_received > max_length:
                    raise exceptions.DownloadLengthMismatchError(
                        f"Downloaded {number_of_bytes_received} bytes exceeding"
                        f" the maximum allowed length of {max_length}"
                    )

                temp_file.write(chunk)

            logger.debug(
                "Downloaded %d out of %d bytes",
                number_of_bytes_received,
                max_length,
            )

            temp_file.seek(0)
            yield temp_file

    def download_bytes(self, url: str, max_length: int) -> bytes:
        """Download bytes from given ``url``.

        Returns the downloaded bytes, otherwise like ``download_file()``.

        Args:
            url: URL string that represents the location of the file.
            max_length: Upper bound of data size in bytes.

        Raises:
            exceptions.DownloadError: An error occurred during download.
            exceptions.DownloadLengthMismatchError: Downloaded bytes exceed
                ``max_length``.
            exceptions.DownloadHTTPError: An HTTP error code was received.

        Returns:
            Content of the file in bytes.
        """
        with self.download_file(url, max_length) as dl_file:
            return dl_file.read()
