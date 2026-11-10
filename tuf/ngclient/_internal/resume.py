# Copyright 2026, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Resumable target downloads.

This module implements optional download resumption for the ngclient.
Partially downloaded target data is kept in a ``<filepath>.part`` file
next to the final destination, with a ``<filepath>.part.json`` sidecar
that binds the partial data to a specific target path, expected length
and all expected hashes. If any of these change (e.g. the target is
replaced by a new version), the partial data is discarded.

Resumption is a transport-level optimization only: the complete file is
always verified against the trusted target metadata (length and all
hashes) before it atomically replaces the final destination file. An
ETag may be used to make resume requests conditional, but is never
trusted as proof of content correctness.

Partial file usability on failure:
  * Download interruption (cancellation or read errors): the partial
    file is kept and a later call can resume from it.
  * Final verification failure: the partial file cannot be trusted and
    is deleted. A previously existing final file (if any) is left
    untouched and remains readable.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tuf.api import exceptions

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tuf.api.metadata import TargetFile
    from tuf.ngclient.fetcher import FetcherInterface

logger = logging.getLogger(__name__)

# How many times an interrupted download is resumed within one call
_MAX_RESUME_ATTEMPTS = 3


@dataclass
class ResumeStats:
    """Statistics of a resumable download.

    Attributes:
        reused_bytes: Bytes taken from the local partial file instead of
            being downloaded again (i.e. bytes of reading saved by
            resuming instead of restarting from scratch).
        downloaded_bytes: Bytes actually read from the remote repository.
    """

    reused_bytes: int = 0
    downloaded_bytes: int = 0


def download_target_file(
    fetcher: FetcherInterface,
    targetinfo: TargetFile,
    url: str,
    filepath: str,
) -> ResumeStats:
    """Download ``url`` to ``filepath``, resuming partial data if possible.

    The final file at ``filepath`` is only replaced (atomically) after
    the downloaded data has been verified against ``targetinfo``.

    Args:
        fetcher: Fetcher used for the download. If it does not support
            range requests, the target is downloaded in full.
        targetinfo: Trusted target metadata used for final verification.
        url: URL to download the target from.
        filepath: Local destination path.

    Raises:
        DownloadError: Download of the target file failed in some way.
        RepositoryError: Downloaded target failed to be verified.
        OSError: Failed to write target to file.

    Returns:
        ``ResumeStats`` describing reused and downloaded bytes.
    """
    part_path = f"{filepath}.part"
    sidecar_path = f"{part_path}.json"
    stats = ResumeStats()

    offset, etag = _load_partial(part_path, sidecar_path, targetinfo)
    _fetch_to_partial(
        fetcher, targetinfo, url, part_path, sidecar_path, offset, etag, stats
    )
    _verify_and_replace(targetinfo, part_path, sidecar_path, filepath)

    if stats.reused_bytes:
        logger.info(
            "Resumed download of %s saved %d bytes of re-downloading "
            "(%d bytes downloaded, %d bytes reused from partial file)",
            targetinfo.path,
            stats.reused_bytes,
            stats.downloaded_bytes,
            stats.reused_bytes,
        )
    return stats


def _fetch_to_partial(
    fetcher: FetcherInterface,
    targetinfo: TargetFile,
    url: str,
    part_path: str,
    sidecar_path: str,
    offset: int,
    etag: str | None,
    stats: ResumeStats,
) -> None:
    """Fetch target data into the partial file until it is complete."""
    attempts = 0
    while offset < targetinfo.length:
        try:
            response = fetcher.fetch_range(
                url, offset, targetinfo.length - 1, etag
            )
        except NotImplementedError:
            # Fetcher does not support range requests: full download
            logger.debug("Fetcher has no range support: full download")
            _download_whole(fetcher, url, targetinfo.length, part_path)
            _write_sidecar(sidecar_path, targetinfo, None)
            stats.downloaded_bytes += os.path.getsize(part_path)
            return
        except exceptions.DownloadHTTPError as e:
            if e.status_code not in (412, 416):
                raise
            # The server rejected the conditional/range request: the
            # partial data cannot be completed from this response.
            # A 416 is only reasonable if we already have every byte,
            # which we do not: discard the partial data and restart.
            logger.debug("Range request rejected (%s): restarting", e)
            attempts = _retry_or_raise(attempts, e)
            offset, etag = _reset_partial(part_path, sidecar_path)
            continue

        if response.status_code == 200:
            # Server ignored the Range request: restart from scratch
            offset = 0
        elif (
            response.total_length is not None
            and response.total_length != targetinfo.length
        ):
            # The remote content does not match the expected length: the
            # partial data cannot be completed from it. Restart.
            logger.debug(
                "Remote length %s does not match expected %d: restarting",
                response.total_length,
                targetinfo.length,
            )
            attempts = _retry_or_raise(
                attempts, exceptions.DownloadError("Remote length changed")
            )
            offset, etag = _reset_partial(part_path, sidecar_path)
            continue
        else:
            # Validated 206 partial response
            stats.reused_bytes += offset
            if offset:
                logger.debug("Resuming %s at byte %d", targetinfo.path, offset)

        if response.etag is not None:
            etag = response.etag

        # Persist the binding (and ETag) before streaming so that an
        # interrupted download can be resumed by a later call
        _write_sidecar(sidecar_path, targetinfo, etag)
        try:
            _stream_to_partial(
                part_path, offset, response.chunks, targetinfo.length, stats
            )
        except exceptions.DownloadError as e:
            attempts = _retry_or_raise(attempts, e)
        offset = _partial_size(part_path)


def _verify_and_replace(
    targetinfo: TargetFile, part_path: str, sidecar_path: str, filepath: str
) -> None:
    """Verify the partial file, then atomically replace the final file."""
    if not os.path.exists(part_path):
        # Zero-length target: nothing was downloaded
        with open(part_path, "wb"):
            pass

    try:
        with open(part_path, "rb") as part_file:
            targetinfo.verify_length_and_hashes(part_file)
    except exceptions.LengthOrHashMismatchError:
        # The partial data cannot be trusted: remove it. The final file
        # (if one exists) is left untouched and remains readable.
        _remove_quietly(part_path)
        _remove_quietly(sidecar_path)
        raise

    # Verification succeeded: atomically replace the final file
    os.replace(part_path, filepath)
    _remove_quietly(sidecar_path)


def _retry_or_raise(attempts: int, error: Exception) -> int:
    """Increment the attempt counter, re-raising ``error`` if exhausted."""
    attempts += 1
    if attempts > _MAX_RESUME_ATTEMPTS:
        raise error
    logger.debug("Download interrupted (%s), resuming", error)
    return attempts


def _reset_partial(part_path: str, sidecar_path: str) -> tuple[int, None]:
    """Discard the partial data so the download restarts from scratch."""
    _remove_quietly(part_path)
    _remove_quietly(sidecar_path)
    return 0, None


def _load_partial(
    part_path: str, sidecar_path: str, targetinfo: TargetFile
) -> tuple[int, str | None]:
    """Return (offset, etag) of reusable partial data, or (0, None).

    Partial data is only reusable if the sidecar binds it to the same
    target path, expected length and all expected hashes as
    ``targetinfo``: a new target version invalidates the partial data.
    """
    try:
        with open(sidecar_path, encoding="utf-8") as sidecar_file:
            state = json.load(sidecar_file)
        if (
            not isinstance(state, dict)
            or state.get("target_path") != targetinfo.path
            or state.get("length") != targetinfo.length
            or state.get("hashes") != dict(targetinfo.hashes)
        ):
            raise ValueError("Partial data does not match targetinfo")
        offset = os.path.getsize(part_path)
        if offset > targetinfo.length:
            raise ValueError("Partial data exceeds expected length")
        etag = state.get("etag")
        return offset, etag if isinstance(etag, str) else None
    except (OSError, ValueError):
        _remove_quietly(part_path)
        _remove_quietly(sidecar_path)
        return 0, None


def _write_sidecar(
    sidecar_path: str, targetinfo: TargetFile, etag: str | None
) -> None:
    """Bind the partial file to the current target path, length, hashes."""
    state = {
        "target_path": targetinfo.path,
        "length": targetinfo.length,
        "hashes": dict(targetinfo.hashes),
        "etag": etag,
    }
    with open(sidecar_path, "w", encoding="utf-8") as sidecar_file:
        json.dump(state, sidecar_file)


def _stream_to_partial(
    part_path: str,
    offset: int,
    chunks: Iterator[bytes],
    length: int,
    stats: ResumeStats,
) -> None:
    """Append ``chunks`` to the partial file, starting at ``offset``.

    Raises:
        exceptions.DownloadLengthMismatchError: More than ``length``
            total bytes were received.
    """
    received = offset
    with open(part_path, "r+b" if offset else "wb") as part_file:
        part_file.seek(offset)
        for chunk in chunks:
            received += len(chunk)
            stats.downloaded_bytes += len(chunk)
            if received > length:
                raise exceptions.DownloadLengthMismatchError(
                    f"Downloaded {received} bytes exceeding the maximum "
                    f"allowed length of {length}"
                )
            part_file.write(chunk)


def _download_whole(
    fetcher: FetcherInterface, url: str, length: int, part_path: str
) -> None:
    """Download all of ``url`` into the partial file in one go."""
    with (
        fetcher.download_file(url, length) as dl_file,
        open(part_path, "wb") as part_file,
    ):
        shutil.copyfileobj(dl_file, part_file)


def _partial_size(part_path: str) -> int:
    """Return the current size of the partial file (0 if missing)."""
    try:
        return os.path.getsize(part_path)
    except OSError:
        return 0


def _remove_quietly(path: str) -> None:
    with contextlib.suppress(OSError):
        os.remove(path)
