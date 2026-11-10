# Copyright 2026, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Test resumable target downloads (Updater.download_target resume=True)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from collections.abc import Iterator
from urllib import parse

from tests import utils
from tests.repository_simulator import RepositorySimulator
from tuf.api.exceptions import (
    DownloadError,
    DownloadHTTPError,
    RepositoryError,
)
from tuf.api.metadata import TargetFile
from tuf.ngclient import Updater
from tuf.ngclient.fetcher import RangeResponse


class RangeSim(RepositorySimulator):
    """In-memory repository simulator with HTTP range request support.

    Only target files are served via range requests: metadata is fetched
    with the regular fetch() like in RepositorySimulator.
    """

    def __init__(self) -> None:
        super().__init__()
        # All range requests made: (start, end, etag)
        self.range_requests: list[tuple[int, int, str | None]] = []
        # Total response body bytes served over the "network"
        self.bytes_served = 0
        self.etag: str | None = '"v1"'
        # If True, respond 412 to If-Match values != self.etag
        self.reject_stale_etag = False
        # Number of times to respond 416 to range requests
        self.respond_416_count = 0
        # If True, ignore Range and respond 200 with the full body
        self.respond_full = False
        # If True, report a wrong Content-Range start
        self.misalign = False
        # If True, send more bytes than the requested range
        self.overlong = False
        # Per-response stream plans, popped by _serve():
        # ("fail", n): yield n bytes then raise DownloadError
        # ("close", n): yield n bytes then cleanly end the stream
        # ("cancel", n): yield n bytes then raise KeyboardInterrupt
        self.stream_plan: list[tuple[str, int]] = []

    def _fetch_range(
        self, url: str, start: int, end: int, etag: str | None = None
    ) -> RangeResponse:
        path = parse.urlparse(url).path
        assert path.startswith("/targets/")
        target_path = path[len("/targets/") :]
        dir_parts, sep, prefixed_filename = target_path.rpartition("/")
        prefix: str | None = None
        filename = prefixed_filename
        if self.root.consistent_snapshot and self.prefix_targets_with_hash:
            prefix, _, filename = prefixed_filename.partition(".")
        target_path = f"{dir_parts}{sep}{filename}"
        data = self.fetch_target(target_path, prefix)

        self.range_requests.append((start, end, etag))

        if self.reject_stale_etag and etag is not None and etag != self.etag:
            raise DownloadHTTPError("Precondition Failed", 412)
        if self.respond_416_count > 0:
            self.respond_416_count -= 1
            raise DownloadHTTPError("Range Not Satisfiable", 416)
        if self.respond_full:
            return RangeResponse(
                status_code=200,
                chunks=self._serve(data),
                total_length=len(data),
                etag=self.etag,
            )

        body = data[start : end + 1]
        if self.overlong:
            body += b"unexpected-extra-data"
        return RangeResponse(
            status_code=206,
            chunks=self._serve(body),
            range_start=start + 1 if self.misalign else start,
            range_end=end,
            total_length=len(data),
            etag=self.etag,
        )

    def _serve(self, data: bytes) -> Iterator[bytes]:
        mode: str | None = None
        limit = len(data)
        if self.stream_plan:
            mode, limit = self.stream_plan.pop(0)
            limit = min(limit, len(data))
        pos = 0
        while pos < limit:
            chunk = data[pos : pos + 100]
            self.bytes_served += len(chunk)
            yield chunk
            pos += len(chunk)
        if limit < len(data):
            if mode == "fail":
                raise DownloadError("simulated connection reset")
            if mode == "cancel":
                raise KeyboardInterrupt("simulated cancellation")


class TestResumeDownload(unittest.TestCase):
    """Test resumable target downloads of the ngclient Updater."""

    dump_dir: str | None = None

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metadata_dir = os.path.join(self.temp_dir.name, "metadata")
        self.targets_dir = os.path.join(self.temp_dir.name, "targets")
        os.mkdir(self.metadata_dir)
        os.mkdir(self.targets_dir)

        self.sim = RangeSim()
        if self.dump_dir is not None:
            name = self.id().split(".")[-1]
            self.sim.dump_dir = os.path.join(self.dump_dir, name)
            os.mkdir(self.sim.dump_dir)

        self.content = bytes(range(256)) * 16  # 4096 bytes
        self.target_path = "targetpath"
        self.filepath = os.path.join(self.targets_dir, self.target_path)
        self.part_path = f"{self.filepath}.part"
        self.sidecar_path = f"{self.part_path}.json"

        self.sim.targets.version += 1
        self.sim.add_target("targets", self.content, self.target_path)
        self.sim.update_snapshot()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _init_updater(self) -> Updater:
        if self.sim.dump_dir is not None:
            self.sim.write()
        return Updater(
            self.metadata_dir,
            "https://example.com/metadata/",
            self.targets_dir,
            "https://example.com/targets/",
            self.sim,
            bootstrap=self.sim.signed_roots[0],
        )

    def _get_targetinfo(self, updater: Updater) -> TargetFile:
        info = updater.get_targetinfo(self.target_path)
        assert info is not None
        return info

    def _publish_new_version(self, content: bytes) -> None:
        self.sim.targets.version += 1
        self.sim.add_target("targets", content, self.target_path)
        self.sim.update_snapshot()

    def test_resume_after_two_interruptions(self) -> None:
        updater = self._init_updater()
        info = self._get_targetinfo(updater)

        # Two interruptions, then a successful final attempt
        self.sim.stream_plan = [("fail", 1000), ("fail", 1000)]
        with self.assertLogs(
            "tuf.ngclient._internal.resume", level="INFO"
        ) as log_cm:
            self.assertEqual(
                self.filepath, updater.download_target(info, resume=True)
            )

        with open(self.filepath, "rb") as f:
            self.assertEqual(f.read(), self.content)

        # Resumed from the partial offsets, with the stored ETag as
        # transport-level assistance
        self.assertEqual(
            self.sim.range_requests,
            [
                (0, 4095, None),
                (1000, 4095, '"v1"'),
                (2000, 4095, '"v1"'),
            ],
        )
        # Every byte was read from the remote exactly once
        self.assertEqual(self.sim.bytes_served, len(self.content))
        # The saved read bytes (1000 + 2000 reused) are recorded
        self.assertTrue(
            any("saved 3000 bytes" in msg for msg in log_cm.output),
            log_cm.output,
        )
        # Partial state is cleaned up after success
        self.assertFalse(os.path.exists(self.part_path))
        self.assertFalse(os.path.exists(self.sidecar_path))

    def test_interrupted_download_resumes_across_calls(self) -> None:
        updater = self._init_updater()
        info = self._get_targetinfo(updater)

        # Read errors exhaust the retry attempts: download fails but the
        # partial file remains usable for a later call
        self.sim.stream_plan = [("fail", 1000)] * 4
        with self.assertRaises(DownloadError):
            updater.download_target(info, resume=True)
        self.assertEqual(os.path.getsize(self.part_path), 4000)
        self.assertTrue(os.path.exists(self.sidecar_path))
        self.assertFalse(os.path.exists(self.filepath))

        # Next call resumes where the previous one stopped
        self.sim.stream_plan = []
        updater.download_target(info, resume=True)
        self.assertEqual(self.sim.range_requests[-1], (4000, 4095, '"v1"'))
        self.assertEqual(self.sim.bytes_served, len(self.content))
        with open(self.filepath, "rb") as f:
            self.assertEqual(f.read(), self.content)

    def test_cancellation_keeps_partial(self) -> None:
        updater = self._init_updater()
        info = self._get_targetinfo(updater)

        self.sim.stream_plan = [("cancel", 1000)]
        with self.assertRaises(KeyboardInterrupt):
            updater.download_target(info, resume=True)
        # Cancellation keeps the partial file usable
        self.assertEqual(os.path.getsize(self.part_path), 1000)
        self.assertTrue(os.path.exists(self.sidecar_path))

        self.sim.stream_plan = []
        updater.download_target(info, resume=True)
        self.assertEqual(self.sim.range_requests[-1], (1000, 4095, '"v1"'))
        with open(self.filepath, "rb") as f:
            self.assertEqual(f.read(), self.content)

    def test_new_target_version_invalidates_partial(self) -> None:
        updater = self._init_updater()
        info = self._get_targetinfo(updater)

        # Leave a partial download of version 1 behind
        self.sim.stream_plan = [("fail", 1000)] * 4
        with self.assertRaises(DownloadError):
            updater.download_target(info, resume=True)
        self.assertEqual(os.path.getsize(self.part_path), 4000)

        # Publish a new version of the target
        new_content = b"new version content" * 100
        self._publish_new_version(new_content)

        updater = self._init_updater()
        info = self._get_targetinfo(updater)
        requests_before = len(self.sim.range_requests)
        updater.download_target(info, resume=True)

        # The stale partial was discarded: restarted from byte 0
        self.assertEqual(self.sim.range_requests[requests_before][0], 0)
        with open(self.filepath, "rb") as f:
            self.assertEqual(f.read(), new_content)
        self.assertFalse(os.path.exists(self.part_path))

    def test_full_response_restarts_download(self) -> None:
        updater = self._init_updater()
        info = self._get_targetinfo(updater)

        # Server ignores the Range header and responds 200: the download
        # restarts from scratch instead of appending
        self.sim.respond_full = True
        self.sim.stream_plan = [("fail", 1000)]
        updater.download_target(info, resume=True)

        self.assertEqual(self.sim.range_requests[-1][0], 1000)
        self.assertEqual(self.sim.bytes_served, 1000 + len(self.content))
        with open(self.filepath, "rb") as f:
            self.assertEqual(f.read(), self.content)

    def test_misaligned_range_rejected(self) -> None:
        updater = self._init_updater()
        info = self._get_targetinfo(updater)

        self.sim.misalign = True
        with self.assertRaises(DownloadError):
            updater.download_target(info, resume=True)
        self.assertFalse(os.path.exists(self.filepath))

    def test_overlong_response_rejected(self) -> None:
        updater = self._init_updater()
        info = self._get_targetinfo(updater)

        self.sim.overlong = True
        with self.assertRaises(DownloadError):
            updater.download_target(info, resume=True)
        self.assertFalse(os.path.exists(self.filepath))

    def test_truncated_stream_is_resumed(self) -> None:
        updater = self._init_updater()
        info = self._get_targetinfo(updater)

        # Server cleanly closes the stream before the range is complete
        self.sim.stream_plan = [("close", 1000)]
        updater.download_target(info, resume=True)

        self.assertEqual(self.sim.range_requests[-1][0], 1000)
        self.assertEqual(self.sim.bytes_served, len(self.content))
        with open(self.filepath, "rb") as f:
            self.assertEqual(f.read(), self.content)

    def test_unreasonable_416_is_not_success(self) -> None:
        updater = self._init_updater()
        info = self._get_targetinfo(updater)

        # A 416 while bytes are still missing must not count as success
        self.sim.respond_416_count = 100
        with self.assertRaises(DownloadError):
            updater.download_target(info, resume=True)
        self.assertFalse(os.path.exists(self.filepath))
        self.assertFalse(os.path.exists(self.part_path))

    def test_416_causes_restart_from_scratch(self) -> None:
        updater = self._init_updater()
        info = self._get_targetinfo(updater)

        self.sim.respond_416_count = 2
        updater.download_target(info, resume=True)

        # Every attempt restarted at byte 0 after a 416
        self.assertEqual([r[0] for r in self.sim.range_requests], [0, 0, 0])
        with open(self.filepath, "rb") as f:
            self.assertEqual(f.read(), self.content)

    def test_stale_etag_causes_restart_from_scratch(self) -> None:
        updater = self._init_updater()
        info = self._get_targetinfo(updater)

        # Interrupt once so a partial file with ETag "v1" is stored
        self.sim.stream_plan = [("fail", 1000)] * 4
        with self.assertRaises(DownloadError):
            updater.download_target(info, resume=True)

        # The remote content now tags differently: If-Match fails with
        # 412 and the client restarts without the stale partial
        self.sim.etag = '"v2"'
        self.sim.reject_stale_etag = True
        self.sim.stream_plan = []
        updater.download_target(info, resume=True)

        self.assertEqual(self.sim.range_requests[-1], (0, 4095, None))
        with open(self.filepath, "rb") as f:
            self.assertEqual(f.read(), self.content)

    def test_corrupt_partial_byte_is_rejected(self) -> None:
        updater = self._init_updater()
        info_v1 = self._get_targetinfo(updater)

        # Version 1 of the target is successfully downloaded
        updater.download_target(info_v1, resume=True)
        with open(self.filepath, "rb") as f:
            self.assertEqual(f.read(), self.content)

        # Publish version 2 with different content
        new_content = b"version two" * 200
        self._publish_new_version(new_content)
        updater = self._init_updater()
        info_v2 = self._get_targetinfo(updater)

        # Plant a complete-looking partial file with one corrupted byte,
        # correctly bound to version 2 (path, length, all hashes, ETag)
        corrupted = bytearray(new_content)
        corrupted[10] ^= 0xFF
        with open(self.part_path, "wb") as f:
            f.write(corrupted)
        with open(self.sidecar_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "target_path": info_v2.path,
                    "length": info_v2.length,
                    "hashes": dict(info_v2.hashes),
                    "etag": self.sim.etag,
                },
                f,
            )

        # Nothing left to download: the partial file goes straight to
        # final verification, which rejects the corrupted byte
        requests_before = len(self.sim.range_requests)
        with self.assertRaises(RepositoryError):
            updater.download_target(info_v2, resume=True)
        self.assertEqual(len(self.sim.range_requests), requests_before)

        # The untrusted partial is deleted ...
        self.assertFalse(os.path.exists(self.part_path))
        self.assertFalse(os.path.exists(self.sidecar_path))
        # ... and the previously downloaded file is untouched and valid
        with open(self.filepath, "rb") as f:
            self.assertEqual(f.read(), self.content)
        self.assertEqual(updater.find_cached_target(info_v1), self.filepath)

    def test_fetcher_without_range_support(self) -> None:
        # Old FetcherInterface implementations (only _fetch) still work:
        # the target is downloaded in full
        sim = RepositorySimulator()
        sim.targets.version += 1
        sim.add_target("targets", self.content, self.target_path)
        sim.update_snapshot()
        updater = Updater(
            self.metadata_dir,
            "https://example.com/metadata/",
            self.targets_dir,
            "https://example.com/targets/",
            sim,
            bootstrap=sim.signed_roots[0],
        )
        info = self._get_targetinfo(updater)
        updater.download_target(info, resume=True)
        with open(self.filepath, "rb") as f:
            self.assertEqual(f.read(), self.content)


if __name__ == "__main__":
    if "--dump" in sys.argv:
        TestResumeDownload.dump_dir = tempfile.mkdtemp()
        print(f"Repository Simulator dumps in {TestResumeDownload.dump_dir}")
        sys.argv.remove("--dump")

    utils.configure_test_logging(sys.argv)
    unittest.main()
