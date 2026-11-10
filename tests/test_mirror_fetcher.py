# Copyright 2026, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Tests for the composable mirror fetcher.

The tests drive MirrorFetcher with scripted fake fetchers: no real network
connections are made. They cover retry after timeout, mirror-specific
content, redirect containment, prefix escape attempts, retry exhaustion,
per-attempt length handling, sanitized attempt records and the interaction
with Updater's root probing workflow.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import urllib3
from urllib3.connectionpool import ConnectionPool

from tests.repository_simulator import RepositorySimulator
from tuf.api import exceptions
from tuf.ngclient import (
    AttemptRecord,
    CrossOriginRedirectError,
    Mirror,
    MirrorFetcher,
    Updater,
)
from tuf.ngclient._internal.proxy import ProxyEnvironment
from tuf.ngclient.mirror_fetcher import (
    FAIL_HTTP_STATUS,
    FAIL_LENGTH,
    FAIL_TIMEOUT,
    _OriginBoundFetcher,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

MIRROR_A = "https://mirror-a.example/tuf/"
MIRROR_B = "https://mirror-b.example/tuf/"
LOGICAL_FILE = "metadata/timestamp.json"
URL_A = f"{MIRROR_A}{LOGICAL_FILE}"
URL_B = f"{MIRROR_B}{LOGICAL_FILE}"


class ScriptedFetcher:
    """Fetcher that replays a scripted sequence of responses or errors."""

    def __init__(self, script: list[Exception | bytes | list[bytes]]) -> None:
        self._script = list(script)
        self.requested_urls: list[str] = []

    def fetch(self, url: str) -> Iterator[bytes]:
        self.requested_urls.append(url)
        action = self._script.pop(0)
        if isinstance(action, Exception):
            raise action
        if isinstance(action, list):
            yield from action
            return
        yield action


def make_fetcher(
    script: list[Exception | bytes | list[bytes]],
    **kwargs: object,
) -> tuple[MirrorFetcher, ScriptedFetcher]:
    inner = ScriptedFetcher(script)
    mirrors = [Mirror(MIRROR_A), Mirror(MIRROR_B)]
    return MirrorFetcher(mirrors, inner_fetcher=inner, **kwargs), inner


class TestMirrorFetcher(unittest.TestCase):
    def test_timeout_then_success_on_second_mirror(self) -> None:
        fetcher, inner = make_fetcher(
            [exceptions.SlowRetrievalError("timeout"), b"fresh"]
        )

        data = fetcher.download_bytes(LOGICAL_FILE, 100)

        self.assertEqual(data, b"fresh")
        self.assertEqual(inner.requested_urls, [URL_A, URL_B])
        self.assertEqual(
            fetcher.attempt_history,
            (
                AttemptRecord("mirror-a.example", LOGICAL_FILE, FAIL_TIMEOUT),
                AttemptRecord("mirror-b.example", LOGICAL_FILE, None),
            ),
        )

    def test_connection_error_then_success(self) -> None:
        fetcher, inner = make_fetcher(
            [exceptions.DownloadError("connection refused"), b"fresh"]
        )

        data = fetcher.download_bytes(LOGICAL_FILE, 100)

        self.assertEqual(data, b"fresh")
        self.assertEqual(inner.requested_urls, [URL_A, URL_B])
        self.assertIsNone(fetcher.attempt_history[1].failure)

    def test_mirrors_serve_different_versions(self) -> None:
        # Mirror A is unreachable, mirror B serves its newer copy: rotation
        # makes the refresh succeed instead of failing on the first mirror.
        fetcher, inner = make_fetcher(
            [
                exceptions.SlowRetrievalError("timeout"),
                b'{"version": 2}',
            ]
        )

        data = fetcher.download_bytes(LOGICAL_FILE, 100)
        self.assertEqual(data, b'{"version": 2}')
        self.assertEqual(inner.requested_urls, [URL_A, URL_B])

        # A second logical file starts rotation from mirror A again and
        # receives that mirror's (older) content.
        inner._script = [b'{"version": 1}']
        data = fetcher.download_bytes("metadata/snapshot.json", 100)
        self.assertEqual(data, b'{"version": 1}')
        self.assertEqual(
            inner.requested_urls[-1],
            f"{MIRROR_A}metadata/snapshot.json",
        )

    def test_404_is_retried_and_exhaustion_preserves_status(self) -> None:
        fetcher, inner = make_fetcher(
            [exceptions.DownloadHTTPError("missing", 404)] * 4
        )

        with self.assertRaises(exceptions.DownloadHTTPError) as ctx:
            fetcher.download_bytes(LOGICAL_FILE, 100)
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(len(inner.requested_urls), 4)
        self.assertTrue(
            all(
                record.failure == FAIL_HTTP_STATUS and record.status_code == 404
                for record in fetcher.attempt_history
            )
        )

    def test_non_retryable_status_fails_fast(self) -> None:
        fetcher, inner = make_fetcher(
            [exceptions.DownloadHTTPError("forbidden", 403)]
        )

        with self.assertRaises(exceptions.DownloadHTTPError) as ctx:
            fetcher.download_bytes(LOGICAL_FILE, 100)
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(len(inner.requested_urls), 1)

    def test_retryable_status_is_configurable(self) -> None:
        fetcher, inner = make_fetcher(
            [exceptions.DownloadHTTPError("busy", 503), b"ok"],
            retry_on_status=frozenset({503}),
        )

        self.assertEqual(fetcher.download_bytes(LOGICAL_FILE, 100), b"ok")
        self.assertEqual(inner.requested_urls, [URL_A, URL_B])

    def test_max_attempts_bounds_retries(self) -> None:
        fetcher, inner = make_fetcher(
            [exceptions.SlowRetrievalError("timeout")] * 4,
            max_attempts=4,
        )

        with self.assertRaises(exceptions.SlowRetrievalError):
            fetcher.download_bytes(LOGICAL_FILE, 100)
        self.assertEqual(len(inner.requested_urls), 4)
        # Two mirrors are contacted round-robin, two attempts each.
        self.assertEqual(inner.requested_urls.count(URL_A), 2)
        self.assertEqual(inner.requested_urls.count(URL_B), 2)

    def test_partial_response_is_discarded_between_attempts(self) -> None:
        class SwitchingFetcher:
            def __init__(self) -> None:
                self.urls: list[str] = []
                self.attempt = 0

            def fetch(self, url: str) -> Iterator[bytes]:
                self.urls.append(url)
                self.attempt += 1
                yield b"partial-"
                if self.attempt == 1:
                    raise exceptions.SlowRetrievalError("timeout")

        switching = SwitchingFetcher()
        fetcher = MirrorFetcher(
            [Mirror(MIRROR_A), Mirror(MIRROR_B)], inner_fetcher=switching
        )

        data = fetcher.download_bytes(LOGICAL_FILE, 100)
        # The first attempt's "partial-" fragment must not be prepended.
        self.assertEqual(data, b"partial-")
        self.assertEqual(switching.urls, [URL_A, URL_B])
        self.assertEqual(
            [record.failure for record in fetcher.attempt_history],
            [FAIL_TIMEOUT, None],
        )

    def test_each_attempt_enforces_caller_max_length(self) -> None:
        fetcher, inner = make_fetcher(
            [[b"abcdefghij", b"k"]],
            max_attempts=2,
        )

        with self.assertRaises(exceptions.DownloadLengthMismatchError):
            fetcher.download_bytes(LOGICAL_FILE, 10)
        # A length violation is a security failure: no mirror rotation hides
        # it and no fragments carry over to another attempt.
        self.assertEqual(len(inner.requested_urls), 1)
        self.assertEqual(
            fetcher.attempt_history[0].failure,
            FAIL_LENGTH,
        )

    def test_cumulative_byte_budget(self) -> None:
        class BudgetFetcher:
            def __init__(self) -> None:
                self.calls = 0

            def fetch(self, _url: str) -> Iterator[bytes]:
                self.calls += 1
                if self.calls == 1:
                    yield b"aaaa"
                    raise exceptions.SlowRetrievalError("timeout")
                # Budget after attempt 1: 3 bytes left.
                yield b"bbbb"

        budget = BudgetFetcher()
        fetcher = MirrorFetcher(
            [Mirror(MIRROR_A), Mirror(MIRROR_B)],
            inner_fetcher=budget,
            max_attempts=2,
            total_byte_budget=7,
        )

        with self.assertRaises(exceptions.DownloadLengthMismatchError):
            fetcher.download_bytes(LOGICAL_FILE, 100)
        self.assertEqual(budget.calls, 2)

    def test_absolute_url_authority_is_ignored(self) -> None:
        fetcher, inner = make_fetcher([b"ok"])

        data = fetcher.download_bytes(
            "https://attacker.example/metadata/timestamp.json", 100
        )
        self.assertEqual(data, b"ok")
        # The configured mirror host replaces the URL authority while the
        # path stays contained inside the mirror prefix.
        self.assertEqual(inner.requested_urls, [URL_A])

    def test_traversal_paths_are_rejected(self) -> None:
        fetcher, inner = make_fetcher([b"ok"])
        malicious = [
            "../escape.json",
            "a/../../escape.json",
            "a/..%2f..%2fescape.json",
            "a%5c..%5cescape.json",
            "%2e%2e/escape.json",
            "dir/./../../x",
        ]

        for path in malicious:
            with self.subTest(path=path):
                with self.assertRaises(exceptions.DownloadError):
                    fetcher.download_bytes(path, 100)
                self.assertEqual(
                    fetcher.attempt_history[0].failure, "invalid_request"
                )
        self.assertEqual(inner.requested_urls, [])

    def test_non_http_schemes_are_rejected(self) -> None:
        fetcher, inner = make_fetcher([b"ok"])

        for url in ["file:///etc/passwd", "ftp://mirror/x.json"]:
            with (
                self.subTest(url=url),
                self.assertRaises(exceptions.DownloadError),
            ):
                fetcher.download_bytes(url, 100)
        self.assertEqual(inner.requested_urls, [])

    def test_query_string_is_preserved_but_not_logged(self) -> None:
        fetcher, inner = make_fetcher([b"ok"])

        fetcher.download_bytes(f"{LOGICAL_FILE}?token=secret", 100)

        self.assertEqual(inner.requested_urls, [f"{URL_A}?token=secret"])
        record = fetcher.attempt_history[0]
        self.assertEqual(record.logical_file, LOGICAL_FILE)
        self.assertNotIn("secret", str(record))
        self.assertNotIn("token", str(record))

    def test_attempt_history_only_holds_safe_fields(self) -> None:
        fetcher, _ = make_fetcher(
            [exceptions.SlowRetrievalError("timeout"), b"ok"]
        )
        fetcher.download_bytes(LOGICAL_FILE, 100)

        for record in fetcher.attempt_history:
            self.assertEqual(
                set(record.__dataclass_fields__),
                {"mirror", "logical_file", "failure", "status_code"},
            )
            self.assertNotIn("Authorization", str(record))

    def test_history_resets_per_call(self) -> None:
        fetcher, _ = make_fetcher([b"ok", b"ok2"])
        fetcher.download_bytes("first.json", 100)
        fetcher.download_bytes("second.json", 100)

        self.assertEqual(len(fetcher.attempt_history), 1)
        self.assertEqual(fetcher.attempt_history[0].logical_file, "second.json")

    def test_invalid_mirror_configuration(self) -> None:
        with self.assertRaises(ValueError):
            MirrorFetcher([])
        with self.assertRaises(ValueError):
            MirrorFetcher([Mirror("file:///tmp/repo/")])
        with self.assertRaises(ValueError):
            MirrorFetcher([Mirror("https://user:pass@mirror-a.example/tuf/")])
        with self.assertRaises(ValueError):
            MirrorFetcher([Mirror("https://mirror-a.example/tuf/?x=1")])
        with self.assertRaises(ValueError):
            MirrorFetcher(
                [
                    Mirror("https://mirror-a.example/tuf/", "same"),
                    Mirror("https://mirror-b.example/tuf/", "same"),
                ]
            )

    def test_cross_origin_redirect_error_is_not_retried(self) -> None:
        class RedirectFetcher:
            def __init__(self) -> None:
                self.urls: list[str] = []

            def fetch(self, url: str) -> Iterator[bytes]:
                self.urls.append(url)
                raise CrossOriginRedirectError("cross origin")

        redirect_fetcher = RedirectFetcher()
        fetcher = MirrorFetcher(
            [Mirror(MIRROR_A), Mirror(MIRROR_B)],
            inner_fetcher=redirect_fetcher,
        )

        with self.assertRaises(CrossOriginRedirectError):
            fetcher.download_bytes(LOGICAL_FILE, 100)
        # Cross-origin redirects are a containment failure, not a transient
        # mirror problem: the second mirror is never contacted.
        self.assertEqual(redirect_fetcher.urls, [URL_A])

    def test_protocol_relative_url_cannot_introduce_host(self) -> None:
        fetcher, inner = make_fetcher([b"ok"])

        data = fetcher.download_bytes("//evil.example/metadata/x.json", 100)

        self.assertEqual(data, b"ok")
        self.assertEqual(
            inner.requested_urls,
            ["https://mirror-a.example/tuf/metadata/x.json"],
        )

    def test_custom_mirror_identifier_is_logged(self) -> None:
        inner = ScriptedFetcher([b"ok"])
        fetcher = MirrorFetcher(
            [Mirror(MIRROR_A, "primary-dc")],
            inner_fetcher=inner,
        )

        fetcher.download_bytes(LOGICAL_FILE, 100)
        self.assertEqual(fetcher.attempt_history[0].mirror, "primary-dc")

    def test_single_mirror_success(self) -> None:
        inner = ScriptedFetcher([b"solo"])
        fetcher = MirrorFetcher([Mirror(MIRROR_A)], inner_fetcher=inner)

        self.assertEqual(fetcher.download_bytes(LOGICAL_FILE, 100), b"solo")
        self.assertEqual(fetcher.attempt_history[0].failure, None)

    def test_fragment_is_stripped(self) -> None:
        fetcher, inner = make_fetcher([b"ok"])

        fetcher.download_bytes(f"{LOGICAL_FILE}#section", 100)

        self.assertEqual(inner.requested_urls, [URL_A])


class TestOriginBoundFetcher(unittest.TestCase):
    def _make_fetcher(
        self, allow_cross_origin_redirects: bool = False
    ) -> _OriginBoundFetcher:
        mirror = Mirror("https://mirror-a.example/tuf/")
        return _OriginBoundFetcher(
            mirror,
            socket_timeout=30,
            chunk_size=400_000,
            allow_cross_origin_redirects=allow_cross_origin_redirects,
            app_user_agent=None,
        )

    def _redirect_response(self, status: int, location: str) -> Mock:
        response = Mock()
        response.status = status
        response.get_redirect_location.return_value = location
        response.close = Mock()
        return response

    @patch.object(ProxyEnvironment, "request")
    def test_same_origin_redirect_is_followed(self, mock_request: Mock) -> None:
        final = Mock()
        final.status = 200
        final.stream.return_value = iter([b"after-redirect"])
        final.release_conn = Mock()

        mock_request.side_effect = [
            self._redirect_response(
                302, "https://mirror-a.example/tuf/other.json"
            ),
            final,
        ]

        fetcher = self._make_fetcher()
        data = b"".join(fetcher.fetch("https://mirror-a.example/tuf/x.json"))
        self.assertEqual(data, b"after-redirect")
        self.assertEqual(mock_request.call_count, 2)

    @patch.object(ProxyEnvironment, "request")
    def test_cross_origin_redirect_rejected_by_default(
        self, mock_request: Mock
    ) -> None:
        mock_request.return_value = self._redirect_response(
            302, "https://evil.example/steal"
        )

        fetcher = self._make_fetcher()
        with self.assertRaises(CrossOriginRedirectError):
            fetcher.fetch("https://mirror-a.example/tuf/x.json")

    @patch.object(ProxyEnvironment, "request")
    def test_cross_origin_redirect_opt_in(self, mock_request: Mock) -> None:
        final = Mock()
        final.status = 200
        final.stream.return_value = iter([b"ok"])
        final.release_conn = Mock()

        mock_request.side_effect = [
            self._redirect_response(302, "https://evil.example/steal"),
            final,
        ]

        fetcher = self._make_fetcher(allow_cross_origin_redirects=True)
        data = b"".join(fetcher.fetch("https://mirror-a.example/tuf/x.json"))
        self.assertEqual(data, b"ok")

    @patch.object(ProxyEnvironment, "request")
    def test_redirect_to_disallowed_scheme(self, mock_request: Mock) -> None:
        mock_request.return_value = self._redirect_response(
            302, "file:///etc/passwd"
        )

        fetcher = self._make_fetcher(allow_cross_origin_redirects=True)
        with self.assertRaises(CrossOriginRedirectError):
            fetcher.fetch("https://mirror-a.example/tuf/x.json")

    @patch.object(ProxyEnvironment, "request")
    def test_connect_timeout_maps_to_slow_retrieval(
        self, mock_request: Mock
    ) -> None:
        mock_request.side_effect = urllib3.exceptions.MaxRetryError(
            ConnectionPool("mirror-a.example"),
            "https://mirror-a.example/tuf/x.json",
            urllib3.exceptions.TimeoutError(),
        )

        fetcher = self._make_fetcher()
        with self.assertRaises(exceptions.SlowRetrievalError):
            fetcher.fetch("https://mirror-a.example/tuf/x.json")

    def test_mirror_fetcher_creates_origin_bound_fetchers(self) -> None:
        # Without an injected inner fetcher, MirrorFetcher builds one
        # origin-bound urllib3 fetcher per contacted mirror and reuses it.
        with patch.object(ProxyEnvironment, "request") as mock_request:

            def make_response(*_args: object, **_kwargs: object) -> Mock:
                response = Mock()
                response.status = 200
                response.stream.return_value = iter([b"default-path"])
                response.release_conn = Mock()
                return response

            mock_request.side_effect = make_response

            fetcher = MirrorFetcher([Mirror(MIRROR_A), Mirror(MIRROR_B)])
            first = fetcher.download_bytes(LOGICAL_FILE, 100)
            second = fetcher.download_bytes("other.json", 100)

            self.assertEqual(first, b"default-path")
            self.assertEqual(second, b"default-path")
            # Mirror A answers both calls, so only its fetcher is created,
            # once, and is reused across the two logical files.
            self.assertEqual(set(fetcher._inner_fetchers), {"mirror-a.example"})
            self.assertEqual(mock_request.call_count, 2)
            for call in mock_request.call_args_list:
                self.assertTrue(call.args[1].startswith(MIRROR_A))


class TestUpdaterRootRotation(unittest.TestCase):
    """Updater-level behavior with two mirrors of different content."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metadata_dir = os.path.join(self.temp_dir.name, "metadata")
        self.targets_dir = os.path.join(self.temp_dir.name, "targets")
        os.makedirs(self.metadata_dir, exist_ok=True)

        # Mirror A hosts root versions 1 and 2; mirror B only has version 1.
        self.sim_a = RepositorySimulator()
        self.sim_a.root.version += 1
        self.sim_a.publish_root()
        self.sim_b = RepositorySimulator()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _routing_fetcher(self) -> MirrorFetcher:
        class RoutingFetcher:
            def __init__(self, sims: dict[str, RepositorySimulator]):
                self.sims = sims

            def fetch(self, url: str) -> Iterator[bytes]:
                host = url.split("/")[2]
                return self.sims[host].fetch(url)

        inner = RoutingFetcher(
            {
                "mirror-a.example": self.sim_a,
                "mirror-b.example": self.sim_b,
            }
        )
        return MirrorFetcher(
            [
                Mirror("https://mirror-a.example/"),
                Mirror("https://mirror-b.example/"),
            ],
            inner_fetcher=inner,
            # 404 must stay retryable so one mirror missing a root version
            # does not end the rotation early.
            max_attempts=4,
        )

    def test_root_probe_continues_after_mirror_404(self) -> None:
        updater = Updater(
            self.metadata_dir,
            "https://unused.invalid/metadata/",
            self.targets_dir,
            "https://unused.invalid/targets/",
            self._routing_fetcher(),
            bootstrap=self.sim_a.signed_roots[0],
        )
        updater.refresh()

        root_path = os.path.join(
            self.metadata_dir, "root_history", "2.root.json"
        )
        self.assertTrue(os.path.exists(root_path))
        self.assertEqual(updater._trusted_set.root.version, 2)

    def test_invalid_downloaded_metadata_is_security_failure(self) -> None:
        class TamperedFetcher:
            def __init__(self, simulator: RepositorySimulator):
                self.sim = simulator
                self.calls = 0

            def fetch(self, url: str) -> Iterator[bytes]:
                self.calls += 1
                if url.endswith("/metadata/timestamp.json"):
                    yield b"this is not signed metadata"
                    return
                yield from self.sim.fetch(url)

        fetcher = MirrorFetcher(
            [Mirror("https://mirror-a.example/")],
            inner_fetcher=TamperedFetcher(self.sim_a),
            max_attempts=4,
        )
        updater = Updater(
            self.metadata_dir,
            "https://mirror-a.example/metadata/",
            self.targets_dir,
            "https://mirror-a.example/targets/",
            fetcher,
            bootstrap=self.sim_a.signed_roots[0],
        )

        with self.assertRaises(exceptions.RepositoryError):
            updater.refresh()
        # The invalid payload is surfaced to the client layer: the fetcher
        # does not rotate mirrors to hide the verification failure.
        self.assertEqual(
            sum(
                record.logical_file == "metadata/timestamp.json"
                for record in fetcher.attempt_history
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
