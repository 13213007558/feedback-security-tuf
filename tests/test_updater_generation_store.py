# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Test ngclient Updater generational metadata storage.

These tests exercise the optional generation store (``UpdaterConfig(
generation_store=True)``): top-level metadata of one refresh is only
usable as a complete, version-consistent set. Interruptions are injected
at every persistence boundary, then a *new* Updater instance is created
on the same cache directory to verify recovery.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import patch

from tests import utils
from tests.repository_simulator import RepositorySimulator
from tuf.api.exceptions import ExpiredMetadataError
from tuf.api.metadata import Metadata, Snapshot, Targets, Timestamp
from tuf.ngclient import Updater
from tuf.ngclient._internal.generation_store import (
    GenerationStore,
    GenerationStoreError,
)
from tuf.ngclient.config import UpdaterConfig

if TYPE_CHECKING:
    from collections.abc import Callable


class TestGenerationStore(unittest.TestCase):
    """Test crash recovery and migration of the generation store."""

    past_datetime = datetime.now(UTC).replace(microsecond=0) - timedelta(days=5)

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metadata_dir = os.path.join(self.temp_dir.name, "metadata")
        self.targets_dir = os.path.join(self.temp_dir.name, "targets")
        self.sim = RepositorySimulator()
        self.config = UpdaterConfig(generation_store=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # -- helpers --------------------------------------------------------

    def _init_updater(
        self, skip_bootstrap: bool = True, config: UpdaterConfig | None = None
    ) -> Updater:
        """Create a new Updater instance (a 'process restart')."""
        return Updater(
            self.metadata_dir,
            "https://example.com/metadata/",
            self.targets_dir,
            "https://example.com/targets/",
            self.sim,
            config or self.config,
            bootstrap=None if skip_bootstrap else self.sim.signed_roots[0],
        )

    def _run_refresh(self, skip_bootstrap: bool = True) -> Updater:
        updater = self._init_updater(skip_bootstrap)
        updater.refresh()
        return updater

    def _bump_repository(self) -> None:
        """Publish new snapshot, targets and timestamp versions."""
        self.sim.targets.version += 1
        self.sim.update_snapshot()

    def _generation_dir(self, number: int) -> str:
        return os.path.join(self.metadata_dir, "generations", str(number))

    def _current_generation(self) -> int:
        with open(
            os.path.join(self.metadata_dir, "generations", "current")
        ) as f:
            return int(f.read().strip())

    def _generation_files(self, number: int) -> list[str]:
        generation_dir = self._generation_dir(number)
        if not os.path.isdir(generation_dir):
            return []
        return sorted(os.listdir(generation_dir))

    def _generation_version(self, number: int, role: str) -> int:
        md = Metadata.from_file(
            os.path.join(self._generation_dir(number), f"{role}.json")
        )
        return md.signed.version

    def _crash_on_persist(
        self, role: str, *, before: bool
    ) -> Callable[[], None]:
        """Patch GenerationStore.persist to 'kill the process' at a
        persistence boundary: raise KeyboardInterrupt (a BaseException,
        so no cleanup runs) right before or after ``role`` is persisted.
        """
        original = GenerationStore.persist

        def faulty_persist(
            store: GenerationStore, rolename: str, data: bytes
        ) -> None:
            if rolename == role and before:
                raise KeyboardInterrupt(f"simulated crash before {role}")
            original(store, rolename, data)
            if rolename == role and not before:
                raise KeyboardInterrupt(f"simulated crash after {role}")

        return patch.object(GenerationStore, "persist", faulty_persist)

    # -- basic behaviour ------------------------------------------------

    def test_refresh_commits_complete_generation(self) -> None:
        self._run_refresh(skip_bootstrap=False)

        self.assertEqual(self._current_generation(), 1)
        self.assertEqual(
            self._generation_files(1),
            ["snapshot.json", "targets.json", "timestamp.json"],
        )
        for role in (Timestamp.type, Snapshot.type, Targets.type):
            self.assertEqual(self._generation_version(1, role), 1)
        # no flat legacy files remain
        self.assertFalse(
            os.path.exists(os.path.join(self.metadata_dir, "timestamp.json"))
        )

        # A no-change refresh does not create a new generation
        self._run_refresh()
        self.assertEqual(self._current_generation(), 1)

        # A repository change is committed as a new self-contained generation
        self._bump_repository()
        self._run_refresh()
        self.assertEqual(self._current_generation(), 2)
        for role in (Timestamp.type, Snapshot.type, Targets.type):
            self.assertEqual(self._generation_version(2, role), 2)
        # the previously committed generation is kept untouched
        self.assertEqual(self._generation_version(1, Timestamp.type), 1)

    def test_generation_reuses_still_valid_files(self) -> None:
        # repository only re-signs timestamp: snapshot/targets stay valid
        self._run_refresh(skip_bootstrap=False)
        self.sim.update_timestamp()
        self._run_refresh()

        self.assertEqual(self._current_generation(), 2)
        self.assertEqual(self._generation_version(2, Timestamp.type), 2)
        # snapshot and targets were copied forward from generation 1
        self.assertEqual(self._generation_version(2, Snapshot.type), 1)
        self.assertEqual(self._generation_version(2, Targets.type), 1)

    # -- interruption at every persistence boundary ---------------------

    def test_recovery_from_interrupted_refresh(self) -> None:
        # boundaries: before/after each top-level persist and at commit
        cases = [
            ("timestamp", True, []),
            ("timestamp", False, ["timestamp.json"]),
            ("snapshot", False, ["snapshot.json", "timestamp.json"]),
            (
                "targets",
                False,
                ["snapshot.json", "targets.json", "timestamp.json"],
            ),
        ]
        for role, before, expected_leftovers in cases:
            with self.subTest(
                boundary=f"{'before' if before else 'after'} {role}"
            ):
                self.setUp()
                self._run_refresh(skip_bootstrap=False)
                self._bump_repository()

                crash = self._crash_on_persist(role, before=before)
                updater = self._init_updater()
                with crash, self.assertRaises(KeyboardInterrupt):
                    updater.refresh()

                # the interrupted generation 2 exists but is not current
                self.assertEqual(self._current_generation(), 1)
                self.assertEqual(self._generation_files(2), expected_leftovers)

                # recovery: a new Updater selects the complete generation 1
                recovered = self._run_refresh()
                self.assertEqual(recovered._trusted_set.snapshot.version, 2)
                self.assertEqual(self._current_generation(), 2)
                for r in (Timestamp.type, Snapshot.type, Targets.type):
                    self.assertEqual(self._generation_version(2, r), 2)
                self.tearDown()

    def test_crash_before_commit_pointer_flip(self) -> None:
        # All files of generation 2 are written but the process dies
        # before the 'current' pointer is flipped.
        self._run_refresh(skip_bootstrap=False)
        self._bump_repository()

        updater = self._init_updater()
        with (
            patch.object(
                GenerationStore,
                "commit",
                side_effect=KeyboardInterrupt("simulated crash at commit"),
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            updater.refresh()

        # generation 2 is fully written on disk but was never committed:
        # recovery must not select it
        self.assertEqual(self._current_generation(), 1)
        self.assertEqual(
            self._generation_files(2),
            ["snapshot.json", "targets.json", "timestamp.json"],
        )

        recovered = self._run_refresh()
        self.assertEqual(recovered._trusted_set.targets.version, 2)
        self.assertEqual(self._current_generation(), 2)

    def test_replayable_interruption_scenario(self) -> None:
        """Replayable crash: root rotation + interrupted top-level refresh.

        Timeline (deterministic, re-runnable):
          1. refresh commits generation 1 (root v1, all roles v1)
          2. repository rotates root (v2) and publishes new top-level v2
          3. a refresh persists root v2 and generation-2 timestamp.json,
             then the process 'dies' before snapshot.json is written
          4. recovery must keep root v2 (verified roots never regress)
             but select generation 1: generation 2 was never committed,
             so its timestamp v2 must not be combined with the v1
             snapshot/targets of generation 1 (a mixed set)
          5. the next refresh re-downloads and commits a consistent
             generation 2
        """
        # step 1
        self._run_refresh(skip_bootstrap=False)
        self.assertEqual(self._current_generation(), 1)

        # step 2
        self.sim.root.version += 1
        self.sim.publish_root()
        self._bump_repository()

        # step 3: crash after timestamp.json of generation 2 is persisted
        crash = self._crash_on_persist(Timestamp.type, before=False)
        updater = self._init_updater()
        with crash, self.assertRaises(KeyboardInterrupt):
            updater.refresh()

        # root v2 was verified by both root v1 and itself: it is kept
        # even though the top-level refresh failed
        self.assertTrue(
            os.path.exists(
                os.path.join(self.metadata_dir, "root_history", "2.root.json")
            )
        )
        with open(os.path.join(self.metadata_dir, "root.json"), "rb") as f:
            self.assertEqual(Metadata.from_bytes(f.read()).signed.version, 2)

        # generation 2 only contains timestamp.json and is NOT current
        self.assertEqual(self._current_generation(), 1)
        self.assertEqual(self._generation_files(2), ["timestamp.json"])
        self.assertEqual(self._generation_version(2, Timestamp.type), 2)

        # step 4: a new Updater recovers. Root stays at v2 (no rollback
        # to the root v1 era of generation 1), and the selected
        # generation is 1: the half-written generation 2 is ignored, so
        # timestamp v2 is never mixed with snapshot/targets v1.
        recovered = self._init_updater()
        self.assertEqual(recovered._trusted_set.root.version, 2)
        store = GenerationStore(self.metadata_dir)
        self.assertEqual(store.current, 1)
        for role in (Timestamp.type, Snapshot.type, Targets.type):
            self.assertEqual(self._generation_version(1, role), 1)

        # step 5: refresh commits a complete, consistent generation 2
        recovered.refresh()
        self.assertEqual(self._current_generation(), 2)
        self.assertEqual(
            self._generation_files(2),
            ["snapshot.json", "targets.json", "timestamp.json"],
        )
        for role in (Timestamp.type, Snapshot.type, Targets.type):
            self.assertEqual(self._generation_version(2, role), 2)
        self.assertEqual(recovered._trusted_set.root.version, 2)

    # -- migration ------------------------------------------------------

    def test_migration_from_legacy_flat_layout(self) -> None:
        # refresh with the legacy flat layout (generation_store=False)
        legacy_config = UpdaterConfig(generation_store=False)
        updater = self._init_updater(skip_bootstrap=False, config=legacy_config)
        updater.refresh()
        self.assertTrue(
            os.path.exists(os.path.join(self.metadata_dir, "timestamp.json"))
        )

        # first start with the generation store migrates the flat files
        migrated = self._run_refresh()
        self.assertEqual(self._current_generation(), 1)
        self.assertEqual(
            self._generation_files(1),
            ["snapshot.json", "targets.json", "timestamp.json"],
        )
        self.assertFalse(
            os.path.exists(os.path.join(self.metadata_dir, "timestamp.json"))
        )
        self.assertEqual(migrated._trusted_set.targets.version, 1)

        # refreshes after migration commit new generations
        self._bump_repository()
        self._run_refresh()
        self.assertEqual(self._current_generation(), 2)

    def test_migration_rejects_incomplete_legacy_cache(self) -> None:
        # a legacy cache with only some top-level files is the result of
        # an interrupted legacy refresh: refuse to migrate it silently
        os.makedirs(self.metadata_dir)
        for role in (Timestamp.type, Snapshot.type):
            data = self.sim.fetch_metadata(role)
            with open(
                os.path.join(self.metadata_dir, f"{role}.json"), "wb"
            ) as f:
                f.write(data)

        with self.assertRaises(GenerationStoreError) as cm:
            self._init_updater(skip_bootstrap=False)
        self.assertIn("missing", str(cm.exception))

    # -- corrupt store errors -------------------------------------------

    def test_corrupt_generation_pointer(self) -> None:
        self._run_refresh(skip_bootstrap=False)
        pointer = os.path.join(self.metadata_dir, "generations", "current")
        with open(pointer, "w") as f:
            f.write("not-a-number")
        with self.assertRaises(GenerationStoreError):
            self._init_updater()

    def test_pointer_to_missing_generation(self) -> None:
        self._run_refresh(skip_bootstrap=False)
        pointer = os.path.join(self.metadata_dir, "generations", "current")
        with open(pointer, "w") as f:
            f.write("7")
        with self.assertRaises(GenerationStoreError):
            self._init_updater()

    def test_committed_generation_missing_file(self) -> None:
        self._run_refresh(skip_bootstrap=False)
        os.remove(os.path.join(self._generation_dir(1), "snapshot.json"))
        with self.assertRaises(GenerationStoreError) as cm:
            self._init_updater()
        self.assertIn("snapshot.json", str(cm.exception))

    def test_committed_generation_with_mixed_versions(self) -> None:
        # Simulate a torn write in the committed generation: timestamp
        # refers to snapshot v2 but snapshot.json is still v1. Recovery
        # must reject the mixed set with a clear error.
        self._run_refresh(skip_bootstrap=False)
        self._bump_repository()
        timestamp_v2 = self.sim.fetch_metadata(Timestamp.type)
        with open(
            os.path.join(self._generation_dir(1), "timestamp.json"), "wb"
        ) as f:
            f.write(timestamp_v2)

        with self.assertRaises(GenerationStoreError) as cm:
            self._init_updater()
        self.assertIn("mixed set", str(cm.exception))

    def test_committed_generation_with_corrupt_metadata(self) -> None:
        self._run_refresh(skip_bootstrap=False)
        with open(
            os.path.join(self._generation_dir(1), "targets.json"), "wb"
        ) as f:
            f.write(b"not json")
        with self.assertRaises(GenerationStoreError):
            self._init_updater()

    # -- delegated roles --------------------------------------------------

    def test_delegated_roles_follow_committed_snapshot(self) -> None:
        from tuf.api.metadata import DelegatedRole

        role = DelegatedRole("role1", [], 1, False, ["*"])
        self.sim.add_delegation(
            Targets.type, role, Targets(expires=self.sim.safe_expiry)
        )
        self.sim.add_target("role1", b"content", "file1")
        self.sim.update_snapshot()

        updater = self._init_updater(skip_bootstrap=False)
        info = updater.get_targetinfo("file1")
        self.assertIsNotNone(info)
        # the delegated role was persisted into the committed generation
        self.assertIn("role1.json", self._generation_files(1))

        # a restarted Updater uses the cached delegated metadata: it
        # belongs to the same generation, hence to the current snapshot
        self.sim.fetch_tracker.metadata.clear()
        restarted = self._init_updater()
        info = restarted.get_targetinfo("file1")
        self.assertIsNotNone(info)
        fetched_roles = [role for role, _ in self.sim.fetch_tracker.metadata]
        self.assertNotIn("role1", fetched_roles)

    # -- expiry is not masked -------------------------------------------

    def test_expired_committed_timestamp_is_not_served(self) -> None:
        # A failed refresh must not be hidden by serving the expired
        # committed generation: recovery surfaces ExpiredMetadataError.
        self._run_refresh(skip_bootstrap=False)
        self.sim.timestamp.expires = self.past_datetime
        self.sim.update_timestamp()

        with self.assertRaises(ExpiredMetadataError):
            self._run_refresh()


if __name__ == "__main__":
    utils.configure_test_logging(sys.argv)
    unittest.main()
