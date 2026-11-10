# Copyright 2026, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Test structured lookup traces produced by ``Updater.get_targetinfo``.

The tests cover, on top of the existing delegation graph fixtures:
same-name targets in several roles, shared sub-roles, cycles, succinct roles,
path pattern / hash prefix misses, terminating truncation, the traversal
limit, download and signature failures, trace immutability and behavior
parity between traced and untraced lookups.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import astuple
from typing import TYPE_CHECKING

from tests.repository_simulator import RepositorySimulator
from tests.test_updater_delegation_graphs import (
    DelegationsTestCase,
    TestDelegation,
    TestTarget,
)
from tuf.api.exceptions import (
    DownloadHTTPError,
    UnsignedMetadataError,
)
from tuf.ngclient import (
    LookupOutcome,
    TargetLookupTrace,
    TraceEvent,
    TraceEventType,
    Updater,
)

if TYPE_CHECKING:
    from tuf.ngclient.fetcher import FetcherInterface


class FailingSimulator(RepositorySimulator):
    """Simulator whose metadata download for one role always fails."""

    def __init__(self, failing_role: str) -> None:
        super().__init__()
        self._failing_role = failing_role

    def fetch_metadata(self, role: str, version: int | None = None) -> bytes:
        if role == self._failing_role:
            raise DownloadHTTPError(f"role {role} unavailable", 404)
        return super().fetch_metadata(role, version)


class TestTargetLookupTrace(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metadata_dir = os.path.join(self.temp_dir.name, "metadata")
        self.targets_dir = os.path.join(self.temp_dir.name, "targets")
        os.mkdir(self.metadata_dir)
        os.mkdir(self.targets_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _init_repo(
        self,
        test_case: DelegationsTestCase,
        sim: RepositorySimulator | None = None,
    ) -> RepositorySimulator:
        sim = sim or RepositorySimulator()
        from tuf.api.metadata import (
            SPECIFICATION_VERSION,
            DelegatedRole,
            Targets,
        )

        for d in test_case.delegations:
            if d.rolename in sim.md_delegates:
                targets = sim.md_delegates[d.rolename].signed
            else:
                targets = Targets(
                    1,
                    ".".join(SPECIFICATION_VERSION),
                    sim.safe_expiry,
                    {},
                    None,
                )
            role = DelegatedRole(*astuple(d)[1:])
            sim.add_delegation(d.delegator, role, targets)

        for target in test_case.target_files:
            sim.add_target(*astuple(target))

        if test_case.target_files:
            sim.targets.version += 1
        sim.update_snapshot()
        return sim

    def _updater(
        self, sim: RepositorySimulator, fetcher: FetcherInterface | None = None
    ) -> Updater:
        return Updater(
            self.metadata_dir,
            "https://example.com/metadata/",
            self.targets_dir,
            "https://example.com/targets/",
            fetcher or sim,
            bootstrap=sim.signed_roots[0],
        )

    @staticmethod
    def _types(trace: TargetLookupTrace) -> list[TraceEventType]:
        return [event.event_type for event in trace.events]

    @staticmethod
    def _event(
        trace: TargetLookupTrace, event_type: TraceEventType, role: str
    ) -> TraceEvent:
        return next(
            event
            for event in trace.events
            if event.event_type is event_type and event.role == role
        )

    def test_same_name_target_picks_most_trusted_role(self) -> None:
        # The same target exists in targets, A and B: preorder DFS must pick
        # the first verified role and the trace must explain why.
        case = DelegationsTestCase(
            delegations=[
                TestDelegation("targets", "A"),
                TestDelegation("targets", "B"),
            ],
            target_files=[
                TestTarget("targets", b"top", "shared.txt"),
                TestTarget("A", b"by-a", "shared.txt"),
                TestTarget("B", b"by-b", "shared.txt"),
            ],
        )
        sim = self._init_repo(case)
        updater = self._updater(sim)

        trace = TargetLookupTrace()
        target = updater.get_targetinfo("shared.txt", trace)

        self.assertIsNotNone(target)
        self.assertEqual(trace.outcome, LookupOutcome.TARGET_FOUND)
        self.assertEqual(trace.found_in_role, "targets")
        self.assertEqual(trace.trusted_path, ("targets",))
        # A and B were never visited because the top-level role matched: the
        # only verified edge is the root-anchored top-level targets role.
        verified = [
            event.role
            for event in trace.events
            if event.event_type is TraceEventType.ROLE_VERIFIED
        ]
        self.assertEqual(verified, ["targets"])

        # A lookup that does not match the top-level role continues into A.
        trace = TargetLookupTrace()
        target = updater.get_targetinfo("only-in-a", trace)
        self.assertIsNone(target)
        verified = [
            event.role
            for event in trace.events
            if event.event_type is TraceEventType.ROLE_VERIFIED
        ]
        self.assertEqual(verified, ["targets", "A", "B"])

    def test_delegated_same_name_target_trusted_path(self) -> None:
        case = DelegationsTestCase(
            delegations=[
                TestDelegation("targets", "A"),
                TestDelegation("targets", "B"),
                TestDelegation("A", "C"),
            ],
            target_files=[
                TestTarget("C", b"c", "x"),
                TestTarget("B", b"b", "x"),
            ],
        )
        sim = self._init_repo(case)
        updater = self._updater(sim)
        updater.refresh()

        trace = TargetLookupTrace()
        target = updater.get_targetinfo("x", trace)
        self.assertIsNotNone(target)
        self.assertEqual(trace.found_in_role, "C")
        self.assertEqual(trace.trusted_path, ("targets", "A", "C"))
        # B is a later sibling and is never reached once C provides the target.
        self.assertNotIn(
            "B",
            [
                event.role
                for event in trace.events
                if event.event_type is TraceEventType.ROLE_VERIFIED
            ],
        )

    def test_path_pattern_and_hash_prefix_misses(self) -> None:
        case = DelegationsTestCase(
            delegations=[
                TestDelegation("targets", "pyrole", paths=["*.py"]),
                TestDelegation(
                    "targets",
                    "binrole",
                    paths=None,
                    path_hash_prefixes=["00"],
                ),
                TestDelegation("targets", "wildcard"),
            ],
        )
        sim = self._init_repo(case)
        updater = self._updater(sim)
        updater.refresh()

        trace = TargetLookupTrace()
        self.assertIsNone(updater.get_targetinfo("notes.txt", trace))

        pattern_miss = self._event(
            trace, TraceEventType.PATH_PATTERN_MISS, "pyrole"
        )
        self.assertEqual(pattern_miss.parent_role, "targets")
        self.assertEqual(pattern_miss.condition.kind, "paths")
        self.assertEqual(pattern_miss.condition.paths, ("*.py",))
        self.assertFalse(pattern_miss.trusted)

        hash_miss = self._event(
            trace, TraceEventType.HASH_PREFIX_MISS, "binrole"
        )
        self.assertEqual(hash_miss.condition.kind, "path_hash_prefixes")
        self.assertEqual(hash_miss.condition.path_hash_prefixes, ("00",))

        # Misses never produce trusted edges or downloads.
        for event in trace.events:
            if event.event_type in (
                TraceEventType.PATH_PATTERN_MISS,
                TraceEventType.HASH_PREFIX_MISS,
            ):
                self.assertFalse(event.trusted)
        verified = [
            event.role
            for event in trace.events
            if event.event_type is TraceEventType.ROLE_VERIFIED
        ]
        self.assertEqual(verified, ["targets", "wildcard"])
        self.assertEqual(trace.outcome, LookupOutcome.NOT_FOUND_EXHAUSTED)

    def test_hash_prefix_match_records_adopted_prefix(self) -> None:
        import hashlib

        prefix = hashlib.sha256(b"hashed").hexdigest()[:2]
        case = DelegationsTestCase(
            delegations=[
                TestDelegation(
                    "targets",
                    "binrole",
                    paths=None,
                    path_hash_prefixes=[prefix],
                ),
            ]
        )
        sim = self._init_repo(case)
        updater = self._updater(sim)
        updater.refresh()

        trace = TargetLookupTrace()
        self.assertIsNone(updater.get_targetinfo("hashed", trace))
        verified = self._event(trace, TraceEventType.ROLE_VERIFIED, "binrole")
        self.assertTrue(verified.trusted)
        self.assertEqual(verified.condition.kind, "path_hash_prefixes")
        self.assertEqual(verified.condition.matched_prefix, prefix)

    def test_cycle_records_already_visited_and_is_exhaustive(self) -> None:
        case = DelegationsTestCase(
            delegations=[
                TestDelegation("targets", "A"),
                TestDelegation("A", "B"),
                TestDelegation("B", "A"),
            ]
        )
        sim = self._init_repo(case)
        updater = self._updater(sim)
        updater.refresh()

        trace = TargetLookupTrace()
        self.assertIsNone(updater.get_targetinfo("loop", trace))

        revisit = self._event(trace, TraceEventType.ROLE_ALREADY_VISITED, "A")
        self.assertEqual(revisit.parent_role, "B")
        self.assertFalse(revisit.trusted)
        # The cycle was fully handled: every reachable branch was checked.
        self.assertEqual(trace.outcome, LookupOutcome.NOT_FOUND_EXHAUSTED)
        # A appears trusted once (the first visit) and as a revisit once.
        trusted_a = [
            event
            for event in trace.events
            if event.role == "A"
            and event.event_type is TraceEventType.ROLE_VERIFIED
        ]
        self.assertEqual(len(trusted_a), 1)
        self.assertTrue(trusted_a[0].trusted)

    def test_shared_subrole_signature_edge_per_parent(self) -> None:
        # 'release' is delegated by both parents with disjoint paths. Both
        # edges must be independently verifiable and traceable.
        case = DelegationsTestCase(
            delegations=[
                TestDelegation("targets", "parent-a", paths=["path-a"]),
                TestDelegation("targets", "parent-b", paths=["path-b"]),
                TestDelegation("parent-a", "release", paths=["path-a"]),
                TestDelegation("parent-b", "release", paths=["path-b"]),
            ],
            target_files=[
                TestTarget("release", b"a", "path-a"),
                TestTarget("release", b"b", "path-b"),
            ],
        )
        sim = self._init_repo(case)
        updater = self._updater(sim)
        updater.refresh()

        trace_a = TargetLookupTrace()
        self.assertIsNotNone(updater.get_targetinfo("path-a", trace_a))
        edge_a = self._event(trace_a, TraceEventType.ROLE_VERIFIED, "release")
        self.assertEqual(edge_a.parent_role, "parent-a")
        self.assertTrue(edge_a.trusted)
        self.assertEqual(edge_a.condition.matched_path, "path-a")

        trace_b = TargetLookupTrace()
        self.assertIsNotNone(updater.get_targetinfo("path-b", trace_b))
        edge_b = self._event(trace_b, TraceEventType.ROLE_VERIFIED, "release")
        # Second delegator link: role is in the trusted set but must still be
        # verified against parent-b; the trace keeps the actual parent edge.
        self.assertEqual(edge_b.parent_role, "parent-b")
        self.assertTrue(edge_b.trusted)
        self.assertEqual(edge_b.condition.matched_path, "path-b")

    def test_shared_subrole_unsigned_edge_is_not_trusted(self) -> None:
        case = DelegationsTestCase(
            delegations=[
                TestDelegation("targets", "parent-a", paths=["path-a"]),
                TestDelegation("targets", "parent-b", paths=["path-b"]),
                TestDelegation("parent-a", "release", paths=["path-a"]),
                TestDelegation("parent-b", "release", paths=["path-b"]),
            ],
            target_files=[
                TestTarget("release", b"a", "path-a"),
                TestTarget("release", b"b", "path-b"),
            ],
        )
        sim = self._init_repo(case)
        # Prime the trusted set via parent-a using a first updater (its local
        # cache is discarded afterwards, mirroring a fresh client process).
        priming = self._updater(sim)
        self.assertIsNotNone(priming.get_targetinfo("path-a"))

        first_keyid = next(iter(sim.signers["release"].keys()))
        sim.signers["release"] = {
            first_keyid: sim.signers["release"][first_keyid]
        }
        sim.md_delegates["release"].signed.version += 1
        sim.update_snapshot()

        updater = self._updater(sim)
        trace = TargetLookupTrace()
        with self.assertRaises(UnsignedMetadataError):
            updater.get_targetinfo("path-b", trace)

        self.assertEqual(trace.outcome, LookupOutcome.FAILED)
        failure = self._event(trace, TraceEventType.SIGNATURE_FAILED, "release")
        self.assertEqual(failure.parent_role, "parent-b")
        self.assertFalse(failure.trusted)
        # No trusted release->parent-b edge may exist in this call.
        self.assertFalse(
            any(
                event.event_type is TraceEventType.ROLE_VERIFIED
                and event.role == "release"
                and event.parent_role == "parent-b"
                for event in trace.events
            )
        )
        self.assertTrue(trace.sealed)

    def test_succinct_roles_record_bin_condition(self) -> None:
        sim = RepositorySimulator()
        sim.add_succinct_roles("targets", 2, "bin")
        sim.update_snapshot()
        updater = self._updater(sim)
        updater.refresh()

        trace = TargetLookupTrace()
        self.assertIsNone(updater.get_targetinfo("foo", trace))

        verified = [
            event
            for event in trace.events
            if event.event_type is TraceEventType.ROLE_VERIFIED
        ]
        self.assertEqual(
            [event.role for event in verified], ["targets", "bin-0"]
        )
        bin_event = verified[1]
        self.assertEqual(bin_event.condition.kind, "succinct_roles")
        self.assertEqual(bin_event.condition.bin, "bin-0")
        self.assertEqual(bin_event.condition.bit_length, 2)
        self.assertTrue(bin_event.condition.terminating)

    def test_terminating_delegation_truncates_and_marks_skips(self) -> None:
        case = DelegationsTestCase(
            delegations=[
                TestDelegation("targets", "A"),
                TestDelegation("targets", "B"),
                TestDelegation("A", "C", terminating=True),
                TestDelegation("A", "D"),
                TestDelegation("targets", "E"),
            ]
        )
        sim = self._init_repo(case)
        updater = self._updater(sim)
        updater.config.max_delegations = 32
        updater.refresh()

        trace = TargetLookupTrace()
        self.assertIsNone(updater.get_targetinfo("missing", trace))

        truncation = next(
            event
            for event in trace.events
            if event.event_type is TraceEventType.TERMINATING_TRUNCATION
        )
        self.assertEqual(truncation.role, "C")
        self.assertEqual(truncation.parent_role, "A")
        self.assertFalse(truncation.trusted)
        # D is not yet a candidate when C terminates enumeration of A: only
        # the already queued B and E are dropped. D is never evaluated either,
        # which the truncated outcome (not an exhaustive miss) expresses.
        self.assertEqual(set(truncation.skipped_roles), {"B", "E"})
        self.assertEqual(trace.outcome, LookupOutcome.NOT_FOUND_TRUNCATED)
        self.assertNotIn(
            "D",
            [
                event.role
                for event in trace.events
                if event.event_type is TraceEventType.ROLE_VERIFIED
            ],
        )
        # The human-readable text must flag the decisive edge explicitly.
        self.assertIn(
            "this terminating edge removes later candidates",
            trace.to_text(),
        )

    def test_max_delegations_limit_is_truncated_not_exhaustive(self) -> None:
        case = DelegationsTestCase(
            delegations=[
                TestDelegation("targets", "A"),
                TestDelegation("targets", "B"),
                TestDelegation("targets", "C"),
                TestDelegation("C", "D"),
                TestDelegation("C", "E"),
            ]
        )
        sim = self._init_repo(case)
        updater = self._updater(sim)
        updater.config.max_delegations = 4
        updater.refresh()

        trace = TargetLookupTrace()
        self.assertIsNone(updater.get_targetinfo("missing", trace))

        limit_event = next(
            event
            for event in trace.events
            if event.event_type is TraceEventType.MAX_DELEGATIONS_REACHED
        )
        self.assertIn("E", limit_event.skipped_roles)
        self.assertEqual(trace.outcome, LookupOutcome.NOT_FOUND_TRUNCATED)

    def test_download_failure_is_recorded_and_not_trusted(self) -> None:
        case = DelegationsTestCase(
            delegations=[
                TestDelegation("targets", "broken"),
                TestDelegation("targets", "healthy"),
            ]
        )
        sim = self._init_repo(case, FailingSimulator("broken"))
        updater = self._updater(sim)
        updater.refresh()

        trace = TargetLookupTrace()
        with self.assertRaises(DownloadHTTPError):
            updater.get_targetinfo("missing", trace)

        failure = self._event(trace, TraceEventType.DOWNLOAD_FAILED, "broken")
        self.assertEqual(failure.parent_role, "targets")
        self.assertFalse(failure.trusted)
        self.assertIn("unavailable", failure.error)
        self.assertEqual(trace.outcome, LookupOutcome.FAILED)
        # The healthy role must not have been visited after the failure.
        self.assertNotIn(
            "healthy",
            [
                event.role
                for event in trace.events
                if event.event_type is TraceEventType.ROLE_VERIFIED
            ],
        )

    def test_trace_does_not_change_visitation(self) -> None:
        case = DelegationsTestCase(
            delegations=[
                TestDelegation("targets", "A", paths=["*.py"]),
                TestDelegation("targets", "B", paths=["pkg/*"]),
                TestDelegation(
                    "B", "C", terminating=True, paths=["pkg/mod.py"]
                ),
                TestDelegation("B", "D"),
            ],
            target_files=[TestTarget("C", b"c", "pkg/mod.py")],
        )

        def run_traced(
            with_trace: bool,
        ) -> tuple[list, TargetLookupTrace | None]:
            sim = self._init_repo(case)
            updater = self._updater(sim)
            updater.refresh()
            sim.fetch_tracker.metadata.clear()
            trace = TargetLookupTrace() if with_trace else None
            updater.get_targetinfo("pkg/mod.py", trace)
            return list(sim.fetch_tracker.metadata), trace

        untraced_calls, _ = run_traced(False)
        traced_calls, trace = run_traced(True)

        self.assertEqual(untraced_calls, traced_calls)
        self.assertEqual(trace.found_in_role, "C")
        self.assertEqual(trace.trusted_path, ("targets", "B", "C"))

    def test_trace_across_all_reference_graphs_visits_no_extra_roles(
        self,
    ) -> None:
        # Sweep the delegation graph fixtures also used by the core traversal
        # tests: tracing must never cause an additional metadata fetch, and
        # every ROLE_VERIFIED edge must be the only kind of trusted event.
        from tests.test_updater_delegation_graphs import (
            TestDelegationsGraphs,
        )

        for name, graph_case in TestDelegationsGraphs.graphs.items():
            with self.subTest(graph=name):

                def run(
                    case: DelegationsTestCase, with_trace: bool
                ) -> tuple[list, TargetLookupTrace | None]:
                    sim = self._init_repo(case)
                    updater = self._updater(sim)
                    updater.config.max_delegations = 4
                    updater.refresh()
                    sim.fetch_tracker.metadata.clear()
                    trace = TargetLookupTrace() if with_trace else None
                    updater.get_targetinfo("missingpath", trace)
                    return list(sim.fetch_tracker.metadata), trace

                untraced_calls, _ = run(graph_case, False)
                traced_calls, trace = run(graph_case, True)
                self.assertEqual(untraced_calls, traced_calls)
                self.assertTrue(trace.sealed)
                for event in trace.events:
                    self.assertEqual(
                        event.trusted,
                        event.event_type is TraceEventType.ROLE_VERIFIED,
                    )

    def test_trace_is_single_use_and_immutable(self) -> None:
        case = DelegationsTestCase(
            delegations=[TestDelegation("targets", "A")]
        )
        sim = self._init_repo(case)
        updater = self._updater(sim)
        updater.refresh()

        trace = TargetLookupTrace()
        updater.get_targetinfo("missing", trace)
        self.assertTrue(trace.sealed)

        with self.assertRaises(ValueError):
            updater.get_targetinfo("missing", trace)

        with self.assertRaises(ValueError):
            trace.record(TraceEvent(TraceEventType.ROLE_VERIFIED, "forged"))
        with self.assertRaises(ValueError):
            trace.seal(LookupOutcome.TARGET_FOUND, "forged")

        # A later lookup with a different trace cannot mutate saved events.
        snapshot = trace.to_dict()
        other = TargetLookupTrace()
        updater.get_targetinfo("other-missing", other)
        self.assertEqual(trace.to_dict(), snapshot)
        self.assertNotEqual(other.events, trace.events)

    def test_trace_serialization(self) -> None:
        import json

        case = DelegationsTestCase(
            delegations=[TestDelegation("targets", "A", paths=["*.md"])],
            target_files=[TestTarget("A", b"readme", "README.md")],
        )
        sim = self._init_repo(case)
        updater = self._updater(sim)

        trace = TargetLookupTrace()
        self.assertIsNotNone(updater.get_targetinfo("README.md", trace))

        encoded = json.dumps(trace.to_dict())
        decoded = json.loads(encoded)
        self.assertEqual(decoded["outcome"], "target_found")
        self.assertEqual(decoded["found_in_role"], "A")
        self.assertEqual(decoded["trusted_path"], ["targets", "A"])
        self.assertTrue(
            all("event_type" in event for event in decoded["events"])
        )
        self.assertIn("Verified trust path: targets -> A", trace.to_text())

    def test_untraced_lookup_keeps_return_type(self) -> None:
        case = DelegationsTestCase(
            delegations=[TestDelegation("targets", "A")],
            target_files=[TestTarget("A", b"a", "file")],
        )
        sim = self._init_repo(case)
        updater = self._updater(sim)
        self.assertIsNotNone(updater.get_targetinfo("file"))
        self.assertIsNone(updater.get_targetinfo("absent"))


if __name__ == "__main__":
    unittest.main()
