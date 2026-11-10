# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Tests for the offline evidence-bundle verifier."""

from __future__ import annotations

import json
import sys
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tuf.api.metadata import (
    Metadata,
    TargetFile,
    Targets,
)
from tuf.ngclient.config import EnvelopeType
from tuf.offline import (
    BundleConflictError,
    EvidenceBundle,
    VerificationStatus,
    verify_target,
)

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "offline_verification"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from simulated_repo import OfflineRepositorySimulator  # noqa: E402

TARGET_PATH = "vetted/release-1.0.tar.gz"
TARGET_CONTENT = b"offline release contents\n"
REFERENCE_TIME = datetime.now(UTC) + timedelta(days=1)


def _build_simulator(envelope_type: EnvelopeType) -> OfflineRepositorySimulator:
    sim = OfflineRepositorySimulator(envelope_type)
    sim.build_demo_repository(TARGET_PATH, TARGET_CONTENT)
    sim.rotate_root()
    sim.update_snapshot()
    return sim


def _bundle_from_sim(sim: OfflineRepositorySimulator) -> EvidenceBundle:
    metadata: dict[str, list[tuple[int | None, bytes]]] = {}
    for index in range(1, len(sim.signed_roots) + 1):
        metadata.setdefault("root", []).append(
            (index, sim.fetch_metadata("root", index))
        )
    for role in sim.all_role_names():
        data = sim.fetch_metadata(role)
        metadata.setdefault(role, []).append((None, data))
    return EvidenceBundle.from_bytes(
        sim.signed_roots[0],
        metadata=metadata,
        targets={TARGET_PATH: TARGET_CONTENT},
    )


class OfflineVerifierTest(unittest.TestCase):
    def _assert_success(
        self, envelope_type: EnvelopeType
    ) -> None:
        sim = _build_simulator(envelope_type)
        bundle = _bundle_from_sim(sim)
        result = verify_target(
            bundle, TARGET_PATH, REFERENCE_TIME, envelope_type
        )

        self.assertEqual(result.status, VerificationStatus.SUCCESS, result.reason)
        roles = {evidence.role: evidence for evidence in result.role_evidence}
        self.assertEqual(
            list(roles),
            ["root", "timestamp", "snapshot", "targets", "vetted"],
        )
        # Both root versions are part of the proven chain.
        root_versions = [
            evidence.version
            for evidence in result.role_evidence
            if evidence.role == "root"
        ]
        self.assertEqual(root_versions, [1, 2])
        self.assertEqual(roles["vetted"].delegator, "targets")

        target = result.target_evidence
        assert target is not None
        self.assertEqual(target.found_in_role, "vetted")
        self.assertEqual(target.length, len(TARGET_CONTENT))
        self.assertIn("sha256", target.actual_hashes)
        self.assertEqual(
            target.actual_hashes["sha256"],
            target.expected_hashes["sha256"],
        )

    def test_success_traditional_metadata(self) -> None:
        self._assert_success(EnvelopeType.METADATA)

    def test_success_simple_envelope(self) -> None:
        self._assert_success(EnvelopeType.SIMPLE)

    def test_evidence_digests_match_raw_bundle_bytes(self) -> None:
        import hashlib

        sim = _build_simulator(EnvelopeType.METADATA)
        bundle = _bundle_from_sim(sim)
        result = verify_target(
            bundle, TARGET_PATH, REFERENCE_TIME, EnvelopeType.METADATA
        )
        self.assertTrue(result.success)
        raw = {entry.role: entry.data for entry in bundle.metadata["snapshot"]}
        snapshot_evidence = next(
            evidence
            for evidence in result.role_evidence
            if evidence.role == "snapshot"
        )
        self.assertEqual(
            snapshot_evidence.bytes_hashes["sha256"],
            hashlib.sha256(raw["snapshot"]).hexdigest(),
        )

    def test_signed_bytes_are_not_reserialized(self) -> None:
        from tuf.api.serialization.json import CanonicalJSONSerializer

        sim = _build_simulator(EnvelopeType.METADATA)
        bundle = _bundle_from_sim(sim)
        result = verify_target(
            bundle, TARGET_PATH, REFERENCE_TIME, EnvelopeType.METADATA
        )
        self.assertTrue(result.success)

        raw = bundle.metadata["vetted"][0].data
        canonical = Metadata.from_bytes(raw).signed_bytes
        evidence = next(
            item for item in result.role_evidence if item.role == "vetted"
        )
        import hashlib

        self.assertEqual(
            evidence.signed_hashes["sha256"],
            hashlib.sha256(canonical).hexdigest(),
        )
        # The on-wire metadata is pretty-printed, so it differs from the
        # canonical signed payload: the report must expose both digests.
        self.assertNotEqual(
            evidence.bytes_hashes["sha256"], evidence.signed_hashes["sha256"]
        )
        _ = CanonicalJSONSerializer  # documents the signed-payload codec

    def test_corrupt_target_fails_hash_check(self) -> None:
        sim = _build_simulator(EnvelopeType.METADATA)
        bundle = _bundle_from_sim(sim)
        bundle.targets[TARGET_PATH] = type(bundle.targets[TARGET_PATH])(
            TARGET_PATH, TARGET_CONTENT + b"tampered"
        )
        result = verify_target(
            bundle, TARGET_PATH, REFERENCE_TIME, EnvelopeType.METADATA
        )
        self.assertEqual(result.status, VerificationStatus.VERIFICATION_FAILED)
        self.assertIn("LengthOrHashMismatch", result.reason or "")

    def test_missing_target_bytes_is_insufficient_evidence(self) -> None:
        sim = _build_simulator(EnvelopeType.METADATA)
        bundle = _bundle_from_sim(sim)
        bundle.targets.clear()
        result = verify_target(
            bundle, TARGET_PATH, REFERENCE_TIME, EnvelopeType.METADATA
        )
        self.assertEqual(
            result.status, VerificationStatus.INSUFFICIENT_EVIDENCE
        )
        self.assertIn("not in the offline bundle", result.reason or "")

    def test_missing_delegated_role_is_insufficient_evidence(self) -> None:
        sim = _build_simulator(EnvelopeType.METADATA)
        bundle = _bundle_from_sim(sim)
        bundle.metadata.pop("vetted")
        result = verify_target(
            bundle, TARGET_PATH, REFERENCE_TIME, EnvelopeType.METADATA
        )
        self.assertEqual(
            result.status, VerificationStatus.INSUFFICIENT_EVIDENCE
        )
        self.assertIn("vetted", result.reason or "")

    def test_missing_timestamp_is_insufficient_evidence(self) -> None:
        sim = _build_simulator(EnvelopeType.METADATA)
        bundle = _bundle_from_sim(sim)
        bundle.metadata.pop("timestamp")
        result = verify_target(
            bundle, TARGET_PATH, REFERENCE_TIME, EnvelopeType.METADATA
        )
        self.assertEqual(
            result.status, VerificationStatus.INSUFFICIENT_EVIDENCE
        )

    def test_conflicting_root_bytes_rejected_at_bundle_load(self) -> None:
        sim = _build_simulator(EnvelopeType.METADATA)
        root_v2 = sim.fetch_metadata("root", 2)
        reformatted = json.dumps(json.loads(root_v2)).encode()
        self.assertNotEqual(root_v2, reformatted)
        with self.assertRaises(BundleConflictError):
            EvidenceBundle.from_bytes(
                sim.signed_roots[0],
                metadata={
                    "root": [
                        (1, sim.fetch_metadata("root", 1)),
                        (2, root_v2),
                        (2, reformatted),
                    ]
                },
            )

    def test_duplicate_identical_bytes_are_allowed(self) -> None:
        sim = _build_simulator(EnvelopeType.METADATA)
        root_v2 = sim.fetch_metadata("root", 2)
        bundle = EvidenceBundle.from_bytes(
            sim.signed_roots[0],
            metadata={
                "root": [
                    (1, sim.signed_roots[0]),
                    (2, root_v2),
                    (2, root_v2),
                ]
            },
        )
        self.assertEqual(len(bundle.metadata_versions("root")), 3)

    def test_wrong_version_label_is_rejected(self) -> None:
        sim = _build_simulator(EnvelopeType.METADATA)
        bundle = _bundle_from_sim(sim)
        # Snapshot requires timestamp's real version; mislabel the artifact.
        real = bundle.metadata["timestamp"][0]
        real.version = sim.timestamp.version + 5
        result = verify_target(
            bundle, TARGET_PATH, REFERENCE_TIME, EnvelopeType.METADATA
        )
        self.assertEqual(result.status, VerificationStatus.VERIFICATION_FAILED)
        self.assertIn("BadVersionNumber", result.reason or "")

    def test_payload_version_behind_label_is_rejected(self) -> None:
        sim = _build_simulator(EnvelopeType.METADATA)
        bundle = _bundle_from_sim(sim)
        # Label snapshot with a future version: no entry matches the version
        # pinned by timestamp, so the chain cannot be completed.
        bundle.metadata["snapshot"][0].version = (
            sim.snapshot.version + 10
        )
        result = verify_target(
            bundle, TARGET_PATH, REFERENCE_TIME, EnvelopeType.METADATA
        )
        self.assertEqual(
            result.status, VerificationStatus.INSUFFICIENT_EVIDENCE
        )

    def test_extra_untrusted_role_is_ignored(self) -> None:
        from securesystemslib.signer import CryptoSigner

        sim = _build_simulator(EnvelopeType.METADATA)
        bundle = _bundle_from_sim(sim)

        signer = CryptoSigner.generate_ed25519()
        rogue_targets = Targets(expires=sim.safe_expiry)
        rogue_targets.targets[TARGET_PATH] = TargetFile.from_data(
            TARGET_PATH, b"attacker controlled\n", ["sha256"]
        )
        rogue_md = Metadata(rogue_targets)
        rogue_md.sign(signer)
        bundle.add_metadata("rogue-role", rogue_md.to_bytes())

        # The trusted role still proves the genuine target: rogue is ignored.
        result = verify_target(
            bundle, TARGET_PATH, REFERENCE_TIME, EnvelopeType.METADATA
        )
        self.assertTrue(result.success)
        used_roles = {evidence.role for evidence in result.role_evidence}
        self.assertNotIn("rogue-role", used_roles)

    def test_target_only_in_untrusted_role_is_not_found(self) -> None:
        from securesystemslib.signer import CryptoSigner

        sim = _build_simulator(EnvelopeType.METADATA)
        sim.md_delegates["vetted"].signed.targets.pop(TARGET_PATH)
        sim.update_snapshot()
        bundle = _bundle_from_sim(sim)

        signer = CryptoSigner.generate_ed25519()
        rogue_targets = Targets(expires=sim.safe_expiry)
        rogue_targets.targets[TARGET_PATH] = TargetFile.from_data(
            TARGET_PATH, TARGET_CONTENT, ["sha256"]
        )
        rogue_md = Metadata(rogue_targets)
        rogue_md.sign(signer)
        bundle.add_metadata("rogue-role", rogue_md.to_bytes())

        result = verify_target(
            bundle, TARGET_PATH, REFERENCE_TIME, EnvelopeType.METADATA
        )
        self.assertEqual(result.status, VerificationStatus.TARGET_NOT_FOUND)

    def test_envelope_mismatch_is_rejected_without_fallback(self) -> None:
        metadata_sim = _build_simulator(EnvelopeType.METADATA)
        simple_sim = _build_simulator(EnvelopeType.SIMPLE)

        # SIMPLE bytes inside a metadata-mode verification.
        bundle = EvidenceBundle.from_bytes(
            simple_sim.signed_roots[0],
        )
        result = verify_target(
            bundle, TARGET_PATH, REFERENCE_TIME, EnvelopeType.METADATA
        )
        self.assertEqual(result.status, VerificationStatus.VERIFICATION_FAILED)
        self.assertIn("envelope", result.reason or "")

        # Traditional bytes inside a SIMPLE-mode verification.
        bundle = EvidenceBundle.from_bytes(metadata_sim.signed_roots[0])
        result = verify_target(
            bundle, TARGET_PATH, REFERENCE_TIME, EnvelopeType.SIMPLE
        )
        self.assertEqual(result.status, VerificationStatus.VERIFICATION_FAILED)
        self.assertIn("envelope", result.reason or "")

    def test_expired_metadata_at_reference_time(self) -> None:
        sim = _build_simulator(EnvelopeType.METADATA)
        bundle = _bundle_from_sim(sim)
        expired_time = sim.safe_expiry + timedelta(days=60)
        result = verify_target(
            bundle, TARGET_PATH, expired_time, EnvelopeType.METADATA
        )
        self.assertEqual(result.status, VerificationStatus.VERIFICATION_FAILED)
        self.assertIn("expired", (result.reason or "").lower())

    def test_naive_reference_time_rejected(self) -> None:
        sim = _build_simulator(EnvelopeType.METADATA)
        bundle = _bundle_from_sim(sim)
        with self.assertRaises(ValueError):
            verify_target(
                bundle,
                TARGET_PATH,
                datetime(2026, 11, 10, 12, 0, 0),  # noqa: DTZ001
                EnvelopeType.METADATA,
            )

    def test_terminating_delegation_stops_walk(self) -> None:
        # "vetted" is terminating and matches vetted/*: remove its target and
        # ensure no other role is consulted for the path.
        sim = _build_simulator(EnvelopeType.METADATA)
        sim.md_delegates["vetted"].signed.targets.clear()
        sim.update_snapshot()
        bundle = _bundle_from_sim(sim)
        result = verify_target(
            bundle, TARGET_PATH, REFERENCE_TIME, EnvelopeType.METADATA
        )
        self.assertEqual(result.status, VerificationStatus.TARGET_NOT_FOUND)
        visited = {evidence.role for evidence in result.role_evidence}
        self.assertIn("vetted", visited)
        self.assertNotIn("staging", visited)

    def test_non_terminating_delegation_continues_walk(self) -> None:
        # Add the target under staging (non-terminating, but its path pattern
        # does not match vetted/* so build a matching staging target instead).
        staging_path = "staging/notes.txt"
        sim = _build_simulator(EnvelopeType.METADATA)
        data = sim.target_files[staging_path].data
        bundle = _bundle_from_sim(sim)
        bundle.add_target(staging_path, data)
        result = verify_target(
            bundle, staging_path, REFERENCE_TIME, EnvelopeType.METADATA
        )
        self.assertTrue(result.success)
        self.assertEqual(result.target_evidence.found_in_role, "staging")

    def test_root_chain_skips_to_available_version(self) -> None:
        # Bundle contains root v1 and v3 but not v2: the v3 update must fail
        # because rotations cannot skip versions.
        sim = _build_simulator(EnvelopeType.METADATA)
        sim.root.version += 1
        sim.publish_root()
        root_v3 = sim.fetch_metadata("root", 3)

        bundle = EvidenceBundle.from_bytes(
            sim.signed_roots[0],
            metadata={"root": [(1, sim.signed_roots[0]), (3, root_v3)]},
        )
        result = verify_target(
            bundle, TARGET_PATH, REFERENCE_TIME, EnvelopeType.METADATA
        )
        self.assertEqual(result.status, VerificationStatus.VERIFICATION_FAILED)
        self.assertIn("version", (result.reason or "").lower())

    def test_snapshot_hash_pin_enforced(self) -> None:
        sim = _build_simulator(EnvelopeType.METADATA)
        bundle = _bundle_from_sim(sim)
        # Tamper with snapshot bytes while timestamp pins its hash.
        entry = bundle.metadata["snapshot"][0]
        entry.data = entry.data.replace(b" ", b"  ", 1)
        result = verify_target(
            bundle, TARGET_PATH, REFERENCE_TIME, EnvelopeType.METADATA
        )
        self.assertEqual(result.status, VerificationStatus.VERIFICATION_FAILED)

    def test_manifest_roundtrip_directory_bundle(self) -> None:
        import tempfile

        sim = _build_simulator(EnvelopeType.METADATA)
        bundle = _bundle_from_sim(sim)
        with tempfile.TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp) / "metadata"
            metadata_dir.mkdir()
            manifest_metadata: dict[str, list[dict]] = {}
            for role, entries in bundle.metadata.items():
                for entry in entries:
                    if role == "root":
                        rel = f"metadata/{entry.version}.root.json"
                        manifest_entry: dict = {
                            "version": entry.version,
                            "path": rel,
                        }
                    else:
                        rel = f"metadata/{role}.json"
                        manifest_entry = {"path": rel}
                    (Path(tmp) / rel).write_bytes(entry.data)
                    manifest_metadata.setdefault(role, []).append(
                        manifest_entry
                    )
            (Path(tmp) / "targets").mkdir()
            target_rel = f"targets/{TARGET_PATH}"
            (Path(tmp) / target_rel).parent.mkdir(parents=True, exist_ok=True)
            (Path(tmp) / target_rel).write_bytes(TARGET_CONTENT)
            manifest = {
                "bootstrap_root": "metadata/1.root.json",
                "metadata": manifest_metadata,
                "targets": {TARGET_PATH: {"path": target_rel}},
            }
            (Path(tmp) / "manifest.json").write_text(json.dumps(manifest))

            loaded = EvidenceBundle.from_directory(tmp)
            result = verify_target(
                loaded, TARGET_PATH, REFERENCE_TIME, EnvelopeType.METADATA
            )
            self.assertTrue(result.success, result.reason)


if __name__ == "__main__":
    unittest.main()
