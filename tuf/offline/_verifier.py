# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Offline target verification on top of the ngclient validation rules.

The verifier drives :class:`tuf.ngclient._internal.trusted_metadata_set.
TrustedMetadataSet` exactly like the online :class:`tuf.ngclient.Updater`
does, but instead of fetching bytes over HTTP it only reads them from the
provided :class:`~tuf.offline.EvidenceBundle`. Consequently:

* every signature is checked against the *original packaged bytes*
  (canonical JSON payload for traditional metadata, PAE for SIMPLE
  envelopes) - parsed-and-reserialized bytes are never signed objects;
* a role that the bundle does not contain causes "insufficient evidence",
  never an implicit lookup in any local cache;
* roles present in the bundle but not reachable through the verified
  root -> timestamp -> snapshot -> targets/delegations chain are ignored;
* delegation lookup order and terminating delegation semantics match
  ``Updater._preorder_depth_first_walk``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, cast

from tuf.api import exceptions
from tuf.api.dsse import SimpleEnvelope
from tuf.api.metadata import (
    Metadata,
    Root,
    Snapshot,
    TargetFile,
    Targets,
    Timestamp,
)
from tuf.api.serialization import DeserializationError
from tuf.ngclient._internal.trusted_metadata_set import TrustedMetadataSet
from tuf.ngclient.config import EnvelopeType, UpdaterConfig

if TYPE_CHECKING:
    from datetime import datetime

    from tuf.api._payload import Signed
    from tuf.offline._bundle import EvidenceBundle, MetadataEntry

DEFAULT_HASH_ALGORITHMS = ("sha256", "sha512")


def _envelope_label(envelope_type: EnvelopeType) -> str:
    """Return the lowercase name of a single, non-combined envelope flag."""
    label = envelope_type.name
    if label is None:
        raise ValueError(
            "Exactly one envelope type must be selected, "
            f"got {envelope_type!r}"
        )
    return label.lower()


class VerificationStatus(str, Enum):
    """Outcome of an offline verification attempt."""

    SUCCESS = "success"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    VERIFICATION_FAILED = "verification_failed"
    TARGET_NOT_FOUND = "target_not_found"


class InsufficientEvidenceError(exceptions.RepositoryError):
    """A role or target required for the proof is absent from the bundle.

    The absence must not be repaired from any source outside the bundle.
    """


@dataclass
class RoleEvidence:
    """Record of one metadata role actually used during verification.

    Attributes:
        role: Role name as used in the TUF delegation graph.
        version: Signed payload version.
        envelope: Envelope mode used, ``"metadata"`` (traditional canonical
            JSON) or ``"simple"`` (DSSE SIMPLE envelope).
        delegator: Name of the role that delegated trust (``"root"`` for
            top-level roles and for every root in the rotation chain).
        bundle_path: Path of the raw bytes inside the bundle.
        bytes_length: Length of the raw packaged metadata bytes.
        bytes_hashes: Hex digests of the raw packaged bytes, by algorithm.
        signed_hashes: Hex digests of the exact bytes that were signature
            inputs (canonical JSON payload or DSSE PAE), by algorithm.
    """

    role: str
    version: int
    envelope: str
    delegator: str
    bundle_path: str
    bytes_length: int
    bytes_hashes: dict[str, str]
    signed_hashes: dict[str, str]


@dataclass
class TargetEvidence:
    """Record of the target file proof."""

    path: str
    found_in_role: str
    length: int
    expected_hashes: dict[str, str]
    actual_hashes: dict[str, str]
    bundle_path: str | None = None


@dataclass
class OfflineVerificationResult:
    """Structured verification result, suitable for third-party recomputation.

    A consumer can independently re-hash every listed artifact: all digests
    are computed over the exact bytes that were verified, not over
    reserialized content.
    """

    status: VerificationStatus
    target_path: str
    reference_time: str
    envelope: str
    reason: str | None = None
    role_evidence: list[RoleEvidence] = field(default_factory=list)
    target_evidence: TargetEvidence | None = None

    @property
    def success(self) -> bool:
        return self.status is VerificationStatus.SUCCESS

    def to_dict(self) -> dict:
        data: dict = {
            "status": self.status.value,
            "target_path": self.target_path,
            "reference_time": self.reference_time,
            "envelope": self.envelope,
            "roles": [role.__dict__ for role in self.role_evidence],
        }
        if self.target_evidence is not None:
            data["target"] = self.target_evidence.__dict__
        if self.reason is not None:
            data["reason"] = self.reason
        return data


def _digests(data: bytes, algorithms: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for algorithm in algorithms:
        digest = hashlib.new(algorithm)
        digest.update(data)
        result[algorithm] = digest.hexdigest()
    return result


def _detect_envelope(data: bytes) -> tuple[str, str, bytes]:
    """Identify the envelope of raw metadata without guessing/fallback.

    Returns a tuple of envelope label, payload ``_type`` and the exact bytes
    that signatures are computed over. A file must parse fully as exactly
    one envelope type: ambiguous or unparseable bytes fail verification.
    """
    try:
        md = Metadata.from_bytes(data)
    except DeserializationError:
        md = None
    try:
        envelope = SimpleEnvelope.from_bytes(data)
    except DeserializationError:
        envelope = None

    if md is not None and envelope is not None:
        raise exceptions.RepositoryError(
            "Metadata bytes parse as both traditional Metadata and SIMPLE "
            "envelope: ambiguous encapsulation, refusing to guess"
        )
    if md is not None:
        return (
            EnvelopeType.METADATA.name.lower(),
            md.signed.type,
            md.signed_bytes,
        )
    if envelope is not None:
        signed = envelope.get_signed()
        return EnvelopeType.SIMPLE.name.lower(), signed.type, envelope.pae()
    raise exceptions.RepositoryError(
        "Metadata bytes match neither traditional Metadata nor SIMPLE envelope"
    )


def _payload_version(data: bytes) -> int:
    """Read the signed payload version from raw bytes, either envelope."""
    try:
        return Metadata.from_bytes(data).signed.version
    except DeserializationError:
        return SimpleEnvelope.from_bytes(data).get_signed().version


def verify_target(
    bundle: EvidenceBundle,
    target_path: str,
    reference_time: datetime,
    envelope_type: EnvelopeType,
    *,
    hash_algorithms: tuple[str, ...] = DEFAULT_HASH_ALGORITHMS,
    config: UpdaterConfig | None = None,
) -> OfflineVerificationResult:
    """Verify whether ``target_path`` has a complete offline chain of trust.

    Args:
        bundle: Evidence bundle containing the bootstrap root, metadata and
            target bytes. Nothing outside ``bundle`` is consulted.
        target_path: TUF target path to prove.
        reference_time: Explicit verification instant used for all expiry
            checks (must be timezone-aware).
        envelope_type: Required envelope mode for all metadata in the bundle;
            no automatic fallback between METADATA and SIMPLE is performed.
        hash_algorithms: Algorithms used for the evidence report digests.
        config: Optional :class:`UpdaterConfig`; delegation/rotation limits
            are reused from it so traversal limits match the online updater.

    Returns:
        :class:`OfflineVerificationResult` with a machine-readable status and
        the full list of roles, versions and digests actually used.
    """
    if reference_time.tzinfo is None or reference_time.utcoffset() is None:
        raise ValueError("reference_time must be timezone-aware")

    cfg = config or UpdaterConfig()
    envelope_label = _envelope_label(envelope_type)
    result = OfflineVerificationResult(
        status=VerificationStatus.VERIFICATION_FAILED,
        target_path=target_path,
        reference_time=reference_time.isoformat(),
        envelope=envelope_label,
    )

    try:
        verifier = _OfflineVerifier(
            bundle, envelope_type, reference_time, hash_algorithms, cfg, result
        )
        verifier.run(target_path)
    except InsufficientEvidenceError as e:
        result.status = VerificationStatus.INSUFFICIENT_EVIDENCE
        result.reason = str(e)
    except _TargetNotFoundError as e:
        result.status = VerificationStatus.TARGET_NOT_FOUND
        result.reason = str(e)
    except exceptions.RepositoryError as e:
        result.status = VerificationStatus.VERIFICATION_FAILED
        result.reason = f"{type(e).__name__}: {e}"
    return result


class _TargetNotFoundError(Exception):
    """Internal signal: delegated graph was fully traversed without a match."""


class _OfflineVerifier:
    """Stateful driver mirroring the ngclient update + target workflow."""

    def __init__(
        self,
        bundle: EvidenceBundle,
        envelope_type: EnvelopeType,
        reference_time: datetime,
        hash_algorithms: tuple[str, ...],
        config: UpdaterConfig,
        result: OfflineVerificationResult,
    ) -> None:
        self.bundle = bundle
        self.envelope_type = envelope_type
        self.hash_algorithms = hash_algorithms
        self.config = config
        self.result = result
        self._picked: dict[str, MetadataEntry] = {}

        # Fail fast with an explicit envelope error rather than relying on the
        # TrustedMetadataSet parser, whose message would not explain the
        # configured-vs-actual envelope mismatch.
        bootstrap_label, _, _ = _detect_envelope(bundle.bootstrap_root)
        if bootstrap_label != _envelope_label(envelope_type):
            raise exceptions.RepositoryError(
                f"Bootstrap root is packaged as {bootstrap_label!r} envelope "
                f"but verification was requested for "
                f"{_envelope_label(envelope_type)!r}: no fallback between "
                "envelope types"
            )

        trusted_set = TrustedMetadataSet(bundle.bootstrap_root, envelope_type)
        trusted_set.reference_time = reference_time
        self.trusted_set = trusted_set

    # -- envelope helpers -------------------------------------------------

    def _check_envelope(self, entry: MetadataEntry) -> bytes:
        """Validate envelope mode of ``entry`` and return signature bytes."""
        label, payload_type, signed_bytes = _detect_envelope(entry.data)
        expected = _envelope_label(self.envelope_type)
        if label != expected:
            raise exceptions.RepositoryError(
                f"Role {entry.role!r} is packaged as {label!r} envelope but "
                f"verification was requested for {expected!r}: no fallback "
                "between envelope types"
            )
        return signed_bytes

    def _record(
        self,
        entry: MetadataEntry,
        signed: Signed,
        delegator: str,
        signed_bytes: bytes,
    ) -> None:
        self.result.role_evidence.append(
            RoleEvidence(
                role=entry.role,
                version=signed.version,
                envelope=_envelope_label(self.envelope_type),
                delegator=delegator,
                bundle_path=entry.path,
                bytes_length=len(entry.data),
                bytes_hashes=_digests(entry.data, self.hash_algorithms),
                signed_hashes=_digests(signed_bytes, self.hash_algorithms),
            )
        )

    # -- offline byte source ---------------------------------------------

    def _pick_entry(self, role: str, expected_version: int) -> MetadataEntry:
        """Select the bundle entry for ``role`` at ``expected_version``."""
        if role in self._picked:
            return self._picked[role]

        candidates = self.bundle.metadata_versions(role)
        if not candidates:
            raise InsufficientEvidenceError(
                f"Required role {role!r} (version {expected_version}) is not "
                "in the offline bundle"
            )

        matching = [
            entry
            for entry in candidates
            if entry.version is None or entry.version == expected_version
        ]
        if not matching:
            raise InsufficientEvidenceError(
                f"Role {role!r} is bundled but none of its versions match "
                f"the required version {expected_version}"
            )

        # Prefer explicitly versioned evidence; otherwise use unversioned
        # entries. Equal bytes collapse to one artifact (duplicate packaging
        # is harmless); distinct bytes for the same role are a hard conflict.
        versioned = [entry for entry in matching if entry.version is not None]
        pool = versioned or matching
        unique = {entry.data: entry for entry in pool}
        if len(unique) > 1:
            raise exceptions.RepositoryError(
                f"Multiple distinct bundle entries for role {role!r} version "
                f"{expected_version}"
            )
        entry = next(iter(unique.values()))
        self._check_envelope(entry)
        self._picked[role] = entry
        return entry

    # -- workflow ---------------------------------------------------------

    def run(self, target_path: str) -> None:
        self._load_root_chain()
        self._load_timestamp()
        self._load_snapshot()
        self._load_targets_role(Targets.type, Root.type)
        target_file, role_name = self._walk_delegations(target_path)
        self._verify_target_bytes(target_path, role_name, target_file)
        self.result.status = VerificationStatus.SUCCESS

    def _load_root_chain(self) -> None:
        bootstrap_version = _payload_version(self.bundle.bootstrap_root)
        self._record_bootstrap(bootstrap_version)

        current = bootstrap_version
        upper_bound = current + self.config.max_root_rotations
        while current < upper_bound:
            next_version = current + 1
            candidates = [
                entry
                for entry in self.bundle.metadata_versions(Root.type)
                if entry.version == next_version
            ]
            if not candidates:
                bundled_newer = [
                    entry.version
                    for entry in self.bundle.metadata_versions(Root.type)
                    if (
                        entry.version is not None
                        and entry.version > next_version
                    )
                ]
                if bundled_newer:
                    # A root rotation chain must be contiguous: accepting a
                    # later version without its predecessor would skip the
                    # predecessor's signature/key-rotation checks.
                    raise exceptions.BadVersionNumberError(
                        f"Root version {next_version} is missing from the "
                        f"bundle but newer versions {sorted(bundled_newer)} "
                        "are present: root rotation chain must be contiguous"
                    )
                # No newer root in the bundle: chain ends here, exactly like a
                # 404 ends the online root update loop.
                return
            if len(candidates) > 1:
                raise exceptions.RepositoryError(
                    f"Multiple distinct root entries for version {next_version}"
                )
            entry = candidates[0]
            signed_bytes = self._check_envelope(entry)
            new_root = self.trusted_set.update_root(entry.data)
            self._record(entry, new_root, Root.type, signed_bytes)
            current = next_version

    def _record_bootstrap(self, version: int) -> None:
        data = self.bundle.bootstrap_root
        label, _, signed_bytes = _detect_envelope(data)
        expected = _envelope_label(self.envelope_type)
        if label != expected:
            raise exceptions.RepositoryError(
                f"Bootstrap root is packaged as {label!r} envelope but "
                f"verification was requested for {expected!r}"
            )
        self.result.role_evidence.append(
            RoleEvidence(
                role=Root.type,
                version=version,
                envelope=label,
                delegator=Root.type,
                bundle_path=self.bundle.bootstrap_path,
                bytes_length=len(data),
                bytes_hashes=_digests(data, self.hash_algorithms),
                signed_hashes=_digests(signed_bytes, self.hash_algorithms),
            )
        )

    def _load_timestamp(self) -> None:
        # Timestamp is not version-prefixed in TUF; the bundle may still
        # carry a version label which must match the signed payload.
        candidates = self.bundle.metadata_versions(Timestamp.type)
        if not candidates:
            raise InsufficientEvidenceError(
                "Required role 'timestamp' is not in the offline bundle"
            )
        entry = self._single_unversioned(Timestamp.type, candidates)
        signed_bytes = self._check_envelope(entry)
        timestamp = self.trusted_set.update_timestamp(entry.data)
        if entry.version is not None and entry.version != timestamp.version:
            raise exceptions.BadVersionNumberError(
                f"Bundle label says timestamp v{entry.version} but payload is "
                f"v{timestamp.version}"
            )
        self._record(entry, timestamp, Root.type, signed_bytes)

    def _load_snapshot(self) -> None:
        expected_version = self.trusted_set.timestamp.snapshot_meta.version
        entry = self._pick_entry(Snapshot.type, expected_version)
        signed_bytes = self._check_envelope(entry)
        snapshot = self.trusted_set.update_snapshot(entry.data)
        self._record(entry, snapshot, Root.type, signed_bytes)

    def _load_targets_role(self, role: str, parent_role: str) -> Targets:
        """Load one targets role using snapshot meta, as the updater does."""
        if self.trusted_set.contains(role, parent_role):
            return cast("Targets", self.trusted_set[role])

        snapshot = self.trusted_set.snapshot
        metainfo = snapshot.meta.get(f"{role}.json")
        if metainfo is None:
            raise InsufficientEvidenceError(
                f"Role {role!r} is delegated by {parent_role!r} but snapshot "
                "does not contain version/hash information for it"
            )

        entry = self._pick_entry(role, metainfo.version)
        signed_bytes = self._check_envelope(entry)
        targets = self.trusted_set.update_delegated_targets(
            entry.data, role, parent_role
        )
        self._record(entry, targets, parent_role, signed_bytes)
        return targets

    def _single_unversioned(
        self, role: str, candidates: list[MetadataEntry]
    ) -> MetadataEntry:
        unversioned = [entry for entry in candidates if entry.version is None]
        versioned = [entry for entry in candidates if entry.version is not None]
        if unversioned and len(unversioned) > 1:
            raise exceptions.RepositoryError(
                f"Multiple unversioned entries for role {role!r}"
            )
        if unversioned:
            return unversioned[0]
        if len(versioned) != 1:
            raise InsufficientEvidenceError(
                f"Role {role!r} requires a single unversioned artifact, "
                f"{len(versioned)} versioned entries found"
            )
        return versioned[0]

    # -- delegation walk (mirrors Updater._preorder_depth_first_walk) -----

    def _walk_delegations(
        self, target_path: str
    ) -> tuple[TargetFile, str]:
        delegations_to_visit: list[tuple[str, str]] = [
            (Targets.type, Root.type)
        ]
        visited: set[str] = set()

        while (
            len(visited) <= self.config.max_delegations
            and delegations_to_visit
        ):
            role_name, parent_role = delegations_to_visit.pop(-1)
            if role_name in visited:
                continue

            targets = self._load_targets_role(role_name, parent_role)
            target = targets.targets.get(target_path)
            if target is not None:
                return target, role_name

            visited.add(role_name)
            if targets.delegations is not None:
                children: list[tuple[str, str]] = []
                for child_name, terminating in (
                    targets.delegations.get_roles_for_target(target_path)
                ):
                    children.append((child_name, role_name))
                    if terminating:
                        # Terminating delegation: stop backtracking, exactly
                        # like the online updater.
                        delegations_to_visit = []
                        break
                children.reverse()
                delegations_to_visit.extend(children)

        if delegations_to_visit:
            raise InsufficientEvidenceError(
                f"Delegation limit of {self.config.max_delegations} reached "
                f"before target {target_path!r} could be resolved"
            )
        raise _TargetNotFoundError(
            f"Target {target_path!r} is not referenced by any reachable "
            "targets role"
        )

    # -- target bytes -----------------------------------------------------

    def _verify_target_bytes(
        self, target_path: str, role_name: str, target_file: TargetFile
    ) -> None:
        target_entry = self.bundle.targets.get(target_path)
        if target_entry is None:
            raise InsufficientEvidenceError(
                f"Target {target_path!r} is signed by role {role_name!r} but "
                "its file bytes are not in the offline bundle"
            )

        target_file.verify_length_and_hashes(target_entry.data)

        actual = _digests(target_entry.data, self.hash_algorithms)
        self.result.target_evidence = TargetEvidence(
            path=target_path,
            found_in_role=role_name,
            length=len(target_entry.data),
            expected_hashes=dict(target_file.hashes),
            actual_hashes=actual,
            bundle_path=target_entry.path,
        )
