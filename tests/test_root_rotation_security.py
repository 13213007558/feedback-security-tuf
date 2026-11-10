# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Regression tests for the security conditions of root trust migration.

These tests pin down the trust migration rules that are easiest to break
while refactoring the root rotation code. They do not introduce any new
signature algorithms: every metadata object is signed with the existing
Ed25519 ``CryptoSigner``.

Two independent layers are exercised for every security condition:

* ``TrustedMetadataSet`` directly, so the in-memory trusted collection can
  be inspected before and after every rejected step.
* ``Updater.refresh()`` against an in-memory repository, so the persisted
  root history and the behaviour of a second client startup are checked.

Every condition is constrained for both supported wire envelopes: the
traditional canonical JSON ``Metadata`` envelope and the DSSE
``SimpleEnvelope``. The expected outcomes below follow the TUF detailed
client workflow and the manually constructed signature relationships in
this module; they are not copied from the implementation under test.

A fixed reference time is used throughout. Expiry of an *intermediate* root
is never fatal: only the final root's expiry is checked, and that check
happens when the timestamp is first loaded. This makes it possible to
migrate trust across an already-expired intermediate root as long as the
final root is currently valid.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import patch
from urllib import parse

from securesystemslib.signer import CryptoSigner

from tuf.api import exceptions
from tuf.api.dsse import SimpleEnvelope
from tuf.api.metadata import (
    Metadata,
    MetaFile,
    Root,
    Snapshot,
    Targets,
    Timestamp,
)
from tuf.api.serialization import DeserializationError
from tuf.ngclient import Updater
from tuf.ngclient._internal.trusted_metadata_set import (
    TrustedMetadataSet,
)
from tuf.ngclient.config import EnvelopeType, UpdaterConfig
from tuf.ngclient.fetcher import FetcherInterface

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from securesystemslib.signer import Signer

# Fixed verification time: all "valid" metadata expires after REFERENCE_TIME
# and all "expired" metadata expires before it. Pinning the clock makes the
# final-root expiry check deterministic.
REFERENCE_TIME = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
FUTURE_EXPIRY = REFERENCE_TIME + timedelta(days=30)
PAST_EXPIRY = REFERENCE_TIME - timedelta(days=1)

ENVELOPES = (EnvelopeType.METADATA, EnvelopeType.SIMPLE)


class SignedMetadata:
    """A piece of signed metadata plus the signers used to produce it.

    The same logical signed object can be serialized into either of the two
    supported envelopes so that the shared security conditions can be tested
    identically.
    """

    def __init__(self, signed: object, signers: Sequence[Signer]) -> None:
        self.signed = signed
        self.signers = list(signers)

    def to_metadata_bytes(self) -> bytes:
        metadata = Metadata(self.signed)  # type: ignore[arg-type]
        for signer in self.signers:
            metadata.sign(signer, append=True)
        return metadata.to_bytes()

    def to_simple_bytes(self) -> bytes:
        envelope = SimpleEnvelope.from_signed(self.signed)  # type: ignore[arg-type]
        for signer in self.signers:
            envelope.sign(signer)
        return envelope.to_bytes()

    def bytes_for(self, envelope_type: EnvelopeType) -> bytes:
        if envelope_type is EnvelopeType.SIMPLE:
            return self.to_simple_bytes()
        return self.to_metadata_bytes()


def make_signers(count: int) -> list[Signer]:
    return [CryptoSigner.generate_ed25519() for _ in range(count)]


def build_root(
    version: int,
    root_key_signers: Sequence[Signer],
    signing_signers: Sequence[Signer],
    threshold: int,
    timestamp_signers: Sequence[Signer] = (),
    expires: datetime = FUTURE_EXPIRY,
) -> SignedMetadata:
    """Construct one root version with explicit root and timestamp keys."""

    root = Root(version=version, expires=expires)
    root.roles[Root.type].threshold = threshold
    for signer in root_key_signers:
        root.add_key(signer.public_key, Root.type)
    for signer in timestamp_signers:
        root.add_key(signer.public_key, Timestamp.type)
    return SignedMetadata(root, signing_signers)


def build_root_with_top_level_key(
    version: int,
    root_key_signers: Sequence[Signer],
    signing_signers: Sequence[Signer],
    threshold: int,
    top_level_signer: Signer,
    expires: datetime = FUTURE_EXPIRY,
) -> SignedMetadata:
    """Construct a root authorizing one signer for every non-root role.

    A full ``Updater.refresh()`` verifies timestamp, snapshot and targets, so
    repositories that complete the workflow need keys for all of those roles.
    """

    root = Root(version=version, expires=expires)
    root.roles[Root.type].threshold = threshold
    for signer in root_key_signers:
        root.add_key(signer.public_key, Root.type)
    for role in (Timestamp.type, Snapshot.type, Targets.type):
        root.add_key(top_level_signer.public_key, role)
    return SignedMetadata(root, signing_signers)


def build_timestamp(
    signers: Sequence[Signer], version: int = 1, expires: datetime = FUTURE_EXPIRY
) -> SignedMetadata:
    timestamp = Timestamp(version=version, expires=expires)
    timestamp.snapshot_meta = MetaFile(1)
    return SignedMetadata(timestamp, signers)


def build_snapshot(signers: Sequence[Signer]) -> SignedMetadata:
    snapshot = Snapshot(version=1, expires=FUTURE_EXPIRY)
    snapshot.meta["targets.json"] = MetaFile(1)
    return SignedMetadata(snapshot, signers)


def build_targets(signers: Sequence[Signer]) -> SignedMetadata:
    targets = Targets(version=1, expires=FUTURE_EXPIRY)
    return SignedMetadata(targets, signers)


def top_level_metadata(timestamp_signer: Signer) -> dict[str, SignedMetadata]:
    """Return a healthy, mutually consistent timestamp/snapshot/targets set."""

    return {
        Timestamp.type: build_timestamp([timestamp_signer]),
        Snapshot.type: build_snapshot([timestamp_signer]),
        Targets.type: build_targets([timestamp_signer]),
    }


def with_duplicate_keyid_signature(
    root: SignedMetadata, envelope_type: EnvelopeType
) -> bytes:
    """Serialize a root carrying two signatures with the same keyid.

    A single private key signs the payload but the signature block lists the
    same keyid twice. Threshold counting must never let one signer count as
    two, so this metadata must be rejected rather than accepted.
    """

    signer = root.signers[0]
    if envelope_type is EnvelopeType.SIMPLE:
        envelope = SimpleEnvelope.from_signed(root.signed)  # type: ignore[arg-type]
        signature = envelope.sign(signer)
        decoded = json.loads(envelope.to_bytes())
    else:
        metadata = Metadata(root.signed)  # type: ignore[arg-type]
        signature = metadata.sign(signer, append=True)
        decoded = json.loads(metadata.to_bytes())

    decoded["signatures"].append(
        {"keyid": signature.keyid, "sig": signature.signature}
    )
    return json.dumps(decoded).encode()


class InMemoryFetcher(FetcherInterface):
    """Serve pre-signed metadata bytes from memory for both envelopes.

    Mirrors the URL layout used by ``Updater``: version-prefixed root files
    (``<version>.root.json``) and non-versioned other roles.
    """

    def __init__(
        self,
        roots: Sequence[SignedMetadata],
        other_metadata: dict[str, SignedMetadata],
        envelope_type: EnvelopeType,
    ) -> None:
        self._roots = list(roots)
        self._other = dict(other_metadata)
        self._envelope_type = envelope_type

    def _fetch(self, url: str) -> Iterator[bytes]:
        path = parse.urlparse(url).path
        if not path.startswith("/metadata/") or not path.endswith(".json"):
            raise exceptions.DownloadHTTPError(f"Unknown path {path}", 404)

        name = path[len("/metadata/") : -len(".json")]
        version_str, dot, role = name.partition(".")

        if dot and role == Root.type:
            version = int(version_str)
            if version < 1 or version > len(self._roots):
                raise exceptions.DownloadHTTPError(
                    f"Unknown root version {version}", 404
                )
            yield self._roots[version - 1].bytes_for(self._envelope_type)
            return

        # Consistent snapshot prefixes non-root roles with a version number
        # (e.g. "1.snapshot.json"); the role name is the part after the dot.
        role_name = role if dot and role != "" else name
        if role_name in self._other:
            yield self._other[role_name].bytes_for(self._envelope_type)
            return

        raise exceptions.DownloadHTTPError(f"Unknown role {name}", 404)


@contextmanager
def frozen_reference_time() -> Iterator[None]:
    """Freeze ``TrustedMetadataSet.reference_time`` at ``REFERENCE_TIME``.

    The trusted set stamps its reference time at construction; patching the
    clock there is the single point that also drives ``Updater`` refresh.
    """

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:  # type: ignore[override]
            if tz is None:
                return REFERENCE_TIME.replace(tzinfo=None)
            return REFERENCE_TIME

    with patch(
        "tuf.ngclient._internal.trusted_metadata_set.datetime", FrozenDatetime
    ):
        yield


class RootRotationTestBase(unittest.TestCase):
    """Shared builders and invariants for the root rotation tests."""

    def make_trusted_set(
        self, bootstrap: SignedMetadata, envelope_type: EnvelopeType
    ) -> TrustedMetadataSet:
        with frozen_reference_time():
            trusted_set = TrustedMetadataSet(
                bootstrap.bytes_for(envelope_type), envelope_type
            )
        trusted_set.reference_time = REFERENCE_TIME
        return trusted_set

    def assert_root_version(
        self, trusted_set: TrustedMetadataSet, version: int
    ) -> None:
        self.assertEqual(trusted_set.root.version, version)

    def assert_role_absent(
        self, trusted_set: TrustedMetadataSet, role: str
    ) -> None:
        self.assertNotIn(role, trusted_set)

    def make_updater(
        self,
        metadata_dir: str,
        roots: Sequence[SignedMetadata],
        other_metadata: dict[str, SignedMetadata],
        envelope_type: EnvelopeType,
        bootstrap: bytes | None = None,
    ) -> Updater:
        fetcher = InMemoryFetcher(roots, other_metadata, envelope_type)
        config = UpdaterConfig(envelope_type=envelope_type)
        bootstrap_bytes = (
            bootstrap
            if bootstrap is not None
            else roots[0].bytes_for(envelope_type)
        )
        return Updater(
            metadata_dir,
            "https://example.com/metadata/",
            fetcher=fetcher,
            config=config,
            bootstrap=bootstrap_bytes,
        )

    def run_updater(
        self,
        roots: Sequence[SignedMetadata],
        other_metadata: dict[str, SignedMetadata],
        envelope_type: EnvelopeType,
        bootstrap: bytes | None = None,
    ) -> tuple[Updater, str]:
        """Run one full ``Updater.refresh()`` against the in-memory repo."""

        metadata_dir = tempfile.mkdtemp()
        updater = self.make_updater(
            metadata_dir, roots, other_metadata, envelope_type, bootstrap
        )
        with frozen_reference_time():
            updater.refresh()
        return updater, metadata_dir

    def persisted_root_versions(self, metadata_dir: str) -> list[int]:
        history = os.path.join(metadata_dir, "root_history")
        if not os.path.isdir(history):
            return []
        versions = []
        for filename in os.listdir(history):
            if filename.endswith(".root.json"):
                versions.append(int(filename.split(".", 1)[0]))
        return sorted(versions)

    def assert_persisted_root(
        self, metadata_dir: str, version: int, expected: bytes
    ) -> None:
        path = os.path.join(
            metadata_dir, "root_history", f"{version}.root.json"
        )
        with open(path, "rb") as f:
            self.assertEqual(f.read(), expected)


class TestRootThresholdMigration(RootRotationTestBase):
    """Both the old and the new root threshold must be met."""

    def setUp(self) -> None:
        # v1 root: keys A,B threshold 2. v2 root rotates to keys C,D thr 2.
        self.signers = make_signers(4)
        self.a, self.b, self.c, self.d = self.signers
        self.root_v1 = build_root(1, [self.a, self.b], [self.a, self.b], 2)

    def _root_v2(self, signing_signers: Sequence[Signer]) -> SignedMetadata:
        return build_root(2, [self.c, self.d], signing_signers, 2)

    def test_valid_migration_signed_by_both_old_and_new_keys(self) -> None:
        # A successful rotation is signed by both threshold sets.
        root_v2 = self._root_v2([self.a, self.b, self.c, self.d])
        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                trusted_set = self.make_trusted_set(
                    self.root_v1, envelope_type
                )
                trusted_set.update_root(root_v2.bytes_for(envelope_type))
                self.assert_root_version(trusted_set, 2)

    def test_old_threshold_only_is_rejected(self) -> None:
        # Signed by the old A,B set (old threshold met) but not by new C,D.
        root_v2 = self._root_v2([self.a, self.b])
        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                trusted_set = self.make_trusted_set(
                    self.root_v1, envelope_type
                )
                with self.assertRaises(exceptions.UnsignedMetadataError):
                    trusted_set.update_root(
                        root_v2.bytes_for(envelope_type)
                    )
                # Rejection must not advance the trusted root.
                self.assert_root_version(trusted_set, 1)

    def test_old_threshold_met_but_new_threshold_not_is_rejected(self) -> None:
        # Precise discriminator for the new-root self-threshold check: the new
        # root carries enough signatures from the previous root's keys (so
        # dropping only the *self* verification would accept it), but it is not
        # signed by its own key set up to its declared threshold.
        signers = make_signers(4)
        old_a, old_b, new_c, new_d = signers
        root_v1 = build_root(1, [old_a, old_b], [old_a, old_b], 2)
        # v2 declares keys C,D at threshold 2 but is signed only by old A,B.
        root_v2 = build_root(2, [new_c, new_d], [old_a, old_b], 2)

        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                trusted_set = self.make_trusted_set(root_v1, envelope_type)
                with self.assertRaises(exceptions.UnsignedMetadataError):
                    trusted_set.update_root(
                        root_v2.bytes_for(envelope_type)
                    )
                self.assert_root_version(trusted_set, 1)

    def test_new_threshold_only_is_rejected(self) -> None:
        # Signed by the new C,D set (new threshold met) but not by old A,B.
        root_v2 = self._root_v2([self.c, self.d])
        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                trusted_set = self.make_trusted_set(
                    self.root_v1, envelope_type
                )
                with self.assertRaises(exceptions.UnsignedMetadataError):
                    trusted_set.update_root(
                        root_v2.bytes_for(envelope_type)
                    )
                self.assert_root_version(trusted_set, 1)

    def test_new_threshold_met_but_old_threshold_not_is_rejected(self) -> None:
        # Precise discriminator for the previous-root threshold check: the new
        # root is fully self-signed (so dropping only the *old* check would
        # accept it), yet it carries no valid signature from the previous root
        # key set. The migration must still be refused.
        signers = make_signers(4)
        old_a, old_b, new_c, new_d = signers
        root_v1 = build_root(1, [old_a, old_b], [old_a, old_b], 2)
        root_v2 = build_root(2, [new_c, new_d], [new_c, new_d], 2)

        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                trusted_set = self.make_trusted_set(root_v1, envelope_type)
                with self.assertRaises(exceptions.UnsignedMetadataError):
                    trusted_set.update_root(
                        root_v2.bytes_for(envelope_type)
                    )
                self.assert_root_version(trusted_set, 1)

    def test_updater_persists_only_verified_roots(self) -> None:
        top_level_signer = make_signers(1)[0]
        good_v2 = build_root_with_top_level_key(
            2,
            [self.c, self.d],
            [self.a, self.b, self.c, self.d],
            2,
            top_level_signer,
        )
        bad_v2 = self._root_v2([self.c, self.d])  # new threshold only
        healthy = top_level_metadata(top_level_signer)

        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                # Successful migration completes refresh and persists v1, v2.
                _, good_dir = self.run_updater(
                    [self.root_v1, good_v2], healthy, envelope_type
                )
                self.assertEqual(
                    self.persisted_root_versions(good_dir), [1, 2]
                )

                # Failing migration must persist v1 only and keep root.json at
                # the bootstrap version; nothing advances to v2.
                fail_dir = tempfile.mkdtemp()
                updater = self.make_updater(
                    fail_dir,
                    [self.root_v1, bad_v2],
                    {},
                    envelope_type,
                )
                with frozen_reference_time():
                    with self.assertRaises(exceptions.UnsignedMetadataError):
                        updater.refresh()
                self.assertEqual(self.persisted_root_versions(fail_dir), [1])
                self.assert_persisted_root(
                    fail_dir, 1, self.root_v1.bytes_for(envelope_type)
                )
                # The non-versioned root.json still resolves to v1.
                linked = os.path.realpath(
                    os.path.join(fail_dir, "root.json")
                )
                with open(linked, "rb") as f:
                    self.assertEqual(
                        f.read(), self.root_v1.bytes_for(envelope_type)
                    )

    def test_second_startup_trusts_persisted_rotated_root(self) -> None:
        # After a successful rotation the persisted v2 root is a valid trust
        # anchor on its own: a fresh Updater bootstrapped with it accepts the
        # current top-level role keys and completes refresh.
        top_level_signer = make_signers(1)[0]
        root_v2 = build_root_with_top_level_key(
            2,
            [self.c, self.d],
            [self.a, self.b, self.c, self.d],
            2,
            top_level_signer,
        )
        other = top_level_metadata(top_level_signer)

        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                _, metadata_dir = self.run_updater(
                    [self.root_v1, root_v2], other, envelope_type
                )
                with open(
                    os.path.join(
                        metadata_dir, "root_history", "2.root.json"
                    ),
                    "rb",
                ) as f:
                    rotated_bootstrap = f.read()

                second_dir = tempfile.mkdtemp()
                updater = self.make_updater(
                    second_dir,
                    [self.root_v1, root_v2],
                    other,
                    envelope_type,
                    bootstrap=rotated_bootstrap,
                )
                with frozen_reference_time():
                    updater.refresh()
                # Bootstrapped directly at v2: no v1 history is written and
                # the trusted root is the rotated one.
                self.assertEqual(
                    self.persisted_root_versions(second_dir), [2]
                )
                self.assertEqual(
                    updater._trusted_set.root.version, 2  # noqa: SLF001
                )


class TestDuplicateSignatureImpersonation(RootRotationTestBase):
    """One signer must never count as multiple signers toward threshold."""

    def test_duplicate_keyid_does_not_meet_threshold(self) -> None:
        signers = make_signers(2)
        root_v1 = build_root(1, signers, signers, 2)
        # v2 keeps both keys at threshold 2 but a single key produces the
        # signature block; a duplicated keyid entry tries to pose as two.
        root_v2 = build_root(2, signers, [signers[0]], 2)

        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                trusted_set = self.make_trusted_set(root_v1, envelope_type)
                malicious = with_duplicate_keyid_signature(
                    root_v2, envelope_type
                )
                with self.assertRaises(
                    (DeserializationError, exceptions.UnsignedMetadataError)
                ):
                    trusted_set.update_root(malicious)
                self.assert_root_version(trusted_set, 1)

    def test_updater_rejects_duplicate_keyid_root(self) -> None:
        signers = make_signers(2)
        root_v1 = build_root(1, signers, signers, 2)
        root_v2 = build_root(2, signers, [signers[0]], 2)

        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                malicious_bytes = with_duplicate_keyid_signature(
                    root_v2, envelope_type
                )
                metadata_dir = tempfile.mkdtemp()
                fetcher = _SingleRootFetcher(malicious_bytes)
                config = UpdaterConfig(envelope_type=envelope_type)
                with frozen_reference_time():
                    updater = Updater(
                        metadata_dir,
                        "https://example.com/metadata/",
                        fetcher=fetcher,
                        config=config,
                        bootstrap=root_v1.bytes_for(envelope_type),
                    )
                    with self.assertRaises(exceptions.RepositoryError):
                        updater.refresh()
                # Only the bootstrap root is persisted; no rotation happened.
                self.assertEqual(
                    self.persisted_root_versions(metadata_dir), [1]
                )


class _SingleRootFetcher(FetcherInterface):
    """Return fixed bytes for the next root version and 404 afterwards."""

    def __init__(self, root_bytes: bytes) -> None:
        self._root_bytes = root_bytes

    def _fetch(self, url: str) -> Iterator[bytes]:
        path = parse.urlparse(url).path
        if path == "/metadata/2.root.json":
            yield self._root_bytes
            return
        raise exceptions.DownloadHTTPError("not found", 404)


class TestSequentialVersions(RootRotationTestBase):
    """Root versions must be consecutive: skipping a version is forbidden."""

    def test_version_skip_rejected_at_trusted_set(self) -> None:
        signers = make_signers(2)
        root_v1 = build_root(1, signers, signers, 1)
        # A well-formed v3 (signed by v1's keys) tries to jump from 1 to 3.
        root_v3 = build_root(3, signers, signers, 1)

        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                trusted_set = self.make_trusted_set(root_v1, envelope_type)
                with self.assertRaises(exceptions.BadVersionNumberError):
                    trusted_set.update_root(root_v3.bytes_for(envelope_type))
                self.assert_root_version(trusted_set, 1)

    def test_updater_stops_at_skipped_version(self) -> None:
        signers = make_signers(2)
        root_v1 = build_root(1, signers, signers, 1)
        root_v3 = build_root(3, signers, signers, 1)

        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                metadata_dir = tempfile.mkdtemp()
                # The client only ever requests the next version (v2). Serve
                # the out-of-sequence v3 bytes in answer so the consecutive
                # version check is what rejects it. A genuine missing v2 would
                # instead end the rotation harmlessly on a 404.
                fetcher = _SingleRootFetcher(
                    root_v3.bytes_for(envelope_type)
                )
                config = UpdaterConfig(envelope_type=envelope_type)
                with frozen_reference_time():
                    updater = Updater(
                        metadata_dir,
                        "https://example.com/metadata/",
                        fetcher=fetcher,
                        config=config,
                        bootstrap=root_v1.bytes_for(envelope_type),
                    )
                    with self.assertRaises(exceptions.BadVersionNumberError):
                        updater.refresh()
                self.assertEqual(
                    self.persisted_root_versions(metadata_dir), [1]
                )


class TestTimestampKeyRotation(RootRotationTestBase):
    """After a timestamp key rotation the old timestamp key is untrusted."""

    def _build_chain(self) -> tuple[list[SignedMetadata], Signer, Signer]:
        signers = make_signers(4)
        old_ts, new_ts = signers[2], signers[3]
        # v1 root authorizes old top-level role key T_old.
        root_v1 = build_root(
            1,
            [signers[0]],
            [signers[0]],
            1,
            timestamp_signers=[old_ts],
        )
        # v2 root rotates the root key and replaces the top-level role key.
        root_v2 = build_root_with_top_level_key(
            2,
            [signers[1]],
            [signers[0], signers[1]],
            1,
            new_ts,
        )
        return [root_v1, root_v2], old_ts, new_ts

    def test_stale_timestamp_key_rejected_at_trusted_set(self) -> None:
        roots, old_ts, new_ts = self._build_chain()
        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                trusted_set = self.make_trusted_set(roots[0], envelope_type)
                trusted_set.update_root(roots[1].bytes_for(envelope_type))
                self.assert_root_version(trusted_set, 2)
                self.assert_role_absent(trusted_set, Timestamp.type)

                # A timestamp signed with the removed T_old key must fail and
                # must not install a timestamp.
                stale_timestamp = build_timestamp([old_ts])
                with self.assertRaises(exceptions.UnsignedMetadataError):
                    trusted_set.update_timestamp(
                        stale_timestamp.bytes_for(envelope_type)
                    )
                self.assert_role_absent(trusted_set, Timestamp.type)
                self.assert_root_version(trusted_set, 2)

                # The same role signed with the current T_new key succeeds.
                fresh_timestamp = build_timestamp([new_ts])
                trusted_set.update_timestamp(
                    fresh_timestamp.bytes_for(envelope_type)
                )
                self.assertIn(Timestamp.type, trusted_set)

    def test_updater_refresh_rejects_stale_timestamp(self) -> None:
        roots, old_ts, new_ts = self._build_chain()
        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                stale = {Timestamp.type: build_timestamp([old_ts])}
                with self.assertRaises(exceptions.UnsignedMetadataError):
                    self.run_updater(roots, stale, envelope_type)

                # A healthy repository with the new timestamp key refreshes.
                healthy = top_level_metadata(new_ts)
                updater, _ = self.run_updater(roots, healthy, envelope_type)
                self.assertIn(Timestamp.type, updater._trusted_set)  # noqa: SLF001


class TestIntermediateRootExpiry(RootRotationTestBase):
    """Expired intermediate roots are usable; only the final root is checked."""

    def _build_chain(
        self, final_expiry: datetime
    ) -> tuple[SignedMetadata, SignedMetadata, SignedMetadata]:
        signers = make_signers(4)
        root_v1 = build_root(1, signers[:2], signers[:2], 2)
        # v2 already expired at REFERENCE_TIME but is signed by both the old
        # (v1) and the new (v2/v3) threshold sets, so the migration is valid.
        root_v2 = build_root(
            2,
            signers[2:],
            signers,
            2,
            expires=PAST_EXPIRY,
        )
        # v3 keeps the new keys and expires as requested.
        root_v3 = build_root(
            3, signers[2:], signers[2:], 2, expires=final_expiry
        )
        return root_v1, root_v2, root_v3

    def _final_root_with_timestamp(
        self,
        root_v3: SignedMetadata,
        timestamp_signer: Signer,
        expires: datetime = FUTURE_EXPIRY,
    ) -> SignedMetadata:
        signed = root_v3.signed
        assert isinstance(signed, Root)
        threshold = signed.roles[Root.type].threshold
        # Authorize the signer for every non-root role so a full refresh can
        # verify timestamp, snapshot and targets against the final root.
        return build_root_with_top_level_key(
            3,
            root_v3.signers,
            root_v3.signers,
            threshold,
            timestamp_signer,
            expires=expires,
        )

    def test_expired_intermediate_with_valid_final_is_accepted(self) -> None:
        root_v1, root_v2, root_v3 = self._build_chain(FUTURE_EXPIRY)
        timestamp_signer = make_signers(1)[0]
        root_v3 = self._final_root_with_timestamp(root_v3, timestamp_signer)

        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                trusted_set = self.make_trusted_set(root_v1, envelope_type)
                # The expired intermediate loads without an expiry error...
                trusted_set.update_root(root_v2.bytes_for(envelope_type))
                self.assert_root_version(trusted_set, 2)
                trusted_set.update_root(root_v3.bytes_for(envelope_type))
                self.assert_root_version(trusted_set, 3)

                # ...and a timestamp load succeeds because only the final root
                # expiry is enforced at this step.
                trusted_set.update_timestamp(
                    build_timestamp([timestamp_signer]).bytes_for(
                        envelope_type
                    )
                )
                self.assertIn(Timestamp.type, trusted_set)

    def test_expired_final_root_is_rejected_when_timestamp_loads(self) -> None:
        root_v1, root_v2, root_v3 = self._build_chain(PAST_EXPIRY)
        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                trusted_set = self.make_trusted_set(root_v1, envelope_type)
                trusted_set.update_root(root_v2.bytes_for(envelope_type))
                trusted_set.update_root(root_v3.bytes_for(envelope_type))
                self.assert_root_version(trusted_set, 3)
                self.assert_role_absent(trusted_set, Timestamp.type)

                timestamp_signer = make_signers(1)[0]
                timestamp = build_timestamp([timestamp_signer])
                # The final root being expired is reported only at the
                # timestamp step, not during the root migration itself.
                with self.assertRaises(exceptions.ExpiredMetadataError):
                    trusted_set.update_timestamp(
                        timestamp.bytes_for(envelope_type)
                    )
                # The timestamp is not installed on a failed final-root check.
                self.assert_role_absent(trusted_set, Timestamp.type)
                self.assert_root_version(trusted_set, 3)

    def test_updater_accepts_expired_intermediate_valid_final(self) -> None:
        root_v1, root_v2, root_v3 = self._build_chain(FUTURE_EXPIRY)
        timestamp_signer = make_signers(1)[0]
        root_v3 = self._final_root_with_timestamp(root_v3, timestamp_signer)
        other = top_level_metadata(timestamp_signer)

        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                updater, metadata_dir = self.run_updater(
                    [root_v1, root_v2, root_v3], other, envelope_type
                )
                self.assertEqual(
                    updater._trusted_set.root.version, 3  # noqa: SLF001
                )
                self.assertEqual(
                    self.persisted_root_versions(metadata_dir), [1, 2, 3]
                )

    def test_updater_rejects_expired_final_root(self) -> None:
        root_v1, root_v2, root_v3 = self._build_chain(PAST_EXPIRY)
        timestamp_signer = make_signers(1)[0]
        # Even an expired final root authorizes the (irrelevant) timestamp key.
        root_v3 = self._final_root_with_timestamp(
            root_v3, timestamp_signer, expires=PAST_EXPIRY
        )
        other = {Timestamp.type: build_timestamp([timestamp_signer])}

        for envelope_type in ENVELOPES:
            with self.subTest(envelope_type=envelope_type):
                metadata_dir = tempfile.mkdtemp()
                updater = self.make_updater(
                    metadata_dir,
                    [root_v1, root_v2, root_v3],
                    other,
                    envelope_type,
                )
                # The full root history is persisted (root migration itself is
                # allowed), but refresh fails at the timestamp expiry check.
                with frozen_reference_time():
                    with self.assertRaises(exceptions.ExpiredMetadataError):
                        updater.refresh()
                self.assertEqual(
                    self.persisted_root_versions(metadata_dir), [1, 2, 3]
                )
                self.assertFalse(
                    os.path.exists(
                        os.path.join(metadata_dir, "timestamp.json")
                    )
                )


if __name__ == "__main__":
    unittest.main()
