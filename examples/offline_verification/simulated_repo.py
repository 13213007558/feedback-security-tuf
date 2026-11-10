# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""In-memory repository simulator for offline verification bundles.

The simulator wraps the test-suite ``RepositorySimulator`` (which already
implements the repository-side signing rules) and adds:

* emission of DSSE SIMPLE envelopes in addition to traditional Metadata;
* a second root version to exercise the root rotation chain;
* a terminating and a non-terminating delegation to exercise the target
  lookup order and terminating semantics of the offline verifier.

Everything stays in memory and requires no network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tuf.api.dsse import SimpleEnvelope
from tuf.api.metadata import (
    DelegatedRole,
    Metadata,
    Root,
    Targets,
)
from tuf.ngclient.config import EnvelopeType

if TYPE_CHECKING:
    from securesystemslib.signer import Signer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TESTS_DIR = _REPO_ROOT / "tests"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from repository_simulator import RepositorySimulator  # noqa: E402


class OfflineRepositorySimulator(RepositorySimulator):
    """Repository simulator able to publish both envelope types."""

    def __init__(self, envelope_type: EnvelopeType) -> None:
        # super().__init__() publishes the initial root, so the envelope mode
        # must exist before the parent constructor runs.
        self.envelope_type = envelope_type
        super().__init__()
        # Keep evidence realistic: timestamp pins snapshot length+hashes and
        # snapshot pins every targets role length+hashes.
        self.compute_metafile_hashes_length = True

    # -- envelope-aware serialization ------------------------------------

    def _serialize(self, role: str, md: Metadata) -> bytes:
        if self.envelope_type is EnvelopeType.SIMPLE:
            envelope = SimpleEnvelope.from_signed(md.signed)
            signers = list(self.signers[role].values())
            if not signers:
                raise ValueError(f"No signers registered for role {role!r}")
            for signer in signers:
                envelope.sign(signer)
            return envelope.to_bytes()
        return super().fetch_metadata(role)

    def fetch_metadata(self, role: str, version: int | None = None) -> bytes:
        if role == Root.type:
            # publish_root() already stored roots in the configured envelope,
            # so the pre-published bytes are returned verbatim.
            return super().fetch_metadata(role, version)
        md = self._metadata_object(role)
        return self._serialize(role, md)

    def _wrap_stored_root(self, data: bytes) -> bytes:
        md = Metadata[Root].from_bytes(data)
        envelope = SimpleEnvelope.from_signed(md.signed)
        for signer in self.signers[Root.type].values():
            envelope.sign(signer)
        return envelope.to_bytes()

    def _metadata_object(self, role: str) -> Metadata:
        if role == "timestamp":
            return self.md_timestamp
        if role == "snapshot":
            return self.md_snapshot
        if role == "targets":
            return self.md_targets
        if role not in self.md_delegates:
            raise KeyError(f"Unknown delegated role {role!r}")
        return self.md_delegates[role]

    def publish_root(self) -> None:  # type: ignore[override]
        super().publish_root()
        if self.envelope_type is EnvelopeType.SIMPLE:
            data = self._wrap_stored_root(self.signed_roots[-1])
            self.signed_roots[-1] = data

    # -- scenario construction -------------------------------------------

    def build_demo_repository(
        self, target_path: str, target_content: bytes
    ) -> None:
        """Create a multi-role repository that delegates ``target_path``.

        Layout::

            targets
              +-- "vetted" (terminating, matches "vetted/*")
              +-- "staging" (non-terminating, matches "staging/*")

        The requested target is served by ``vetted``. A second target under
        ``staging`` demonstrates that extra bundled roles gain no trust
        unless the delegation graph actually leads to them.
        """
        vetted = DelegatedRole(
            "vetted", [], 1, terminating=True, paths=["vetted/*"]
        )
        staging = DelegatedRole(
            "staging", [], 1, terminating=False, paths=["staging/*"]
        )
        self.add_delegation(
            "targets", vetted, Targets(expires=self.safe_expiry)
        )
        self.add_delegation(
            "targets", staging, Targets(expires=self.safe_expiry)
        )

        self.add_target("vetted", target_content, target_path)
        self.add_target(
            "staging", b"staging placeholder\n", "staging/notes.txt"
        )
        # Bump delegated metadata versions and propagate hashes into snapshot.
        self.update_snapshot()

    def rotate_root(self) -> None:
        """Publish root v2 with a fresh timestamp key (a real key rotation)."""
        self.root.version += 1
        self.root.expires = self.safe_expiry
        self.rotate_keys("timestamp")
        self.publish_root()
        # Re-sign timestamp with the rotated key and refresh snapshot chain.
        self.update_timestamp()

    def all_role_names(self) -> list[str]:
        return ["timestamp", "snapshot", "targets", *self.md_delegates]

    def signers_for(self, role: str) -> list[Signer]:
        return list(self.signers[role].values())
