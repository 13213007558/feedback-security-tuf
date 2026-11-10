# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Offline TUF release verification.

This package verifies a TUF target against an *offline evidence bundle*:
the bundle is only a transport container, the trust anchor is exclusively
the caller-provided bootstrap root. All TUF validation rules (root version
chain, timestamp -> snapshot -> targets/delegations, target length and
hashes, delegation lookup order and terminating semantics) are the same
rules :mod:`tuf.ngclient` applies online; no network access or local
metadata cache is ever used.
"""

from tuf.offline._bundle import (
    BundleConflictError,
    EvidenceBundle,
    MalformedBundleError,
)
from tuf.offline._verifier import (
    OfflineVerificationResult,
    RoleEvidence,
    TargetEvidence,
    VerificationStatus,
    verify_target,
)

__all__ = [
    "BundleConflictError",
    "EvidenceBundle",
    "MalformedBundleError",
    "OfflineVerificationResult",
    "RoleEvidence",
    "TargetEvidence",
    "VerificationStatus",
    "verify_target",
]
