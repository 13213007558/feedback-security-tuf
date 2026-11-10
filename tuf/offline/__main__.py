# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

r"""Command line entry point for offline TUF target verification.

Example::

    python -m tuf.offline \
        --bundle ./release-bundle \
        --target release-1.0.tar.gz \
        --at 2026-11-10T12:00:00Z \
        --envelope metadata

The bundle directory must contain a ``manifest.json``; see
:class:`tuf.offline.EvidenceBundle`. The command never opens a network
connection and never reads metadata outside the bundle. Exit status is 0
only when the target has a complete, verified chain of trust.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from tuf.api.exceptions import RepositoryError
from tuf.ngclient.config import EnvelopeType
from tuf.offline import (
    BundleConflictError,
    EvidenceBundle,
    OfflineVerificationResult,
    VerificationStatus,
    verify_target,
)

_EXIT_OK = 0
_EXIT_INVALID = 2
_EXIT_UNTRUSTED = 3
_EXIT_INSUFFICIENT = 4
_EXIT_NOT_FOUND = 5


def _parse_time(value: str) -> datetime:
    # datetime.fromisoformat handles offsets and, on modern Python, trailing Z
    normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(
            f"time {value!r} must include a timezone (e.g. +00:00 or Z)"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the offline verifier argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Offline TUF verifier: prove a target against a trusted bootstrap "
            "root using only bytes contained in an evidence bundle."
        )
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help=(
            "Directory containing manifest.json and raw metadata/target bytes"
        ),
    )
    parser.add_argument(
        "--target",
        required=True,
        help="TUF target path (as listed in targets metadata) to verify",
    )
    parser.add_argument(
        "--at",
        required=True,
        type=_parse_time,
        dest="reference_time",
        help="Explicit verification time, ISO-8601 with timezone "
        "(e.g. 2026-11-10T12:00:00Z)",
    )
    parser.add_argument(
        "--envelope",
        required=True,
        choices=[
            EnvelopeType.METADATA.name.lower(),
            EnvelopeType.SIMPLE.name.lower(),
        ],
        help="Required metadata envelope; no automatic fallback is attempted",
    )
    parser.add_argument(
        "--digests",
        nargs="+",
        default=["sha256", "sha512"],
        metavar="ALG",
        help="Hash algorithms reported in the evidence output",
    )
    parser.add_argument(
        "--output",
        choices=["json", "text"],
        default="text",
        help="Report format (default: human-readable text)",
    )
    return parser


def _render_text(result: OfflineVerificationResult) -> str:
    """Render a human-readable evidence report."""
    lines = [
        f"status: {result.status.value}",
        f"target: {result.target_path}",
        f"reference_time: {result.reference_time}",
        f"envelope: {result.envelope}",
        "",
        "roles used in verification:",
    ]
    for role in result.role_evidence:
        lines.append(
            f"  - {role.role} v{role.version} ({role.envelope}, "
            f"delegated by {role.delegator})"
        )
        lines.append(
            f"      source: {role.bundle_path} "
            f"({role.bytes_length} bytes)"
        )
        for algorithm, digest in role.bytes_hashes.items():
            lines.append(f"      file {algorithm}: {digest}")
        for algorithm, digest in role.signed_hashes.items():
            lines.append(f"      signed-payload {algorithm}: {digest}")

    if result.target_evidence is not None:
        target = result.target_evidence
        lines.extend(
            [
                "",
                "target evidence:",
                f"  path: {target.path}",
                f"  found in role: {target.found_in_role}",
                f"  source: {target.bundle_path}",
                f"  length: {target.length}",
            ]
        )
        for algorithm, digest in target.actual_hashes.items():
            expected = target.expected_hashes.get(algorithm)
            if expected is None:
                marker = "not asserted by metadata"
            elif expected == digest:
                marker = "matches metadata"
            else:
                marker = "MISMATCH"
            lines.append(f"  {algorithm}: {digest} ({marker})")

    if result.reason is not None:
        lines.extend(["", f"reason: {result.reason}"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run verification and return the process exit code."""
    args = build_parser().parse_args(argv)
    envelope_type = EnvelopeType[args.envelope.upper()]

    try:
        bundle = EvidenceBundle.from_directory(args.bundle)
    except (BundleConflictError, OSError) as e:
        print(f"bundle error: {e}", file=sys.stderr)  # noqa: T201
        return _EXIT_INVALID

    try:
        result = verify_target(
            bundle,
            args.target,
            args.reference_time,
            envelope_type,
            hash_algorithms=tuple(args.digests),
        )
    except (ValueError, RepositoryError) as e:
        print(  # noqa: T201
            f"verification error: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return _EXIT_INVALID

    report = (
        json.dumps(result.to_dict(), indent=2, sort_keys=True)
        if args.output == "json"
        else _render_text(result)
    )
    print(report)  # noqa: T201

    if result.status is VerificationStatus.SUCCESS:
        return _EXIT_OK
    if result.status is VerificationStatus.INSUFFICIENT_EVIDENCE:
        return _EXIT_INSUFFICIENT
    if result.status is VerificationStatus.TARGET_NOT_FOUND:
        return _EXIT_NOT_FOUND
    return _EXIT_UNTRUSTED


if __name__ == "__main__":
    sys.exit(main())
