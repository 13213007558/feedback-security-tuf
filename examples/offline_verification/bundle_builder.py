# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Build offline evidence bundles from the simulated repository.

A bundle is a directory with raw signed bytes plus ``manifest.json``. The
builder deliberately writes *original* signed bytes: traditional metadata is
the canonical-JSON ``Metadata`` serialization and SIMPLE mode writes DSSE
envelope bytes, so signature inputs survive packaging untouched.
"""

from __future__ import annotations

import json
import os
from urllib import parse

from simulated_repo import OfflineRepositorySimulator

from tuf.api.metadata import Root


def write_bundle(
    sim: OfflineRepositorySimulator,
    bundle_dir: str,
    target_path: str,
    *,
    include_target: bool = True,
    extra_roles: dict[str, bytes] | None = None,
) -> None:
    """Serialize the current simulator state into an evidence bundle.

    Args:
        sim: Simulator whose state is packaged.
        bundle_dir: Output directory (created if needed).
        target_path: Target path whose bytes are placed under ``targets/``.
        include_target: When ``False`` the target bytes are omitted, yielding
            an insufficient-evidence bundle.
        extra_roles: Additional ``{role: bytes}`` metadata dropped into the
            bundle but not connected to the trust chain (used to prove that
            bundled presence alone grants no trust).
    """
    metadata_dir = os.path.join(bundle_dir, "metadata")
    targets_dir = os.path.join(bundle_dir, "targets")
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(targets_dir, exist_ok=True)

    manifest_metadata: dict[str, list[dict]] = {}

    # Root history: every published root version, version-prefixed.
    for index, _ in enumerate(sim.signed_roots, start=1):
        data = sim.fetch_metadata(Root.type, index)
        rel = f"metadata/{index}.root.json"
        _write(os.path.join(bundle_dir, rel), data)
        manifest_metadata.setdefault(Root.type, []).append(
            {"version": index, "path": rel}
        )

    # Top-level and delegated roles, labeled with the signed version.
    for role in sim.all_role_names():
        data = sim.fetch_metadata(role)
        version = _payload_version(data, sim)
        encoded = parse.quote(role, "")
        rel = f"metadata/{encoded}.json"
        _write(os.path.join(bundle_dir, rel), data)
        manifest_metadata.setdefault(role, []).append(
            {"version": version, "path": rel}
        )

    if extra_roles:
        for role, data in extra_roles.items():
            encoded = parse.quote(role, "")
            rel = f"metadata/{encoded}.extra.json"
            _write(os.path.join(bundle_dir, rel), data)
            manifest_metadata.setdefault(role, []).append({"path": rel})

    manifest_targets: dict[str, dict] = {}
    if include_target:
        repo_target = sim.target_files[target_path]
        rel = f"targets/{target_path}"
        _write(os.path.join(bundle_dir, rel), repo_target.data)
        manifest_targets[target_path] = {"path": rel}

    manifest = {
        "bootstrap_root": "metadata/1.root.json",
        "metadata": manifest_metadata,
        "targets": manifest_targets,
    }
    _write(
        os.path.join(bundle_dir, "manifest.json"),
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
    )


def _payload_version(data: bytes, sim: OfflineRepositorySimulator) -> int:
    from tuf.api.dsse import SimpleEnvelope
    from tuf.api.metadata import Metadata
    from tuf.ngclient.config import EnvelopeType

    if sim.envelope_type is EnvelopeType.SIMPLE:
        return SimpleEnvelope.from_bytes(data).get_signed().version
    return Metadata.from_bytes(data).signed.version


def _write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def make_simulator(envelope_name: str) -> OfflineRepositorySimulator:
    from tuf.ngclient.config import EnvelopeType

    envelope_type = EnvelopeType[envelope_name.upper()]
    sim = OfflineRepositorySimulator(envelope_type)
    sim.rotate_root()
    return sim
