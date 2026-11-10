#!/usr/bin/env python3
# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Generate offline verification demo bundles.

For each supported envelope type (traditional Metadata and DSSE SIMPLE) the
script emits one successful bundle and several failure bundles, each of
which mutates exactly one piece of evidence relative to the success bundle:

* ``02-corrupt-target``      target bytes altered (length/hash mismatch);
* ``03-tampered-timestamp``  timestamp bytes altered without re-signing;
* ``04-missing-delegation``  delegated role metadata removed;
* ``05-missing-target``      target file bytes removed;
* ``06-conflicting-roots``   same root version shipped with conflicting bytes;
* ``07-untrusted-extra-role`` an extra role is bundled but not delegated.

Run with the repository root on PYTHONPATH (the script adds it automatically)
and an output directory as argument::

    python examples/offline_verification/generate_bundles.py ./offline-bundles
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from tuf.api.metadata import Metadata
from tuf.ngclient.config import EnvelopeType

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bundle_builder import write_bundle
from simulated_repo import OfflineRepositorySimulator

TARGET_PATH = "vetted/release-1.0.tar.gz"
TARGET_CONTENT = b"offline-release-1.0 contents\n"


def _copy_tree(src: str, dst: str) -> None:
    import shutil

    shutil.copytree(src, dst)


def _manifest_path(bundle_dir: str) -> str:
    return os.path.join(bundle_dir, "manifest.json")


def _load_manifest(bundle_dir: str) -> dict:
    with open(_manifest_path(bundle_dir), "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def _save_manifest(bundle_dir: str, manifest: dict) -> None:
    with open(_manifest_path(bundle_dir), "wb") as f:
        f.write(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))


def _mutate_tail(path: str) -> None:
    with open(path, "rb") as f:
        data = f.read()
    with open(path, "wb") as f:
        f.write(data[:-1] + bytes([data[-1] ^ 0xFF]))


def _reformat_json(path: str) -> None:
    """Rewrite JSON with different whitespace: same document, different bytes.

    Signature bytes still cover the canonical payload so TUF would accept the
    document on its own; the point is that the bundle now holds two different
    byte artifacts for the same role/version.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    compact = json.dumps(document, separators=(",", ":"))
    Path(path).write_text(compact, encoding="utf-8")


def _tamper_signature(path: str) -> None:
    """Flip one bit inside a signature while keeping the document parseable.

    The signed payload stays byte-identical but the signature no longer
    verifies: this isolates signature/trust failure from parse failure.
    Traditional Metadata stores ``sig`` as hex while DSSE SIMPLE envelopes
    store it base64-encoded; the right codec is chosen from the document
    shape (envelopes have a ``payload`` field).
    """
    import base64

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    sig_field = document["signatures"][0]["sig"]
    if "payload" in document:
        raw = bytearray(base64.b64decode(sig_field))
        raw[-1] ^= 0x01
        new_sig = base64.b64encode(bytes(raw)).decode()
    else:
        raw = bytearray.fromhex(sig_field)
        raw[-1] ^= 0x01
        new_sig = raw.hex()
    document["signatures"][0]["sig"] = new_sig
    Path(path).write_text(json.dumps(document), encoding="utf-8")


def generate_envelope_bundles(
    sim: OfflineRepositorySimulator, out_dir: str
) -> None:
    # 01 - success bundle, including an unrelated extra role that must not be
    # trusted merely because it is present.
    success_dir = os.path.join(out_dir, "01-success")
    rogue = _build_rogue_role(sim)
    write_bundle(
        sim,
        success_dir,
        TARGET_PATH,
        extra_roles={"rogue-role": rogue},
    )

    # 02 - target byte corruption (single-byte flip).
    corrupt_dir = os.path.join(out_dir, "02-corrupt-target")
    _copy_tree(success_dir, corrupt_dir)
    _mutate_tail(os.path.join(corrupt_dir, "targets", TARGET_PATH))

    # 03 - tampered timestamp: invalidate one signature while keeping the
    # signed payload and JSON structure untouched.
    tampered_dir = os.path.join(out_dir, "03-tampered-timestamp")
    _copy_tree(success_dir, tampered_dir)
    _tamper_signature(os.path.join(tampered_dir, "metadata", "timestamp.json"))

    # 04 - missing delegated role: remove vetted metadata and its manifest
    # entry so the chain cannot reach the target.
    missing_role_dir = os.path.join(out_dir, "04-missing-delegation")
    _copy_tree(success_dir, missing_role_dir)
    os.remove(os.path.join(missing_role_dir, "metadata", "vetted.json"))
    manifest = _load_manifest(missing_role_dir)
    manifest["metadata"].pop("vetted", None)
    _save_manifest(missing_role_dir, manifest)

    # 05 - missing target bytes.
    missing_target_dir = os.path.join(out_dir, "05-missing-target")
    _copy_tree(success_dir, missing_target_dir)
    os.remove(os.path.join(missing_target_dir, "targets", TARGET_PATH))
    manifest = _load_manifest(missing_target_dir)
    manifest["targets"] = {}
    _save_manifest(missing_target_dir, manifest)

    # 06 - conflicting root v2 bytes: the manifest lists two entries for the
    # same role/version with different (but parseable) bytes. Bundle loading
    # must reject this before any TUF verification runs.
    conflict_dir = os.path.join(out_dir, "06-conflicting-roots")
    _copy_tree(success_dir, conflict_dir)
    root_path = os.path.join(conflict_dir, "metadata", "2.root.json")
    _reformat_json(root_path)
    manifest = _load_manifest(conflict_dir)
    manifest["metadata"]["root"].append(
        {"version": 2, "path": "metadata/2.root.duplicate.json"}
    )
    # Keep the original bytes as the duplicate: entries now disagree.
    original = os.path.join(success_dir, "metadata", "2.root.json")
    shutil.copyfile(
        original,
        os.path.join(conflict_dir, "metadata", "2.root.duplicate.json"),
    )
    _save_manifest(conflict_dir, manifest)

    # 07 - target is referenced only by a well-formed role that is not part
    # of any delegation graph and not listed in snapshot. Verification must
    # ignore that bundled role and report the target as not found through any
    # trusted path.
    untrusted_dir = os.path.join(out_dir, "07-untrusted-extra-role")
    write_bundle(sim, untrusted_dir, TARGET_PATH)
    _strip_target_from_trusted_graph(sim, untrusted_dir)
    _inject_target_only_in_rogue(sim, untrusted_dir)


def _build_rogue_role(sim: OfflineRepositorySimulator) -> bytes:
    """Produce a well-formed but never-delegated targets role."""
    from securesystemslib.signer import CryptoSigner

    from tuf.api.metadata import TargetFile, Targets

    signer = CryptoSigner.generate_ed25519()
    targets = Targets(expires=sim.safe_expiry)
    targets.targets[TARGET_PATH] = TargetFile.from_data(
        TARGET_PATH, TARGET_CONTENT, ["sha256"]
    )
    md = Metadata(targets)
    md.sign(signer)
    if sim.envelope_type.name == "SIMPLE":
        from tuf.api.dsse import SimpleEnvelope

        envelope = SimpleEnvelope.from_signed(targets)
        envelope.sign(signer)
        return envelope.to_bytes()
    return md.to_bytes()


def _inject_target_only_in_rogue(
    sim: OfflineRepositorySimulator, bundle_dir: str
) -> None:
    rogue = _build_rogue_role(sim)
    path = os.path.join(bundle_dir, "metadata", "rogue-role.extra.json")
    with open(path, "wb") as f:
        f.write(rogue)
    manifest = _load_manifest(bundle_dir)
    manifest["metadata"].setdefault("rogue-role", []).append(
        {"path": "metadata/rogue-role.extra.json"}
    )
    _save_manifest(bundle_dir, manifest)


def _strip_target_from_trusted_graph(
    sim: OfflineRepositorySimulator, bundle_dir: str
) -> None:
    """Remove the demo target from the delegated, snapshot-pinned role.

    The target bytes remain in the bundle and the rogue extra role still
    claims them, but no role reachable through the trust graph references the
    target. Snapshot meta for ``vetted.json`` is refreshed so the bundle is
    internally consistent: the only difference from 01 is the missing target
    claim in trusted metadata.
    """
    from tuf.api.metadata import MetaFile

    vetted_md = sim.md_delegates["vetted"]
    vetted_md.signed.targets.pop(TARGET_PATH, None)
    vetted_bytes = sim.fetch_metadata("vetted")
    with open(os.path.join(bundle_dir, "metadata", "vetted.json"), "wb") as f:
        f.write(vetted_bytes)

    import hashlib

    digest = hashlib.sha256(vetted_bytes).hexdigest()
    snapshot_md = sim.md_snapshot
    snapshot_md.signed.meta["vetted.json"] = MetaFile(
        version=vetted_md.signed.version,
        length=len(vetted_bytes),
        hashes={"sha256": digest},
    )
    snapshot_bytes = sim.fetch_metadata("snapshot")
    with open(os.path.join(bundle_dir, "metadata", "snapshot.json"), "wb") as f:
        f.write(snapshot_bytes)

    # Timestamp pins snapshot length+hashes: refresh it as well, signed by the
    # current (possibly rotated) timestamp key.
    sim.update_timestamp()
    timestamp_bytes = sim.fetch_metadata("timestamp")
    with open(
        os.path.join(bundle_dir, "metadata", "timestamp.json"), "wb"
    ) as f:
        f.write(timestamp_bytes)

    # Refresh version labels in the manifest for the rewritten roles.
    manifest = _load_manifest(bundle_dir)
    for role, data in (
        ("vetted", vetted_bytes),
        ("snapshot", snapshot_bytes),
        ("timestamp", timestamp_bytes),
    ):
        version = _read_version(sim, data)
        for entry in manifest["metadata"][role]:
            entry["version"] = version
    _save_manifest(bundle_dir, manifest)


def _read_version(sim: OfflineRepositorySimulator, data: bytes) -> int:
    from tuf.api.dsse import SimpleEnvelope
    from tuf.api.metadata import Metadata as TufMetadata

    if sim.envelope_type is EnvelopeType.SIMPLE:
        return SimpleEnvelope.from_bytes(data).get_signed().version
    return TufMetadata.from_bytes(data).signed.version


def main() -> int:
    out_root = sys.argv[1] if len(sys.argv) > 1 else "./offline-bundles"
    shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(out_root, exist_ok=True)

    for envelope_type in (EnvelopeType.METADATA, EnvelopeType.SIMPLE):
        envelope_name = envelope_type.name.lower()
        sim = OfflineRepositorySimulator(envelope_type)
        sim.build_demo_repository(TARGET_PATH, TARGET_CONTENT)
        sim.rotate_root()
        # Rotation re-signs timestamp; snapshot/timestamp meta hashes must
        # describe the current bytes, so refresh once more.
        sim.update_snapshot()

        out_dir = os.path.join(out_root, envelope_name)
        generate_envelope_bundles(sim, out_dir)
        print(f"generated {envelope_name} bundles under {out_dir}")

    print(f"target path: {TARGET_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
