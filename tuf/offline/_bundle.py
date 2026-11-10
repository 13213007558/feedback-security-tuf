# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Offline evidence bundle loading.

An evidence bundle is a plain directory (or an equivalent in-memory mapping)
that carries *bytes* only:

* one trusted bootstrap root (the sole trust anchor),
* raw, signed metadata bytes indexed by role name and version,
* raw target file bytes indexed by target path.

The bundle itself never grants trust: a role present in the bundle is only
trusted once it passes the standard TUF verification chain. A role that is
not reachable through the verified chain is ignored, no matter how it is
named.

Multiple root versions are allowed. Two entries for the same role and
version with different bytes are a hard error (:class:`BundleConflictError`):
silently picking one would hide evidence tampering.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from urllib import parse

MANIFEST_NAME = "manifest.json"
BOOTSTRAP_FIELD = "bootstrap_root"
METADATA_FIELD = "metadata"
TARGETS_FIELD = "targets"
VERSION_FIELD = "version"
PATH_FIELD = "path"


class MalformedBundleError(ValueError):
    """Bundle layout or manifest is not structurally valid."""


class BundleConflictError(MalformedBundleError):
    """Same role/version (or target path) is present with conflicting bytes."""


@dataclass
class MetadataEntry:
    """One raw metadata artifact in a bundle.

    Attributes:
        role: TUF role name (e.g. ``"root"``, ``"targets"``, ``"vetted"``).
        version: Metadata version declared for this entry, or ``None`` when
            the entry is not version-prefixed (the version is then taken from
            the signed payload during verification).
        path: Bundle-relative path of the raw bytes (for reporting).
        data: Original signed bytes, exactly as packaged. The bytes are never
            re-serialized before signature verification.
    """

    role: str
    version: int | None
    path: str
    data: bytes


@dataclass
class TargetEntry:
    """One raw target artifact in a bundle.

    Attributes:
        path: Bundle-relative path of the raw bytes (for reporting).
        data: Original target bytes, exactly as packaged.
    """

    path: str
    data: bytes


@dataclass
class EvidenceBundle:
    """Container of raw offline evidence.

    The class only organizes and cross-checks bytes; it performs no TUF
    verification. Use :func:`tuf.offline.verify_target` for that.
    """

    bootstrap_root: bytes
    metadata: dict[str, list[MetadataEntry]] = field(default_factory=dict)
    targets: dict[str, TargetEntry] = field(default_factory=dict)
    bootstrap_path: str = BOOTSTRAP_FIELD

    # -- construction -----------------------------------------------------

    @classmethod
    def from_directory(cls, bundle_dir: str) -> EvidenceBundle:
        """Load a bundle from a directory containing ``manifest.json``.

        The manifest maps role names to one or more metadata artifacts
        (versioned or not) and target paths to target artifacts::

            {
              "bootstrap_root": "root.bootstrap.json",
              "metadata": {
                "root": [
                  {"version": 1, "path": "metadata/1.root.json"},
                  {"version": 2, "path": "metadata/2.root.json"}
                ],
                "timestamp": [{"path": "metadata/timestamp.json"}],
                "vetted": [{"version": 3, "path": "metadata/vetted.json"}]
              },
              "targets": {
                "release-1.0.tar.gz": {"path": "targets/release-1.0.tar.gz"}
              }
            }

        Raises:
            MalformedBundleError: Manifest is missing required fields or a
                listed file cannot be read.
            BundleConflictError: Two entries share role/version (or target
                path) but carry different bytes.
        """
        manifest_path = os.path.join(bundle_dir, MANIFEST_NAME)
        try:
            with open(manifest_path, "rb") as f:
                manifest = json.loads(f.read().decode("utf-8"))
        except OSError as e:
            raise MalformedBundleError(f"Cannot read manifest: {e}") from e
        except json.JSONDecodeError as e:
            raise MalformedBundleError(
                f"Manifest is not valid JSON: {e}"
            ) from e

        if not isinstance(manifest, dict) or BOOTSTRAP_FIELD not in manifest:
            raise MalformedBundleError(
                f"Manifest must be an object containing '{BOOTSTRAP_FIELD}'"
            )

        bootstrap_rel = manifest[BOOTSTRAP_FIELD]
        if not isinstance(bootstrap_rel, str):
            raise MalformedBundleError("bootstrap_root must be a path string")
        bootstrap = _read_bundle_file(bundle_dir, bootstrap_rel)

        bundle = cls(bootstrap, bootstrap_path=bootstrap_rel)
        bundle.load_manifest_metadata(bundle_dir, manifest)
        bundle.load_manifest_targets(bundle_dir, manifest)
        return bundle

    def load_manifest_metadata(
        self, bundle_dir: str, manifest: dict
    ) -> None:
        """Validate and ingest the manifest's metadata section."""
        raw_metadata = manifest.get(METADATA_FIELD, {})
        if not isinstance(raw_metadata, dict):
            raise MalformedBundleError("metadata must be an object")
        for role, entries in raw_metadata.items():
            if not isinstance(role, str) or not isinstance(entries, list):
                raise MalformedBundleError(
                    "metadata entries must map role names to lists"
                )
            for entry in entries:
                rel, version = self._parse_manifest_entry(role, entry)
                data = _read_bundle_file(bundle_dir, rel)
                self.add_metadata(role, data, version, rel)

    @staticmethod
    def _parse_manifest_entry(
        role: str, entry: object
    ) -> tuple[str, int | None]:
        if not isinstance(entry, dict) or PATH_FIELD not in entry:
            raise MalformedBundleError(
                f"metadata entry for {role!r} lacks '{PATH_FIELD}'"
            )
        rel = entry[PATH_FIELD]
        if not isinstance(rel, str):
            raise MalformedBundleError("metadata path must be a string")
        version = entry.get(VERSION_FIELD)
        if version is not None and not isinstance(version, int):
            raise MalformedBundleError(
                f"metadata version for {role!r} must be an integer"
            )
        return rel, version

    def load_manifest_targets(
        self, bundle_dir: str, manifest: dict
    ) -> None:
        """Validate and ingest the manifest's targets section."""
        raw_targets = manifest.get(TARGETS_FIELD, {})
        if not isinstance(raw_targets, dict):
            raise MalformedBundleError("targets must be an object")
        for target_path, entry in raw_targets.items():
            if not isinstance(target_path, str) or not isinstance(
                entry, dict
            ):
                raise MalformedBundleError("targets must map paths to entries")
            rel = entry.get(PATH_FIELD)
            if not isinstance(rel, str):
                raise MalformedBundleError(
                    f"target entry for {target_path!r} lacks '{PATH_FIELD}'"
                )
            self.add_target(
                target_path, _read_bundle_file(bundle_dir, rel), rel
            )

    @classmethod
    def from_bytes(
        cls,
        bootstrap_root: bytes,
        metadata: dict[str, list[tuple[int | None, bytes]]] | None = None,
        targets: dict[str, bytes] | None = None,
    ) -> EvidenceBundle:
        """Build a bundle directly from raw bytes (no filesystem).

        ``metadata`` maps role names to a list of ``(version, bytes)`` pairs
        (use ``None`` for the version when not version-prefixed). This is the
        programmatic entry point used by tests and embedding applications.
        """
        bundle = cls(bootstrap_root)
        for role, entries in (metadata or {}).items():
            for version, data in entries:
                bundle.add_metadata(role, data, version)
        for target_path, data in (targets or {}).items():
            bundle.add_target(target_path, data)
        return bundle

    # -- mutation ---------------------------------------------------------

    def add_metadata(
        self,
        role: str,
        data: bytes,
        version: int | None = None,
        path: str | None = None,
    ) -> None:
        """Add raw metadata bytes for ``role``.

        Raises:
            BundleConflictError: An entry with the same role and version but
                different bytes already exists.
        """
        if not isinstance(data, bytes):
            raise MalformedBundleError("metadata bytes must be bytes")
        label = path or _metadata_label(role, version)
        entry = MetadataEntry(role, version, label, data)

        existing = self.metadata.setdefault(role, [])
        if version is not None:
            for other in existing:
                if other.version == version and other.data != data:
                    raise BundleConflictError(
                        f"Conflicting bytes for role {role!r} version "
                        f"{version}: {other.path!r} vs {label!r}"
                    )
        else:
            # Unversioned entries identify the same artifact: duplicates must
            # be byte-identical.
            for other in existing:
                if other.version is None and other.data != data:
                    raise BundleConflictError(
                        f"Conflicting unversioned bytes for role {role!r}: "
                        f"{other.path!r} vs {label!r}"
                    )
        existing.append(entry)

    def add_target(
        self, target_path: str, data: bytes, path: str | None = None
    ) -> None:
        """Add raw target bytes for ``target_path``.

        Raises:
            BundleConflictError: The same target path is present with
                different bytes.
        """
        if not isinstance(data, bytes):
            raise MalformedBundleError("target bytes must be bytes")
        label = path or f"targets/{target_path}"
        if target_path in self.targets:
            other = self.targets[target_path]
            if other.data != data:
                raise BundleConflictError(
                    f"Conflicting bytes for target {target_path!r}: "
                    f"{other.path!r} vs {label!r}"
                )
            return
        self.targets[target_path] = TargetEntry(label, data)

    # -- access -----------------------------------------------------------

    def metadata_versions(self, role: str) -> list[MetadataEntry]:
        """Return all packaged entries for ``role`` (possibly empty)."""
        return list(self.metadata.get(role, ()))


def _read_bundle_file(bundle_dir: str, rel_path: str) -> bytes:
    full_path = os.path.join(bundle_dir, rel_path)
    try:
        with open(full_path, "rb") as f:
            return f.read()
    except OSError as e:
        raise MalformedBundleError(
            f"Cannot read bundle file {rel_path!r}: {e}"
        ) from e


def _metadata_label(role: str, version: int | None) -> str:
    encoded = parse.quote(role, "")
    if version is None:
        return f"metadata/{encoded}.json"
    return f"metadata/{version}.{encoded}.json"
