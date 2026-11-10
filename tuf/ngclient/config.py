# Copyright 2021, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Configuration options for ``Updater`` class."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Flag, unique


@unique
class EnvelopeType(Flag):
    """Configures deserialization and verification mode of TUF metadata.

    Args:
        METADATA: Traditional canonical JSON -based TUF Metadata.
        SIMPLE: Dead Simple Signing Envelope. (experimental)
    """

    METADATA = 1
    SIMPLE = 2


@dataclass
class UpdaterConfig:
    """Used to store ``Updater`` configuration.

    Args:
        max_root_rotations: Maximum number of root rotations.
        max_delegations: Maximum number of delegations.
        root_max_length: Maximum length of a root metadata file.
        timestamp_max_length: Maximum length of a timestamp metadata file.
        snapshot_max_length: Maximum length of a snapshot metadata file.
        targets_max_length: Maximum length of a targets metadata file.
        prefix_targets_with_hash: When `consistent snapshots
            <https://theupdateframework.github.io/specification/latest/#consistent-snapshots>`_
            are used, target download URLs are formed by prefixing the filename
            with a hash digest of file content by default. This can be
            overridden by setting ``prefix_targets_with_hash`` to ``False``.
        envelope_type: Configures deserialization and verification mode of TUF
            metadata. Per default, it is treated as traditional canonical JSON
            -based TUF Metadata.
        app_user_agent: Application user agent, e.g. "MyApp/1.0.0". This will be
            prefixed to ngclient user agent when the default fetcher is used.
        generation_store: Store top-level metadata in generations so that
            the timestamp, snapshot and targets metadata of one ``refresh()``
            only become visible as a complete, version-consistent set. A
            process exiting mid-refresh then cannot leave a half-written
            set behind that a later process would mistake for valid cache.
            Enabling this on a legacy (flat) metadata directory migrates
            it on first use. Root metadata is not generational: verified
            root versions are append-only and never regress. Intended for
            a single writing process with any number of read-only
            processes; concurrent writers are not supported.
    """

    max_root_rotations: int = 256
    max_delegations: int = 32
    root_max_length: int = 512000  # bytes
    timestamp_max_length: int = 16384  # bytes
    snapshot_max_length: int = 2000000  # bytes
    targets_max_length: int = 5000000  # bytes
    prefix_targets_with_hash: bool = True
    envelope_type: EnvelopeType = EnvelopeType.METADATA
    app_user_agent: str | None = None
    generation_store: bool = False
