# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Generational local storage for top-level metadata.

``GenerationStore`` is an optional local metadata layout for ``Updater``
that makes crash recovery deterministic: the timestamp, snapshot and
targets metadata produced by one ``refresh()`` are only considered usable
as a set. A refresh writes the new top-level metadata into a new
*generation* directory and only once every file is durably written, the
``current`` pointer is atomically switched to the new generation. A
process that exits mid-refresh leaves an incomplete generation behind:
recovery ignores it and keeps using the last fully committed generation.

Layout inside the metadata directory::

    root.json                   # symlink to root_history/<N>.root.json
    root_history/               # append-only, shared across generations
    generations/
        current                 # text file: committed generation number
        1/
            timestamp.json
            snapshot.json
            targets.json
            <delegated>.json    # persisted on demand, matches snapshot
        2/
            ...

Root metadata is deliberately not part of a generation: every root in
``root_history`` was individually verified against both the previous and
the new root threshold, so root trust is monotonic and must never regress
when a top-level refresh fails. Recovery always starts from the newest
trusted root and never reloads an older root from an older generation.

The store supports a single writing process and multiple read-only
processes. Concurrent writers are not coordinated: that is left to the
application. Committed generation directories are immutable; generations
newer than ``current`` are incomplete leftovers of an interrupted refresh
and are removed when the next refresh starts.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
from urllib import parse

from tuf.api import exceptions
from tuf.api.metadata import Metadata, Snapshot, Targets, Timestamp

logger = logging.getLogger(__name__)

TOP_LEVEL_ROLES = (Timestamp.type, Snapshot.type, Targets.type)


class GenerationStoreError(exceptions.RepositoryError):
    """The generation store on disk is inconsistent or corrupt."""


class GenerationStore:
    """Manages the generational metadata layout inside ``metadata_dir``.

    Args:
        metadata_dir: Local metadata directory used by ``Updater``.
    """

    def __init__(self, metadata_dir: str):
        self._dir = metadata_dir
        self._generations_dir = os.path.join(metadata_dir, "generations")
        self._pointer_file = os.path.join(self._generations_dir, "current")
        # generation number currently being written, None if no refresh
        # is in progress
        self._pending: int | None = None

    # ------------------------------------------------------------------
    # initialization, migration and recovery validation

    def initialize(self) -> None:
        """Prepare the store, migrating a legacy flat layout if found.

        Raises:
            GenerationStoreError: The on-disk state (legacy or
                generational) is corrupt, incomplete or inconsistent.
            OSError: The store could not be read or written.
        """
        if not os.path.isdir(self._generations_dir):
            self._migrate_flat_layout()
        self._validate_current()

    def _migrate_flat_layout(self) -> None:
        """Move legacy flat metadata files into generation 1.

        A legacy directory either contains all of timestamp, snapshot and
        targets metadata (a usable set that becomes generation 1) or none
        of them (a fresh client that starts with an empty store). Anything
        in between is the result of an interrupted legacy refresh and is
        rejected with a clear error instead of being silently discarded.
        """
        present = {
            role
            for role in TOP_LEVEL_ROLES
            if os.path.isfile(os.path.join(self._dir, f"{role}.json"))
        }
        os.makedirs(self._generations_dir, exist_ok=True)
        if not present:
            logger.debug("No legacy metadata found: starting empty store")
            return
        missing = set(TOP_LEVEL_ROLES) - present
        if missing:
            raise GenerationStoreError(
                "Cannot migrate legacy metadata cache: incomplete top-level"
                f" set, missing {sorted(missing)}. Remove the cache and"
                " bootstrap again."
            )

        # Validate before moving anything so a failed migration leaves
        # the legacy layout untouched
        self._validate_generation_data(
            1,
            {
                role: self._read_file(os.path.join(self._dir, f"{role}.json"))
                for role in TOP_LEVEL_ROLES
            },
        )

        generation_dir = self._generation_dir(1)
        os.mkdir(generation_dir)
        for entry in os.scandir(self._dir):
            # move all role metadata (including delegated roles, which
            # belong to the legacy snapshot) but not root.json or dirs
            if (
                entry.is_file()
                and entry.name.endswith(".json")
                and entry.name != "root.json"
            ):
                os.replace(entry.path, os.path.join(generation_dir, entry.name))
        self._write_pointer(1)
        logger.info("Migrated legacy metadata cache into generation 1")

    def _validate_current(self) -> None:
        """Validate the committed generation, if one exists.

        Raises:
            GenerationStoreError: The committed generation is missing
                files, contains unparseable metadata or the versions of
                timestamp, snapshot and targets do not agree.
        """
        current = self.current
        if current is None:
            return
        data: dict[str, bytes] = {}
        for role in TOP_LEVEL_ROLES:
            path = os.path.join(self._generation_dir(current), f"{role}.json")
            if not os.path.isfile(path):
                raise GenerationStoreError(
                    f"Committed generation {current} is missing"
                    f" {role}.json: the cache is corrupt, remove it and"
                    " bootstrap again"
                )
            data[role] = self._read_file(path)
        self._validate_generation_data(current, data)

    @staticmethod
    def _validate_generation_data(number: int, data: dict[str, bytes]) -> None:
        """Check that a generation is parseable and version-consistent.

        Signature verification is not done here: it happens when the
        metadata is loaded into ``TrustedMetadataSet`` during refresh.
        """
        parsed: dict[str, Metadata] = {}
        for role, raw in data.items():
            try:
                parsed[role] = Metadata.from_bytes(raw)
            except Exception as e:
                raise GenerationStoreError(
                    f"Generation {number}: {role}.json is not valid"
                    f" metadata: {e}"
                ) from e

        timestamp = parsed[Timestamp.type].signed
        snapshot = parsed[Snapshot.type].signed
        targets = parsed[Targets.type].signed
        if not isinstance(timestamp, Timestamp):
            raise GenerationStoreError(
                f"Generation {number}: timestamp.json contains"
                f" '{timestamp.type}' metadata"
            )
        if not isinstance(snapshot, Snapshot):
            raise GenerationStoreError(
                f"Generation {number}: snapshot.json contains"
                f" '{snapshot.type}' metadata"
            )
        if not isinstance(targets, Targets):
            raise GenerationStoreError(
                f"Generation {number}: targets.json contains"
                f" '{targets.type}' metadata"
            )
        if timestamp.snapshot_meta.version != snapshot.version:
            raise GenerationStoreError(
                f"Generation {number}: timestamp refers to snapshot"
                f" v{timestamp.snapshot_meta.version} but snapshot.json is"
                f" v{snapshot.version}: refusing to use a mixed set"
            )
        targets_meta = snapshot.meta.get(f"{Targets.type}.json")
        if targets_meta is None or targets_meta.version != targets.version:
            raise GenerationStoreError(
                f"Generation {number}: snapshot refers to targets"
                f" v{None if targets_meta is None else targets_meta.version}"
                f" but targets.json is v{targets.version}: refusing to use"
                " a mixed set"
            )

    # ------------------------------------------------------------------
    # reading

    @property
    def current(self) -> int | None:
        """Committed generation number, None if none was committed yet.

        Raises:
            GenerationStoreError: The pointer exists but is corrupt or
                points to a generation that does not exist.
        """
        try:
            with open(self._pointer_file, encoding="utf-8") as f:
                raw = f.read().strip()
        except FileNotFoundError:
            return None
        try:
            number = int(raw)
        except ValueError as e:
            raise GenerationStoreError(
                f"Generation pointer {self._pointer_file} is corrupt:"
                f" {raw!r} is not a generation number"
            ) from e
        if number < 1 or not os.path.isdir(self._generation_dir(number)):
            raise GenerationStoreError(
                f"Generation pointer refers to generation {number} but"
                f" {self._generation_dir(number)} does not exist: the cache"
                " is corrupt, remove it and bootstrap again"
            )
        return number

    def load(self, rolename: str) -> bytes:
        """Read ``rolename`` from the committed generation.

        Raises:
            FileNotFoundError: No generation is committed or the role is
                not part of it. Callers treat this as a cache miss.
        """
        current = self.current
        if current is None:
            raise FileNotFoundError(
                f"No committed generation: {rolename} is not cached"
            )
        encoded_name = parse.quote(rolename, "")
        path = os.path.join(
            self._generation_dir(current), f"{encoded_name}.json"
        )
        with open(path, "rb") as f:
            return f.read()

    # ------------------------------------------------------------------
    # writing (single writer process only)

    def begin(self) -> None:
        """Start a new pending generation and remove interrupted ones.

        Generation directories newer than ``current`` can only be
        leftovers of a refresh that was interrupted before commit: they
        were never usable and are deleted.
        """
        current = self.current
        for entry in os.scandir(self._generations_dir):
            if not entry.is_dir():
                continue
            try:
                number = int(entry.name)
            except ValueError:
                number = -1
            if number < 1 or (current is not None and number > current):
                logger.debug("Removing incomplete generation %s", entry.name)
                shutil.rmtree(entry.path, ignore_errors=True)
        self._pending = 1 if current is None else current + 1

    def persist(self, rolename: str, data: bytes) -> None:
        """Write ``rolename`` atomically into the applicable generation.

        During a refresh (after ``begin()``) metadata goes to the pending
        generation and only becomes visible at ``commit()``. Outside a
        refresh (delegated roles fetched on demand) metadata is added to
        the committed generation it was verified against.
        """
        encoded_name = parse.quote(rolename, "")
        if self._pending is not None:
            generation = self._pending
        else:
            current = self.current
            if current is None:
                raise GenerationStoreError(
                    f"Cannot persist {rolename}: no refresh in progress and"
                    " no committed generation exists"
                )
            generation = current
        generation_dir = self._generation_dir(generation)
        os.makedirs(generation_dir, exist_ok=True)
        filename = os.path.join(generation_dir, f"{encoded_name}.json")
        _persist_file(filename, data)

    def commit(self) -> None:
        """Atomically make the pending generation the committed one.

        Files that the refresh did not rewrite (because the previously
        committed file was still valid under the new timestamp/snapshot)
        are copied from the previous generation so every committed
        generation is self-contained.
        """
        if self._pending is None:
            return
        pending_dir = self._generation_dir(self._pending)
        if not os.path.isdir(pending_dir):
            # refresh did not need to write anything: nothing to commit
            self._pending = None
            return

        current = self.current
        if current is not None:
            current_dir = self._generation_dir(current)
            for entry in os.scandir(current_dir):
                target = os.path.join(pending_dir, entry.name)
                if entry.is_file() and not os.path.exists(target):
                    shutil.copyfile(entry.path, target)

        data = {}
        for role in TOP_LEVEL_ROLES:
            path = os.path.join(pending_dir, f"{role}.json")
            if not os.path.isfile(path):
                raise GenerationStoreError(
                    f"Cannot commit generation {self._pending}: {role}.json"
                    " was not persisted and no previous generation provides"
                    " it"
                )
            data[role] = self._read_file(path)
        self._validate_generation_data(self._pending, data)

        self._write_pointer(self._pending)
        logger.debug("Committed generation %d", self._pending)
        self._pending = None

    def abort(self) -> None:
        """Discard the pending generation after a failed refresh."""
        if self._pending is None:
            return
        shutil.rmtree(self._generation_dir(self._pending), ignore_errors=True)
        self._pending = None

    # ------------------------------------------------------------------

    def _generation_dir(self, number: int) -> str:
        return os.path.join(self._generations_dir, str(number))

    def _write_pointer(self, number: int) -> None:
        _persist_file(self._pointer_file, f"{number}\n".encode())

    @staticmethod
    def _read_file(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()


def _persist_file(filename: str, data: bytes) -> None:
    """Write a file to disk atomically to avoid data loss."""
    temp_file_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=os.path.dirname(filename), delete=False
        ) as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(data)
        os.replace(temp_file.name, filename)
    except OSError as e:
        if temp_file_name is not None:
            with contextlib.suppress(FileNotFoundError):
                os.remove(temp_file_name)
        raise e
