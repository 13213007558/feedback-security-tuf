# Copyright 2020, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Client update workflow implementation.

The ``Updater`` class provides an implementation of the
`TUF client workflow
<https://theupdateframework.github.io/specification/latest/#detailed-client-workflow>`_.
``Updater`` provides an API to query available targets and to download them in a
secure manner: All downloaded files are verified by signed metadata.

High-level description of ``Updater`` functionality:
  * Initializing an ``Updater`` loads and validates the trusted local root
    metadata: This root metadata is used as the source of trust for all other
    metadata. Updater should always be initialized with the ``bootstrap``
    argument: pass ``bootstrap=None`` only to explicitly opt into using the
    cached root.json as the trust anchor.
  * ``refresh()`` can optionally be called to update and load all top-level
    metadata as described in the specification, using both locally cached
    metadata and metadata downloaded from the remote repository. If refresh is
    not done explicitly, it will happen automatically during the first target
    info lookup.
  * ``Updater`` can be used to download targets. For each target:

      * ``Updater.get_targetinfo()`` is first used to find information about a
        specific target. This will load new targets metadata as needed (from
        local cache or remote repository).
      * ``Updater.find_cached_target()`` can optionally be used to check if a
        target file is already locally cached.
      * ``Updater.download_target()`` downloads a target file and ensures it is
        verified correct by the metadata.

Note that applications using ``Updater`` should be 'single instance'
applications: running multiple instances that use the same cache directories at
the same time is not supported.

A simple example of using the Updater to implement a Python TUF client that
downloads target files is available in `examples/client
<https://github.com/theupdateframework/python-tuf/tree/develop/examples/client>`_.

Notes on how Updater uses HTTP by default:
  * urllib3 is the HTTP library
  * Typically all requests are retried by urllib3 three times (in cases where
    this seems useful)
  * Operating system certificate store is used for TLS, in other words
    ``certifi`` is not used as the certificate source
  * Proxy use can be configured with ``https_proxy`` and other similar
    environment variables

All of the HTTP decisions can be changed with ``fetcher`` argument:
Custom ``FetcherInterface`` implementations are possible. The alternative
``RequestsFetcher`` implementation is also provided (although deprecated).
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib import parse

from tuf.api import exceptions
from tuf.api.metadata import (
    Delegations,
    Root,
    Snapshot,
    TargetFile,
    Targets,
    Timestamp,
)
from tuf.ngclient._internal.trusted_metadata_set import TrustedMetadataSet
from tuf.ngclient.config import EnvelopeType, UpdaterConfig
from tuf.ngclient.trace import (
    DelegationCondition,
    LookupOutcome,
    TargetLookupTrace,
    TraceEvent,
    TraceEventType,
)
from tuf.ngclient.urllib3_fetcher import Urllib3Fetcher

if TYPE_CHECKING:
    from tuf.ngclient.fetcher import FetcherInterface

logger = logging.getLogger(__name__)


class Updater:
    """Creates a new ``Updater`` instance and loads trusted root metadata.

    Args:
        metadata_dir: Local metadata directory. Directory must be
            writable. If ``bootstrap`` is ``None``, this directory must contain
            a trusted root.json file.
        metadata_base_url: Base URL for all remote metadata downloads
        target_dir: Local targets directory. Directory must be writable. It
            will be used as the default target download directory by
            ``find_cached_target()`` and ``download_target()``
        target_base_url: ``Optional``; Default base URL for all remote target
            downloads. Can be individually set in ``download_target()``
        fetcher: ``Optional``; ``FetcherInterface`` implementation used to
            download both metadata and targets. Default is ``Urllib3Fetcher``
        config: ``Optional``; ``UpdaterConfig`` could be used to setup common
            configuration options.
        bootstrap: Initial root metadata bytes. This argument is required.
            Pass the embedded root metadata bytes for secure initialization.
            Pass ``None`` only if you explicitly want to use the cached
            root.json as the trust anchor (not recommended for most
            deployments).

    Raises:
        OSError: Local root.json cannot be read
        RepositoryError: Local root.json is invalid
    """

    def __init__(  # noqa: PLR0917
        self,
        metadata_dir: str,
        metadata_base_url: str,
        target_dir: str | None = None,
        target_base_url: str | None = None,
        fetcher: FetcherInterface | None = None,
        config: UpdaterConfig | None = None,
        *,
        bootstrap: bytes | None,
    ):
        self._dir = metadata_dir
        self._metadata_base_url = _ensure_trailing_slash(metadata_base_url)
        self.target_dir = target_dir
        if target_base_url is None:
            self._target_base_url = None
        else:
            self._target_base_url = _ensure_trailing_slash(target_base_url)

        self.config = config or UpdaterConfig()
        if fetcher is not None:
            self._fetcher = fetcher
        else:
            self._fetcher = Urllib3Fetcher(
                app_user_agent=self.config.app_user_agent
            )
        supported_envelopes = [EnvelopeType.METADATA, EnvelopeType.SIMPLE]
        if self.config.envelope_type not in supported_envelopes:
            raise ValueError(
                f"config: envelope_type must be one of {supported_envelopes}, "
                f"got '{self.config.envelope_type}'"
            )

        if bootstrap is None:
            # if no root was provided, use the cached non-versioned root.json
            bootstrap = self._load_local_metadata(Root.type)

        # Load the initial root, make sure it's cached
        self._trusted_set = TrustedMetadataSet(
            bootstrap, self.config.envelope_type
        )
        self._persist_root(self._trusted_set.root.version, bootstrap)
        self._update_root_symlink()

    def refresh(self) -> None:
        """Refresh top-level metadata.

        Downloads, verifies, and loads metadata for the top-level roles in the
        specified order (root -> timestamp -> snapshot -> targets) implementing
        all the checks required in the TUF client workflow.

        A ``refresh()`` can be done only once during the lifetime of an Updater.
        If ``refresh()`` has not been explicitly called before the first
        ``get_targetinfo()`` call, it will be done implicitly at that time.

        The metadata for delegated roles is not updated by ``refresh()``:
        that happens on demand during ``get_targetinfo()``. However, if the
        repository uses `consistent_snapshot
        <https://theupdateframework.github.io/specification/latest/#consistent-snapshots>`_,
        then all metadata downloaded by the Updater will use the same consistent
        repository state.

        Raises:
            OSError: New metadata could not be written to disk
            RepositoryError: Metadata failed to verify in some way
            DownloadError: Download of a metadata file failed in some way
        """

        self._load_root()
        self._load_timestamp()
        self._load_snapshot()
        self._load_targets(Targets.type, Root.type)

    def _generate_target_file_path(self, targetinfo: TargetFile) -> str:
        if self.target_dir is None:
            raise ValueError("target_dir must be set if filepath is not given")

        # Use URL encoded target path as filename
        filename = parse.quote(targetinfo.path, "")
        return os.path.join(self.target_dir, filename)

    def get_targetinfo(
        self,
        target_path: str,
        trace: TargetLookupTrace | None = None,
    ) -> TargetFile | None:
        """Return ``TargetFile`` instance with information for ``target_path``.

        The return value can be used as an argument to
        ``download_target()`` and ``find_cached_target()``.

        If ``refresh()`` has not been called before calling
        ``get_targetinfo()``, the refresh will be done implicitly.

        As a side-effect this method downloads all the additional (delegated
        targets) metadata it needs to return the target information.

        Args:
            target_path: `path-relative-URL string
                <https://url.spec.whatwg.org/#path-relative-url-string>`_
                that uniquely identifies the target within the repository.
            trace: ``Optional`` :class:`TargetLookupTrace` recording the
                structured delegation walk: every role visited, why branches
                were skipped (path pattern or hash prefix miss, cycles,
                terminating delegations, traversal limit) and any download or
                metadata verification failure. Only verified edges are marked
                trusted. The trace is single-use: pass a new instance for
                every call. When omitted, lookup behavior is unchanged.

        Raises:
            OSError: New metadata could not be written to disk
            RepositoryError: Metadata failed to verify in some way
            DownloadError: Download of a metadata file failed in some way
            ValueError: ``trace`` was already used by an earlier call

        Returns:
            ``TargetFile`` instance or ``None``.
        """

        if trace is not None:
            trace.begin(target_path)
        try:
            if Targets.type not in self._trusted_set:
                self.refresh()
            return self._preorder_depth_first_walk(target_path, trace)
        except BaseException:
            if trace is not None:
                trace.seal(LookupOutcome.FAILED)
            raise

    def find_cached_target(
        self,
        targetinfo: TargetFile,
        filepath: str | None = None,
    ) -> str | None:
        """Check whether a local file is an up to date target.

        Args:
            targetinfo: ``TargetFile`` from ``get_targetinfo()``.
            filepath: Local path to file. If ``None``, a file path is
                generated based on ``target_dir`` constructor argument.

        Raises:
            ValueError: Incorrect arguments

        Returns:
            Local file path if the file is an up to date target file.
            ``None`` if file is not found or it is not up to date.
        """

        if filepath is None:
            filepath = self._generate_target_file_path(targetinfo)

        try:
            with open(filepath, "rb") as target_file:
                targetinfo.verify_length_and_hashes(target_file)
            return filepath
        except (OSError, exceptions.LengthOrHashMismatchError):
            return None

    def download_target(
        self,
        targetinfo: TargetFile,
        filepath: str | None = None,
        target_base_url: str | None = None,
    ) -> str:
        """Download the target file specified by ``targetinfo``.

        Args:
            targetinfo: ``TargetFile`` from ``get_targetinfo()``.
            filepath: Local path to download into. If ``None``, the file is
                downloaded into directory defined by ``target_dir`` constructor
                argument using a generated filename. If file already exists,
                it is overwritten.
            target_base_url: Base URL used to form the final target
                download URL. Default is the value provided in ``Updater()``

        Raises:
            ValueError: Invalid arguments
            DownloadError: Download of the target file failed in some way
            RepositoryError: Downloaded target failed to be verified in some way
            OSError: Failed to write target to file

        Returns:
            Local path to downloaded file
        """

        if filepath is None:
            filepath = self._generate_target_file_path(targetinfo)
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)

        if target_base_url is None:
            if self._target_base_url is None:
                raise ValueError(
                    "target_base_url must be set in either "
                    "download_target() or constructor"
                )

            target_base_url = self._target_base_url
        else:
            target_base_url = _ensure_trailing_slash(target_base_url)

        target_filepath = targetinfo.path
        consistent_snapshot = self._trusted_set.root.consistent_snapshot
        if consistent_snapshot and self.config.prefix_targets_with_hash:
            hashes = list(targetinfo.hashes.values())
            dirname, sep, basename = target_filepath.rpartition("/")
            target_filepath = f"{dirname}{sep}{hashes[0]}.{basename}"
        full_url = f"{target_base_url}{target_filepath}"

        with self._fetcher.download_file(
            full_url, targetinfo.length
        ) as target_file:
            targetinfo.verify_length_and_hashes(target_file)

            target_file.seek(0)
            with open(filepath, "wb") as destination_file:
                shutil.copyfileobj(target_file, destination_file)

        logger.debug("Downloaded target %s", targetinfo.path)
        return filepath

    def _download_metadata(
        self, rolename: str, length: int, version: int | None = None
    ) -> bytes:
        """Download a metadata file and return it as bytes."""
        encoded_name = parse.quote(rolename, "")
        if version is None:
            url = f"{self._metadata_base_url}{encoded_name}.json"
        else:
            url = f"{self._metadata_base_url}{version}.{encoded_name}.json"
        return self._fetcher.download_bytes(url, length)

    def _load_local_metadata(self, rolename: str) -> bytes:
        encoded_name = parse.quote(rolename, "")
        with open(os.path.join(self._dir, f"{encoded_name}.json"), "rb") as f:
            return f.read()

    def _persist_metadata(self, rolename: str, data: bytes) -> None:
        """Write metadata to disk atomically to avoid data loss.

        Use a filename _not_ prefixed with version (e.g. "timestamp.json")
        . Encode the rolename to avoid issues with e.g. path separators
        """

        encoded_name = parse.quote(rolename, "")
        filename = os.path.join(self._dir, f"{encoded_name}.json")
        self._persist_file(filename, data)

    def _persist_root(self, version: int, data: bytes) -> None:
        """Write root metadata to disk atomically to avoid data loss.

        The metadata is stored with version prefix (e.g.
        "root_history/1.root.json").
        """
        rootdir = Path(self._dir, "root_history")
        rootdir.mkdir(exist_ok=True, parents=True)
        self._persist_file(str(rootdir / f"{version}.root.json"), data)

    def _persist_file(self, filename: str, data: bytes) -> None:
        """Write a file to disk atomically to avoid data loss."""
        temp_file_name = None

        try:
            with tempfile.NamedTemporaryFile(
                dir=self._dir, delete=False
            ) as temp_file:
                temp_file_name = temp_file.name
                temp_file.write(data)
            os.replace(temp_file.name, filename)
        except OSError as e:
            # remove tempfile if we managed to create one,
            # then let the exception happen
            if temp_file_name is not None:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(temp_file_name)
            raise e

    def _update_root_symlink(self) -> None:
        """Symlink root.json to current trusted root version in root_history/"""
        linkname = os.path.join(self._dir, "root.json")
        version = self._trusted_set.root.version
        current = os.path.join("root_history", f"{version}.root.json")
        with contextlib.suppress(FileNotFoundError):
            os.remove(linkname)
        try:
            os.symlink(current, linkname)
        except OSError as e:
            if getattr(e, "winerror", None) == 1314:
                # Fallback for NTFS "required privilege is not held by client"
                os.link(os.path.join(self._dir, current), linkname)
            else:
                raise

    def _load_root(self) -> None:
        """Load root metadata.

        Sequentially load newer root metadata versions. First try to load from
        local cache and if that does not work, from the remote repository.

        If metadata is loaded from remote repository, store it in local cache.
        """

        # Update the root role
        lower_bound = self._trusted_set.root.version + 1
        upper_bound = lower_bound + self.config.max_root_rotations

        try:
            for next_version in range(lower_bound, upper_bound):
                # look for next_version in local cache
                try:
                    root_path = os.path.join(
                        self._dir, "root_history", f"{next_version}.root.json"
                    )
                    with open(root_path, "rb") as f:
                        self._trusted_set.update_root(f.read())
                    continue
                except (OSError, exceptions.RepositoryError) as e:
                    # this root did not exist locally or is invalid
                    logger.debug("Local root is not valid: %s", e)

                # next_version was not found locally, try remote
                try:
                    data = self._download_metadata(
                        Root.type,
                        self.config.root_max_length,
                        next_version,
                    )
                    self._trusted_set.update_root(data)
                    self._persist_root(next_version, data)

                except exceptions.DownloadHTTPError as exception:
                    if exception.status_code not in {403, 404}:
                        raise
                    # 404/403 means current root is newest available
                    break
        finally:
            # Make sure the non-versioned root.json links to current version
            self._update_root_symlink()

    def _load_timestamp(self) -> None:
        """Load local and remote timestamp metadata."""
        try:
            data = self._load_local_metadata(Timestamp.type)
            self._trusted_set.update_timestamp(data)
        except (OSError, exceptions.RepositoryError) as e:
            # Local timestamp does not exist or is invalid
            logger.debug("Local timestamp not valid as final: %s", e)

        # Load from remote (whether local load succeeded or not)
        data = self._download_metadata(
            Timestamp.type, self.config.timestamp_max_length
        )
        try:
            self._trusted_set.update_timestamp(data)
        except exceptions.EqualVersionNumberError:
            # If the new timestamp version is the same as current, discard the
            # new timestamp. This is normal and it shouldn't raise any error.
            return

        self._persist_metadata(Timestamp.type, data)

    def _load_snapshot(self) -> None:
        """Load local (and if needed remote) snapshot metadata."""
        try:
            data = self._load_local_metadata(Snapshot.type)
            self._trusted_set.update_snapshot(data, trusted=True)
            logger.debug("Local snapshot is valid: not downloading new one")
        except (OSError, exceptions.RepositoryError) as e:
            # Local snapshot does not exist or is invalid: update from remote
            logger.debug("Local snapshot not valid as final: %s", e)

            snapshot_meta = self._trusted_set.timestamp.snapshot_meta
            length = snapshot_meta.length or self.config.snapshot_max_length
            version = None
            if self._trusted_set.root.consistent_snapshot:
                version = snapshot_meta.version

            data = self._download_metadata(Snapshot.type, length, version)
            self._trusted_set.update_snapshot(data)
            self._persist_metadata(Snapshot.type, data)

    def _load_targets(self, role: str, parent_role: str) -> Targets:
        """Load local (and if needed remote) metadata for ``role``."""

        # Avoid loading 'role' more than once for the same parent_role
        if self._trusted_set.contains(role, parent_role):
            return cast("Targets", self._trusted_set[role])

        try:
            data = self._load_local_metadata(role)
            delegated_targets = self._trusted_set.update_delegated_targets(
                data, role, parent_role
            )
            logger.debug("Local %s is valid: not downloading new one", role)
            return delegated_targets
        except (OSError, exceptions.RepositoryError) as e:
            # Local 'role' does not exist or is invalid: update from remote
            logger.debug("Failed to load local %s: %s", role, e)

            snapshot = self._trusted_set.snapshot
            metainfo = snapshot.meta.get(f"{role}.json")
            if metainfo is None:
                raise exceptions.RepositoryError(
                    f"Role {role} was delegated but is not part of snapshot"
                ) from None

            length = metainfo.length or self.config.targets_max_length
            version = None
            if self._trusted_set.root.consistent_snapshot:
                version = metainfo.version

            data = self._download_metadata(role, length, version)
            delegated_targets = self._trusted_set.update_delegated_targets(
                data, role, parent_role
            )
            self._persist_metadata(role, data)

            return delegated_targets

    def _preorder_depth_first_walk(
        self,
        target_filepath: str,
        trace: TargetLookupTrace | None = None,
    ) -> TargetFile | None:
        """
        Interrogates the tree of target delegations in order of appearance
        (which implicitly order trustworthiness), and returns the matching
        target found in the most trusted role.
        """

        # List of delegations to be interrogated. A (role, parent role,
        # condition) tuple is needed to load and verify the delegated targets
        # metadata and to explain why the role was selected.
        delegations_to_visit: list[tuple[str, str, DelegationCondition]] = [
            (
                Targets.type,
                Root.type,
                DelegationCondition(kind="root"),
            )
        ]
        visited_role_names: set[str] = set()
        truncated = False

        # Preorder depth-first traversal of the graph of target delegations.
        while (
            len(visited_role_names) <= self.config.max_delegations
            and len(delegations_to_visit) > 0
        ):
            # Pop the role name from the top of the stack.
            role_name, parent_role, condition = delegations_to_visit.pop(-1)

            # Skip any visited current role to prevent cycles.
            if role_name in visited_role_names:
                self._handle_visited_role(
                    role_name, parent_role, condition, trace
                )
                continue

            # The metadata for 'role_name' must be downloaded/updated before
            # its targets, delegations, and child roles can be inspected.
            targets = self._load_traced_role(
                role_name, parent_role, condition, trace
            )

            target = targets.targets.get(target_filepath)
            if target is not None:
                logger.debug("Found target in current role %s", role_name)
                if trace is not None:
                    trace.record(
                        TraceEvent(
                            TraceEventType.TARGET_FOUND,
                            role_name,
                            version=targets.version,
                        )
                    )
                    trace.seal(LookupOutcome.TARGET_FOUND, role_name)
                return target

            # After preorder check, add current role to set of visited roles.
            visited_role_names.add(role_name)

            if targets.delegations is not None:
                if trace is not None:
                    self._record_delegation_misses(
                        targets.delegations,
                        target_filepath,
                        role_name,
                        trace,
                    )
                children, terminating = self._child_roles_to_visit(
                    targets.delegations, target_filepath, role_name
                )
                if terminating is not None:
                    truncated = True
                    if trace is not None:
                        self._record_terminating_truncation(
                            terminating,
                            delegations_to_visit,
                            children,
                            role_name,
                            trace,
                        )
                    delegations_to_visit = []
                # Push children in reverse order of appearance onto the
                # stack: roles are popped from the end of the list.
                delegations_to_visit.extend(reversed(children))

        truncated |= self._handle_traversal_limit(delegations_to_visit, trace)

        if trace is not None:
            outcome = (
                LookupOutcome.NOT_FOUND_TRUNCATED
                if truncated
                else LookupOutcome.NOT_FOUND_EXHAUSTED
            )
            trace.seal(outcome)

        # If this point is reached then target is not found, return None
        return None

    def _handle_visited_role(
        self,
        role_name: str,
        parent_role: str,
        condition: DelegationCondition,
        trace: TargetLookupTrace | None,
    ) -> None:
        """Record a role skipped by cycle protection, if tracing.

        A cycle is not a truncation: the role was already checked through
        another edge, so every reachable branch can still be evaluated.
        """
        logger.debug("Skipping visited current role %s", role_name)
        if trace is not None:
            trace.record(
                TraceEvent(
                    TraceEventType.ROLE_ALREADY_VISITED,
                    role_name,
                    parent_role,
                    condition=condition,
                )
            )

    def _load_traced_role(
        self,
        role_name: str,
        parent_role: str,
        condition: DelegationCondition,
        trace: TargetLookupTrace | None,
    ) -> Targets:
        """Load role metadata and record the outcome on the trace.

        Download and verification failures are recorded before being re-raised
        so the trace never presents an unverified role as trusted.
        """
        from_cache = self._trusted_set.contains(role_name, parent_role)
        try:
            targets = self._load_targets(role_name, parent_role)
        except exceptions.DownloadError as e:
            if trace is not None:
                trace.record(
                    TraceEvent(
                        TraceEventType.DOWNLOAD_FAILED,
                        role_name,
                        parent_role,
                        condition=condition,
                        error=str(e),
                    )
                )
            raise
        except exceptions.RepositoryError as e:
            if trace is not None:
                trace.record(
                    TraceEvent(
                        TraceEventType.SIGNATURE_FAILED,
                        role_name,
                        parent_role,
                        condition=condition,
                        error=str(e),
                    )
                )
            raise

        if trace is not None:
            trace.record(
                TraceEvent(
                    TraceEventType.ROLE_VERIFIED,
                    role_name,
                    parent_role,
                    trusted=True,
                    condition=condition,
                    version=targets.version,
                    from_cache=from_cache,
                )
            )
        return targets

    def _child_roles_to_visit(
        self,
        delegations: Delegations,
        target_filepath: str,
        parent_role: str,
    ) -> tuple[
        list[tuple[str, str, DelegationCondition]],
        tuple[str, str, DelegationCondition] | None,
    ]:
        """Return ordered child edges and the terminating edge, if any.

        Only the in-memory parent metadata is inspected, so this performs no
        download and visits no additional role.
        """
        children: list[tuple[str, str, DelegationCondition]] = []
        terminating: tuple[str, str, DelegationCondition] | None = None
        # NOTE: This may be a slow operation if there are many delegated roles.
        for (
            child_name,
            is_terminating,
            child_condition,
        ) in self._get_roles_for_target(delegations, target_filepath):
            logger.debug("Adding child role %s", child_name)
            edge = (child_name, parent_role, child_condition)
            children.append(edge)
            if is_terminating:
                logger.debug("Not backtracking to other roles")
                terminating = edge
                break
        return children, terminating

    def _record_terminating_truncation(
        self,
        terminating: tuple[str, str, DelegationCondition],
        delegations_to_visit: list[tuple[str, str, DelegationCondition]],
        children: list[tuple[str, str, DelegationCondition]],
        parent_role: str,
        trace: TargetLookupTrace,
    ) -> None:
        """Record the terminating edge and the candidates it removes.

        Roles still on the stack and matching siblings not enumerated yet are
        never evaluated; they are listed but not marked as visited or trusted.
        """
        child_name, _, child_condition = terminating
        skipped = [name for name, _, _ in delegations_to_visit]
        skipped.extend(name for name, _, _ in children if name != child_name)
        trace.record(
            TraceEvent(
                TraceEventType.TERMINATING_TRUNCATION,
                child_name,
                parent_role,
                condition=child_condition,
                skipped_roles=tuple(skipped),
            )
        )

    def _handle_traversal_limit(
        self,
        delegations_to_visit: list[tuple[str, str, DelegationCondition]],
        trace: TargetLookupTrace | None,
    ) -> bool:
        """Record unvisited roles when the delegation limit is reached.

        Returns whether the walk was truncated.
        """
        if not delegations_to_visit:
            return False
        logger.debug(
            "%d roles left to visit, but allowed at most %d delegations",
            len(delegations_to_visit),
            self.config.max_delegations,
        )
        if trace is not None:
            trace.record(
                TraceEvent(
                    TraceEventType.MAX_DELEGATIONS_REACHED,
                    delegations_to_visit[-1][0],
                    skipped_roles=tuple(
                        name for name, _, _ in delegations_to_visit
                    ),
                )
            )
        return True

    def _get_roles_for_target(
        self,
        delegations: Delegations,
        target_filepath: str,
    ) -> list[tuple[str, bool, DelegationCondition]]:
        """Return matching child roles with the adopted delegation condition.

        Mirrors ``Delegations.get_roles_for_target`` but additionally returns
        the condition that made each child a candidate, so a trace can explain
        the edge. Only in-memory metadata is inspected: this performs no
        downloads and visits no additional roles.
        """
        results: list[tuple[str, bool, DelegationCondition]] = []

        if delegations.roles is not None:
            digest = hashlib.sha256(target_filepath.encode()).hexdigest()
            for role in delegations.roles.values():
                if role.paths is not None:
                    matched = next(
                        (
                            pattern
                            for pattern in role.paths
                            if role.matches_path_pattern(
                                target_filepath, pattern
                            )
                        ),
                        None,
                    )
                    if matched is None:
                        continue
                    condition = DelegationCondition(
                        kind="paths",
                        terminating=role.terminating,
                        paths=tuple(role.paths),
                        matched_path=matched,
                    )
                else:
                    prefixes = role.path_hash_prefixes or []
                    matched_prefix = next(
                        (
                            prefix
                            for prefix in prefixes
                            if digest.startswith(prefix)
                        ),
                        None,
                    )
                    if matched_prefix is None:
                        continue
                    condition = DelegationCondition(
                        kind="path_hash_prefixes",
                        terminating=role.terminating,
                        path_hash_prefixes=tuple(prefixes),
                        matched_prefix=matched_prefix,
                    )
                results.append((role.name, role.terminating, condition))

        elif delegations.succinct_roles is not None:
            succinct = delegations.succinct_roles
            bin_name = succinct.get_role_for_target(target_filepath)
            # All succinct roles are treated as terminating (see TAP 15).
            condition = DelegationCondition(
                kind="succinct_roles",
                terminating=True,
                name_prefix=succinct.name_prefix,
                bit_length=succinct.bit_length,
                bin=bin_name,
            )
            results.append((bin_name, True, condition))

        return results

    def _record_delegation_misses(
        self,
        delegations: Delegations,
        target_filepath: str,
        parent_role: str,
        trace: TargetLookupTrace,
    ) -> None:
        """Record roles whose delegation condition did not match.

        Only delegations present in the already loaded parent metadata are
        evaluated: a miss event means the role was never a candidate and its
        metadata must not be downloaded.
        """
        if delegations.roles is None:
            # Succinct roles always assign a target to exactly one bin.
            return

        for role in delegations.roles.values():
            if role.is_delegated_path(target_filepath):
                continue
            if role.paths is not None:
                event_type = TraceEventType.PATH_PATTERN_MISS
                condition = DelegationCondition(
                    kind="paths",
                    terminating=role.terminating,
                    paths=tuple(role.paths),
                )
            else:
                event_type = TraceEventType.HASH_PREFIX_MISS
                condition = DelegationCondition(
                    kind="path_hash_prefixes",
                    terminating=role.terminating,
                    path_hash_prefixes=tuple(role.path_hash_prefixes or ()),
                )
            trace.record(
                TraceEvent(
                    event_type,
                    role.name,
                    parent_role,
                    condition=condition,
                )
            )


def _ensure_trailing_slash(url: str) -> str:
    """Return url guaranteed to end in a slash."""
    return url if url.endswith("/") else f"{url}/"
