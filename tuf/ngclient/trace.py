# Copyright 2026, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Structured lookup trace for ``Updater.get_targetinfo()``.

When the same target is delegated to multiple roles, it can be difficult to
explain why the client selected one candidate over the others. A
:class:`TargetLookupTrace` can be passed to
``Updater.get_targetinfo(target_path, trace=trace)`` to record a structured,
machine-readable account of the delegation walk.

The trace is an ordered list of immutable :class:`TraceEvent` objects. Events
describe both the edges that were followed (role metadata loaded and verified)
and the reasons other branches were not followed: path pattern misses, hash
prefix misses, already visited roles (cycle protection), terminating
delegation truncation, the traversal limit, download failures and metadata
verification failures.

Trust is recorded conservatively: only an edge whose metadata was
successfully verified is marked ``trusted``. Roles that merely appear in a
delegation list, that were skipped, or whose verification failed are never
marked trusted.

A target lookup that returns ``None`` is not necessarily exhaustive: the
``outcome`` distinguishes ``NOT_FOUND_EXHAUSTED`` (every reachable branch was
checked) from ``NOT_FOUND_TRUNCATED`` (a terminating delegation or the
``max_delegations`` limit stopped the walk before all branches were checked).

A trace belongs to exactly one ``get_targetinfo()`` call. It is sealed when
the call ends and cannot be modified or reused afterwards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class TraceEventType(str, Enum):
    """Types of events recorded during a target lookup.

    The first seven values explain why a branch was not followed or why the
    lookup failed; the last two mark successful progress.
    """

    # No path pattern of the role's ``paths`` matched the target path.
    PATH_PATTERN_MISS = "path_pattern_miss"
    # None of the role's ``path_hash_prefixes`` matched the target path hash.
    HASH_PREFIX_MISS = "hash_prefix_miss"
    # The role was already visited: cycle protection skipped it.
    ROLE_ALREADY_VISITED = "role_already_visited"
    # A terminating delegation was adopted and later candidates were dropped.
    TERMINATING_TRUNCATION = "terminating_truncation"
    # The ``max_delegations`` traversal limit was reached.
    MAX_DELEGATIONS_REACHED = "max_delegations_reached"
    # Role metadata could not be downloaded.
    DOWNLOAD_FAILED = "download_failed"
    # Role metadata failed verification (signatures, version, expiry, ...).
    SIGNATURE_FAILED = "signature_failed"
    # Role metadata was loaded and verified: the edge is trusted.
    ROLE_VERIFIED = "role_verified"
    # The target was found in this role's targets metadata.
    TARGET_FOUND = "target_found"


class LookupOutcome(str, Enum):
    """Final outcome of a traced ``get_targetinfo()`` call."""

    # Target was found in ``found_in_role``.
    TARGET_FOUND = "target_found"
    # Target was not found and every reachable delegation branch was checked.
    NOT_FOUND_EXHAUSTED = "not_found_exhausted"
    # Target was not found and some branches were never checked because a
    # terminating delegation or the max_delegations limit truncated the walk.
    NOT_FOUND_TRUNCATED = "not_found_truncated"
    # The lookup aborted with an exception; see the last trace event.
    FAILED = "failed"


@dataclass(frozen=True)
class DelegationCondition:
    """The delegation condition that made a role a lookup candidate.

    Exactly one delegation mechanism applies per delegator:

      * ``kind == "paths"``: ``paths`` lists the delegated path patterns and
        ``matched_path`` is the pattern that matched the target (if any).
      * ``kind == "path_hash_prefixes"``: ``path_hash_prefixes`` lists the
        delegated prefixes and ``matched_prefix`` is the prefix that matched
        the target path hash (if any).
      * ``kind == "succinct_roles"``: the target was assigned to ``bin`` of
        a succinct roles delegation with ``bit_length`` bits.
      * ``kind == "root"``: the top-level targets role, anchored in root.

    Args:
        kind: One of ``"paths"``, ``"path_hash_prefixes"``,
            ``"succinct_roles"`` or ``"root"``.
        terminating: Whether adopting this condition stops backtracking.
        paths: Delegated path patterns for ``paths`` delegations.
        path_hash_prefixes: Delegated hash prefixes for hash-prefix
            delegations.
        matched_path: The specific path pattern that matched.
        matched_prefix: The specific hash prefix that matched.
        name_prefix: Bin name prefix for succinct roles delegations.
        bit_length: Number of bin-selection bits for succinct roles.
        bin: Selected bin name for succinct roles delegations.
    """

    kind: str
    terminating: bool = False
    paths: tuple[str, ...] | None = None
    path_hash_prefixes: tuple[str, ...] | None = None
    matched_path: str | None = None
    matched_prefix: str | None = None
    name_prefix: str | None = None
    bit_length: int | None = None
    bin: str | None = None


@dataclass(frozen=True)
class TraceEvent:
    """One immutable step of a target lookup.

    Args:
        event_type: What happened, see :class:`TraceEventType`.
        role: Name of the role the event concerns.
        parent_role: Name of the delegating role, when the event is about an
            edge between two roles.
        trusted: ``True`` only if the edge's metadata was verified
            successfully. No other event type is trusted.
        condition: The adopted delegation condition, for edges and for
            candidates that were considered.
        version: Metadata version of ``role``, known once the role metadata
            was loaded. ``None`` if it was never loaded.
        from_cache: For a ``ROLE_VERIFIED`` event, whether the metadata was
            already trusted in the metadata set (so no new download or
            signature verification happened during this call).
        skipped_roles: Roles that were dropped from the candidate set, for
            truncation and traversal-limit events.
        error: Error message for failure events.
    """

    event_type: TraceEventType
    role: str
    parent_role: str | None = None
    trusted: bool = False
    condition: DelegationCondition | None = None
    version: int | None = None
    from_cache: bool = False
    skipped_roles: tuple[str, ...] = ()
    error: str | None = None


class TargetLookupTrace:
    """Structured record of one ``Updater.get_targetinfo()`` lookup.

    Create an instance, pass it as the ``trace`` argument of
    ``get_targetinfo()``, and inspect it once the call returns. The trace is
    sealed at the end of that call and cannot be reused: each lookup needs its
    own ``TargetLookupTrace`` so that saved records are never mutated by later
    lookups.
    """

    def __init__(self) -> None:
        self._target_path: str | None = None
        self._events: list[TraceEvent] = []
        self._outcome: LookupOutcome | None = None
        self._found_in_role: str | None = None
        self._sealed = False

    @property
    def target_path(self) -> str | None:
        """Target path of the lookup, or ``None`` before the first call."""
        return self._target_path

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        """Chronologically ordered, immutable events of the lookup."""
        return tuple(self._events)

    @property
    def outcome(self) -> LookupOutcome | None:
        """Final :class:`LookupOutcome`, or ``None`` while the call runs."""
        return self._outcome

    @property
    def found_in_role(self) -> str | None:
        """Role that provided the target, or ``None`` if it was not found."""
        return self._found_in_role

    @property
    def sealed(self) -> bool:
        """Whether the owning lookup call has finished."""
        return self._sealed

    @property
    def trusted_path(self) -> tuple[str, ...]:
        """Roles on the verified chain from top-level targets to the target.

        Empty if the target was not found. Only edges represented by trusted
        ``ROLE_VERIFIED`` events participate.
        """
        if self._found_in_role is None:
            return ()

        parents: dict[str, str] = {}
        for event in self._events:
            if (
                event.event_type is TraceEventType.ROLE_VERIFIED
                and event.trusted
                and event.parent_role is not None
                # Root is the trust anchor, not part of the delegation chain
                and event.condition is not None
                and event.condition.kind != "root"
            ):
                parents.setdefault(event.role, event.parent_role)

        path = [self._found_in_role]
        current = self._found_in_role
        while current in parents:
            current = parents[current]
            path.append(current)
        path.reverse()
        return tuple(path)

    def begin(self, target_path: str) -> None:
        """Bind the trace to a new lookup. Used by the updater."""
        if self._sealed:
            raise ValueError(
                "TargetLookupTrace is single-use: create a new trace for "
                "each get_targetinfo() call"
            )
        self._target_path = target_path

    def record(self, event: TraceEvent) -> None:
        """Append an event. Used by the updater; not part of the read API."""
        if self._sealed:
            raise ValueError("Cannot record into a sealed TargetLookupTrace")
        self._events.append(event)

    def seal(
        self,
        outcome: LookupOutcome,
        found_in_role: str | None = None,
    ) -> None:
        """Finalize the trace. Used by the updater; not part of the read API."""
        if self._sealed:
            raise ValueError("TargetLookupTrace is already sealed")
        self._outcome = outcome
        self._found_in_role = found_in_role
        self._sealed = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the trace."""
        return {
            "target_path": self._target_path,
            "outcome": self._outcome.value if self._outcome else None,
            "found_in_role": self._found_in_role,
            "trusted_path": list(self.trusted_path),
            "events": [_jsonable(asdict(event)) for event in self._events],
        }

    def to_text(self) -> str:
        """Return a human-readable representation for an on-call hand-off.

        The representation highlights the terminating edge that removed later
        candidates from selection, if one was encountered.
        """
        lines = [
            f"Target lookup {self._target_path!r}: "
            f"{self._outcome.value if self._outcome else 'in progress'}"
        ]
        if self.trusted_path:
            lines.append(
                "Verified trust path: " + " -> ".join(self.trusted_path)
            )
        if self._outcome is LookupOutcome.NOT_FOUND_TRUNCATED:
            lines.append(
                "Target not found, but NOT all branches were checked: the "
                "walk was truncated (see truncation events below)."
            )
        elif self._outcome is LookupOutcome.NOT_FOUND_EXHAUSTED:
            lines.append(
                "Target not found: every reachable delegation branch was "
                "checked."
            )

        for index, event in enumerate(self._events):
            lines.append(f"  [{index:>2}] {_describe_event(event)}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _describe_condition(condition: DelegationCondition) -> str:
    if condition.kind == "root":
        return "root trust anchor"
    if condition.kind == "paths":
        text = f"paths {list(condition.paths or ())}"
        if condition.matched_path is not None:
            text += f", matched {condition.matched_path!r}"
        return text
    if condition.kind == "path_hash_prefixes":
        text = f"hash prefixes {list(condition.path_hash_prefixes or ())}"
        if condition.matched_prefix is not None:
            text += f", matched {condition.matched_prefix!r}"
        return text
    return (
        f"succinct bin {condition.bin} "
        f"({condition.bit_length} bits, prefix {condition.name_prefix!r})"
    )


def _describe_event(event: TraceEvent) -> str:
    label = event.event_type.value
    text = f"{label:<28} role={event.role}"
    if event.parent_role is not None:
        text += f" parent={event.parent_role}"
    if event.version is not None:
        text += f" version={event.version}"
    if event.event_type is TraceEventType.ROLE_VERIFIED:
        text += " TRUSTED"
        if event.from_cache:
            text += " (already in trusted metadata set)"
    if event.condition is not None:
        text += f" condition=[{_describe_condition(event.condition)}]"
    if event.event_type is TraceEventType.TERMINATING_TRUNCATION:
        text += (
            "  <-- this terminating edge removes later candidates "
            f"from selection: {list(event.skipped_roles)}"
        )
    elif event.skipped_roles:
        text += f" skipped={list(event.skipped_roles)}"
    if event.error is not None:
        text += f" error={event.error!r}"
    return text
