"""Persisted, line-oriented change chunks backed by Jujutsu (JJ)."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


# What: Describe one author's contiguous edit to one repository file.
# How: Store a 1-based inclusive line range and replacement text. An empty
#      range uses end_line == start_line - 1 and represents an insertion.
# Why: A small, single-file unit gives AVECON precise attribution boundaries.
@dataclass(frozen=True)
class ChangeChunk:
    """One author's edit to one contiguous range in one file."""

    author_name: str
    author_email: str
    file_path: str
    start_line: int
    end_line: int
    replacement: str
    timestamp: str | datetime | None = None
    message: str = "Automated checkpoint"
    chunk_id: str = field(default_factory=lambda: uuid4().hex)
    parent_id: str | None = None
    child_id: str | None = None

    # What: Normalize an optional timestamp to an ISO-8601 UTC value.
    # How: Accept timezone-aware datetime objects or timezone-aware strings.
    # Why: Ordering must not depend on the machine's local timezone.
    def __post_init__(self) -> None:
        if self.timestamp is not None:
            object.__setattr__(self, "timestamp", _normalise_timestamp(self.timestamp))

    # What: Serialize the public chunk fields for JSON persistence.
    # How: Convert the frozen dataclass to a dictionary and normalize timestamps.
    # Why: The persistence layer should not need to understand datetime objects.
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.timestamp is not None:
            data["timestamp"] = _normalise_timestamp(self.timestamp)
        return data

    # What: Recreate a chunk from persisted JSON data.
    # How: Route fields through the normal constructor and its validation hooks.
    # Why: Loaded data must follow the same normalization rules as new data.
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChangeChunk:
        return cls(**data)


# What: Normalize a timestamp to a timezone-aware UTC ISO string.
# How: Parse strings with datetime.fromisoformat and reject naive values.
# Why: Naive timestamps make collision ordering ambiguous across collaborators.
def _normalise_timestamp(value: str | datetime) -> str:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


# What: Count logical text lines while preserving the empty-text case.
# How: Use splitlines with newline preservation; non-empty text is one or more
#      lines and empty text is zero lines.
# Why: Range validation and line shifting need one shared definition of lines.
def _line_count(contents: str) -> int:
    return len(contents.splitlines(keepends=True))


# What: Replace or insert a contiguous range in a text value.
# How: Convert the 1-based range to Python indexes and splice replacement lines.
# Why: This is the only line arithmetic used by materialization and replay.
def _replace_lines(
    contents: str, start_line: int, end_line: int, replacement: str
) -> str:
    lines = contents.splitlines(keepends=True)
    start = start_line - 1
    end = start if end_line == start_line - 1 else end_line
    return "".join(lines[:start] + replacement.splitlines(keepends=True) + lines[end:])


# What: Detect whether two inclusive ranges or insertion points collide.
# How: Treat an empty range as a point and compare it with the other point or
#      interval using inclusive line coordinates.
# Why: Insertions must participate in collision detection without replacing a
#      line that they do not target.
def _ranges_overlap(
    first_start: int, first_end: int, second_start: int, second_end: int
) -> bool:
    first_point = first_start if first_end < first_start else None
    second_point = second_start if second_end < second_start else None
    if first_point is not None and second_point is not None:
        return first_point == second_point
    if first_point is not None:
        return second_start <= first_point <= second_end
    if second_point is not None:
        return first_start <= second_point <= first_end
    return max(first_start, second_start) <= min(first_end, second_end)


# What: Divide a colliding range at every overlapping chunk boundary.
# How: Build sorted interval boundaries and emit the smallest contiguous pieces
#      together with the IDs of chunks touching each piece.
# Why: Collision metadata must preserve where attribution was split even when a
#      replacement's text cannot be safely divided without semantic context.
def _collision_segments(
    start_line: int,
    end_line: int,
    collisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if end_line < start_line:
        return [
            {
                "start_line": start_line,
                "end_line": end_line,
                "collision_ids": [record["chunk_id"] for record in collisions],
            }
        ]
    boundaries = {start_line, end_line + 1}
    for record in collisions:
        if record["current_end_line"] < record["current_start_line"]:
            continue
        left = max(start_line, record["current_start_line"])
        right = min(end_line, record["current_end_line"])
        if left <= right:
            boundaries.update((left, right + 1))
    ordered = sorted(boundaries)
    segments: list[dict[str, Any]] = []
    for left, right in itertools.pairwise(ordered):
        segment_end = right - 1
        segment_collisions = [
            record["chunk_id"]
            for record in collisions
            if _ranges_overlap(
                left,
                segment_end,
                record["current_start_line"],
                record["current_end_line"],
            )
        ]
        segments.append(
            {
                "start_line": left,
                "end_line": segment_end,
                "collision_ids": segment_collisions,
            }
        )
    return segments


# What: Coordinate persisted pending chunks and their JJ commits.
# How: Capture file baselines, materialize each accepted edit immediately, keep
#      internal snapshots for replay, and store all state inside .jj/avecon.
# Why: Pending attribution must survive process restarts without becoming a JJ
#      working-tree file itself.
class VersionControl:
    """Manage persisted, attributed pending edits in a JJ repository."""

    _STATE_VERSION = 1

    # What: Initialize a repository-backed version-control coordinator.
    # How: Resolve the repository, verify JJ access, then load AVECON state.
    # Why: All public operations should fail early when the selected repository is
    #      not a usable JJ repository.
    def __init__(self, repository: str = ".") -> None:
        self.repository = Path(repository).resolve()
        self._run_jj("root")
        self._state_path = self.repository / ".jj" / "avecon" / "blocks.json"
        self._state = self._load_state()

    # What: Add one pending chunk.
    # How: Delegate to the batch path so timestamp and persistence behavior match.
    # Why: A one-item batch should not have a special collision or link behavior.
    def add_chunk(self, chunk: ChangeChunk) -> ChangeChunk:
        return self.add_chunks([chunk])[0]

    # What: Add and materialize pending chunks in caller order.
    # How: Validate each edit against the current expected file, detect overlaps,
    #      apply timestamp-winner materialization, relink the chain, and persist.
    # Why: Processing a batch together gives omitted timestamps one shared value
    #      while preserving deterministic insertion order as the tie-breaker.
    def add_chunks(self, chunks: Iterable[ChangeChunk]) -> list[ChangeChunk]:
        incoming = list(chunks)
        if not incoming:
            return []
        if any(not isinstance(chunk, ChangeChunk) for chunk in incoming):
            raise TypeError("all pending changes must be ChangeChunk instances")

        state = self._copy_state()
        records = state["chunks"]
        baselines = state["baselines"]
        expected = state["expected_contents"]
        working_contents: dict[str, str | None] = {}
        batch_timestamp = datetime.now(timezone.utc).isoformat()
        added: list[ChangeChunk] = []

        for original in incoming:
            chunk = (
                original
                if original.timestamp is not None
                else replace(original, timestamp=batch_timestamp)
            )
            self._validate_chunk(chunk)
            if any(record["chunk_id"] == chunk.chunk_id for record in records):
                raise ValueError(f"duplicate chunk ID: {chunk.chunk_id}")
            path = self._safe_path(chunk.file_path)
            actual = (
                working_contents[chunk.file_path]
                if chunk.file_path in working_contents
                else self._read_file(path)
            )
            if chunk.file_path in expected and actual != expected[chunk.file_path]:
                raise RuntimeError(
                    f"working tree changed outside AVECON for {chunk.file_path}; "
                    "refresh or resolve it before adding another chunk"
                )
            if chunk.file_path not in baselines:
                baselines[chunk.file_path] = actual
            current = actual or ""
            self._validate_range(chunk, current)

            collisions = [
                record
                for record in records
                if record["file_path"] == chunk.file_path
                and _ranges_overlap(
                    chunk.start_line,
                    chunk.end_line,
                    record["current_start_line"],
                    record["current_end_line"],
                )
            ]
            new_key = (chunk.timestamp or "", len(records))
            losing_collisions = [
                record
                for record in collisions
                if (record["timestamp"], record["sequence"]) > new_key
            ]
            applies = not losing_collisions
            before = actual
            after = before
            if applies:
                after = _replace_lines(
                    current, chunk.start_line, chunk.end_line, chunk.replacement
                )
                working_contents[chunk.file_path] = after
                expected[chunk.file_path] = after

            output_lines = _line_count(chunk.replacement)
            old_lines = (
                0
                if chunk.end_line == chunk.start_line - 1
                else chunk.end_line - chunk.start_line + 1
            )
            line_delta = output_lines - old_lines
            if applies and line_delta:
                for record in records:
                    if record["file_path"] != chunk.file_path:
                        continue
                    shift_after = (
                        chunk.start_line
                        if chunk.end_line == chunk.start_line - 1
                        else chunk.end_line
                    )
                    if record["current_start_line"] > shift_after:
                        record["current_start_line"] += line_delta
                        record["current_end_line"] += line_delta
            records.append(
                {
                    **chunk.to_dict(),
                    "sequence": len(records),
                    "current_start_line": chunk.start_line,
                    "current_end_line": chunk.start_line + output_lines - 1,
                    "collision_ids": [record["chunk_id"] for record in collisions],
                    "collision_segments": _collision_segments(
                        chunk.start_line, chunk.end_line, collisions
                    ),
                    "applied": applies,
                    "before_content": before,
                    "after_content": after,
                }
            )
            added.append(chunk)

        for filename, contents in working_contents.items():
            if contents != self._read_file(self._safe_path(filename)):
                self._write_file(self._safe_path(filename), contents)
        self._relink(records)
        self._state = state
        self._persist_state()
        return self.pending_chunks[-len(added) :]

    # What: Return all pending chunks in serial order.
    # How: Strip replay-only fields from each JSON record and rebuild dataclasses.
    # Why: Callers need stable attribution data, not internal snapshots.
    @property
    def pending_chunks(self) -> list[ChangeChunk]:
        return [
            ChangeChunk.from_dict(self._public_record(record))
            for record in self._state["chunks"]
            if not record.get("committed", False)
        ]

    # What: Verify and reapply the expected materialized pending files.
    # How: Refuse unexpected external changes, then write each expected snapshot.
    # Why: Materialization is safe and idempotent rather than an unconditional
    #      overwrite of edits AVECON does not know about.
    def materialize_pending(self) -> None:
        self._assert_expected_working_tree()
        for filename, contents in self._state["expected_contents"].items():
            self._write_file(self._safe_path(filename), contents)

    # What: Commit each pending edit that changes content as an individual JJ
    #      commit and return its change IDs.
    # How: Replay stored snapshots, write each intermediate prefix, commit with
    #      the chunk's author, and save a cursor after every successful commit.
    # Why: A mid-chain JJ failure must leave the remaining pending attribution
    #      recoverable rather than silently discarding it.
    def commit_pending(self) -> list[str]:
        if not self._state["chunks"]:
            return []
        records = self._state["chunks"]
        commit_records = [
            record
            for record in records
            if not record.get("committed", False)
            and record["after_content"] != record["before_content"]
        ]
        states = dict(self._state["baselines"])
        for record in records:
            if record.get("committed", False):
                states[record["file_path"]] = record["after_content"]
        if not any(record.get("committed", False) for record in records):
            self._assert_expected_working_tree()
        else:
            for filename, contents in states.items():
                if self._read_file(self._safe_path(filename)) != contents:
                    raise RuntimeError(
                        f"working tree does not match AVECON commit cursor for {filename}"
                    )
        commit_ids: list[str] = []

        for index, record in enumerate(commit_records):
            states[record["file_path"]] = record["after_content"]
            self._write_states(states)
            chunk = ChangeChunk.from_dict(self._public_record(record))
            try:
                self._run_jj(
                    "commit",
                    "-m",
                    chunk.message,
                    config={
                        "user.name": json.dumps(chunk.author_name),
                        "user.email": json.dumps(chunk.author_email),
                    },
                )
            except Exception:
                self._persist_state()
                raise
            record["committed"] = True
            self._persist_state()
            commit_ids.append(
                self._run_jj(
                    "log", "-r", "@-", "-T", "change_id", "--no-graph"
                ).stdout.strip()
            )

        self._state = self._empty_state()
        self._remove_state_file()
        return commit_ids

    # What: Discard pending metadata and, by default, restore the captured files.
    # How: Write each baseline snapshot and remove the private state file.
    # Why: Clearing state must not strand changes that AVECON can no longer track.
    def clear_pending(self, restore: bool = True) -> None:
        if restore:
            for filename, contents in self._state["baselines"].items():
                self._write_file(self._safe_path(filename), contents)
        self._state = self._empty_state()
        self._remove_state_file()

    # What: Validate chunk metadata before it enters persistent state.
    # How: Check identity, path, message, ID, and the inclusive/empty range form.
    # Why: Invalid attribution is cheaper to reject before files are modified.
    @staticmethod
    def _validate_chunk(chunk: ChangeChunk) -> None:
        if not chunk.author_name.strip() or not chunk.author_email.strip():
            raise ValueError("each chunk requires an author name and email")
        if not chunk.file_path.strip():
            raise ValueError("each chunk requires a file path")
        if not chunk.chunk_id.strip():
            raise ValueError("each chunk requires a chunk ID")
        if not chunk.message.strip():
            raise ValueError("each chunk requires a commit message")
        if chunk.start_line < 1:
            raise ValueError("start_line must be at least 1")
        if chunk.end_line < 0 or chunk.end_line < chunk.start_line - 1:
            raise ValueError("range must be inclusive or an empty insertion range")

    # What: Validate a range against the current file.
    # How: Permit insertion through EOF and require replacement ranges to fit.
    # Why: A chunk must refer to real current working-tree lines.
    @staticmethod
    def _validate_range(chunk: ChangeChunk, contents: str) -> None:
        line_count = _line_count(contents)
        if chunk.end_line == chunk.start_line - 1:
            if chunk.start_line > line_count + 1:
                raise ValueError("insertion point is outside the file")
            return
        if chunk.end_line > line_count:
            raise ValueError("chunk range extends beyond the file")

    # What: Resolve and validate a repository-relative user path.
    # How: Reject paths outside the root and all paths inside .jj metadata.
    # Why: Chunk operations must not write arbitrary files or JJ internals.
    def _safe_path(self, filename: str) -> Path:
        candidate = (self.repository / filename).resolve()
        try:
            candidate.relative_to(self.repository)
        except ValueError as error:
            raise ValueError(f"file path escapes repository: {filename}") from error
        jj_directory = self.repository / ".jj"
        if candidate == jj_directory or jj_directory in candidate.parents:
            raise ValueError("file path may not target JJ metadata")
        return candidate

    # What: Run one JJ subprocess in the configured repository.
    # How: Add repository and command-local config arguments, capture text output,
    #      and raise a readable error for non-zero results.
    # Why: Centralizing subprocess behavior makes commit behavior consistent and
    #      keeps it straightforward to replace with a fake in tests.
    def _run_jj(
        self, *arguments: str, config: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        command = ["jj", "-R", str(self.repository)]
        for key, value in (config or {}).items():
            command.extend(("--config", f"{key}={value}"))
        command.extend(arguments)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"jj {' '.join(arguments)} failed: {detail}")
        return result

    # What: Load persisted state or return clean defaults.
    # How: Parse JSON and verify its version and required top-level fields.
    # Why: Corrupt or incompatible attribution state must fail loudly.
    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return self._empty_state()
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"could not load AVECON state: {error}") from error
        if state.get("version") != self._STATE_VERSION:
            raise RuntimeError("unsupported AVECON state version")
        for key in ("chunks", "baselines", "expected_contents", "commit_cursor"):
            if key not in state:
                raise RuntimeError(f"AVECON state is missing {key}")
        return state

    # What: Create an isolated copy for a transactional add operation.
    # How: Round-trip the JSON-compatible state through json.
    # Why: A failed batch must not leave partially updated in-memory collections.
    def _copy_state(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._state))

    # What: Return the canonical empty state shape.
    # How: Initialize collections and the resumable commit cursor.
    # Why: Initialization, clearing, and completed commits must share defaults.
    @classmethod
    def _empty_state(cls) -> dict[str, Any]:
        return {
            "version": cls._STATE_VERSION,
            "chunks": [],
            "baselines": {},
            "expected_contents": {},
            "commit_cursor": 0,
        }

    # What: Rebuild the single parent/child chain.
    # How: Point each record at its immediate neighbors and leave the endpoints
    #      nullable.
    # Why: Re-linking after every append guarantees one parent and one child max.
    @staticmethod
    def _relink(records: list[dict[str, Any]]) -> None:
        for index, record in enumerate(records):
            record["parent_id"] = records[index - 1]["chunk_id"] if index else None
            record["child_id"] = (
                records[index + 1]["chunk_id"] if index + 1 < len(records) else None
            )

    # What: Read a UTF-8 repository file while preserving missing-file state.
    # How: Return None for a missing path and text for an existing regular file.
    # Why: Replay must distinguish a deleted path from an empty file.
    @staticmethod
    def _read_file(path: Path) -> str | None:
        if not path.exists():
            return None
        if not path.is_file():
            raise ValueError(f"chunk path is not a regular file: {path}")
        return path.read_text(encoding="utf-8")

    # What: Write one exact file snapshot.
    # How: Create parent directories for text, or unlink only the requested file
    #      when the snapshot is None.
    # Why: The same operation supports additions, edits, and deletions in replay.
    @staticmethod
    def _write_file(path: Path, contents: str | None) -> None:
        if contents is None:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    # What: Write every touched file for one intermediate commit state.
    # How: Apply each stored snapshot through the safe repository path.
    # Why: JJ snapshots all working-tree changes, so other pending files must be
    #      restored to the correct prefix before each individual commit.
    def _write_states(self, states: dict[str, str | None]) -> None:
        for filename, contents in states.items():
            self._write_file(self._safe_path(filename), contents)

    # What: Detect unexpected edits to files under AVECON management.
    # How: Compare current UTF-8 contents to the persisted expected snapshots.
    # Why: Never overwrite changes that have not been attributed to a chunk.
    def _assert_expected_working_tree(self) -> None:
        for filename, expected in self._state["expected_contents"].items():
            if self._read_file(self._safe_path(filename)) != expected:
                raise RuntimeError(
                    f"working tree changed outside AVECON for {filename}; "
                    "refresh or resolve it before continuing"
                )

    # What: Persist state atomically in the JJ-private AVECON directory.
    # How: Flush a temporary JSON file and replace the destination in one rename.
    # Why: Interrupted processes should not leave an unreadable pending chain.
    def _persist_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._state, indent=2, sort_keys=True) + "\n"
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self._state_path.parent, delete=False
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, self._state_path)

    # What: Remove the exact AVECON state file after pending work is cleared.
    # How: Unlink the file and tolerate an already-clean repository.
    # Why: Completed or discarded collaboration state must not be reloaded later.
    def _remove_state_file(self) -> None:
        self._state_path.unlink(missing_ok=True)

    # What: Select the public fields from an internal replay record.
    # How: Filter out snapshots, collision metadata, and cursor bookkeeping.
    # Why: Internal persistence details should not leak into the public API.
    @staticmethod
    def _public_record(record: dict[str, Any]) -> dict[str, Any]:
        fields = {
            "author_name",
            "author_email",
            "file_path",
            "start_line",
            "end_line",
            "replacement",
            "timestamp",
            "message",
            "chunk_id",
            "parent_id",
            "child_id",
        }
        return {key: value for key, value in record.items() if key in fields}


__all__ = ["ChangeChunk", "VersionControl"]
