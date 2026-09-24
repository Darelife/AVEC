"""Tests for the persisted AVECON version-control layer."""

import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from versionControl import ChangeChunk, VersionControl


# What: Provide a temporary real JJ repository for behavior tests.
# How: Initialize a colocated JJ/Git repository and seed one text file.
# Why: The public implementation is specifically a JJ adapter, so tests should
#      exercise its actual command boundary rather than only mock subprocesses.
class VersionControlTestMixin:

	# What: Create an isolated repository before each test.
	# How: Use TemporaryDirectory and jj git init, then write predictable content.
	# Why: Tests must not affect the user's working repository or each other.
	def setUp(self) -> None:
		self.temporary = tempfile.TemporaryDirectory()
		self.root = Path(self.temporary.name)
		subprocess.run(
			["jj", "git", "init", "--colocate", str(self.root)],
			check=True,
			capture_output=True,
			text=True,
		)
		(self.root / "sample.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

	# What: Remove the temporary repository after a test.
	# How: Delegate cleanup to TemporaryDirectory.
	# Why: Temporary JJ metadata and working files should never remain behind.
	def tearDown(self) -> None:
		self.temporary.cleanup()

	# What: Verify persistence, insertion, and serial links.
	# How: Add two chunks, reload the coordinator, and inspect the public chain.
	# Why: Restart safety and one-parent/one-child ordering are core invariants.
	def test_persists_chunks_and_links_them_serially(self) -> None:
		vc = VersionControl(str(self.root))
		vc.add_chunk(ChangeChunk("Alice", "alice@example.com", "sample.txt", 2, 2, "TWO\n"))
		vc.add_chunk(ChangeChunk("Bob", "bob@example.com", "sample.txt", 4, 3, "four\n"))

		reloaded = VersionControl(str(self.root))
		chunks = reloaded.pending_chunks
		self.assertEqual(len(chunks), 2)
		self.assertIsNone(chunks[0].parent_id)
		self.assertEqual(chunks[0].child_id, chunks[1].chunk_id)
		self.assertEqual(chunks[1].parent_id, chunks[0].chunk_id)
		self.assertIsNone(chunks[1].child_id)
		self.assertEqual((self.root / "sample.txt").read_text(), "one\nTWO\nthree\nfour\n")

	# What: Verify same-line collision resolution.
	# How: Add two timestamped replacements and assert the newer text wins while
	#      both authored chunks remain pending.
	# Why: This is the deterministic interim policy before CRDT/OT integration.
	def test_newer_timestamp_wins_same_line_collision(self) -> None:
		vc = VersionControl(str(self.root))
		first_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
		second_time = first_time + timedelta(seconds=1)
		vc.add_chunk(ChangeChunk("Alice", "alice@example.com", "sample.txt", 2, 2, "old\n", first_time))
		vc.add_chunk(ChangeChunk("Bob", "bob@example.com", "sample.txt", 2, 2, "new\n", second_time))

		self.assertEqual((self.root / "sample.txt").read_text(), "one\nnew\nthree\n")
		self.assertEqual(len(vc.pending_chunks), 2)
		self.assertEqual(vc._state["chunks"][1]["collision_ids"], [vc._state["chunks"][0]["chunk_id"]])
		self.assertEqual(vc._state["chunks"][1]["collision_segments"][0]["start_line"], 2)
		self.assertEqual(vc._state["chunks"][1]["collision_segments"][0]["end_line"], 2)

	# What: Verify insertion, deletion, and invalid path protection.
	# How: Apply an insertion and a deletion, then attempt a path traversal.
	# Why: These operations cover the line-range edge cases and repository safety.
	def test_line_operations_and_path_validation(self) -> None:
		vc = VersionControl(str(self.root))
		vc.add_chunk(ChangeChunk("Alice", "alice@example.com", "sample.txt", 2, 1, "inserted\n"))
		vc.add_chunk(ChangeChunk("Bob", "bob@example.com", "sample.txt", 3, 3, ""))
		self.assertEqual((self.root / "sample.txt").read_text(), "one\ninserted\nthree\n")

		with self.assertRaises(ValueError):
			vc.add_chunk(ChangeChunk("Eve", "eve@example.com", "../outside", 1, 0, "x"))

	# What: Verify one JJ commit is produced per changing pending chunk.
	# How: Commit two edits and inspect the resulting change IDs and clean state.
	# Why: Individual commits preserve author attribution in JJ history.
	def test_commit_pending_creates_individual_commits(self) -> None:
		vc = VersionControl(str(self.root))
		vc.add_chunk(ChangeChunk("Alice", "alice@example.com", "sample.txt", 1, 1, "ONE\n"))
		vc.add_chunk(ChangeChunk("Bob", "bob@example.com", "sample.txt", 2, 2, "TWO\n"))

		commit_ids = vc.commit_pending()
		self.assertEqual(len(commit_ids), 2)
		self.assertEqual(vc.pending_chunks, [])
		self.assertFalse((self.root / ".jj" / "avecon" / "blocks.json").exists())

	# What: Verify unexpected edits are rejected instead of overwritten.
	# How: Modify a materialized file outside the coordinator before adding again.
	# Why: Attribution must never silently lose another process's work.
	def test_external_working_tree_change_is_rejected(self) -> None:
		vc = VersionControl(str(self.root))
		vc.add_chunk(ChangeChunk("Alice", "alice@example.com", "sample.txt", 1, 1, "ONE\n"))
		(self.root / "sample.txt").write_text("external\n", encoding="utf-8")

		with self.assertRaises(RuntimeError):
			vc.add_chunk(ChangeChunk("Bob", "bob@example.com", "sample.txt", 1, 1, "BOB\n"))


# What: Simulate a JJ failure after the first commit.
# How: Let normal JJ behavior handle repository validation and fail only the
#      second commit command.
# Why: The test needs to verify that the persisted pending suffix survives.
class FailingSecondCommitVersionControl(VersionControl):

	# What: Count commit calls and fail on the second one.
	# How: Intercept only the commit subcommand and delegate all other commands.
	# Why: This keeps the failure test focused on AVECON's recovery behavior.
	def __init__(self, *args: str) -> None:
		self.commit_calls = 0
		super().__init__(*args)

	# What: Inject one deterministic commit failure.
	# How: Raise on the second commit and use the real runner otherwise.
	# Why: A realistic failure boundary is needed to test resumable persistence.
	def _run_jj(self, *arguments: str, config: dict[str, str] | None = None):
		if arguments and arguments[0] == "commit":
			self.commit_calls += 1
			if self.commit_calls == 2:
				raise RuntimeError("expected test failure")
		return super()._run_jj(*arguments, config=config)


# What: Exercise commit recovery after the injected failure.
# How: Verify the first chunk is no longer pending and a new instance reloads
#      the uncommitted suffix.
# Why: Successful work must be removed from the pending view incrementally.
class VersionControlTestCase(VersionControlTestMixin, unittest.TestCase):
	"""Standard behavior tests."""


# What: Exercise commit recovery after the injected failure.
# How: Verify the first chunk is no longer pending and a new instance reloads
#      the uncommitted suffix.
# Why: Successful work must be removed from the pending view incrementally.
class CommitRecoveryTests(unittest.TestCase):

	# What: Create an isolated repository for the recovery test.
	# How: Repeat the shared fixture setup without inheriting behavior tests.
	# Why: The recovery suite should run only its focused failure scenario.
	def setUp(self) -> None:
		self.temporary = tempfile.TemporaryDirectory()
		self.root = Path(self.temporary.name)
		subprocess.run(
			["jj", "git", "init", "--colocate", str(self.root)],
			check=True,
			capture_output=True,
			text=True,
		)
		(self.root / "sample.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

	# What: Remove the recovery test's temporary repository.
	# How: Delegate cleanup to TemporaryDirectory.
	# Why: Failure simulation must not leave JJ state on disk.
	def tearDown(self) -> None:
		self.temporary.cleanup()

	# What: Preserve the pending suffix after a partial commit.
	# How: Add two edits, fail the second commit, and reload the coordinator.
	# Why: This is the critical crash/failure recovery acceptance case.
	def test_failed_commit_preserves_remaining_chunks(self) -> None:
		vc = FailingSecondCommitVersionControl(str(self.root))
		vc.add_chunk(ChangeChunk("Alice", "alice@example.com", "sample.txt", 1, 1, "ONE\n"))
		vc.add_chunk(ChangeChunk("Bob", "bob@example.com", "sample.txt", 2, 2, "TWO\n"))

		with self.assertRaises(RuntimeError):
			vc.commit_pending()
		self.assertEqual([chunk.author_name for chunk in vc.pending_chunks], ["Bob"])

		reloaded = VersionControl(str(self.root))
		self.assertEqual([chunk.author_name for chunk in reloaded.pending_chunks], ["Bob"])


if __name__ == "__main__":
	unittest.main()
