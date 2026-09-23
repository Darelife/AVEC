"""Create small, ordered JJ commits for collaborative changes."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


# ChangeChunk is the small data object used to describe one checkpoint.
# What: Store the identity of the person making a change, the files to change,
#       and the message that should appear on the resulting JJ commit.
# How: Use a dataclass so Python creates the initializer and useful comparison
#      behavior from the annotated fields automatically.
# Why: Keeping these related values together makes each requested change
#      explicit and lets VersionControl process a list of uniform objects.
#
# The class is frozen so a chunk cannot be modified accidentally after it has
# been created; that keeps the requested author, files, and message stable
# while the commit is being produced.
@dataclass(frozen=True)
class ChangeChunk:
	"""One group of changes made by one person."""

	author_name: str
	author_email: str
	files: dict[str, str | None]
	message: str = "Automated checkpoint"

# VersionControl coordinates the conversion of ChangeChunk objects into JJ
# commits.
# What: Validate the repository, write each requested file change, stage it,
#       create a commit, and return the commit identifier.
# How: Keep the repository path on the instance and delegate JJ operations to
#      _run_jj, while the public commit_chunks method preserves input order.
# Why: Centralizing the workflow gives callers one simple API for producing
#      reproducible, ordered commits and keeps subprocess details private.
class VersionControl:
	"""Write change chunks as JJ commits, in the order received."""

	# What: Initialize a VersionControl object for one repository.
	# How: Convert the supplied path to an absolute Path, then ask JJ for the
	#      repository root as an immediate validity check.
	# Why: An absolute path makes later file operations unambiguous, and the
	#      early JJ command fails fast when the path is not a JJ repository.
	def __init__(self, repository: str = ".") -> None:
		self.repository = Path(repository).resolve()
		self._run_jj("root")

	# What: Create one JJ commit for every supplied change chunk and return all
	#       resulting commit IDs.
	# How: Start with an empty result list, visit chunks in their existing order,
	#      delegate each individual commit to _commit_chunk, and append its ID.
	# Why: Serial processing preserves the caller's intended history order and
	#      returning IDs lets the caller refer to the commits afterward.
	def commit_chunks(self, chunks: list[ChangeChunk]) -> list[str]:
		"""Create one commit for each chunk, serially."""
		commit_ids = []
		for chunk in chunks:
			commit_ids.append(self._commit_chunk(chunk))
		return commit_ids

	# What: Apply one ChangeChunk and create its corresponding JJ commit.
	# How: Validate metadata first; for every file, either delete it when its
	#      contents are None or create parent directories and write text content;
	#      let JJ snapshot the working copy, set the change's author through
	#      command-local configuration, commit with the requested message, and
	#      read the committed change ID.
	# Why: Validation prevents incomplete commits, handling None provides a clear
	#      deletion operation, mkdir supports nested paths, UTF-8 gives stable
	#      text encoding, JJ's working-copy snapshot captures additions/edits/
	#      deletions, explicit identity preserves attribution, and reading @-
	#      returns the exact change created by this operation.
	def _commit_chunk(self, chunk: ChangeChunk) -> str:
		self._validate_chunk(chunk)

		for filename, contents in chunk.files.items():
			# Build the path inside the selected repository so the requested file
			# operation is performed relative to that repository.
			path = self.repository / filename
			if contents is None:
				# Missing files are ignored because the requested end state is
				# deletion, even if the file has already disappeared.
				path.unlink(missing_ok=True)
				continue
			# Create missing parent folders so nested file names work without
			# requiring callers to prepare the directory structure themselves.
			path.parent.mkdir(parents=True, exist_ok=True)
			# Write the requested text as UTF-8 so file contents are predictable
			# across machines and Python's default encoding is not relied upon.
			path.write_text(contents, encoding="utf-8")

		# JJ automatically snapshots the working copy before a command runs, so
		# there is no separate add/stage step to maintain.
		jj_config = {
			"user.name": json.dumps(chunk.author_name),
			"user.email": json.dumps(chunk.author_email),
		}
		# Use command-local configuration so one contributor's identity does not
		# overwrite the repository or the next contributor's identity.
		self._run_jj(
			"commit",
			"-m",
			chunk.message,
			config=jj_config,
		)
		# jj commit creates a new working-copy change, so @- identifies the change
		# that was just committed. The template prints only its stable change ID.
		return self._run_jj(
			"log",
			"-r",
			"@-",
			"-T",
			"change_id",
			"--no-graph",
		).stdout.strip()

	# What: Check that a ChangeChunk contains the metadata required to make a
	#       meaningful commit.
	# How: Strip surrounding whitespace before checking the author name, email,
	#      and message; separately test that the files mapping is non-empty and
	#      raise ValueError with a targeted explanation when a check fails.
	# Why: Rejecting invalid input before touching files or JJ avoids partial
	#      work and gives callers a clear correction to make.
	@staticmethod
	def _validate_chunk(chunk: ChangeChunk) -> None:
		# Whitespace-only names and emails cannot provide useful JJ attribution.
		if not chunk.author_name.strip() or not chunk.author_email.strip():
			raise ValueError("Each chunk requires an author name and email")
		# An empty mapping would produce no requested file change to commit.
		if not chunk.files:
			raise ValueError("Each chunk must contain at least one file change")
		# A whitespace-only commit message would create poor or invalid history.
		if not chunk.message.strip():
			raise ValueError("Each chunk requires a commit message")

	# What: Run one JJ command in the configured repository and return its
	#       completed-process result.
	# How: Build a command beginning with jj and -R, add any command-local config,
	#      capture standard output and errors as text, and disable automatic
	#      exception raising so a custom error can be produced.
	# Why: One helper keeps subprocess behavior consistent, ensures every command
	#      targets the same repository, and turns JJ failures into readable
	#      RuntimeErrors containing the useful command detail.
	def _run_jj(
		self,
		*arguments: str,
		config: dict[str, str] | None = None,
	) -> subprocess.CompletedProcess[str]:
		# -R selects the repository without changing the process's working
		# directory; command-local config values apply only to this JJ command.
		command = ["jj", "-R", str(self.repository)]
		for key, value in (config or {}).items():
			command.extend(("--config", f"{key}={value}"))
		command.extend(arguments)

		result = subprocess.run(
			command,
			capture_output=True,
			text=True,
			check=False,
		)
		# JJ reports failure through its return code; prefer stderr, but fall back
		# to stdout because some JJ commands put diagnostics there.
		if result.returncode != 0:
			detail = result.stderr.strip() or result.stdout.strip()
			raise RuntimeError(f"jj {' '.join(arguments)} failed: {detail}")
		return result
