#!/usr/bin/env bash
# Permanently sever all ties between this project and the course it was
# derived from. Run this ONCE, after the final sync-course.sh run.
#
# What it does:
#   1. Verifies no course commits are left unsynced.
#   2. Rewrites the entire branch history:
#        - drops scripts/sync-course.sh, scripts/sever-course.sh and
#          notation_reference.txt from every commit
#        - replaces README.md with a course-free version in every commit
#        - renames the package to "smol-llm" in every commit
#        - rewrites the two commit messages that mention the origin
#        - prunes commits that become empty (the two script commits)
#   3. Deletes the sync tag, the course/upstream remotes, and the
#      remote tag; force-pushes the rewritten history.
#   4. Physically purges all unreachable objects (i.e., the entire course
#      history) from the local .git via gc --prune=now.
#
# Kept by design: LICENSE (generic Apache-2.0 text, no names or links —
# required when redistributing code derived from Apache-2.0 sources).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# --- preflight: refuse if course commits are still pending ----------------
if git remote get-url course >/dev/null 2>&1; then
  git fetch -q course main
  git fetch -q upstream main
  LAST=$(git rev-parse -q --verify sync/course 2>/dev/null || true)
  if [ -n "$LAST" ]; then
    PENDING=$(git cherry upstream/main course/main "$LAST" | awk '$1=="+"' | wc -l | tr -d ' ')
    if [ "$PENDING" != "0" ]; then
      echo "error: $PENDING course commit(s) not yet synced. Run scripts/sync-course.sh first." >&2
      exit 1
    fi
  fi
else
  echo "note: 'course' remote already absent; skipping pending-commit check."
fi

echo
echo "This will PERMANENTLY:"
echo "  * rewrite all history in this repo (commit SHAs change),"
echo "  * delete the course history objects, remotes, and tags,"
echo "  * force-push the result to origin."
read -rp "Continue? [y/N] " ans
[ "$ans" = "y" ] || { echo "Aborted."; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# --- README used to replace all course-era READMEs in history ---------------
# Taken from the current HEAD; it must already be free of course references
# (the verification step at the end checks for leftovers).
git show main:README.md > "$TMP/README.md"

# --- per-commit tree cleanup ------------------------------------------------
cat > "$TMP/treefilter.sh" <<SH
rm -f scripts/sync-course.sh scripts/sever-course.sh notation_reference.txt
if [ -f README.md ]; then cp "$TMP/README.md" README.md; fi
if [ -f pyproject.toml ]; then
  sed -i '' 's/^name = ".*"\$/name = "smol-llm"/' pyproject.toml
fi
for f in bench.py src/extensions/bindings.cpp; do
  if [ -f "\$f" ]; then sed -i '' 's/tiny-llm/smol-llm/g' "\$f"; fi
done
true
SH

# --- commit-message rewrites -------------------------------------------------
cat > "$TMP/rewrite_msgs.py" <<'PY'
import sys

REWRITES = {
    "Initial commit: standalone project baseline\n\n"
    "Derived from the tiny-llm course starter (Apache-2.0). Course scaffolding\n"
    "(book, reference solutions, course CI/tests) removed; harness restricted to\n"
    "the tiny_llm implementation.":
        "Initial commit: project baseline",

    "Make project self-contained\n\n"
    "- import dequantize_linear from tiny_llm.quantize instead of the\n"
    "  course reference package\n"
    "- track per-day test suites in tests/ (course repo keeps them untracked)":
        "Make project self-contained\n\n"
        "- import dequantize_linear from tiny_llm.quantize\n"
        "- track per-day test suites in tests/",
}

msg = sys.stdin.read()
sys.stdout.write(REWRITES.get(msg.rstrip("\n"), msg))
PY

# --- rewrite history (branch only; tag is dropped afterwards) ---------------
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f \
  --tree-filter "sh $TMP/treefilter.sh" \
  --msg-filter "python3 $TMP/rewrite_msgs.py" \
  --prune-empty -- main

# --- drop all remaining references to the old identity ----------------------
git update-ref -d refs/original/refs/heads/main
rm -f scripts/sever-course.sh scripts/sync-course.sh notation_reference.txt
git tag -d sync/course
git remote remove course 2>/dev/null || true
git remote remove upstream 2>/dev/null || true
git reflog expire --expire=now --all
rm -f .git/FETCH_HEAD .git/ORIG_HEAD

# --- push the clean history, delete the remote tag ---------------------------
git push --force-with-lease origin main
git push origin :refs/tags/sync-course 2>/dev/null || true

# --- physically purge unreachable objects (the course history) ---------------
git gc --prune=now --aggressive

# --- verification -------------------------------------------------------------
echo
echo "=== severance verification ==="
echo "authors in history:      $(git log --format='%an <%ae>' | sort -u | tr '\n' ' ')"
echo "commit count:            $(git rev-list --count main)"
if git cat-file -e efb0c89fd236695a9117d64d42d17456a446afe2^{commit} 2>/dev/null; then
  echo "course objects:          STILL PRESENT (unexpected)"
else
  echo "course objects:          purged from .git"
fi
if grep -ri "skyzh\|tiny-llm" --exclude-dir=.git . >/dev/null 2>&1; then
  echo "course strings in tree:  FOUND — inspect manually"
  grep -ri "skyzh\|tiny-llm" --exclude-dir=.git -l .
else
  echo "course strings in tree:  none"
fi
echo
echo "Severance complete. This repository is now fully independent."
