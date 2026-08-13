#!/usr/bin/env bash
# Sync my course implementation commits from the course repo into this project.
#
# Run from the standalone repo root. Usage: scripts/sync-course.sh [--no-push]
#
# Mechanism:
#   - The tag `sync/course` points at the last course-repo commit that has
#     already been transferred here.
#   - `git cherry upstream/main course/main sync/course` lists commits on
#     course/main since the last sync, marking with '+' only those whose patch
#     is NOT equivalent to an upstream (course-infra/refsol) commit. Upstream
#     updates merged into the course branch can therefore never leak across.
#   - Commits are cherry-picked one at a time; the tag advances after EACH
#     success, so a conflict mid-run can be resolved (`git cherry-pick
#     --continue`) and the script re-run to resume exactly where it stopped.
#     Use `git cherry-pick --skip` to drop an obsolete commit, `--abort` to
#     bail out entirely (the tag stays at the last fully-applied commit).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

LAST=$(git rev-parse -q --verify sync/course) \
  || { echo "error: sync/course tag missing — run one-time setup first" >&2; exit 1; }

git fetch -q course main
git fetch -q upstream main

# Commits to transfer, oldest-first.
PICKS=$(git cherry upstream/main course/main "$LAST" \
        | awk '$1=="+" {a[i++]=$2} END {for (j=i-1; j>=0; j--) print a[j]}')

if [ -z "$PICKS" ]; then
  echo "Nothing to sync — standalone repo already contains all course commits."
  exit 0
fi

echo "Commits to transfer:"
for sha in $PICKS; do
  echo "  $(git log -1 --format='%h %ad %s' --date=short "$sha")"
done
echo
echo "Files affected:"
git log --no-walk --name-only --format='' $PICKS | grep -v '^$' | sort -u | sed 's/^/  /'
echo
read -rp "Proceed? [y/N] " ans
[ "$ans" = "y" ] || { echo "Aborted."; exit 1; }

for sha in $PICKS; do
  echo "==> cherry-picking $sha: $(git log -1 --format=%s $sha)"
  git cherry-pick "$sha"
  git tag -f sync/course "$sha"
done

git push -qf origin sync/course
if [ "${1:-}" = "--no-push" ]; then
  echo "Sync complete (not pushed; run 'git push origin main' when satisfied)."
else
  git push -q origin main
  echo "Sync complete and pushed. Rebuild extensions and run tests to verify."
fi
