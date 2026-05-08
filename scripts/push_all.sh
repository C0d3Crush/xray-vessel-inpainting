#!/bin/bash
# Push to both GitHub (origin) and GitLab remotes

set -e

BRANCH="${1:-main}"

echo "Pushing branch '$BRANCH' to all remotes..."

# Push to GitHub (origin)
echo "→ Pushing to origin (GitHub)..."
git push origin "$BRANCH"

# Push to GitLab (maps main → lukas/main due to protected branch)
echo "→ Pushing to gitlab (Uni Heidelberg)..."
git push gitlab "$BRANCH"

echo "✓ Successfully pushed to all remotes"
