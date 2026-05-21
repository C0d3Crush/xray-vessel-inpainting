#!/bin/bash
# Version-aware git commit helper
# Usage: ./scripts/commit-with-version.sh "feat: add new feature"

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 \"commit message\""
    echo "Examples:"
    echo "  $0 \"feat: add new feature\"     # Bumps minor version"
    echo "  $0 \"fix: resolve bug\"          # Bumps patch version" 
    echo "  $0 \"docs: update readme\"       # Bumps patch version"
    exit 1
fi

COMMIT_MSG="$1"

# Extract commit type from message (before the colon)
COMMIT_TYPE=$(echo "$COMMIT_MSG" | sed 's/:.*//' | tr '[:upper:]' '[:lower:]')

# Get current version
CURRENT_VERSION=$(python scripts/version.py)

# Get next version based on commit type
NEXT_VERSION=$(python scripts/version.py next "$COMMIT_TYPE")

echo "Current version: $CURRENT_VERSION"
echo "Commit type: $COMMIT_TYPE" 
echo "Next version: $NEXT_VERSION"
echo "Commit message: $COMMIT_MSG"
echo

# Confirm before proceeding
read -p "Proceed with commit and version bump? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Update version file
python scripts/version.py next "$COMMIT_TYPE" --write

# Add version file to commit
git add VERSION

# Create commit with version in message
git commit -m "$COMMIT_MSG [v$NEXT_VERSION]"

echo "✓ Committed with version $NEXT_VERSION"
echo "To push: git push origin $(git branch --show-current)"