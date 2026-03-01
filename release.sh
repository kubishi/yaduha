#!/usr/bin/env bash
set -euo pipefail

# Usage: ./release.sh [patch|minor|major]
# Defaults to "patch" if no argument given.

BUMP_TYPE="${1:-patch}"

# -- Read current version from pyproject.toml --
CURRENT=$(grep -m1 '^version' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"
MAJOR="${MAJOR:-0}"
MINOR="${MINOR:-0}"
PATCH="${PATCH:-0}"

case "$BUMP_TYPE" in
  patch) PATCH=$((PATCH + 1)) ;;
  minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
  major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
  *)     echo "Usage: $0 [patch|minor|major]"; exit 1 ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
TAG="v$NEW_VERSION"

echo "Bumping version: $CURRENT → $NEW_VERSION"

# -- Update pyproject.toml --
sed -i "s/^version = \"$CURRENT\"/version = \"$NEW_VERSION\"/" pyproject.toml

# -- Commit, tag, push --
git add pyproject.toml
git commit -m "Release $TAG"
git tag "$TAG"
git push origin main
git push origin "$TAG"

# -- Create GitHub release (triggers publish workflow) --
gh release create "$TAG" \
  --title "$TAG" \
  --generate-notes

echo "Released $TAG — PyPI publish workflow triggered."
