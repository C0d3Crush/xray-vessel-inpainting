#!/usr/bin/env python3
"""
Semantic versioning utility for ARCADE inpainting project.

Usage:
    python scripts/version.py                    # Show current version
    python scripts/version.py bump patch         # 1.0.0 -> 1.0.1
    python scripts/version.py bump minor         # 1.0.0 -> 1.1.0  
    python scripts/version.py bump major         # 1.0.0 -> 2.0.0
    python scripts/version.py next feat          # Get next version for feat: commit
    python scripts/version.py next fix           # Get next version for fix: commit
    python scripts/version.py next docs          # Get next version for docs: commit
"""

import sys
import argparse
from pathlib import Path

VERSION_FILE = Path(__file__).parent.parent / "VERSION"

def get_current_version():
    """Read current version from VERSION file."""
    if not VERSION_FILE.exists():
        return "0.0.0"
    return VERSION_FILE.read_text().strip()

def parse_version(version_str):
    """Parse version string into (major, minor, patch) tuple."""
    try:
        parts = version_str.split('.')
        return tuple(int(x) for x in parts[:3])
    except (ValueError, IndexError):
        return (0, 0, 0)

def format_version(major, minor, patch):
    """Format version tuple into string."""
    return f"{major}.{minor}.{patch}"

def bump_version(current, bump_type):
    """Bump version based on type."""
    major, minor, patch = parse_version(current)
    
    if bump_type == "major":
        return format_version(major + 1, 0, 0)
    elif bump_type == "minor":
        return format_version(major, minor + 1, 0)
    elif bump_type == "patch":
        return format_version(major, minor, patch + 1)
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")

def get_next_version_for_commit_type(commit_type):
    """Get next version based on conventional commit type."""
    current = get_current_version()
    
    # Semantic versioning rules for conventional commits
    if commit_type in ["feat", "feature"]:
        return bump_version(current, "minor")
    elif commit_type in ["fix", "bugfix", "hotfix"]:
        return bump_version(current, "patch")
    elif commit_type in ["docs", "style", "refactor", "test", "chore", "ci"]:
        return bump_version(current, "patch")
    elif commit_type in ["breaking", "major"]:
        return bump_version(current, "major")
    else:
        # Default to patch for unknown types
        return bump_version(current, "patch")

def update_version_file(new_version):
    """Update VERSION file with new version."""
    VERSION_FILE.write_text(new_version + "\n")
    print(f"Updated VERSION to {new_version}")

def main():
    parser = argparse.ArgumentParser(description="Semantic versioning utility")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Show current version (default)
    parser.set_defaults(command="show")
    
    # Bump version
    bump_parser = subparsers.add_parser("bump", help="Bump version")
    bump_parser.add_argument("type", choices=["major", "minor", "patch"], 
                           help="Type of version bump")
    bump_parser.add_argument("--write", action="store_true", 
                           help="Write new version to file")
    
    # Get next version for commit type
    next_parser = subparsers.add_parser("next", help="Get next version for commit type")
    next_parser.add_argument("commit_type", help="Conventional commit type (feat, fix, docs, etc.)")
    next_parser.add_argument("--write", action="store_true",
                           help="Write new version to file")
    
    args = parser.parse_args()
    
    current_version = get_current_version()
    
    if args.command == "show" or not args.command:
        print(current_version)
        
    elif args.command == "bump":
        new_version = bump_version(current_version, args.type)
        if args.write:
            update_version_file(new_version)
        else:
            print(new_version)
            
    elif args.command == "next":
        new_version = get_next_version_for_commit_type(args.commit_type)
        if args.write:
            update_version_file(new_version)
        else:
            print(new_version)

if __name__ == "__main__":
    main()