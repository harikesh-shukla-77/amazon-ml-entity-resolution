#!/bin/zsh
set -euo pipefail

cd /Users/harikeshshukla/mla

echo "=== STATUS ==="
git status --short

echo "=== FETCH ==="
git fetch origin

echo "=== REBASE ==="
git pull --rebase origin main

echo "=== STAGE ==="
git add .

echo "=== STAGED ==="
git diff --cached --name-status

echo "=== LARGE FILE CHECK ==="
git diff --cached --name-only | while read f; do
    [ -f "$f" ] || continue
    size=$(stat -f%z "$f")
    if [ "$size" -gt 95000000 ]; then
        echo "ERROR: large file: $f ($size bytes)"
        exit 1
    fi
done

echo "=== COMMIT ==="
git diff --cached --quiet || git commit -m "Update Amazon ML Challenge 2026 solution"

echo "=== PUSH ==="
git push origin main

echo "=== FINAL ==="
git status
git log -1 --oneline

echo "✅ GitHub update complete"
