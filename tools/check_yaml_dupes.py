#!/usr/bin/env python3
"""Detect duplicate mapping keys in YAML files.

YAML spec discourages duplicate keys; many parsers either override silently or behave
in a non-obvious way. This script finds duplicates with file/line context.

Implementation notes:
- Uses PyYAML with a custom loader that records keys and raises on duplicates.
- Falls back to a lightweight text scan if PyYAML isn't available.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Dupe:
    file: str
    line: int
    key: str
    message: str


def _scan_text(path: str) -> List[Dupe]:
    # Very lightweight heuristic: only catches duplicates within the same indentation level
    # of a contiguous mapping block.
    dupes: List[Dupe] = []
    stack: List[Tuple[int, Dict[str, int]]] = []  # (indent, {key: first_line})

    key_re = re.compile(r"^(?P<indent>\s*)(?P<key>[^#:\n]+?)\s*:\s*(?:#.*)?$")

    with open(path, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue

            m = key_re.match(line)
            if not m:
                continue

            indent = len(m.group("indent"))
            key = m.group("key").strip().strip("\"'")

            # unwind stack to current indent
            while stack and indent < stack[-1][0]:
                stack.pop()

            if not stack or indent > stack[-1][0]:
                stack.append((indent, {}))

            mapping = stack[-1][1]
            if key in mapping:
                dupes.append(
                    Dupe(
                        file=path,
                        line=idx,
                        key=key,
                        message=f"duplicate key '{key}' (first seen at line {mapping[key]})",
                    )
                )
            else:
                mapping[key] = idx

    return dupes


def _scan_pyyaml(path: str) -> List[Dupe]:
    try:
        import yaml
    except Exception:
        return _scan_text(path)

    dupes: List[Dupe] = []

    class Loader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: Loader, node, deep=False):
        mapping = {}
        seen: Dict[str, int] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if isinstance(key, str):
                if key in seen:
                    dupes.append(
                        Dupe(
                            file=path,
                            line=key_node.start_mark.line + 1,
                            key=key,
                            message=(
                                f"duplicate key '{key}' (first at line {seen[key]})"
                            ),
                        )
                    )
                else:
                    seen[key] = key_node.start_mark.line + 1
            value = loader.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

    Loader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )

    try:
        with open(path, "r", encoding="utf-8") as f:
            yaml.load(f, Loader=Loader)
    except Exception:
        # Parsing failed; still return dupes collected so far.
        pass

    return dupes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="*",
        default=["configs/*.yaml"],
        help="Glob(s) of yaml files to check",
    )
    args = ap.parse_args()

    files: List[str] = []
    for p in args.paths:
        files.extend(glob.glob(p))

    files = sorted({os.path.normpath(f) for f in files})
    if not files:
        print("No files matched.", file=sys.stderr)
        return 2

    all_dupes: List[Dupe] = []
    for f in files:
        all_dupes.extend(_scan_pyyaml(f))

    if not all_dupes:
        print("OK: no duplicate keys found")
        return 0

    for d in all_dupes:
        rel = os.path.relpath(d.file)
        print(f"{rel}:{d.line}: {d.message}")

    print(f"Found {len(all_dupes)} duplicate key(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
