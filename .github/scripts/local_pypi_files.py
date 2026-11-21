"""Compare an approved package pair with PyPI before upload and after publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import urlopen

# Resolve the sibling module from the pinned checkout under isolated Python.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_release_assets import expected_package_names  # noqa: E402


def check_files(expected: dict, rows: list, *, read_back: bool) -> bool:
    """Refuse foreign or changed bytes; report if both approved distributions are present."""
    seen = set()
    for row in rows:
        name = row["filename"]
        if name not in expected or name in seen:
            raise ValueError("unexpected or duplicate PyPI distribution")
        path = expected[name]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if row["size"] != path.stat().st_size or row["digests"]["sha256"] != digest or row.get("yanked") is not False:
            raise ValueError("existing PyPI file disagrees with the approved release")
        if read_back:
            url = urlsplit(row["url"])
            if url.scheme != "https" or url.netloc != "files.pythonhosted.org":
                raise ValueError("unexpected PyPI file origin")
            with urlopen(row["url"], timeout=30) as response:
                if hashlib.file_digest(response, "sha256").hexdigest() != digest:
                    raise ValueError("PyPI download digest mismatch")
        seen.add(name)
    return seen == set(expected)


def main() -> None:
    """Preflight succeeds for missing files. Final confirmation downloads the complete pair."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages-dir", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--read-back", action="store_true")
    args = parser.parse_args()
    if re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", args.tag) is None:
        raise ValueError("invalid tag")
    version = args.tag[1:]
    expected = {path.name: path for path in args.packages_dir.iterdir()}
    if set(expected) != set(expected_package_names(version)):
        raise ValueError("invalid staged distributions")
    try:
        with urlopen(f"https://pypi.org/pypi/hermes-scope-recall/{version}/json", timeout=30) as response:
            rows = json.load(response)["urls"]
    except HTTPError as exc:
        if exc.code != 404 or args.read_back:
            raise
        rows = []
    complete = check_files(expected, rows, read_back=args.read_back)
    if args.read_back and not complete:
        raise ValueError("PyPI has not published both approved distributions")
    if not args.read_back:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            stream.write(f"complete={str(complete).lower()}\n")


if __name__ == "__main__":
    main()
