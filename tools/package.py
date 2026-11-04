#!/usr/bin/env python3
"""Packs the mod into a native `.hmod`: `horizon-native.json` plus `mixins/steam_audio/`.

The file follows Horizon's mod-file rules (docs/api/mod-files.md): a zip with stored or deflated entries,
relative `/`-separated paths, and under `mixins/<id>/` only `horizon-mixin.toml`, `.rs` and `.toml` files, files
under `data/` or a top-level folder's `data/`, and README/LICENSE/NOTICE files. The output is byte-for-byte
reproducible: sorted entries, fixed timestamps and permissions.

Usage: python tools/package.py [--out DIR]
"""

import argparse
import json
import pathlib
import re
import sys
import tomllib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
MOD_ID = "steam_audio"
BUNDLE = ROOT / "mixins" / MOD_ID
# Minecraft version of the pinned Horizon commit (`horizon_core::version::VERSION_ID`).
GAME = "26.3"
# horizon-mixins revision the pinned Horizon launcher builds with (`launcher/src/native_mods.rs` `MIXINS_REV`).
MIXINS_REV = "e8cd0aa38cffef49cb2e088ef716baa107148c02"

NOTICE_NAMES = ("README", "LICENSE", "LICENCE", "COPYING", "NOTICE")
SKIPPED_DIRS = {"target", ".git", ".local", "node_modules"}
LIMIT_FILES = 512
LIMIT_BYTES = 64 * 1024 * 1024
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def allowed(rel: pathlib.PurePosixPath) -> bool:
    """Whether a path under `mixins/<id>/` may be in a native bundle."""
    parts = rel.parts
    if rel.suffix in (".rs", ".toml"):
        return True
    if parts[0] == "data" or (len(parts) > 2 and parts[1] == "data"):
        return True
    if len(parts) <= 2 and (rel.name.startswith(NOTICE_NAMES) or rel.name == ".gitignore"):
        return True
    return False


def bundle_files() -> list[pathlib.Path]:
    files = []
    for path in sorted(BUNDLE.rglob("*")):
        rel = path.relative_to(BUNDLE)
        if any(part in SKIPPED_DIRS for part in rel.parts) or path.is_dir():
            continue
        if path.is_symlink():
            raise SystemExit(f"symlinks are not allowed: {rel}")
        if not allowed(pathlib.PurePosixPath(rel.as_posix())):
            raise SystemExit(f"not allowed in a native bundle: {rel.as_posix()}")
        files.append(path)
    return files


def native_manifest() -> dict:
    manifest = tomllib.loads((BUNDLE / "horizon-mixin.toml").read_text(encoding="utf-8"))
    mod = manifest["mod"]
    revision = manifest["compatibility"]["horizon_revision"]
    if mod["id"] != MOD_ID or not re.fullmatch(r"[a-z0-9_]{1,64}", mod["id"]):
        raise SystemExit(f"bad mod id {mod['id']!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(f"bad horizon_revision {revision!r}")
    return {
        "schema": 1,
        "id": mod["id"],
        "version": mod["version"],
        "game": GAME,
        "horizon": revision,
        "mixins": MIXINS_REV,
    }


def entry(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16  # a regular file, rw-r--r--
    info.create_system = 3
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "dist")
    parser.add_argument(
        "--hnmod",
        action="store_true",
        help="also write the same file as .hnmod (the registry's native-only extension)",
    )
    args = parser.parse_args()

    manifest = native_manifest()
    files = bundle_files()
    total = sum(f.stat().st_size for f in files)
    if len(files) + 1 > LIMIT_FILES or total > LIMIT_BYTES:
        raise SystemExit(f"over the native limits: {len(files) + 1} entries, {total} bytes")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{manifest['id']}-{manifest['version']}.hmod"
    with zipfile.ZipFile(out, "w", compresslevel=9) as zf:
        zf.writestr(entry("horizon-native.json"), json.dumps(manifest, indent=2) + "\n")
        for path in files:
            name = f"mixins/{MOD_ID}/{path.relative_to(BUNDLE).as_posix()}"
            zf.writestr(entry(name), path.read_bytes())
    print(f"{out} ({len(files) + 1} entries, {total} bytes unpacked, {out.stat().st_size} bytes)")
    if args.hnmod:
        hnmod = out.with_suffix(".hnmod")
        hnmod.write_bytes(out.read_bytes())
        print(hnmod)
    return 0


if __name__ == "__main__":
    sys.exit(main())
