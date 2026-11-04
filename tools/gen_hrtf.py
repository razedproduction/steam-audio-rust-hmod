#!/usr/bin/env python3
"""Converts Steam Audio's built-in HRTF (`core/src/core/cipic_124.inl`, a C byte-array
initializer) into the binary `.hrtf` file the engine crate embeds.

This does what steam-audio-rust's `build.rs` does at build time. A Horizon native bundle
crate cannot read files outside its own folder, so the binary is generated once and
committed as `mixins/steam_audio/engine/data/cipic_124.hrtf`.

Usage: python tools/gen_hrtf.py <steam-audio-rust checkout> [output]
"""

import pathlib
import re
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    checkout = pathlib.Path(sys.argv[1])
    root = pathlib.Path(__file__).resolve().parent.parent
    out = (
        pathlib.Path(sys.argv[2])
        if len(sys.argv) > 2
        else root / "mixins/steam_audio/engine/data/cipic_124.hrtf"
    )
    text = (checkout / "core/src/core/cipic_124.inl").read_text(encoding="ascii")
    start = text.index("{") + 1
    end = text.index("}", start)
    values = [int(v) for v in re.findall(r"\d+", text[start:end])]
    if any(v > 255 for v in values):
        raise SystemExit("byte value out of range in cipic_124.inl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(values))
    print(f"{out}: {len(values)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
