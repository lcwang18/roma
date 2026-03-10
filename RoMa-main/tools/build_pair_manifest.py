#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--a-dir", required=True)
    parser.add_argument("--b-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ext", nargs="*", default=[".jpg", ".jpeg", ".png", ".tif", ".tiff"])
    args = parser.parse_args()

    data_root = Path(args.data_root)
    a_dir = data_root / args.a_dir
    b_dir = data_root / args.b_dir
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ext_set = {e.lower() for e in args.ext}
    a_files = sorted([p for p in a_dir.rglob("*") if p.suffix.lower() in ext_set])

    pairs = []
    for a in a_files:
        rel = a.relative_to(a_dir)
        b = b_dir / rel
        if b.exists():
            pairs.append((a.relative_to(data_root).as_posix(), b.relative_to(data_root).as_posix()))

    with out.open("w", encoding="utf-8") as f:
        for a, b in pairs:
            f.write(f"{a} {b}\n")

    print(f"Wrote {len(pairs)} pairs to {out}")


if __name__ == "__main__":
    main()
