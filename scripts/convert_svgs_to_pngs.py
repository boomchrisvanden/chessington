#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _normalize_piece_stem(stem: str) -> str | None:
    """
    Normalize common chess piece stems so they match the GUI's expectations:
      wp.svg -> wP.png, bn.svg -> bN.png, etc.
    Returns None if it's not a 2-char piece code.
    """
    if len(stem) != 2:
        return None
    color = stem[0].lower()
    piece = stem[1].lower()
    if color not in ("w", "b"):
        return None
    if piece not in "pnbrqk":
        return None
    return f"{color}{piece.upper()}"


def svg_to_png(svg_path: Path, png_path: Path, size_px: int) -> bool:
    png_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cairosvg  # type: ignore

        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            output_width=size_px,
            output_height=size_px,
        )
        return png_path.exists()
    except Exception:
        pass

    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        subprocess.run(
            [rsvg, "-w", str(size_px), "-h", str(size_px), "-o", str(png_path), str(svg_path)],
            check=False,
        )
        return png_path.exists()

    inkscape = shutil.which("inkscape")
    if inkscape:
        subprocess.run(
            [
                inkscape,
                str(svg_path),
                "--export-type=png",
                f"--export-filename={png_path}",
                "-w",
                str(size_px),
                "-h",
                str(size_px),
            ],
            check=False,
        )
        return png_path.exists()

    raise RuntimeError(
        "No SVG->PNG converter found. Install one of: "
        "cairosvg (pip), rsvg-convert (librsvg), or inkscape."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert all .svg files under a directory to .png.")
    parser.add_argument(
        "assets_dir",
        nargs="?",
        default="assets",
        help="Directory to scan (default: ./assets)",
    )
    parser.add_argument("--size", type=int, default=256, help="Output PNG width/height in px (default: 256).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if PNG exists and is newer than SVG.",
    )

    args = parser.parse_args()

    assets_dir = Path(args.assets_dir)
    if not assets_dir.exists():
        raise SystemExit(f"Directory not found: {assets_dir}")

    svg_files = [p for p in assets_dir.rglob("*.svg") if p.is_file()]
    if not svg_files:
        print(f"No .svg files found under {assets_dir}")
        return 0

    converted = 0
    for svg_path in sorted(svg_files):
        normalized = _normalize_piece_stem(svg_path.stem)
        if normalized is None:
            png_path = svg_path.with_suffix(".png")
        else:
            png_path = svg_path.with_name(normalized).with_suffix(".png")

        if not args.force and png_path.exists():
            try:
                if png_path.stat().st_mtime >= svg_path.stat().st_mtime:
                    continue
            except OSError:
                pass

        ok = svg_to_png(svg_path, png_path, args.size)
        if ok:
            converted += 1
            print(f"{svg_path} -> {png_path}")

    print(f"Converted {converted} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

