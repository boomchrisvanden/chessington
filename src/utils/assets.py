"""
Shared asset-loading helpers for chess GUIs.

Handles piece-image discovery (PNG / SVG→PNG fallback) and
text-based fallback surfaces.  Used by both ``gui.py`` (main game)
and ``src/theory/gui.py`` (opening practice).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import pygame

# Piece names used for candidate filename matching
_PIECE_NAMES = {
    "P": "pawn",
    "N": "knight",
    "B": "bishop",
    "R": "rook",
    "Q": "queen",
    "K": "king",
}

_PIECE_CODES = ("P", "N", "B", "R", "Q", "K")


def _convert_svg_to_png(svg_path: Path, png_path: Path, size_px: int) -> bool:
    """Best-effort SVG→PNG conversion.  Returns True if *png_path* exists afterwards."""
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return png_path.exists()

    return False


def load_piece_images(
    piece_dir: str,
    square_size: int,
    *,
    color_values: Optional[Tuple] = None,
    piece_type_values: Optional[Tuple] = None,
) -> Dict:
    """Load piece PNGs (and auto-convert SVGs) from *piece_dir*.

    Parameters
    ----------
    piece_dir
        Directory containing piece image files.
    square_size
        Target pixel size (width = height) each image is scaled to.
    color_values
        ``(white_val, black_val)`` used as dict keys.
        Defaults to ``(0, 1)`` (matching ``Color.WHITE / Color.BLACK``).
    piece_type_values
        Sequence of piece-type values in P/N/B/R/Q/K order.
        Defaults to ``(1, 2, 3, 4, 5, 6)`` (matching ``PieceType``).

    Returns a ``dict[(color_val, pt_val), pygame.Surface]``.
    """
    if color_values is None:
        color_values = (0, 1)
    if piece_type_values is None:
        piece_type_values = (1, 2, 3, 4, 5, 6)

    pieces: Dict = {}
    piece_dir_path = Path(piece_dir)

    if not piece_dir_path.exists():
        return pieces

    png_by_name: Dict[str, Path] = {}
    for p in piece_dir_path.iterdir():
        if p.is_file() and p.suffix.lower() == ".png":
            png_by_name[p.name.lower()] = p

    white_val, black_val = color_values

    for color_val, ccode in ((white_val, "w"), (black_val, "b")):
        is_white = color_val == white_val
        for pt_val, pcode in zip(piece_type_values, _PIECE_CODES):
            piece_name = _PIECE_NAMES[pcode]
            human_color = "white" if is_white else "black"
            wiki_color = "l" if is_white else "d"
            wiki_re = re.compile(rf"^chess_{pcode.lower()}{wiki_color}t\d+\.png$")

            candidate_names = [
                f"{ccode}{pcode}.png",
                f"{ccode}{pcode.lower()}.png",
                f"{ccode}_{piece_name}.png",
                f"{ccode}-{piece_name}.png",
                f"{human_color}_{piece_name}.png",
                f"{human_color}-{piece_name}.png",
                f"{piece_name}_{human_color}.png",
                f"{piece_name}-{human_color}.png",
            ]

            png_path = next(
                (png_by_name.get(n.lower()) for n in candidate_names if n.lower() in png_by_name),
                None,
            )

            if png_path is None:
                wiki_match = next(
                    (p for n, p in png_by_name.items() if wiki_re.match(n)), None
                )
                png_path = wiki_match

            if png_path is None:
                svg_candidates = [
                    piece_dir_path / f"{ccode}{pcode}.svg",
                    piece_dir_path / f"{ccode}{pcode.lower()}.svg",
                ]
                svg_path = next((p for p in svg_candidates if p.exists()), None)
                if svg_path is not None:
                    out_png = piece_dir_path / f"{ccode}{pcode}.png"
                    if _convert_svg_to_png(svg_path, out_png, square_size):
                        png_path = out_png

            if png_path is None or not png_path.exists():
                continue

            img = pygame.image.load(str(png_path)).convert_alpha()
            img = pygame.transform.smoothscale(img, (square_size, square_size))
            pieces[(color_val, pt_val)] = img

    return pieces


def build_fallback_piece_surfaces(
    square_size: int,
    piece_font: pygame.font.Font,
    *,
    color_values: Optional[Tuple] = None,
    piece_type_values: Optional[Tuple] = None,
) -> Dict:
    """Build text-glyph fallback surfaces for every piece.

    Parameters mirror :func:`load_piece_images`.
    """
    if color_values is None:
        color_values = (0, 1)
    if piece_type_values is None:
        piece_type_values = (1, 2, 3, 4, 5, 6)

    white_val = color_values[0]

    fallback: Dict = {}
    for color_val in color_values:
        for pt_val, pcode in zip(piece_type_values, _PIECE_CODES):
            fg = (245, 245, 245) if color_val == white_val else (20, 20, 20)
            outline = (20, 20, 20) if color_val == white_val else (245, 245, 245)
            base = piece_font.render(pcode, True, fg)
            shadow = piece_font.render(pcode, True, outline)

            surf = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
            rect = base.get_rect(center=(square_size // 2, square_size // 2))
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                surf.blit(shadow, rect.move(dx, dy))
            surf.blit(base, rect)
            fallback[(color_val, pt_val)] = surf

    return fallback


def get_piece_dir() -> str:
    """Return the path to the directory containing piece images."""
    base_dir = Path(__file__).resolve().parents[2]
    assets_dir = base_dir / "assets"
    pieces_dir = assets_dir / "pieces"

    def has_piece_images(d: Path) -> bool:
        if not d.exists():
            return False
        for p in d.iterdir():
            if not p.is_file() or p.suffix.lower() != ".png":
                continue
            name = p.name.lower()
            if re.match(r"^[wb][pnbrqk]\.png$", name):
                return True
            if re.match(r"^chess_[pnbrqk][ld]t\d+\.png$", name):
                return True
            if re.match(r"^(white|black)[_-](pawn|knight|bishop|rook|queen|king)\.png$", name):
                return True
            if re.match(r"^(pawn|knight|bishop|rook|queen|king)[_-](white|black)\.png$", name):
                return True
            if re.match(r"^[wb][_-](pawn|knight|bishop|rook|queen|king)\.png$", name):
                return True
        return False

    if has_piece_images(assets_dir):
        return str(assets_dir)
    elif has_piece_images(pieces_dir):
        return str(pieces_dir)
    else:
        return str(assets_dir)
