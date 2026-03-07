from __future__ import annotations

from src.core.board import Board
from src.core.types import Color, Move, PieceType

# Piece values for SEE (centipawns), indexed by PieceType int value.
# PieceType: PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6
SEE_VAL = [0, 100, 300, 300, 500, 900, 10_000]

_KNIGHT_OFFSETS = (
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2),
)
_DIAG_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_ORTHO_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_attackers(board: Board, sq: int, occ: int) -> int:
    """Return bitboard of all pieces (both colors) attacking *sq* given *occ*."""
    rank = sq >> 3
    file = sq & 7
    attackers = 0

    # --- Pawns ---
    # White pawns attack from rank-1, black pawns from rank+1.
    for color_int, pawn_dr in ((0, -1), (1, 1)):
        pr = rank + pawn_dr
        if 0 <= pr < 8:
            pawn_bb = board.piece_bb[color_int][1]  # PieceType.PAWN = 1
            for df in (-1, 1):
                pf = file + df
                if 0 <= pf < 8:
                    bit = 1 << (pr * 8 + pf)
                    if occ & pawn_bb & bit:
                        attackers |= bit

    # --- Knights ---
    knight_bb = board.piece_bb[0][2] | board.piece_bb[1][2]
    for dr, df in _KNIGHT_OFFSETS:
        r = rank + dr
        f = file + df
        if 0 <= r < 8 and 0 <= f < 8:
            bit = 1 << (r * 8 + f)
            if occ & knight_bb & bit:
                attackers |= bit

    # --- Diagonal sliders (bishop + queen) ---
    diag_bb = (
        board.piece_bb[0][3] | board.piece_bb[1][3]
        | board.piece_bb[0][5] | board.piece_bb[1][5]
    )
    for dr, df in _DIAG_DIRS:
        r, f = rank + dr, file + df
        while 0 <= r < 8 and 0 <= f < 8:
            bit = 1 << (r * 8 + f)
            if occ & bit:
                if diag_bb & bit:
                    attackers |= bit
                break
            r += dr
            f += df

    # --- Orthogonal sliders (rook + queen) ---
    ortho_bb = (
        board.piece_bb[0][4] | board.piece_bb[1][4]
        | board.piece_bb[0][5] | board.piece_bb[1][5]
    )
    for dr, df in _ORTHO_DIRS:
        r, f = rank + dr, file + df
        while 0 <= r < 8 and 0 <= f < 8:
            bit = 1 << (r * 8 + f)
            if occ & bit:
                if ortho_bb & bit:
                    attackers |= bit
                break
            r += dr
            f += df

    # --- Kings ---
    king_bb = board.piece_bb[0][6] | board.piece_bb[1][6]
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            r, f = rank + dr, file + df
            if 0 <= r < 8 and 0 <= f < 8:
                bit = 1 << (r * 8 + f)
                if occ & king_bb & bit:
                    attackers |= bit

    return attackers


def _consider_xray(
    board: Board, to_sq: int, removed_sq: int, occ: int,
) -> int:
    """After removing *removed_sq* from occ, find any sliding x-ray attacker
    revealed behind it on the ray toward *to_sq*.  Returns a single-bit mask
    or 0."""
    to_rank = to_sq >> 3
    to_file = to_sq & 7
    rm_rank = removed_sq >> 3
    rm_file = removed_sq & 7

    dr = rm_rank - to_rank
    df = rm_file - to_file

    # Must be on a straight line (rank, file, or diagonal).
    if dr != 0 and df != 0 and abs(dr) != abs(df):
        return 0

    # Normalise to unit direction.
    if dr:
        dr = 1 if dr > 0 else -1
    if df:
        df = 1 if df > 0 else -1

    is_diag = dr != 0 and df != 0
    if is_diag:
        slider_bb = (
            board.piece_bb[0][3] | board.piece_bb[1][3]
            | board.piece_bb[0][5] | board.piece_bb[1][5]
        )
    else:
        slider_bb = (
            board.piece_bb[0][4] | board.piece_bb[1][4]
            | board.piece_bb[0][5] | board.piece_bb[1][5]
        )

    # Scan outward from to_sq in the direction of removed_sq.
    r, f = to_rank + dr, to_file + df
    while 0 <= r < 8 and 0 <= f < 8:
        bit = 1 << (r * 8 + f)
        if occ & bit:
            return bit if slider_bb & bit else 0
        r += dr
        f += df

    return 0


def see(board: Board, move: Move) -> int:
    """Static Exchange Evaluation for a capture move.

    Returns the expected material gain/loss in centipawns when both sides
    play the best recapture sequence.  Positive = good for the side making
    the initial capture.
    """
    from_sq = move.from_sq
    to_sq = move.to_sq

    attacker = board.squares[from_sq]
    if attacker is None:
        return 0

    # --- initial capture value ---
    target = board.squares[to_sq]
    if target is not None:
        gain = SEE_VAL[int(target[1])]
    elif (
        board.ep_square is not None
        and to_sq == board.ep_square
        and attacker[1] == PieceType.PAWN
    ):
        gain = SEE_VAL[1]
    else:
        return 0  # not a capture

    # Promotion: attacker becomes the promoted piece.
    if move.promotion is not None:
        attacker_val = SEE_VAL[int(move.promotion)]
        gain += attacker_val - SEE_VAL[1]  # promotion bonus
    else:
        attacker_val = SEE_VAL[int(attacker[1])]

    # --- set up occupancy ---
    occ = board.occ_all ^ (1 << from_sq)

    # En passant: also remove the captured pawn.
    if (
        board.ep_square is not None
        and to_sq == board.ep_square
        and attacker[1] == PieceType.PAWN
    ):
        rank_step = 1 if attacker[0] == Color.WHITE else -1
        occ ^= 1 << (to_sq - 8 * rank_step)

    # All current attackers of to_sq (both colours).
    attackers = _get_attackers(board, to_sq, occ)
    attackers &= ~(1 << from_sq)

    # Discover x-ray behind the initial attacker.
    attackers |= _consider_xray(board, to_sq, from_sq, occ)

    side = int(attacker[0]) ^ 1  # side to recapture

    # --- build swap list ---
    swap = [gain]

    while True:
        # Least valuable attacker for the current side.
        side_atk = attackers & board.occ_bb[side]
        if not side_atk:
            break

        atk_sq = -1
        for pt_int in (1, 2, 3, 4, 5, 6):
            pt_bb = side_atk & board.piece_bb[side][pt_int]
            if pt_bb:
                atk_sq = (pt_bb & -pt_bb).bit_length() - 1
                attacker_val_next = SEE_VAL[pt_int]
                break

        if atk_sq < 0:
            break

        swap.append(attacker_val - swap[-1])

        # Pruning: if the current side is losing no matter what, stop.
        if max(-swap[-2], swap[-1]) < 0:
            break

        attacker_val = attacker_val_next

        # Remove this attacker and discover x-rays.
        bit = 1 << atk_sq
        occ ^= bit
        attackers ^= bit
        attackers |= _consider_xray(board, to_sq, atk_sq, occ)

        side ^= 1

    # --- evaluate swap list back-to-front ---
    i = len(swap) - 1
    while i > 0:
        swap[i - 1] = -max(-swap[i - 1], swap[i])
        i -= 1

    return swap[0]
