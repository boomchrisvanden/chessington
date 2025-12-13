'''
Author: Chris Vanden Boom
11/14/2025
'''


class Zobrist:
    """Holds random keys; board xor's them to maintain hash incrementally."""
    def __init__(self):
        self.piece_square = [[random.getrandbits(64) for _ in range(64)] for _ in range(12)]  # 6 pieces * 2 colors
        self.stm = random.getrandbits(64)
        self.castling = [random.getrandbits(64) for _ in range(16)]
        self.ep_file = [random.getrandbits(64) for _ in range(8)]

    @staticmethod
    def seed() -> "Zobrist":
        random.seed(0xC0FFEE)  # deterministic dev runs; change later if desired
        return Zobrist()