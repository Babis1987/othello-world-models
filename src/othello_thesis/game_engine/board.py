import numpy as np

DIRECTIONS = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]


class OthelloBoardState:
    def __init__(self, n=8):
        assert n % 2 == 0, f"Board size must be even, got {n}"
        self.n = n
        self.state = np.zeros((n, n), dtype=int)
        self.next_hand_color = 1  # 1=black, -1=white, black moves first
        self.history = []
        # initial 4 pieces in the center
        self.state[n//2 - 1, n//2 - 1] = -1
        self.state[n//2 - 1, n//2    ] =  1
        self.state[n//2    , n//2 - 1] =  1
        self.state[n//2    , n//2    ] = -1

    def _find_flips(self, r, c, color):
        # find all enemy pieces that would be flipped by placing color at (r, c)
        to_flip = []
        for dr, dc in DIRECTIONS:
            buffer = []
            cr, cc = r + dr, c + dc
            while 0 <= cr < self.n and 0 <= cc < self.n:
                if self.state[cr, cc] == -color:
                    buffer.append((cr, cc))
                    cr += dr
                    cc += dc
                elif self.state[cr, cc] == color:
                    if buffer:
                        to_flip.extend(buffer)
                    break
                else:
                    break
        return to_flip

    def tentative_move(self, move):
        # check if move is legal for the current player without changing state
        if not (0 <= move < self.n * self.n):
            return False
        r, c = move // self.n, move % self.n
        if self.state[r, c] != 0:
            return False
        return bool(self._find_flips(r, c, self.next_hand_color))

    def get_valid_moves(self):
        return [move for move in range(self.n * self.n) if self.tentative_move(move)]

    def umpire(self, move):
        # execute a move, handling forfeit if needed
        assert 0 <= move < self.n * self.n, f"Invalid move: {move}"
        r, c = move // self.n, move % self.n
        assert self.state[r, c] == 0, f"Cell ({r},{c}) is already occupied!"

        color = self.next_hand_color
        if not self.get_valid_moves():
            self.next_hand_color *= -1
            color = self.next_hand_color

        to_flip = self._find_flips(r, c, color)

        assert to_flip, "Illegal move!"

        for fr, fc in to_flip:
            self.state[fr, fc] *= -1
        self.state[r, c] = color
        self.next_hand_color *= -1
        self.history.append(move)

    def update(self, moves):
        for move in moves:
            self.umpire(move)

    def get_state(self):
        # returns flat board: -1=white, 0=empty, 1=black
        return self.state.flatten().tolist()

    def get_gt(self, moves, func):
        # play through moves and collect the result of func() after each move
        container = []
        for move in moves:
            self.umpire(move)
            container.append(getattr(self, func)())
        return container

    def is_game_over(self):
        # game over when neither player has valid moves
        if self.get_valid_moves():
            return False
        self.next_hand_color *= -1
        has_moves = bool(self.get_valid_moves())
        self.next_hand_color *= -1
        return not has_moves

    def get_winner(self):
        # returns 1 (black wins), -1 (white wins), 0 (draw)
        black = np.sum(self.state == 1)
        white = np.sum(self.state == -1)
        if black > white:
            return 1
        elif white > black:
            return -1
        else:
            return 0

    def __repr__(self):
        symbols = {1: "X", -1: "O", 0: "."}
        rows = []
        for row in self.state:
            rows.append(" ".join(symbols[v] for v in row))
        return "\n".join(rows)
