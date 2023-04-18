import numpy as np
import torch

PIECES = np.r_[-15, -14, -11:0, 1:12, 14, 15][:, None]
def state_onehot(board, hand):
    return torch.Tensor(np.r_[(board==PIECES).flatten(), hand[1:-1]])

MAJOR = dict((p, [None]*81) for p in (2,6,7,14,15))
MINOR = dict((p, [None]*81) for p in (1,3,4,5,8,9,10,11,12,14,15))
# ↙↘↖↗↑←→↓  銀ならds[:5], 金ならds[2:]
ds = [(1,-1,8), (1,1,10), (-1,-1,-10), (-1,1,-8), (-1,0,-9), (0,-1,-1), (0,1,1), (1,0,9)]
for n0 in range(81): # p: 移動元
    (i, j), raid = divmod(n0, 9), n0<27
    # ctrl[↖↗↘↙][動数]: (n1, 成れるかprom, 列j1)
    ctrl = [[(n1, n1<27, n1%9) for n1 in range(n0-10, n0-10*(1+min(i,   j)), -10)],
            [(n1, n1<27, n1%9) for n1 in range(n0- 8, n0- 8*(1+min(i, 8-j)), - 8)],
            [(n1,  raid, n1%9) for n1 in range(n0+10, n0+10*(9-max(i,   j)),  10)],
            [(n1,  raid, n1%9) for n1 in range(n0+ 8, n0+ 8*(9-max(i, 8-j)),   8)]]
    # major[角馬][p0][方向][動数]: (n1, p1, j1)
    MAJOR[ 6][n0] = [[(n1, (6, 14)[prom], j1) for n1, prom, j1 in ns] for ns in ctrl]
    MAJOR[14][n0] = [[(n1,     14       , j1) for n1, prom, j1 in ns] for ns in ctrl]
    # ctrl[↑↓←→][動数]: (n, prom, j1)
    ctrl = [[(n1, n1<27,    j) for n1 in range(n0-9,     -1, -9)],
            [(n1,  raid,    j) for n1 in range(n0+9,     81,  9)],
            [(n1,  raid, n1%9) for n1 in range(n0-1, n0-j-1, -1)],
            [(n1,  raid, n1%9) for n1 in range(n0+1, n0-j+9,  1)]]
    # major[飛竜香][p0][方向][動数]: (n, n1, j1)
    MAJOR[ 2][n0] = [[(n1, (2, 10)[prom], j1) for n1, prom, j1 in ctrl[0]]]
    MAJOR[ 7][n0] = [[(n1, (7, 15)[prom], j1) for n1, prom, j1 in ns] for ns in ctrl]
    MAJOR[15][n0] = [[(n1,     15       , j1) for n1, prom, j1 in ns] for ns in ctrl]
    # 歩桂：成れたら成る
    MINOR[ 1][n0] = [[n0-9, (1, 9)[i<=3], j, i<=3]] if i else []
    MINOR[ 3][n0] = [[n0+dj-18, (3, 11)[i<=4], j+dj, None] for dj in (-1, 1) if 0<=j+dj<=8] if i>=2 else []
    # 金銀
    c = [(n0+d, j+dj, 0<=i+di<=8 and 0<=j+dj<=8) for di, dj, d in ds]   # (n1, j1, on-board)
    for p1, ctrl in (4,c[:5]), (5,c[2:]), (8,c), (9,c[2:]), (10,c[2:]), (11,c[2:]), (12,c[2:]), (14,c[4:]), (15,c[:4]):
        MINOR[p1][n0] = [[n1, p1, j1, None] for n1, j1, on_board in ctrl if on_board]


class Env:
    def __init__(self, state, side=1):
        self.board, self.hand, self.nonp = state
        self.side = side
        self.legalmoves = list(self._movegen())
    
    def __repr__(self):
        board = self.side * self.board[::self.side].reshape(9, 9)
        hand = self.hand[::self.side]
        player = ["_", "1st", "2nd"][self.side]
        return f"{board}\n{hand[1:9]}\n{hand[-2:-10:-1]}\nplayer: {player}"
    
    def _movegen(self):
        # 時刻0に位置n0に駒p0があり，時刻1に位置n1に移動し，駒p1になったとする．
        # 返り値 (n1, p1, 列j1, n0, -p0, 歩成pp)
        for n, p0 in enumerate(self.board):
            if not p0:  # nに打つ
                for p1 in 7, 6, 5, 4:   # 飛角金銀: 常に可
                    if self.hand[p1]:
                        yield n, p1, None, None, None, None
                if n >= 9:
                    if n >= 18 and self.hand[3]:        # 桂: >2段目
                        yield n, 3, None, None, None, None
                    if self.hand[2]:                    # 香: >1段目
                        yield n, 2, None, None, None, None
                    if self.nonp[n%9] and self.hand[1]: # 歩: >1段目∧非二歩
                        yield n, 1, n%9, None, None, None
            else:   # nから移動
                if p0 in MAJOR: # 大駒  各方位，動数について
                    for direction in MAJOR[p0][n]:
                        for n1, p1, j1 in direction:
                            # 相手の駒か空なら移動可，相手の駒なら移動終了
                            if self.board[n1] <= 0:
                                yield n1, p1, j1, n, -self.board[n1], None
                            if self.board[n1]:
                                break
                if p0 in MINOR: # 小駒
                    for n1, p1, j1, pp in MINOR[p0][n]:
                        if self.board[n1] <= 0:
                            yield n1, p1, j1, n, -self.board[n1], pp
    
    def next_state(self, a, board=None, hand=None, nonp=None):
        n1, p1, j1, n0, p0, pp = a
        if board is None:
            board, hand, nonp = -self.board[::-1], self.hand.copy()[::-1], self.nonp.copy()[::-1]
        board[80-n1] = -p1
        if p0 == None:  # 駒打ち（歩ならnonp更新）
            hand[-1-p1] -= 1
            if p1 == 1:
                nonp[-1-j1] = False
        else:
            board[80-n0] = 0   # 移動元を空に
            if p0 >= 9:             # 成駒取り
                hand[7-p0] += 1
            elif p0:                # 生駒取り（歩ならnonp更新）
                hand[-1-p0] += 1
                if p0 == 1:
                    nonp[8-j1] = True
            # if p0 == 1 and p1 == 9: # 歩成ならnonp更新
            if pp:                  # 歩成ならnonp更新
                nonp[-1-j1] = True
        return board, hand, nonp
        
    def step(self, a: int|tuple):
        # sが既知ならstep_from_s()を使うと少し高速．
        self.step_from_s(self.next_state(self.legalmoves[a] if type(a) == int else a))
    
    def step_from_s(self, s):
        self.side *= -1
        self.board, self.hand, self.nonp = s
        self.legalmoves = list(self._movegen()) # 新しい局面になったらすぐに合法手を生成しておく．
    
    def s(self):
        return state_onehot(self.board, self.hand)
    
    def Snew(self):
        S = [self.next_state(a) for a in self.legalmoves]
        X = torch.stack([state_onehot(*s[:2]) for s in S])
        return S, X
