import sys
sys.path.append(".")
from env import Env
import numpy as np

def sfen_to_state(sfen_board, sfen_hand, sfen_turn) -> Env:
    pieces = {'.':0,'P': 1,'L': 2,'N': 3,'S': 4,'G': 5,'B': 6,'R': 7,'K': 8,'+P': 9,'+L': 10,'+N': 11,'+S': 12,'+B': 14,'+R': 15,
                    'p':-1,'l':-2,'n':-3,'s':-4,'g':-5,'b':-6,'r':-7,'k':-8,'+p':-9,'+l':-10,'+n':-11,'+s':-12,'+b':-14,'+r':-15}
    state = board, hand, nonp = np.zeros(81, int), np.zeros(18, int), np.ones(18, bool)
    for n in range(1, 10):
        sfen_board = sfen_board.replace(str(n), "."*n)
    sfen_board = sfen_board.replace("/", "")
    for n in range(81):
        j = 1 + (sfen_board[0] == "+")
        piece = pieces[sfen_board[:j]]
        if piece == 1:
            nonp[n%9] = False
        elif piece == -1:
            nonp[-1-n%9] = False
        board[n] = piece
        sfen_board = sfen_board[j:]
    if sfen_hand != "-":
        while sfen_hand:
            if sfen_hand[0].isdigit():
                j = 1 + sfen_hand[1].isdigit()
                hand[pieces[sfen_hand[j]]] = int(sfen_hand[:j])
                sfen_hand = sfen_hand[1+j:]
            else:
                hand[pieces[sfen_hand[0]]] = 1
                sfen_hand = sfen_hand[1:]
    if sfen_turn == "b":
        return Env(state, 1)
    return Env((-board[::-1], hand[::-1], nonp[::-1]), -1)

PIECES_SFEN, COL, ROW = "_PLNSGBR", "987654321", "abcdefghi"

def move_to_sfen(board, turn, n1, p1, j1=None, p0=None, n0=None, pp=None) -> str:
    i1, j1 = divmod(n1, 9)
    ns = COL[::turn][j1] + ROW[::turn][i1]
    if n0 == None:
        return PIECES_SFEN[p1] + "*" + ns
    i0, j0 = divmod(p0, 9)
    ps = COL[::turn][j0] + ROW[::turn][i0]
    if board[p0] < p1:
        return ps + ns + "+"
    return ps + ns

def sfen_to_move(board, turn, sfen) -> tuple:
    i1, j1 = ROW[::turn].find(sfen[3]), COL[::turn].find(sfen[2])
    n1 = 9 * i1 + j1    # 移動先
    if sfen[1] == "*":  # 打ち
        return n1, PIECES_SFEN.find(sfen[0]), j1, None, None, None
    n0 = 9 * ROW[::turn].find(sfen[1]) + COL[::turn].find(sfen[0])  # 移動元
    if sfen[-1] == "+": # 成り
        return n1, board[n0]+8, j1, n0, -board[n1], n1==1
    return n1, board[n0], j1, n0, -board[n1], None

DEPTH_LIMIT = 3
TEMPERATURE = 0.01
# 初期化
from agent import Agent, QNetwork
agent = Agent()
env = sfen_to_state("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL", "-", "b")

while True:
    cmd = input().split()
    if not cmd:
        continue
    if cmd[0] == "usi":
        print("id name DQN")
        print("option name DepthLimit type spin default 3 min 1 max 8")
        print("option name Temperature(*0.01) type spin default 1 min 0 max 200")
        print("usiok")
    elif cmd[0] == "isready":
        print("readyok")
    elif cmd[0] == "setoption": # 設定
        if cmd[2] == "DepthLimit":
            DEPTH_LIMIT = int(cmd[-1])
        elif cmd[2] == "Temperature(*0.01)":
            TEMPERATURE = float(cmd[-1]) / 100
    elif cmd[0] == "position":
        # 環境初期化
        if cmd[1] == "startpos":
            cmd[1:2] = ["sfen", "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL", "b", "-", "1"]
        env = sfen_to_state(cmd[2], cmd[4], cmd[3])
        for move in cmd[7:]:
            env.step(sfen_to_move(env.board, env.side, move))
    elif cmd[0] == "go":
        Q, a = agent.act_boltzmann_negamax(*env.Snew(), TEMPERATURE, DEPTH_LIMIT)
        score = int((1420/np.pi) * np.tan(Q*np.pi/2))   # https://t.co/0XLsQvFjLg に基づくQ値∈(-1, 1)→評価値換算
        print("info score cp", score)
        if Q == -1:
            print("bestmove resign")
        else:
            print("bestmove", move_to_sfen(env.board, env.side, *env.legalmoves[a]))
    elif cmd[0] == "debug":
        print(env)
        print(f"DepthLimit = {DEPTH_LIMIT}, Temperature = {TEMPERATURE}")
    elif cmd[0] == "quit":
        sys.exit()

