import torch, env

device = "cuda" if torch.cuda.is_available() else "cpu"

class QNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2122, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 1)
        self.to(device)

    def forward(self, x):
        x = x.to(device)
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x).tanh()
        return x

class Agent:
    def __init__(self, wfile="./model.pth"):
        self.Q = torch.load(wfile, device)
    
    def negamax(self, S=None, X=None, depth=1, alpha=-1, beta=1):
        if X[:, -8].any():  # 玉を取る手がある＝勝ち
            return 1.0
        Q = -self.Q(X).flatten()
        depth1 = depth - 1
        if depth1 == 0:
            return float(max(Q))
        for a in Q.argsort(descending=True)[:3]:    # 前向き枝刈り
            alpha = max(alpha, -self.negamax(*env.Env(S[a]).Snew(), depth1, -beta, -alpha))
            if alpha >= beta:
                return alpha
        return alpha
    
    def act_boltzmann_negamax(self, S=None, X=None, T=0.0, depth=1):
        depth1 = depth - 1
        if depth1:
            Q = -torch.Tensor([self.negamax(*env.Env(s).Snew(), depth1) for s in S])
        else:
            Q = -self.Q(X).flatten()
        # 最善手
        if T == 0:
            Q, a = Q.max(0)
            return float(Q), int(a)
        # Boltzmann方策
        p = (Q / T).softmax(0)
        a = int(torch.multinomial(p, 1))
        return float(Q[a]), a
