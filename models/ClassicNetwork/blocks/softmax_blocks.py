import torch
import torch.nn as nn
import torch.nn.functional as F

class softmax_layer(nn.Module):
    """Constructs a ECA module.
        Args:
            input: [B,K,F]
           output: [B,F]
        """
    def __init__(self, dim=512):
        super(softmax_layer, self).__init__()

        self.dim = dim
        self.w_omega = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.u_omega = nn.Parameter(torch.Tensor(self.dim, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, x):
        # inputs的形状是[B,K,F]
        # Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega))
        # u形状是[B,K,F]
        att = torch.matmul(u, self.u_omega)
        # att形状是[B,K,1]
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为[B,K,1]
        scored_x = x * att_score
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束
        outs = torch.sum(scored_x, dim=1)
        return outs
