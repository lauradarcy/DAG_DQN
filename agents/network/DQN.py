from agents.network.GCN import GCN, MatMul
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, D_in, H1, H2, H3, H4, H5):
        super(DQN, self).__init__()
        # Deep network weights and biases

        self.gcn1 = GCN(D_in * 3, H1)

        self.gcn2 = GCN(H1 * 3, H2)

        self.matmul1 = MatMul(H2, H3)
        self.matmul2 = MatMul(H3, H4)
        self.matmul3 = MatMul(H4, H5)

    def forward(self, state):

        y, in_adj_mat, out_adj_mat = state

        # First convolution layer
        y = self.gcn1(y, in_adj_mat, out_adj_mat)

        # Second convolution layer
        y = self.gcn2(y, in_adj_mat, out_adj_mat)

        # Output layer
        y = self.matmul1(y)
        # For each vertex you have a vertex of length H3. This is the vertex embedding.

        # Perform pooling
        y = torch.sum(y, dim=0).view(1, -1)

        y = self.matmul2(y).clamp(min=0)
        y = self.matmul3(y)  # Could insert a ReLu layer before this.

        # this is the Q(s,a) value
        return y
