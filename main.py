import torch
import torch.nn as nn
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class WisePooling(Module):

    def __init__(self):
        super(WisePooling, self).__init__()

    def forward(self, input, graph):
        batch_list = []
        for i in range(input.shape[0]):
            tensor_list = list()
            for j in range(graph[i].shape[0]):
                shot_boundary = graph[i][j]
                tensor_list.append(
                    torch.div(torch.sum(input[shot_boundary[0]:shot_boundary[1] + 1], dim=0).requires_grad_(True),
                              shot_boundary[1] - shot_boundary[0] + 1).requires_grad_(True) + 6e-3)
            batch_list.append(torch.stack(tensor_list, dim=0).requires_grad_(True))

        return torch.stack(batch_list, dim=0).requires_grad_(True)


class WiseConvolution(Module):

    def __init__(self, input_size, output_size):
        super(WiseConvolution, self).__init__()
        self.WiseConv = nn.Linear(input_size, output_size)

    def forward(self, input, graph):
        batch_list = []
        for i in range(input.shape[0]):
            tensor_list = list()
            for j in range(graph[i].shape[0]):
                shot_boundary = graph[i][j]

                tensor_list.append(torch.sum(self.WiseConv(input[i][shot_boundary[0]:shot_boundary[1] + 1]), dim=0).requires_grad_(True))
            batch_list.append(torch.stack(tensor_list, dim=0).requires_grad_(True))

        return torch.stack(batch_list, dim=0).requires_grad_(True)


class NodeConvolution(Module):

    def __init__(self, kernel, input_size, pooling_size=2):
        super(NodeConvolution, self).__init__()
        self.pooling_size = pooling_size
        self.weight1 = Parameter(torch.FloatTensor(kernel, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)

    def forward(self, input):
        batch_list = list()
        for i in range(input.shape[0]):
            tensor = input[i]
            batch_size = tensor.shape[0]
            steps = batch_size // self.pooling_size
            left_tensor = False
            if batch_size % self.pooling_size != 0:
                steps += 1
                left_tensor = True
            tensor_list = list()
            for j in range(steps):
                if left_tensor is True and j == steps - 1:
                    tensor_ = torch.zeros(self.pooling_size, tensor.shape[1], requires_grad=True).cuda()
                    for i in range(batch_size % self.pooling_size):
                        tensor_[i] = tensor[self.pooling_size * j + i]
                    tensor_list.append(torch.sum(tensor_ * self.weight1, dim=0).requires_grad_(True))
                else:
                    tensor_list.append(
                        torch.sum(
                            tensor[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size] * self.weight1,
                            dim=0).requires_grad_(True))
            batch_list.append(torch.stack(tensor_list, dim=0))

        return torch.stack(batch_list, dim=0).requires_grad_(True)


class GraphPooling(Module):

    def __init__(self, pooling_size=3):
        super(GraphPooling, self).__init__()
        self.pooling_size = pooling_size

    def forward(self, input, graph):
        batch_list = []
        for i in range(input.shape[0]):
            tensor = input[i]
            batch_size = tensor.shape[0]
            steps = batch_size // self.pooling_size
            left_tensor = False
            if batch_size % self.pooling_size != 0:
                steps += 1
                left_tensor = True
            tensor_list = list()
            for j in range(steps):
                shot_boundary = graph[i][j]
                tensor_list.append(
                    torch.div(torch.sum(input[shot_boundary[0]:shot_boundary[1] + 1], dim=0).requires_grad_(True),
                              shot_boundary[1] - shot_boundary[0] + 1).requires_grad_(True) + 6e-3)
                if left_tensor is True and j == steps - 1:
                    tensor_ = torch.zeros(self.pooling_size, tensor.shape[1], requires_grad=True).cuda()
                    for i in range(batch_size % self.pooling_size):
                        tensor_[i] = tensor[self.pooling_size * j + i]

                    tensor_list.append(torch.div(torch.sum(tensor_, dim=0).requires_grad_(True),self.pooling_size))
                else:
                    tensor_list.append(
                        torch.div(torch.sum(tensor[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size], dim=0).requires_grad_(True),self.pooling_size))
            batch_list.append(torch.stack(tensor_list, dim=0).requires_grad_(True))

        return torch.stack(batch_list, dim=0).requires_grad_(True)

class GraphAttentionPooling(Module):

    def __init__(self, in_features, pooling_size=3):
        super(GraphAttentionPooling, self).__init__()
        self.in_features = in_features
        self.W = nn.Linear(in_features, 1, bias=True)
        self.pooling_size = pooling_size

    def forward(self, batch_tensor):
        batch_list = list()
        for i in range(batch_tensor.shape[0]):
            tensor = batch_tensor[i]
            batch_size = tensor.shape[0]
            steps = batch_size // self.pooling_size
            left_tensor = False
            if batch_size % self.pooling_size != 0:
                steps += 1
                left_tensor = True

            tensor_list = list()
            for j in range(steps):
                if left_tensor is True and j == steps - 1:
                    tensor_ = torch.zeros(self.pooling_size, tensor.shape[1], requires_grad=True).cuda()
                    for i in range(batch_size % self.pooling_size):
                        tensor_[i] = tensor[self.pooling_size * j + i]
                    att_w = F.softmax(self.W(tensor_), dim=0).requires_grad_(True)
                    tensor_list.append(tensor_.T @ att_w)
                else:
                    att_w = F.softmax(
                        self.W(tensor[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size]),
                        dim=0).requires_grad_(True)
                    tensor_list.append(
                        tensor[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size].T @ att_w)
            batch_list.append(torch.stack(tensor_list, dim=0))
        return torch.stack(batch_list, dim=0).requires_grad_(True)

class DCGNPropagate(nn.Module):
    def __init__(self, input, output):
        super(DCGNPropagate, self).__init__()
        self.Weight = Parameter(torch.FloatTensor(input, output))
        self.Bias = Parameter(torch.FloatTensor(output))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Weight.size(1))
        self.Weight.data.uniform_(-stdv, stdv)

    def forward(self, adj, x):
        graphed_net = torch.einsum('kb,abc->abd',adj,x)
        return torch.einsum('abc,cd->abd',graphed_net,self.Weight)+self.Bias


class DCGN(nn.Module):
    def __init__(self, input, nclass, pooling_size=3):
        super(DCGN, self).__init__()

        self.nodewiseconvolution = NodeConvolution(pooling_size, input, pooling_size=pooling_size)
        self.WisePooling = GraphAttentionPooling(input, pooling_size=pooling_size)
        self.Propagate1 = DCGNPropagate(input, 784)

        self.NodeConvolution2 = NodeConvolution(pooling_size, 784, pooling_size=pooling_size)
        self.AttentionPooling2 = GraphAttentionPooling(784, pooling_size=pooling_size)
        self.Propagate2 = DCGNPropagate(784, 28)

        self.classifier = nn.Sequential( nn.Linear(6*28,32),
                                         nn.GELU(),
                                         nn.Linear(32,nclass))

    def forward(self, x, device):

        adj = self.WisePooling(x)
        x = self.nodewiseconvolution(x)  # 2,256
        adj = self.get_adjacent(adj).to(device).requires_grad_(True)  # 2,2
        x = self.Propagate1(adj, x)
        x = F.gelu(x)

        adj = self.AttentionPooling2(x)  # 2,64
        x = self.NodeConvolution2(x)  # 2,64
        adj = self.get_adjacent(adj).to(device).requires_grad_(True)  # 2,32.
        x = self.Propagate2(adj, x)
        x = F.gelu(x)

        x = x.view(-1, 6*28)
        x = self.classifier(x)
        return x

    def cosine_similarity_adjacent(self, matrix1, matrix2):
        squaresum1 = torch.sum(torch.square(matrix1), dim=1)  # 1024 to 1

        squaresum2 = torch.sum(torch.square(matrix2), dim=1)  # 1024 to 1

        multiplesum = torch.sum(torch.multiply(matrix1, matrix2), dim=1)

        Matrix1DotProduct = torch.sqrt(squaresum1)
        Matrix2DotProduct = torch.sqrt(squaresum2)
        cosine_similarity = torch.div(multiplesum, torch.multiply(Matrix1DotProduct, Matrix2DotProduct))
        return cosine_similarity

    def get_adjacent(self, matrix):
        batch_list = list()
        for i in range(matrix.shape[0]):
            tensor = matrix[i]
            matrix_frame = tensor.shape[0]  # 4,2,1024
            AdjacentMatrix = torch.zeros(matrix_frame, matrix_frame)  # 2 X 2

            chunks = torch.chunk(tensor, matrix_frame, dim=0)
            for i in range(matrix_frame):
                for j in range(matrix_frame - i):
                    AdjacentMatrix[j][i] = self.cosine_similarity_adjacent(chunks[i], chunks[j])
                    if not i == j:
                        AdjacentMatrix[j][i] = AdjacentMatrix[i][j]
            I = torch.eye(AdjacentMatrix.shape[0], requires_grad=True)

            AdjacentMatrix += I
            AdjacentMatrix = AdjacentMatrix.requires_grad_(True)
            D_hat = torch.sum(AdjacentMatrix, dim=0)
            D_hat = torch.linalg.inv(torch.sqrt(torch.diag(D_hat)))
            batch_list.append(D_hat @ AdjacentMatrix @ D_hat)
        return torch.stack(batch_list, dim=0).requires_grad_(True)
