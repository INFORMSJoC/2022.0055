import torch
import torch.nn as nn
import math

import numpy as np

'''The use the RIMCell implementation from https://github.com/anirudh9119/RIMs'''

class blocked_grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0


class GroupLinearLayer(nn.Module):

    def __init__(self, din, dout, num_blocks):
        super(GroupLinearLayer, self).__init__()

        self.w = nn.Parameter(0.01 * torch.randn(num_blocks, din, dout)).cuda()

    def forward(self, x):
        x = x.permute(1, 0, 2)

        x = torch.bmm(x, self.w)
        return x.permute(1, 0, 2)


class RIMCell(nn.Module):
    def __init__(self,
                 device, input_size, hidden_size, num_units=4, rnn_cell='LSTM', input_key_size=64, input_value_size=100,
                 input_query_size=64,
                 num_input_heads=1, input_dropout=0.1, comm_key_size=32, comm_value_size=75, comm_query_size=32, #comm_value_size=64
                 num_comm_heads=4, comm_dropout=0.1,
                 k=2
                 ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_units = num_units
        self.rnn_cell = rnn_cell
        self.key_size = input_key_size
        self.k = k
        self.num_input_heads = num_input_heads
        self.num_comm_heads = num_comm_heads
        self.input_key_size = input_key_size
        self.input_query_size = input_query_size
        self.input_value_size = input_value_size

        self.comm_key_size = comm_key_size
        self.comm_query_size = comm_query_size
        self.comm_value_size = comm_value_size

        self.key = nn.Linear(input_size, num_input_heads * input_query_size).to(self.device)
        self.value = nn.Linear(input_size, num_input_heads * input_value_size).to(self.device)

        if self.rnn_cell == 'GRU':
            self.rnn = nn.ModuleList([nn.GRUCell(input_value_size, hidden_size) for _ in range(num_units)])
            self.query = GroupLinearLayer(hidden_size, input_key_size * num_input_heads, self.num_units)
        else:
            self.rnn = nn.ModuleList([nn.LSTMCell(input_value_size, hidden_size) for _ in range(num_units)])
            self.query = GroupLinearLayer(hidden_size, input_key_size * num_input_heads, self.num_units)
        self.query_ = GroupLinearLayer(hidden_size, comm_query_size * num_comm_heads, self.num_units)
        self.key_ = GroupLinearLayer(hidden_size, comm_key_size * num_comm_heads, self.num_units)
        self.value_ = GroupLinearLayer(hidden_size, comm_value_size * num_comm_heads, self.num_units)
        self.comm_attention_output = GroupLinearLayer(num_comm_heads * comm_value_size, comm_value_size, self.num_units)
        self.comm_dropout = nn.Dropout(p=input_dropout)
        self.input_dropout = nn.Dropout(p=comm_dropout)

    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def input_attention_mask(self, x, h):
        """
	    Input : x (batch_size, 2, input_size) [The null input is appended along the first dimension]
	    		h (batch_size, num_units, hidden_size)
	    Output: inputs (list of size num_units with each element of shape (batch_size, input_value_size))
	    		mask_ binary array of shape (batch_size, num_units) where 1 indicates active and 0 indicates inactive
		"""

        key_layer = self.key(x)
        value_layer = self.value(x)
        query_layer = self.query(h)

        key_layer = self.transpose_for_scores(key_layer, self.num_input_heads, self.input_key_size)

        value_layer = torch.mean(self.transpose_for_scores(value_layer, self.num_input_heads, self.input_value_size),
                                 dim=1)

        query_layer = self.transpose_for_scores(query_layer, self.num_input_heads, self.input_query_size)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.input_key_size)
        attention_scores = torch.mean(attention_scores, dim=1)
        mask_ = torch.zeros(x.size(0), self.num_units).to(self.device)

        not_null_scores = attention_scores[:, :, 0]
        topk1 = torch.topk(not_null_scores, self.k, dim=1)
        row_index = np.arange(x.size(0))
        row_index = np.repeat(row_index, self.k)

        mask_[row_index, topk1.indices.view(-1)] = 1

        attention_probs = self.input_dropout(nn.Softmax(dim=-1)(attention_scores))
        inputs = torch.matmul(attention_probs, value_layer) * mask_.unsqueeze(2)
        inputs = torch.split(inputs, 1, 1)
        return inputs, mask_

    def communication_attention(self, h, mask):
        """
	    Input : h (batch_size, num_units, hidden_size)
	    	    mask obtained from the input_attention_mask() function
	    Output: context_layer (batch_size, num_units, hidden_size). New hidden states after communication
	    """
        query_layer = []
        key_layer = []
        value_layer = []

        query_layer = self.query_(h)
        key_layer = self.key_(h)
        value_layer = self.value_(h)

        query_layer = self.transpose_for_scores(query_layer, self.num_comm_heads, self.comm_query_size)
        key_layer = self.transpose_for_scores(key_layer, self.num_comm_heads, self.comm_key_size)
        value_layer = self.transpose_for_scores(value_layer, self.num_comm_heads, self.comm_value_size)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.comm_key_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        mask = [mask for _ in range(attention_probs.size(1))]
        mask = torch.stack(mask, dim=1)

        attention_probs = attention_probs * mask.unsqueeze(3)
        attention_probs = self.comm_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.num_comm_heads * self.comm_value_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.comm_attention_output(context_layer)

        context_layer = context_layer + h

        return context_layer

    def forward(self, x, hs, cs=None):
        """
		Input : x (batch_size, 1 , input_size)
				hs (batch_size, num_units, hidden_size)
				cs (batch_size, num_units, hidden_size)
		Output: new hs, cs for LSTM
				new hs for GRU
		"""
        size = x.size()
        null_input = torch.zeros(size[0], 1, size[2]).float().to(self.device)
        x = torch.cat((x, null_input), dim=1)

        # Compute input attention
        inputs, mask = self.input_attention_mask(x, hs)
        h_old = hs * 1.0
        if cs is not None:
            c_old = cs * 1.0
        hs = list(torch.split(hs, 1, 1))
        if cs is not None:
            cs = list(torch.split(cs, 1, 1))

        # Compute RNN(LSTM or GRU) output
        for i in range(self.num_units):
            if cs is None:
                hs[:, i, :] = self.rnn[i](inputs[i], hs[:, i, :])
            else:
                hs[i], cs[i] = self.rnn[i](inputs[i].squeeze(1).to(self.device), (hs[i].squeeze(1).to(self.device), cs[i].squeeze(1).to(self.device)))

        hs = torch.stack(hs, dim=1)
        if cs is not None:
            cs = torch.stack(cs, dim=1)

        # Block gradient through inactive units
        mask = mask.unsqueeze(2)
        h_new = blocked_grad.apply(hs, mask)

        # Compute communication attention
        h_new = self.communication_attention(h_new, mask.squeeze(2))

        hs = mask * h_new + (1 - mask) * h_old
        if cs is not None:
            cs = mask * cs + (1 - mask) * c_old
            return hs, cs

        return hs
