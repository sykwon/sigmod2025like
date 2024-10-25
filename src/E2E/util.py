import numpy as np
from src.plan_encoding.encoding_predicates import get_representation
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import re
import os
import pickle
from src.training.train_and_test import qerror_loss, normalize_label, unnormalize
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def read_strings(filepath):
    lines = []
    with open(filepath) as f:
        for line in f:
            lines.append(line.rstrip('\n'))
    return lines

def query_type(query):
    if query[0] == '%' and query[-1] == '%':
        return 0    # substring
    elif query[-1] == '%':
        return 1    # prefix
    elif query[-1] == '%':
        return 2    # suffix
    else:
        return 3    # exact

def query_type_onehot_vector(query):
    output = np.zeros(4, dtype=np.float32)
    output[query_type(query)] = 1.
    return output

def parse_like_query(qry, split_beta=False):
    parsed = []
    curr = qry[0]
    is_wild = (curr == '_' or curr == '%')
    for ch in qry[1:]:
        if ch == '_' or ch == '%':
            if is_wild:
                curr += ch
            else:
                parsed.append(curr)
                is_wild = True
                curr = ch
        else:
            if is_wild:
                parsed.append(curr)
                is_wild = False
                curr = ch
            else:
                curr += ch
    parsed.append(curr)
    if split_beta:
        parsed_bak = parsed
        parsed = []
        for token in parsed_bak:
            if '%' in token or '_' in token:
                parsed.append(token)
            else:
                parsed.extend(list(token))
    return parsed


def canonicalize_like_query(qry, is_last_flip=False):
    parsed = parse_like_query(qry)

    out_tokens = []
    for token in parsed:
        if '_' in token or '%' in token:
            if '%' in token:
                new_token = '%'
            else:
                new_token = ''
            new_token += '_' * token.count('_')
            out_tokens.append(new_token)
        else:
            out_tokens.append(token)
    if is_last_flip and '%' in out_tokens[-1]:
        out_tokens[-1] = '_' * (len(out_tokens[-1])-1) + '%'
    return ''.join(out_tokens)


def get_str_representation(value, word_vectors):
    #  str in query
    # value Din%in
    # output average vectors in tokens split by '%'
    value = canonicalize_like_query(value, is_last_flip=True)
    vec = np.array([])
    count = 0
    qtype_size = 4
    hash_size = 500 - qtype_size
    # for v in value.split('%'):
    for v in re.split("%|_", value):
        if len(v) > 0:
            if len(vec) == 0:
                vec = get_representation(v, word_vectors, hash_size)
                count = 1
            else:
                new_vec = get_representation(v, word_vectors, hash_size)
                vec = vec + new_vec
                count += 1
    if count > 0:
        vec /= float(count)
    
    qtype_vec = query_type_onehot_vector(value)
    vec = np.concatenate([vec, qtype_vec])
    return vec


class Representation(nn.Module):
    def __init__(self, input_dim, hidden_dim, hid_dim, middle_result_dim=None, task_num=None):
        super(Representation, self).__init__()
        self.hidden_dim = hidden_dim    # 256
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True) # batch first
        self.batch_norm1 = nn.BatchNorm1d(hid_dim)  # 128
        # The linear layer that maps from hidden state space to tag space

        self.sample_mlp = nn.Linear(1000, hid_dim)
        self.condition_mlp = nn.Linear(hidden_dim, hid_dim)
        #         self.out_mlp1 = nn.Linear(hidden_dim, middle_result_dim)
        #         self.hid_mlp1 = nn.Linear(15+108+2*hid_dim, hid_dim)
        #         self.out_mlp1 = nn.Linear(hid_dim, middle_result_dim)

        self.lstm2 = nn.LSTM(15 + 108 + 2 * hid_dim, hidden_dim, batch_first=True)
        #         self.lstm2_binary = nn.LSTM(15+108+2*hid_dim, hidden_dim, batch_first=True)
        #         self.lstm2_binary = nn.LSTM(15+108+2*hid_dim, hidden_dim, batch_first=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        # The linear layer that maps from hidden state space to tag space
        self.hid_mlp2_task1 = nn.Linear(hidden_dim, hid_dim)
        # self.hid_mlp2_task2 = nn.Linear(hidden_dim, hid_dim)
        self.batch_norm3 = nn.BatchNorm1d(hid_dim)
        self.hid_mlp3_task1 = nn.Linear(hid_dim, hid_dim)
        # self.hid_mlp3_task2 = nn.Linear(hid_dim, hid_dim)
        self.out_mlp2_task1 = nn.Linear(hid_dim, 1)
        # self.out_mlp2_task2 = nn.Linear(hid_dim, 1)

    #         self.hidden2values2 = nn.Linear(hidden_dim, action_num)

    def init_hidden(self, hidden_dim, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, hidden_dim),
                torch.zeros(1, batch_size, hidden_dim))

    # def forward(self, operators, extra_infos, condition1s, condition2s, samples, condition_masks, mapping):
    def forward(self, conditions, samples):
        # conditions: [batch_size, 1000]
        # samples   : [batch_size, 1000]
        
        batch_size = len(conditions)
        # batch_size = 0
        # for i in range(operators.size()[1]):
        #     if operators[0][i].sum(0) != 0:
        #         batch_size += 1
        #     else:
        #         break
        # print('batch_size: ', batch_size)

        #         print (operators.size())          # operation     : X
        #         print (extra_infos.size())        # meta_data     : X
        #         print (condition1s.size())        # predicate     : O
        #         print (condition2s.size())        # predicate2    : X
        #         print (samples.size())            # sample bitmap : O
        #         print (condition_masks.size())    # mask          : X
        #         print (mapping.size())            # child_info    : X

        #         torch.Size([14, 133, 15])         # node_size, batch_size, ?
        #         torch.Size([14, 133, 108])        # [num_level, num_node_per_level, dim]
        #         torch.Size([14, 133, 13, 1119])   # [num_level, num_node_per_level, num_condition_per_node, condition_op_length]
        #         torch.Size([14, 133, 13, 1119])
        #         torch.Size([14, 133, 1000])
        #         torch.Size([14, 133, 1])
        #         torch.Size([14, 133, 2])

        # num_level = conditions.size()[0]               # 14
        # num_node_per_level = conditions.size()[1]      # 133
        # num_condition_per_node = conditions.size()[2]  # 13
        # condition_op_length = conditions.size()[3]     # 1119

        inputs = conditions.view(batch_size, 1, -1) # [batch_size, 1, 1000]
        # inputs = conditions.view(num_level * num_node_per_level, num_condition_per_node, condition_op_length) # [14 * 133, 13, 1119]
        # hidden = self.init_hidden(self.hidden_dim, num_level * num_node_per_level) # [256, 14 * 133]

        out, (hid, cid) = self.lstm1(inputs) # hid: [1, batch_size, hidden_dim]
        # last_output1 = hid[0].view(num_level * num_node_per_level, -1)
        last_output = hid.squeeze(0) #

        last_output = F.relu(self.condition_mlp(last_output))
        last_output = self.batch_norm1(last_output)
        # last_output = self.batch_norm1(last_output).view(num_level, num_node_per_level, -1)

        #         print (last_output.size())
        #         torch.Size([14, 133, 256]) # [num_level, num_node_per_level, hidden_dim]

        sample_output = F.relu(self.sample_mlp(samples)) # [batch_size, 1000] -> [batch_size, hid_dim=128]
        # 
        # sample_output = sample_output * condition_masks

        out = torch.cat((last_output, sample_output), -1)
        # out = torch.cat((operators, extra_infos, last_output, sample_output), 2)
        #         print (out.size())
        #         torch.Size([14, 133, 635])
        #         out = out * node_masks

        # start = time.time()
        # hidden = self.init_hidden(self.hidden_dim, num_node_per_level)
        # last_level = out[num_level - 1].view(num_node_per_level, 1, -1)
        # #         torch.Size([133, 1, 635])
        # _, (hid, cid) = self.lstm2(last_level, hidden)
        # mapping = mapping.long()
        # for idx in reversed(range(0, num_level - 1)):
        #     mapp_left = mapping[idx][:, 0]
        #     mapp_right = mapping[idx][:, 1]
        #     pad = torch.zeros_like(hid)[:, 0].unsqueeze(1)
        #     next_hid = torch.cat((pad, hid), 1)
        #     pad = torch.zeros_like(cid)[:, 0].unsqueeze(1)
        #     next_cid = torch.cat((pad, cid), 1)
        #     hid_left = torch.index_select(next_hid, 1, mapp_left)
        #     cid_left = torch.index_select(next_cid, 1, mapp_left)
        #     hid_right = torch.index_select(next_hid, 1, mapp_right)
        #     cid_right = torch.index_select(next_cid, 1, mapp_right)
        #     hid = (hid_left + hid_right) / 2
        #     cid = (cid_left + cid_right) / 2
        #     last_level = out[idx].view(num_node_per_level, 1, -1)
        #     _, (hid, cid) = self.lstm2(last_level, (hid, cid))
        # output = hid[0]
        # #         print (output.size())
        # #         torch.Size([133, 128])
        # end = time.time()
        # print('Forest Evaluate Running Time: ', end - start)
        # last_output = output[0:batch_size]

        out = self.batch_norm2(out)
        # out = self.batch_norm2(last_output)

        out_task = F.relu(self.hid_mlp2_task1(out))
        out_task = self.batch_norm3(out_task)
        out_task = F.relu(self.hid_mlp3_task1(out_task))
        out_task = self.out_mlp2_task1(out_task)
        out_task = torch.sigmoid(out_task)

        # out_task2 = F.relu(self.hid_mlp2_task2(out))
        # out_task2 = self.batch_norm3(out_task2)
        # out_task2 = F.relu(self.hid_mlp3_task2(out_task2))
        # out_task2 = self.out_mlp2_task2(out_task2)
        # out_task2 = F.sigmoid(out_task2)
        #         print 'out: ', out.size()
        # batch_size * task_num
        return out_task


def compile_LIKE_query(qry):
    query_pat = qry.replace("%", "(.*?)").replace("_", ".")
    qry_compiled = re.compile(query_pat)
    return qry_compiled


def eval_compiled_LIKE_query(qry_re, rec):
    return qry_re.fullmatch(rec)

def get_sample_vector(qry, samples):
    output = []
    qry_re = compile_LIKE_query(qry)
    for sample in samples:
        if eval_compiled_LIKE_query(qry_re, sample):
            output.append(1)
        else:
            output.append(0)
    return np.array(output)

        
if __name__ == "__main__":
    pass