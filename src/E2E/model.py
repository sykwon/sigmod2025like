from gensim.models import Word2Vec
from gensim.models import keyedvectors
from torch.nn import Module
from src.estimator import Estimator
import src.util as ut
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import re
import os
import pickle
from src.E2E.training.train_and_test import (
    qerror_loss,
    normalize_label,
    unnormalize,
    detached_qerror_loss_without_reduction,
)
from src.E2E.plan_encoding.encoding_predicates import get_representation
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from src.util import compile_LIKE_query, eval_compiled_LIKE_query


import os


def query_type(query):
    if query[0] == "%" and query[-1] == "%":
        return 0  # substring
    elif query[-1] == "%":
        return 1  # prefix
    elif query[-1] == "%":
        return 2  # suffix
    else:
        return 3  # exact


def query_type_onehot_vector(query):
    output = np.zeros(4, dtype=np.float32)
    output[query_type(query)] = 1.0
    return output


def get_sample_vector(qry, samples):
    output = []
    qry_re = compile_LIKE_query(qry)
    for sample in samples:
        if eval_compiled_LIKE_query(qry_re, sample):
            output.append(1)
        else:
            output.append(0)
    return np.array(output)


def get_str_representation(value, word_vectors):
    #  str in query
    # value Din%in
    # output average vectors in tokens split by '%'
    value = ut.canonicalize_like_query(value, is_last_flip=True)
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
    else:
        vec = np.array([0.0 for _ in range(500 + hash_size)])

    qtype_vec = query_type_onehot_vector(value)
    vec = np.concatenate([vec, qtype_vec])
    return vec


class E2Edataset(Dataset):
    def __init__(
        self,
        queries,
        cards,
        KV,
        samples,
        n_db,
        data_name,
        seed,
        train_type,
        is_cache=True,
    ):
        super().__init__()
        self.queries = queries
        self.cards = cards
        self.KV = KV
        self.samples = samples
        self.n_db = n_db
        self.data_name = data_name
        self.seed = seed
        self.mini = np.log(1)
        self.maxi = np.log(n_db)
        self.train_type = train_type
        self.is_cache = is_cache

        if self.is_cache:
            self.q_vec_list = [
                get_str_representation(query, self.KV).astype(np.float32)
                for query in queries
            ]
            print("q_vec_list done")

        if is_cache:
            cache_path = f"res/{data_name}/bit_map/{train_type}.pkl"
            if not os.path.exists(cache_path):
                self.q_sp_list = []
                for query in tqdm(queries):
                    self.q_sp_list.append(
                        get_sample_vector(query, samples).astype(np.float32)
                    )
                print("q_sp_list done")
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(self.q_sp_list, f)
                print(f"q_sp_list saved. {cache_path = }")

            with open(cache_path, "rb") as f:
                self.q_sp_list = pickle.load(f)
            print(f"q_sp_list loaded. {cache_path = }")

    def __getitem__(self, index):
        if self.cards is not None:
            card = normalize_label(
                max(self.cards[index], 1), self.mini, self.maxi)
        else:
            card = 0
        if self.is_cache:
            q_vec = self.q_vec_list[index]
            q_sp = self.q_sp_list[index]
        else:
            query = self.queries[index]
            q_vec = get_str_representation(query, self.KV)
            q_vec = q_vec.astype(np.float32)
            q_sp = get_sample_vector(query, self.samples)
            q_sp = q_sp.astype(np.float32)

        return q_vec, q_sp, card

    def __len__(self):
        return len(self.queries)


def evaluate_E2E_model(model, dl_test, mini, maxi):
    estimates = []
    q_errs = []
    model.eval()

    for batch in dl_test:
        # print(f"{batch = }")
        q_vec_batch, q_sp_batch, card_batch = batch
        q_vec_batch = q_vec_batch.cuda()
        q_sp_batch = q_sp_batch.cuda()
        card_batch = card_batch.cuda()
        normalized_estimates_batch = model.forward(q_vec_batch, q_sp_batch)
        estimates_batch = unnormalize(
            normalized_estimates_batch.detach(), mini, maxi)
        estimates.extend(estimates_batch.tolist())
        q_errs_batch = detached_qerror_loss_without_reduction(
            normalized_estimates_batch, card_batch, mini, maxi
        )
        q_errs.extend(q_errs_batch)

    return q_errs, estimates


class E2Eestimator(Module, Estimator):
    def __init__(
        self,
        data_name,
        n_sample,
        input_dim,
        hidden_dim,
        hid_dim,
        wv_size,
        min_count,
        seed,
        model_path,
        wv_path,
        packed,
    ):
        super(E2Eestimator, self).__init__()

        self.data_name = data_name
        self.n_sample = n_sample
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hid_dim = hid_dim
        self.wv_size = wv_size
        self.min_count = min_count
        self.seed = seed
        self.model_path = model_path
        self.wv_path = wv_path
        self.packed = packed

        db = ut.read_strings(f"data/{data_name}/{data_name}.txt")

        np.random.seed(seed)

        self.KV = None
        self.n_db = len(db)
        self.samples = np.random.choice(db, n_sample)
        self.create_model()

    def create_model(self):
        input_dim = self.input_dim
        hidden_dim = self.hidden_dim
        hid_dim = self.hid_dim
        n_sample = self.n_sample

        self.lstm1 = nn.LSTM(input_dim, hidden_dim,
                             batch_first=True)  # batch first
        self.batch_norm1 = nn.BatchNorm1d(hid_dim)  # 128
        # The linear layer that maps from hidden state space to tag space

        self.sample_mlp = nn.Linear(n_sample, hid_dim)
        self.condition_mlp = nn.Linear(hidden_dim, hid_dim)
        #         self.out_mlp1 = nn.Linear(hidden_dim, middle_result_dim)
        #         self.hid_mlp1 = nn.Linear(15+108+2*hid_dim, hid_dim)
        #         self.out_mlp1 = nn.Linear(hid_dim, middle_result_dim)

        # self.lstm2 = nn.LSTM(15 + 108 + 2 * hid_dim, hidden_dim, batch_first=True)
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

        inputs = conditions.view(batch_size, 1, -1)  # [batch_size, 1, 1000]
        # inputs = conditions.view(num_level * num_node_per_level, num_condition_per_node, condition_op_length) # [14 * 133, 13, 1119]
        # hidden = self.init_hidden(self.hidden_dim, num_level * num_node_per_level) # [256, 14 * 133]

        # hid: [1, batch_size, hidden_dim]
        out, (hid, cid) = self.lstm1(inputs)
        # last_output1 = hid[0].view(num_level * num_node_per_level, -1)
        last_output = hid.squeeze(0)

        last_output = F.relu(self.condition_mlp(last_output))
        last_output = self.batch_norm1(last_output)
        # last_output = self.batch_norm1(last_output).view(num_level, num_node_per_level, -1)

        #         print (last_output.size())
        #         torch.Size([14, 133, 256]) # [num_level, num_node_per_level, hidden_dim]

        sample_output = F.relu(
            self.sample_mlp(samples)
        )  # [batch_size, 1000] -> [batch_size, hid_dim=128]
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

    def build(
        self,
        train_data,
        valid_data,
        test_data,
        lr,
        n_epoch,
        n_epoch_wv,
        bs,
        patience,
        sw,
    ):
        start_time = time.time()
        self.build_wv(n_epoch_wv)
        self.build_model(
            train_data, valid_data, test_data, lr, n_epoch, bs, patience, sw
        )
        end_time = time.time()
        build_time = end_time - start_time
        return build_time

    def build_wv(self, n_epoch_wv):
        data_name = self.data_name
        min_count = self.min_count
        wv_size = self.wv_size
        wv_path = self.wv_path

        db = ut.read_strings(f"data/{data_name}/{data_name}.txt")
        if data_name == "GENE":
            sentences = [ut.split_string_by_length(x, 5) for x in db]
        else:
            sentences = [x.split() for x in db]
        self.mini = np.log(1)
        self.maxi = np.log(len(db))

        w2v_model = Word2Vec(
            min_count=min_count,
            window=5,
            vector_size=wv_size,
            alpha=0.03,
            min_alpha=0.0007,
            negative=20,
            workers=1,
        )

        print(f"build vocab")
        w2v_model.build_vocab(sentences, progress_per=10000)

        print(f"train w2v_model")
        w2v_model.train(
            sentences,
            total_examples=w2v_model.corpus_count,
            epochs=n_epoch_wv,
            report_delay=1,
        )

        os.makedirs(os.path.dirname(wv_path), exist_ok=True)
        w2v_model.wv.save(wv_path)
        print(f"word saved at {wv_path = }")

        self.KV = keyedvectors.KeyedVectors.load(self.wv_path)

    def build_model(
        self, train_data, valid_data, test_data, lr, n_epoch, bs, patience, sw
    ):
        model = self
        seed = self.seed
        data_name = self.data_name
        model_path = self.model_path
        KV = self.KV
        n_db = self.n_db
        samples = self.samples

        mini = self.mini
        maxi = self.maxi

        train_queries, train_true_cards = list(zip(*train_data))
        valid_queries, valid_true_cards = list(zip(*valid_data))
        test_queries, test_true_cards = list(zip(*test_data))

        train_type = "train"
        if self.packed:
            train_type += "_packed"

        ds_train = E2Edataset(
            train_queries,
            train_true_cards,
            KV,
            samples,
            n_db,
            data_name,
            seed,
            train_type,
        )
        ds_valid = E2Edataset(
            valid_queries,
            valid_true_cards,
            KV,
            samples,
            n_db,
            data_name,
            seed,
            "valid",
        )
        ds_test = E2Edataset(
            test_queries,
            test_true_cards,
            KV,
            samples,
            n_db,
            data_name,
            seed,
            "test",
        )

        dl_train = DataLoader(ds_train, bs)
        dl_valid = DataLoader(ds_valid, bs)
        dl_test = DataLoader(ds_test, batch_size=1)

        ut.set_seed(seed)
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        bs_step = 0
        best_val_score = float("inf")
        best_epoch = -1
        best_test_score = -1

        epochs_since_improvement = 0

        build_time = 0
        for epoch in range(1, n_epoch + 1):
            start_time = time.time()
            model.train()

            # card_loss_total_train = 0.
            # card_loss_total_valid = 0.
            for batch in tqdm(
                dl_train, total=len(dl_train), desc=f"[Epoch {epoch:3d}/{n_epoch:3d}]"
            ):
                optimizer.zero_grad()

                q_vec_batch, q_sp_batch, card_batch = batch
                q_vec_batch = q_vec_batch.cuda()
                q_sp_batch = q_sp_batch.cuda()
                card_batch = card_batch.cuda()

                estimate_cardinality = model.forward(q_vec_batch, q_sp_batch)
                loss, card_loss_median, card_loss_max, card_max_idx = qerror_loss(
                    estimate_cardinality, card_batch, mini, maxi
                )
                loss.backward()
                optimizer.step()

                # if model.training and model.packed:
                #     assert False
                #     # mask = xs[2].cpu().numpy()
                #     # q_err_ = ut.mean_Q_error(ys.cpu(), preds_, mask=mask)
                # else:
                #     q_err_ = ut.mean_Q_error(ys.cpu(), estimate_cardinality)
                bs_step += 1
                loss_ = float(loss.detach().cpu().numpy())
                # sw.add_scalars(f"Loss", {'train': loss_}, global_step=bs_step)
                sw.add_scalars(f"ACC", {"train": loss_}, global_step=bs_step)

                # card_loss_total_train += loss.item() * len(q_vec_batch)

                # print(f"{loss.item() = :.3f}, {card_loss_median.item() = :.3f}", end="\r")
                # print(f"{loss.item(), card_loss_median.item(), card_loss_max.item() = } ", end="\r")
            # card_loss_total_train /= len(train_queries)
            # print(" " * 60, end="\r")
            # print(f"{epoch = }, {card_loss_total_train = :.3f}")

            # valid score
            q_errs, estimates = evaluate_E2E_model(model, dl_valid, mini, maxi)
            q_err_ = sum(q_errs) / len(q_errs)
            sw.add_scalars(f"ACC", {"valid": q_err_}, global_step=bs_step)

            end_time = time.time()
            build_time += end_time - start_time

            if q_err_ < best_val_score:
                best_val_score = q_err_
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # test score
            q_errs, estimates = evaluate_E2E_model(model, dl_test, mini, maxi)
            q_err_ = sum(q_errs) / len(q_errs)
            sw.add_scalars(f"ACC", {"test": q_err_}, global_step=bs_step)

            if epochs_since_improvement == 0:
                best_test_score = q_err_

            if epochs_since_improvement >= patience:
                print(f"Early stopping triggered.")
                break

        print(
            f"{best_val_score = }, {best_test_score = }, {patience = }, {best_epoch = }"
        )

        model.load_state_dict(torch.load(model_path))

        return build_time

    def estimate(self, test_queries, is_tqdm=True):
        model = self
        KV = self.KV
        data_name = self.data_name
        seed = self.seed
        samples = self.samples
        n_db = self.n_db
        mini = self.mini
        maxi = self.maxi

        ds_test = E2Edataset(
            test_queries,
            None,
            KV,
            samples,
            n_db,
            data_name,
            seed,
            "test",
            is_cache=False,
        )

        dl_test = DataLoader(ds_test, batch_size=1)
        estimates = []
        for batch in dl_test:
            q_vec_batch, q_sp_batch, card_batch = batch
            normalized_estimates_batch = model.forward(q_vec_batch, q_sp_batch)
            estimates_batch = unnormalize(
                normalized_estimates_batch.detach(), mini, maxi
            ).squeeze(-1)
            estimates.extend(estimates_batch.tolist())

        return estimates

    def model_size(self):
        size_model = os.path.getsize(self.model_path)
        size_wv = os.path.getsize(self.wv_path)
        size_total = size_model + size_wv
        print(f"{size_model = }")
        print(f"{size_wv = }")
        print(f"{size_total = }")
        return size_total
