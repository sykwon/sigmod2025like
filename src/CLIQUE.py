from src.estimator import Estimator
from torch.nn import (
    Module,
    LSTM,
    GRU,
    Linear,
    Embedding,
    Sequential,
    ReLU,
    LeakyReLU,
    BatchNorm1d,
)
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import yaml
import os
import pickle
import time
import src.util as ut
import numpy as np
from src.model import init_weights, msle_loss
import torch.nn.functional as F
from torch.optim import Adam
import torch
from torch.utils.data import Dataset, DataLoader
import src.sampling as sp
from src.util import (
    esc_ec,
    esc_ac,
    esc_sc,
    filter_infrequent_entry_PT,
    getCachedExtendedNgramTable,
    getCachedPrunedExtendedNgramTable,
    getCachedPrunedExtendedNgramTableMaxEntry,
)


def is_LIKEquery_in_Table(query, hashtable, N):
    query_cano = ut.canonicalize_like_query(query, True).replace("_", esc_ac)

    is_prefix_query = query_cano[0] != "%"
    is_suffix_query = query_cano[-1] != "%"

    tokens = query_cano.strip("%").split("%")
    is_exact = False
    if len(tokens) == 1 and len(tokens[0]) <= N:
        token = tokens[0]
        if is_prefix_query:
            token = esc_sc + token
        if is_suffix_query:
            token = token + esc_ec
        if token in hashtable:
            is_exact = True
    return is_exact


def get_bounds_from_table_PTs(hashtable, query, PTs):
    max_count = hashtable[""]
    N = len(PTs) - 1

    query_cano = ut.canonicalize_like_query(query, True).replace("_", esc_ac)

    tokens = query_cano.strip("%").split("%")

    is_prefix_query = query_cano[0] != "%"
    is_suffix_query = query_cano[-1] != "%"
    for k, token in enumerate(tokens):
        max_sub_token_length = min(len(token), N)

        is_prefix_token = is_prefix_query and k == 0
        is_suffix_token = is_suffix_query and k == len(tokens) - 1

        for sub_token_length in range(1, max_sub_token_length + 1):
            PT = PTs[sub_token_length]
            for start_pos in range(len(token) - sub_token_length + 1):
                sub_token = token[start_pos: start_pos + sub_token_length]
                if start_pos == 0 and is_prefix_token:
                    sub_token = esc_sc + sub_token
                if start_pos + sub_token_length == len(token) and is_suffix_token:
                    sub_token = sub_token + esc_ec

                if sub_token in hashtable:
                    count = hashtable[sub_token]
                else:
                    # print(f"{query = }, {token = }, {sub_token = }")
                    # assert False
                    count = PT - 1

                if max_count > count:
                    max_count = count

    return max_count


def get_bounds_from_table_PT(hashtable, query, N, PT):
    max_count = hashtable[""]

    query_cano = ut.canonicalize_like_query(query, True).replace("_", esc_ac)

    # if query_cano[0] != '%':
    #     query_cano = esc_sc + query_cano
    # if query_cano[-1] != '%':
    #     query_cano = query_cano + esc_ec

    tokens = query_cano.strip("%").split("%")

    is_prefix_query = query_cano[0] != "%"
    is_suffix_query = query_cano[-1] != "%"
    for k, token in enumerate(tokens):
        sub_token_length = min(len(token), N)

        is_prefix_token = is_prefix_query and k == 0
        is_suffix_token = is_suffix_query and k == len(tokens) - 1

        for start_pos in range(len(token) - sub_token_length + 1):
            sub_token = token[start_pos: start_pos + sub_token_length]
            if start_pos == 0 and is_prefix_token:
                sub_token = esc_sc + sub_token
            if start_pos + sub_token_length == len(token) and is_suffix_token:
                sub_token = sub_token + esc_ec

            if sub_token in hashtable:
                count = hashtable[sub_token]
            else:
                # print(f"{query = }, {token = }, {sub_token = }")
                # assert False
                count = PT - 1

            if max_count > count:
                max_count = count

    return max_count


def get_bounds_PT(counts_per_len, like_predicates, N, PTs, cards):
    u_bnds = []
    ratios = []
    sub_ratios = []
    for query, card in zip(like_predicates, cards):  # input should filpped
        # print(query)
        if query[0] == "%" and query[-1] == "%":
            query_type = "sub"
        elif query[0] == "%":
            query_type = "suf"
        elif query[-1] == "%":
            query_type = "pre"
        tokens = query.strip("%").split("%")
        l_bnd = 0
        u_bnd = counts_per_len[0][""]
        if len(tokens) == 0:  # only percent
            l_bnd = u_bnd

        # print(tokens)
        for k, token in enumerate(tokens):
            token_type = "sub"
            if k == 0 and query_type == "pre":
                token_type = "pre"
            if k == len(tokens) - 1 and query_type == "suf":
                token_type = "suf"

            sub_token_length = min(len(token), N)
            for n_length in range(1, sub_token_length + 1):
                counts = counts_per_len[n_length]
                PT = PTs[n_length]
                for p in range(len(token) - n_length + 1):
                    ngram = token[p: p + n_length]
                    if p == 0 and token_type == "pre":
                        ngram = esc_sc + ngram
                    if p == len(token) - n_length and token_type == "suf":
                        ngram = ngram + esc_ec
                    if ngram in counts:
                        count = counts[ngram]
                    else:
                        count = PT - 1
                    # print(f"{ngram = } {count = }")
                    if u_bnd > count:
                        u_bnd = count

        u_bnds.append(u_bnd)
        ratio = u_bnd / card
        assert u_bnd >= card, f"{query = }, {u_bnd = }, {card =}"
        if len(tokens) == 1:
            sub_ratios.append(ratio)

        ratios.append(ratio)
        # if ratio > 9900:
        #     print(f"{query_type = }")
        #     print(f"{query = }, {l_bnd = }, {u_bnd = }, {card = }, {ratio = }")
        # print(f"{query = }, {l_bnd = }, {u_bnd = }, {card = }, {ratio = }")

    ratios = np.array(ratios)
    avg_ratio = np.average(ratios)
    std_ratio = np.std(ratios)
    max_ratio = np.max(ratios)

    avg_sub_ratio = np.average(sub_ratios)
    std_sub_ratio = np.std(sub_ratios)
    max_sub_ratio = np.max(sub_ratios)

    print(
        f"{N = }, {PTs = }, {avg_sub_ratio = :.1f}, {std_sub_ratio = :.1f}, {max_sub_ratio = :.1f}"
    )
    print(
        f"{N = }, {PTs = }, {avg_ratio = :.1f}, {std_ratio = :.1f}, {max_ratio = :.1f}"
    )
    return u_bnds


class CLIQUEdataset(Dataset):
    def __init__(self, data, char_dict, packed=False):
        super().__init__()

        # data
        # base: (string, card, bound)
        # packed: (string, cards, bounds, b_in_tables)

        self.data = data
        self.patterns = [
            ut.canonicalize_like_query(x[0], is_last_flip=True) for x in data
        ]
        self.lens = [len(x) for x in self.patterns]
        self.ys = [x[1] for x in data]
        self.max_bounds = [x[2] for x in data]
        if packed:
            self.b_in_tables = [x[3] for x in data]
        self.encoded_sequences = ut.string_encoding(self.patterns, char_dict)

        self.char_dict = char_dict
        self.packed = packed

    def __getitem__(self, index):
        encoded_sequence = self.encoded_sequences[index]
        max_bound = self.max_bounds[index]
        pattern = self.patterns[index]
        if self.packed:
            return (
                encoded_sequence,
                self.lens[index],
                max_bound,
                self.b_in_tables[index],
                self.ys[index],
            )
        else:
            return (
                encoded_sequence,
                self.lens[index],
                max_bound,
                self.ys[index],
            )

    def __len__(self):
        return len(self.data)


def evaluate_hybrid_model(model, dl_test):
    losses = []
    q_errs = []
    estimates = []
    model.eval()
    to_cuda = next(model.parameters()).is_cuda
    with torch.no_grad():
        for batch in dl_test:
            xs, ys = ut.batch2cuda(batch, to_cuda=to_cuda)
            loss = model.loss(xs, ys, reduction="none")
            loss_ = loss.cpu().numpy()
            losses.extend(loss_)
            preds = model(xs)
            preds_ = preds.cpu().numpy()
            estimates.extend(preds_)
            q_err = ut.mean_Q_error(ys.cpu(), preds_, reduction="none")
            q_errs.extend(q_err)
    return losses, q_errs, estimates


def collate_fn_hybrid_pack(batch):
    batch = tuple(zip(*batch))
    xs, lens, max_bounds, b_in_tables, ys_map = batch

    # print(f"before {xs = }")
    xs = ut.keras_pad_sequences(xs, padding="post")
    # print(f"after {xs = }")

    max_bounds = ut.keras_pad_sequences(max_bounds, padding="post")
    b_in_tables = ut.keras_pad_sequences(b_in_tables, padding="post")

    # print(f"before {ys_map = }")
    ys_map = ut.keras_pad_sequences(ys_map, padding="post")
    # print(f"after {ys_map = }")
    # print(f"before {mask = }")
    # mask = ut.keras_pad_sequences(mask, padding='post')
    # print(f"after {mask = }")

    batch = (
        torch.LongTensor(xs),
        lens,
        torch.FloatTensor(max_bounds),
        b_in_tables,
        torch.FloatTensor(ys_map),
    )
    return batch


def collate_fn_hybrid(batch):
    batch = tuple(zip(*batch))
    xs, lens, max_bounds, ys = batch

    if len(xs) > 1:
        xs = ut.keras_pad_sequences(xs, padding="post")

    batch = (
        torch.LongTensor(xs),
        lens,
        torch.FloatTensor(max_bounds),
        torch.FloatTensor(ys),
    )
    return batch


def get_hybrid_dl_train(train_data, bs, char_dict, packed, shuffle=True):
    ds_train = CLIQUEdataset(train_data, char_dict, packed=packed)
    if packed:
        train_collate_fn = collate_fn_hybrid_pack
    else:
        train_collate_fn = collate_fn_hybrid

    dl_train = DataLoader(
        ds_train, batch_size=bs, shuffle=shuffle, collate_fn=train_collate_fn
    )

    return dl_train


def get_hybrid_dl_test(test_data, bs, char_dict):
    packed = False
    return get_hybrid_dl_train(
        test_data, bs, char_dict, packed, shuffle=False
    )


def get_hybrid_dl_valid(valid_data, bs, char_dict):
    return get_hybrid_dl_test(valid_data, bs, char_dict)


class CLIQUEestimator(Module, Estimator):
    def __init__(self):
        super().__init__()

    def set_params(
        self,
        N,
        PT,
        seed,
        char_dict,
        ch_es,
        n_rnn,
        packed,
        rnn_hs,
        pred_hs,
        n_pred_layer,
        lr,
        l2,
        n_epoch,
        bs,
        patience,
        last_flip,
        sw,
        count_path,
        model_path,
        bound_in,
        pack_all,
        dynamicPT,
        max_entry_ratio,
        batch_norm,
    ):
        self.N = N
        self.PT = PT

        self.seed = seed
        self.char_dict = char_dict
        self.n_char = len(char_dict)
        self.ch_es = ch_es
        self.n_rnn = n_rnn
        self.packed = packed
        self.seq_out = packed
        self.rnn_hs = rnn_hs
        self.pred_hs = pred_hs
        self.n_pred_layer = n_pred_layer
        self.lr = lr
        self.l2 = l2
        self.n_epoch = n_epoch
        self.bs = bs
        self.patience = patience
        self.last_flip = last_flip
        self.sw = sw
        self.count_path = count_path
        self.model_path = model_path
        self.bound_in = bound_in
        self.pack_all = pack_all
        self.dynamicPT = dynamicPT
        self.max_entry_raio = max_entry_ratio
        self.batch_norm = batch_norm

        self.create_model()

    def set_model_params(
        self,
        seed,
        n_char,
        emb_size,
        cell_size,
        n_rnn,
        packed,
        pred_hs,
        n_pred_layer,
        bound_in,
        pack_all,
        batch_norm,
    ):
        self.seed = seed
        self.n_char = n_char
        self.ch_es = emb_size
        self.rnn_hs = cell_size
        self.n_rnn = n_rnn
        self.packed = packed
        self.pred_hs = pred_hs
        self.n_pred_layer = n_pred_layer
        self.bound_in = bound_in
        self.pack_all = pack_all
        self.batch_norm = batch_norm

    def create_model(self):
        seed = self.seed
        n_char = self.n_char
        emb_size = self.ch_es
        cell_size = self.rnn_hs
        n_rnn = self.n_rnn
        packed = self.packed
        pred_hs = self.pred_hs
        n_pred_layer = self.n_pred_layer
        bound_in = self.bound_in
        batch_norm = self.batch_norm

        ut.set_seed(seed)
        self.embedding = Embedding(n_char, emb_size, padding_idx=0)

        self.rnn: LSTM = LSTM(
            emb_size, cell_size, batch_first=True, num_layers=n_rnn
        )

        self.pred_layer: Sequential = Sequential()

        if bound_in:
            pred_hs_list = [cell_size + 1]
        else:
            pred_hs_list = [cell_size]

        pred_hs_list.extend([pred_hs] * (n_pred_layer - 1))
        pred_hs_list.append(1)

        for i, (in_size, out_size) in enumerate(
            zip(pred_hs_list[:-1], pred_hs_list[1:])
        ):
            if batch_norm and i == 0:
                self.pred_layer.add_module(f"BN-{i}", BatchNorm1d(in_size))
            if i > 0:
                self.pred_layer.add_module(f"LeakyReLU-{i}", LeakyReLU())
            self.pred_layer.add_module(f"PRED-{i}", Linear(in_size, out_size))
        self.pred_layer.add_module("LeakyReLU", LeakyReLU())
        self.pred_layer.apply(init_weights)

    def logit(self, x):
        if self.packed and self.training:
            xs, lens, max_bounds, b_in_tables = x
        else:
            xs, lens, max_bounds = x

        embeds = self.embedding(xs)

        lens = torch.LongTensor(lens)
        bs = lens.shape[0]
        packed = pack_padded_sequence(
            embeds, lens, batch_first=True, enforce_sorted=False
        )

        output, hidden_ = self.rnn(packed)

        if self.seq_out and self.training:
            hidden = pad_packed_sequence(output)[
                0
            ]  # [L, bs, cell_size] // [L, bs, cell_size * 2] if biLSTM
            hidden = hidden.transpose(0, 1)  # [bs, L, cell size]
            # hidden_ = h_n, c_n if self.rnn = LSTM
        else:
            if isinstance(self.rnn, LSTM):
                (
                    hidden,
                    cell,
                ) = hidden_
            else:
                hidden = hidden_
            hidden = hidden.reshape(bs, -1)

        if self.bound_in:
            # hidden =  torch.concat([hidden, torch.unsqueeze(max_bounds / self.n_db, -1)], dim=-1) # hidden
            # if packed:
            #     hidden =  torch.concat([hidden, torch.log(max_bounds+1)], dim=-1) # hidden
            # else:
            hidden = torch.concat(
                [hidden, torch.unsqueeze(torch.log(max_bounds + 1), -1)], dim=-1
            )  # hidden

        if self.bound_in and self.packed:
            pred_shape = hidden.shape[:-1]
            hidden_dim = hidden.shape[-1]
            hidden = hidden.reshape(-1, hidden_dim)

        preds = self.pred_layer(hidden)

        if self.bound_in and self.packed:
            preds = preds.reshape(pred_shape)
        preds = torch.squeeze(preds, dim=-1)
        preds = torch.sigmoid(preds)
        preds = max_bounds * preds
        return preds

    def forward(self, x):
        out = self.logit(x)
        if not self.training:
            out = F.relu(out)
        return out

    def get_mask(self, lens):
        assert self.packed
        lens = torch.LongTensor(lens)
        max_len = max(lens)
        mask = torch.arange(max_len).expand(
            len(lens), max_len) < lens.unsqueeze(1)
        return mask

    def loss(self, x, y, reduction="mean", mask=None):
        if self.training and self.packed:
            mask = self.get_mask(x[1])
            if not self.pack_all:
                b_in_tables = torch.BoolTensor(x[-1])
                mask = torch.logical_and(mask, torch.logical_not(b_in_tables))

        preds = self.forward(x)

        res = msle_loss(y, preds, reduction=reduction)

        if mask is not None:
            mask = mask.to(res.device)
            res = msle_loss(y, preds, reduction="none")
            res = torch.masked_select(res, mask=mask)
            if reduction == "mean":
                res = torch.mean(res)
            elif reduction == "sum":
                res = torch.sum(res)
            elif reduction == "none":
                pass
            else:
                raise
        else:
            res = msle_loss(y, preds, reduction=reduction)

        return res

    def build_count(self, data_name):
        N = self.N
        dynamicPT = self.dynamicPT
        max_entry_ratio = self.max_entry_raio
        PT = self.PT
        count_path = self.count_path

        assert N is not None
        assert PT is not None

        if dynamicPT:
            hashtable, PTs, build_time = getCachedPrunedExtendedNgramTableMaxEntry(
                data_name, N, max_entry_ratio
            )
            self.PTs = PTs
        else:
            hashtable, build_time = getCachedPrunedExtendedNgramTable(
                data_name, N, PT)

        self.hashtable = hashtable
        return build_time

    def get_bound(self, query):
        N = self.N
        PT = self.PT
        hashtable = self.hashtable
        dynamicPT = self.dynamicPT
        if dynamicPT:
            PTs = self.PTs
            max_count = get_bounds_from_table_PTs(
                hashtable, query, PTs)
        else:
            max_count = get_bounds_from_table_PT(
                hashtable, query, N, PT
            )

        n_db = self.n_db

        return max_count

    def append_bound(self, labeled_data, packed=False, is_tqdm=True):
        n_db = self.n_db
        last_flip = self.last_flip

        output = []

        if is_tqdm:
            labeled_data_p = tqdm(labeled_data)
            labeled_data_p.set_description("[Append Bound]")
        else:
            labeled_data_p = labeled_data

        if packed:
            for query, cards in labeled_data_p:
                query_cano = ut.canonicalize_like_query(query, last_flip)
                max_bounds = []
                b_in_tables = []
                for query_len in range(1, len(query) + 1):
                    sub_query = query_cano[:query_len]
                    max_bound = self.get_bound(sub_query)
                    b_in_table = is_LIKEquery_in_Table(
                        sub_query, self.hashtable, self.N)
                    b_in_tables.append(b_in_table)
                    max_bounds.append(max_bound)
                output.append([query, cards, max_bounds, b_in_tables])
        else:
            for query, card in labeled_data_p:
                max_bound = self.get_bound(query)
                output.append([query, card, max_bound])

        return output

    def build_model(self, train_data, valid_data, test_data):
        model = self
        lr = self.lr
        l2 = self.l2
        n_epoch = self.n_epoch
        bs = self.bs
        patience = self.patience
        sw = self.sw
        seed = self.seed
        char_dict = self.char_dict
        model_path = self.model_path
        packed = self.packed

        train_data = self.append_bound(train_data, packed=self.packed)
        valid_data = self.append_bound(valid_data)
        test_data = self.append_bound(test_data)

        dl_train = get_hybrid_dl_train(train_data, bs, char_dict, packed)
        dl_valid = get_hybrid_dl_valid(valid_data, bs, char_dict)
        dl_test = get_hybrid_dl_test(test_data, bs, char_dict)

        ut.set_seed(seed)
        model.cuda()
        optim = Adam(model.parameters(), lr=lr, weight_decay=l2)
        bs_step = 0
        best_val_score = float("inf")
        best_epoch = -1
        best_test_score = -1

        epochs_since_improvement = 0

        build_time = 0
        for epoch in range(1, n_epoch + 1):
            start_time = time.time()
            model.train()

            for batch in tqdm(
                dl_train, total=len(dl_train), desc=f"[Epoch {epoch:3d}/{n_epoch:3d}]"
            ):
                optim.zero_grad()

                xs, ys = ut.batch2cuda(batch)
                # print(f"{xs = }")
                # print(f"{ys = }")
                # preds = model(xs)
                # print(f"{preds = }")

                loss = model.loss(xs, ys)
                loss.backward()
                loss_ = float(loss.detach().cpu().numpy())
                # print(f"[loss]: {loss}")
                optim.step()
                with torch.no_grad():
                    preds_ = model(xs).cpu().numpy()

                if model.training and model.packed:
                    mask = self.get_mask(xs[1]).numpy()
                    avg_q_err_ = ut.mean_Q_error(ys.cpu(), preds_, mask=mask)
                    max_q_err_ = ut.mean_Q_error(
                        ys.cpu(), preds_, mask=mask, reduction="max"
                    )
                else:
                    avg_q_err_ = ut.mean_Q_error(ys.cpu(), preds_)
                    max_q_err_ = ut.mean_Q_error(
                        ys.cpu(), preds_, reduction="max")
                bs_step += 1
                sw.add_scalars(f"Loss", {"train": loss_}, global_step=bs_step)
                sw.add_scalars(
                    f"ACC", {"train": avg_q_err_}, global_step=bs_step)
                sw.add_scalars(
                    f"ACC", {"train_max": max_q_err_}, global_step=bs_step)

            # valid score
            losses, q_errs, estimates = evaluate_hybrid_model(model, dl_valid)
            loss_ = sum(losses) / len(losses)
            avg_q_err_ = sum(q_errs) / len(q_errs)
            max_q_err_ = max(q_errs)

            sw.add_scalars(f"Loss", {"valid": loss_}, global_step=bs_step)
            sw.add_scalars(f"ACC", {"valid": avg_q_err_}, global_step=bs_step)
            sw.add_scalars(
                f"ACC", {"valid_max": max_q_err_}, global_step=bs_step)

            end_time = time.time()
            build_time += end_time - start_time

            q_err_ = avg_q_err_

            if q_err_ < best_val_score:
                best_val_score = q_err_
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # test score
            losses, q_errs, estimates = evaluate_hybrid_model(model, dl_test)
            loss_ = sum(losses) / len(losses)
            avg_q_err_ = sum(q_errs) / len(q_errs)
            max_q_err_ = max(q_errs)
            sw.add_scalars(f"Loss", {"test": loss_}, global_step=bs_step)
            sw.add_scalars(f"ACC", {"test": avg_q_err_}, global_step=bs_step)
            sw.add_scalars(
                f"ACC", {"test_max": max_q_err_}, global_step=bs_step)

            if epochs_since_improvement == 0:
                best_test_score = avg_q_err_

            if epochs_since_improvement >= patience:
                print(f"Early stopping triggered.")
                break

        print(
            f"{best_val_score = }, {best_test_score = }, {patience = }, {best_epoch = }"
        )

        model.load_state_dict(torch.load(model_path))

        return build_time

    def build(self, data_name, db, train_data, valid_data, test_data):
        seed = self.seed
        self.n_db = len(db)
        np.random.seed(seed)

        build_time_count = self.build_count(data_name)
        print(f"{build_time_count = }")
        build_time_model = self.build_model(train_data, valid_data, test_data)

        build_time = build_time_count + build_time_model

        return build_time

    def estimate(self, test_queries, to_cuda=True, is_tqdm=True, **kwargs):
        char_dict = self.char_dict

        test_data = []
        estimates = []
        for q_id, test_query in enumerate(test_queries):
            b_in_table = is_LIKEquery_in_Table(
                test_query, self.hashtable, self.N)
            if b_in_table:
                test_query_norm = ut.flip_last_canonicalized_like_query(
                    test_query)
                ngram = test_query_norm.strip("%").replace("_", esc_ac)
                is_prefix_query = test_query_norm[0] != "%"
                is_suffix_query = test_query_norm[-1] != "%"
                if is_prefix_query:
                    ngram = esc_sc + ngram
                if is_suffix_query:
                    ngram = ngram + esc_ec
                card = self.hashtable[ngram]
                estimates.append((card, q_id))
            else:
                test_data.append((test_query, q_id))

        if len(test_data) > 0:
            test_data = self.append_bound(test_data, is_tqdm=is_tqdm)
            if not to_cuda and test_data[0][-2] == test_data[0][-1]:
                return [test_data[0][-1]]

            ds_test = CLIQUEdataset(test_data, char_dict, packed=False)
            dl_test = DataLoader(
                ds_test, batch_size=1, shuffle=False, collate_fn=collate_fn_hybrid
            )

            model = self

            for batch in dl_test:
                if to_cuda:
                    xs, ys = ut.batch2cuda(batch, to_cuda=to_cuda)
                else:
                    xs, ys = batch[:-1], batch[-1]
                preds = model(xs)
                preds_ = preds.detach().cpu().numpy()
                estimates.extend(zip(preds_, ys.tolist()))

        estimates = [x[0] for x in sorted(estimates, key=lambda x: x[1])]

        return estimates

    def model_size(self):
        count_path = self.count_path
        model_path = self.model_path
        count_size = os.path.getsize(count_path)
        model_size = os.path.getsize(model_path)
        print(f"{count_size = } {model_size = }")
        total_size = count_size + model_size
        return total_size
