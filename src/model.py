import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import time
import os
from torch.nn import Module, LSTM, Linear, Embedding, Sequential, ReLU, LeakyReLU
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_
from torch.nn.functional import mse_loss
import src.util as ut
from src.util import Qtype, batch2cuda
import numpy as np
from src.estimator import Estimator
from tqdm import tqdm


class LikeDataset(Dataset):
    def __init__(
        self,
        data,
        char_dict,
        packed=False,
        qc_dict=None,
        cf_dicts=None,
    ):
        super().__init__()

        self.data = data
        self.patterns = [x[0] for x in data]

        self.lens = [len(x) for x in self.patterns]
        self.ys = [x[1] for x in data]
        self.encoded_sequences = ut.string_encoding(self.patterns, char_dict)
        self.qc_dict = qc_dict
        if self.qc_dict is None:
            self.qc_dict = {
                ut.canonicalize_like_query(x[0], is_last_flip=True): x[1]
                for x in data
            }  # for LSTM
        self.cf_dicts = cf_dicts
        if self.cf_dicts is not None:
            self.char_features = []
            for pattern in self.patterns:
                char_feature = []
                for x in pattern:
                    char_feature.append([cf_dict[x] for cf_dict in self.cf_dicts])
                self.char_features.append(char_feature)

        self.char_dict = char_dict
        self.packed = packed

    def __getitem__(self, index):
        encoded_sequence = self.encoded_sequences[index]
        pattern = self.patterns[index]
        if self.packed:
            mask = np.zeros(len(pattern))
            ys_map = np.zeros(len(pattern))
            for i in range(len(pattern)):
                prefix_query = pattern[: i + 1]
                if prefix_query in self.qc_dict:
                    mask[i] = 1
                    ys_map[i] = self.qc_dict[prefix_query]
            return encoded_sequence, self.lens[index], mask, ys_map
        else:
            return encoded_sequence, self.lens[index], self.ys[index]

    def __len__(self):
        return len(self.data)


class BaseDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return tuple(self.data[index])

    def __len__(self):
        return len(self.data)


def collate_fn_Like_pack(batch):
    batch = tuple(zip(*batch))
    xs, lens, mask, ys_map = batch

    # print(f"before {xs = }")
    xs = ut.keras_pad_sequences(xs, padding="post")
    # print(f"after {xs = }")

    # print(f"before {ys_map = }")
    ys_map = ut.keras_pad_sequences(ys_map, padding="post")
    # print(f"after {ys_map = }")
    # print(f"before {mask = }")
    mask = ut.keras_pad_sequences(mask, padding="post")
    # print(f"after {mask = }")

    batch = (
        torch.LongTensor(xs),
        lens,
        torch.LongTensor(mask),
        torch.FloatTensor(ys_map),
    )
    return batch


def collate_fn_Like_cf(batch):
    batch = tuple(zip(*batch))
    xs, cfs, lens, types, ys = batch

    xs = ut.keras_pad_sequences(xs, padding="post")
    cfs = ut.keras_pad_sequences(cfs, value=0.0, padding="post")

    batch = (
        torch.LongTensor(xs),
        torch.FloatTensor(cfs),
        lens,
        types,
        torch.FloatTensor(ys),
    )
    return batch


def collate_fn_Like(batch):
    batch = tuple(zip(*batch))
    xs, lens, ys = batch

    if len(xs) > 1:
        xs = ut.keras_pad_sequences(xs, padding="post")

    batch = torch.LongTensor(xs), lens, torch.FloatTensor(ys)
    return batch


def collate_fn_Like_aug(batch):
    batch = tuple(zip(*batch))
    xs, lens, ys = batch

    xs = ut.keras_pad_sequences(xs, padding="post")
    ys = ut.keras_pad_sequences(ys, padding="post")

    batch = torch.LongTensor(xs), lens, torch.FloatTensor(ys)
    return batch


def collate_fn_Like_pl(batch, training=True):
    batch = tuple(zip(*batch))
    xs, lens, pls, ys = batch

    xs = ut.keras_pad_sequences(xs, padding="post")
    pls = ut.keras_pad_sequences(pls, padding="post")
    if training:
        ys = ut.keras_pad_sequences(ys, padding="post")

    batch = torch.LongTensor(xs), lens, torch.LongTensor(pls), torch.FloatTensor(ys)
    return batch


def get_LSTM_dl_train(train_data, bs, char_dict, packed, qc_dict):
    ds_train = LikeDataset(train_data, char_dict, packed=packed, qc_dict=qc_dict)
    if packed:
        train_collate_fn = collate_fn_Like_pack
    else:
        train_collate_fn = collate_fn_Like
    dl_train = DataLoader(
        ds_train, batch_size=bs, shuffle=True, collate_fn=train_collate_fn
    )

    return dl_train


def get_LSTM_dl_valid(valid_data, bs, char_dict):
    dl_valid = get_LSTM_dl_test(valid_data, bs, char_dict)
    return dl_valid


def get_LSTM_dl_test(test_data, bs, char_dict):
    ds_test = LikeDataset(test_data, char_dict)

    eval_collate_fn = collate_fn_Like

    dl_test = DataLoader(
        ds_test, batch_size=bs, shuffle=False, collate_fn=eval_collate_fn
    )
    return dl_test


def msle_loss(input, target, reduction="mean", mask=None):
    eps = 1e-7
    input = torch.log(torch.clamp_min(input, eps) + 1)
    target = torch.log(torch.clamp_min(target, eps) + 1)
    # print("input:", input)
    # print("target:", target)
    # print(f"{input.device = } {target.device = }")
    if mask is not None:
        loss = mse_loss(input, target, reduction="none")
        # print("1", loss)
        loss *= mask
        # print()
        # print("2", loss)
        # print(f"{mask.sum() = }")
        loss = loss.sum() / mask.sum()
        # print("3", loss)
    else:
        loss = mse_loss(input, target, reduction=reduction)

    return loss


@torch.no_grad()
def init_bias(model):
    if type(model) == nn.Linear:
        torch.nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(0.01)


def get_estimates_from_model(model, dl_test):
    estimates = []
    for batch in dl_test:
        xs, ys = batch2cuda(batch, to_cuda=False)
        preds = model(xs)
        preds_ = preds.detach().cpu().numpy()
        estimates.extend(preds_)
    return estimates


def evaluate_DREAMmodel(model, dl_test):
    losses = []
    q_errs = []
    estimates = []
    model.eval()
    to_cuda = next(model.parameters()).is_cuda
    with torch.no_grad():
        for batch in dl_test:
            xs, ys = batch2cuda(batch, to_cuda=to_cuda)
            loss = model.loss(xs, ys, reduction="none")
            losses.extend(loss)
            preds = model(xs)
            preds_ = preds.cpu().numpy()
            estimates.extend(preds_)
            q_err = ut.mean_Q_error(ys.cpu(), preds_, reduction="none")
            q_errs.extend(q_err)
    return losses, q_errs, estimates


class DREAMmodel(Module, Estimator):
    def __init__(self):
        super().__init__()

    def set_params(
        self,
        char_dict,
        emb_size,
        cell_size,
        pred_hs,
        n_rnn=1,
        n_pred_layer=1,
        pack_type="q",
        n_db=None,
        seed=None,
        packed=False,
        model_path=None,
        **kwargs,
    ):
        n_char = len(char_dict)
        self.n_char = n_char
        self.emb_size = emb_size
        self.pred_hs = pred_hs
        self.n_rnn = n_rnn
        self.n_pred_layer = n_pred_layer
        self.char_dict = char_dict
        self.n_db = n_db
        self.seed = seed
        self.pack_type = pack_type
        self.packed = packed
        self.seq_out = packed or (pack_type != "q" and "ru" not in pack_type)
        self.cell_size = cell_size
        self.model_path = model_path

        self.create_model()

    def set_model_params(
        self,
        n_char,
        emb_size,
        cell_size,
        pred_hs,
        n_rnn=1,
        n_pred_layer=1,
        pack_type="q",
        seed=None,
        packed=False,
    ):
        self.seed = seed
        self.n_char = n_char
        self.emb_size = emb_size
        self.pack_type = pack_type
        self.packed = packed
        self.cell_size = cell_size
        self.n_rnn = n_rnn
        self.pred_hs = pred_hs
        self.n_pred_layer = n_pred_layer

    def create_model(self):
        seed = self.seed
        n_char = self.n_char
        emb_size = self.emb_size
        pack_type = self.pack_type
        packed = self.packed
        cell_size = self.cell_size
        n_rnn = self.n_rnn
        pred_hs = self.pred_hs
        n_pred_layer = self.n_pred_layer

        ut.set_seed(seed)
        if pack_type == "pl":
            max_qry_len = 30
            pl_emb_size = 5
            self.embedding = Embedding(n_char, emb_size - pl_emb_size, padding_idx=0)
            self.embedding_pl = Embedding(max_qry_len, pl_emb_size, padding_idx=0)
        else:
            self.embedding = Embedding(n_char, emb_size, padding_idx=0)
            self.embedding_pl = None

        self.rnn: LSTM = LSTM(
            emb_size, cell_size, batch_first=True, num_layers=n_rnn
        )

        self.pred_layer: Sequential = Sequential()

        pred_hs_list = [cell_size]
        pred_hs_list.extend([pred_hs] * (n_pred_layer - 1))
        pred_hs_list.append(1)

        for i, (in_size, out_size) in enumerate(
            zip(pred_hs_list[:-1], pred_hs_list[1:])
        ):
            if i > 0:
                self.pred_layer.add_module(f"LeakyReLU-{i}", LeakyReLU())
            self.pred_layer.add_module(f"PRED-{i}", Linear(in_size, out_size))
        self.pred_layer.add_module("LeakyReLU", LeakyReLU())
        self.pred_layer.apply(init_weights)

    def logit(self, x, hx=None, out_h=False, **kwargs):
        if self.packed:
            if self.training:
                xs, lens, mask = x
            else:
                xs, lens = x
            embeds = self.embedding(xs)
        else:
            if self.pack_type == "pl":
                xs, lens, pls = x
                embeds_xs = self.embedding(xs)
                embeds_pls = self.embedding_pl(pls)
                embeds = torch.concat([embeds_xs, embeds_pls], dim=-1)
            else:
                xs, lens = x
                embeds = self.embedding(xs)
        lens = torch.LongTensor(lens)
        bs = lens.shape[0]
        packed = pack_padded_sequence(
            embeds, lens, batch_first=True, enforce_sorted=False
        )
        output, hidden_ = self.rnn(packed, hx=hx)

        if self.seq_out and self.training:
            hidden = pad_packed_sequence(output)[
                0
            ]  # [L, bs, cell_size] // [L, bs, cell_size * 2] if biLSTM
            hidden = hidden.transpose(0, 1)  # [bs, L, cell size]
            # hidden_ = h_n, c_n if self.rnn = LSTM

        else:
            if isinstance(self.rnn, LSTM):
                hidden, cell = hidden_
                # hidden = torch.transpose(hidden, 0, 1)
            # hidden = hidden[:, -1, :]
            hidden = hidden.reshape(bs, -1)

        preds = self.pred_layer(hidden)
        preds = torch.squeeze(preds, dim=-1)

        return preds

    def forward(self, x, hx=None, out_h=False, **kwargs):
        out = self.logit(x, hx=hx, out_h=out_h)
        # print(f"{out = }")
        if not self.training:
            out = F.relu(out)
        return out

    def loss(self, x, y, reduction="mean", mask=None):
        if self.training and self.packed:
            mask = x[2].bool()
        preds = self.forward(x)

        res = msle_loss(y, preds, reduction=reduction)

        if mask is not None:
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

    def estimate(self, test_queries, is_tqdm=True):
        char_dict = self.char_dict
        test_data = [(x, 0) for x in test_queries]
        dl_test = get_LSTM_dl_test(test_data, 1, char_dict)
        model = self

        if is_tqdm:
            dl_test = tqdm(dl_test)

        estimates = get_estimates_from_model(model, dl_test)

        return estimates

    def model_size(self, *args):
        size = os.path.getsize(self.model_path)
        return size

    def build(
        self,
        train_data,
        valid_data,
        test_data,
        lr,
        l2,
        n_epoch,
        bs,
        patience,
        sw,
        qc_dict=None,
    ):
        model = self
        char_dict = self.char_dict
        packed = self.packed
        seed = self.seed
        model_path = self.model_path

        dl_train = get_LSTM_dl_train( train_data, bs, char_dict, packed, qc_dict)
        dl_valid = get_LSTM_dl_valid(valid_data, bs, char_dict)
        dl_test = get_LSTM_dl_test(test_data, bs, char_dict)

        # for batch in dl_train:
        #     print(batch)
        #     break

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

                xs, ys = batch2cuda(batch)
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
                    mask = xs[2].cpu().numpy()
                    q_err_ = ut.mean_Q_error(ys.cpu(), preds_, mask=mask)
                else:
                    q_err_ = ut.mean_Q_error(ys.cpu(), preds_)
                bs_step += 1
                sw.add_scalars(f"Loss", {"train": loss_}, global_step=bs_step)
                sw.add_scalars(f"ACC", {"train": q_err_}, global_step=bs_step)

            # valid score
            losses, q_errs, estimates = evaluate_DREAMmodel(model, dl_valid)
            loss_ = sum(losses) / len(losses)
            q_err_ = sum(q_errs) / len(q_errs)
            sw.add_scalars(f"Loss", {"valid": loss_}, global_step=bs_step)
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
            losses, q_errs, estimates = evaluate_DREAMmodel(model, dl_test)
            loss_ = sum(losses) / len(losses)
            q_err_ = sum(q_errs) / len(q_errs)
            sw.add_scalars(f"Loss", {"test": loss_}, global_step=bs_step)
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

    def load(self, device=torch.device("cuda")):
        model = self
        model_path = self.model_path
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, mask):
        key: torch.Tensor = key
        sqrt_dim = np.sqrt(key.shape[-1])
        score: torch.Tensor = torch.bmm(query, key.transpose(1, 2)) / sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float("Inf"))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class Positional_Enocding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        args:
            x: (batch_size, seq_len, embedding_dim)
        """
        x = x + self.pe[:, : x.size(1), :]
        return x


@torch.no_grad()
def init_weights(model):
    if type(model) == nn.Linear:
        torch.nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(0.01)


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention()
        self.query_proj = nn.Linear(input_dim, self.d_head * num_heads)
        self.key_proj = nn.Linear(input_dim, self.d_head * num_heads)
        self.value_proj = nn.Linear(input_dim, self.d_head * num_heads)

    def forward(self, query, key, value, mask=None):
        bs = value.shape[0]

        query: torch.Tensor = self.query_proj(query).view(
            bs, -1, self.num_heads, self.d_head
        )
        key: torch.Tensor = self.key_proj(key).view(bs, -1, self.num_heads, self.d_head)
        value: torch.Tensor = self.value_proj(value).view(
            bs, -1, self.num_heads, self.d_head
        )

        query = (
            query.permute(2, 0, 1, 3)
            .contiguous()
            .view(bs * self.num_heads, -1, self.d_head)
        )
        key = (
            key.permute(2, 0, 1, 3)
            .contiguous()
            .view(bs * self.num_heads, -1, self.d_head)
        )
        value = (
            value.permute(2, 0, 1, 3)
            .contiguous()
            .view(bs * self.num_heads, -1, self.d_head)
        )

        if mask is not None:
            mask: torch.Tensor = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context: torch.Tensor = context.view(self.num_heads, bs, -1, self.d_head)
        context = (
            context.permute(1, 2, 0, 3)
            .contiguous()
            .view(bs, -1, self.num_heads * self.d_head)
        )

        return context, attn


class Attention_module(nn.Module):
    # def __init__(self, n_char, emb_size, cell_size, pred_hs, n_rnn=1, n_pred_layer=1, **kwargs):
    def __init__(
        self, n_char, emb_size, cell_size, pred_hs, n_pred_layer=1, n_heads=8, **kwargs
    ):
        super().__init__()

        self.embedding = Embedding(n_char, emb_size, padding_idx=0)
        self.pos_embedding = Positional_Enocding(emb_size)

        nn.init.xavier_normal_(self.embedding.weight)

        self.attns: MultiHeadAttention = MultiHeadAttention(
            emb_size, cell_size, num_heads=n_heads
        )

        for name, param in self.attns.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            else:
                nn.init.xavier_uniform_(param)

        self.pred_layer = nn.Sequential()
        for id in range(n_pred_layer):
            id += 1
            if id == 1:
                self.pred_layer.add_module(f"PRED-{id}", nn.Linear(cell_size, pred_hs))
            else:
                self.pred_layer.add_module(f"PRED-{id}", nn.Linear(pred_hs, pred_hs))
            self.pred_layer.add_module(f"LeakyReLU-{id}", nn.LeakyReLU())
        self.pred_layer.add_module(f"PRED-OUT", nn.Linear(pred_hs, 8))
        self.pred_layer.add_module("ReLU", ReLU())
        self.pred_layer.apply(init_weights)

    def logit(self, x, hx=None, out_h=False, **kwargs):
        data, lengths = x
        embed = self.embedding(data)
        query = self.pos_embedding(embed)

        bs = data.shape[0]
        max_len = max(lengths)
        mask = torch.triu(
            torch.ones(bs, max_len, max_len, dtype=torch.bool), diagonal=1
        )
        mask = mask.to(data.device)

        context, attns = self.attns.forward(
            query=query, key=query, value=query, mask=mask
        )

        context = context[[range(bs), [l - 1 for l in lengths]]]

        output = self.pred_layer(context)
        output = torch.mean(output, dim=-1)
        return output

    def forward(self, x, prfx_train=False):
        out = self.logit(x, prfx_train=prfx_train)
        out = F.leaky_relu(out)
        out = torch.squeeze(out)
        return out

    def loss(self, x, y, reduction="mean"):
        preds = self.forward(x)
        res = msle_loss(y, preds, reduction=reduction)
        return res


class AugmentableModel(Module):
    def __init__(self, model, prev_model=None):
        super().__init__()
        self.prev_model = prev_model
        self.model = model

    def logit(self, x, out_h, **kwargs):
        xs, len = x
        hidden = None
        if self.prev_model:
            len1 = (
                ut.find_k_index_in_encoded_strings(xs, 2, 2) + 1
            )  # plus 1 to get length
            logit, hidden = self.prev_model.logit((xs, len1), out_h=True)
        logit = self.model.logit(x, hx=hidden)
        if out_h:
            return logit, hidden
        else:
            return logit

    def forward(self, x, out_h=False, **kwargs):
        return self.logit(x, out_h=out_h)

    def loss(self, x, y, reduction="mean"):
        preds = self.forward(x)
        res = msle_loss(y, preds, reduction=reduction)
        return res
