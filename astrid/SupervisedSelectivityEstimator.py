import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import astrid.misc_utils as misc_utils
from tensorboardX import SummaryWriter
import src.util as ut
import os
from astrid.misc_utils import SelectivityEstimatorConfigs


# We choose a simple DL model.
# More complex models could give better results
class SelectivityEstimator(nn.Module):
    def __init__(self, string_helper, configs: SelectivityEstimatorConfigs, est_scale):
        super(SelectivityEstimator, self).__init__()
        self.embedding_dimension = configs.embedding_dimension
        self.string_helper = string_helper
        self.max_string_length = self.string_helper.max_string_length
        self.alphabet_size = self.string_helper.alphabet_size
        self.est_scale = est_scale
        if configs.agg == "attn":
            self.attn = torch.nn.MultiheadAttention()

        layer_sizes = [128, 64, 32, 16, 8]
        layer_sizes = [x * est_scale for x in layer_sizes]

        self.model = nn.Sequential(
            nn.Linear(self.embedding_dimension, layer_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_sizes[0]),
            nn.Dropout(0.001),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_sizes[1]),
            nn.Dropout(0.01),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_sizes[2]),
            nn.Dropout(0.01),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_sizes[3]),
            nn.Dropout(0.01),
            nn.Linear(layer_sizes[3], layer_sizes[4]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_sizes[4]),
            nn.Dropout(0.01),

            nn.Linear(layer_sizes[4], 1),
        )

    def forward(self, x, agg=None):
        if agg == "pool":
            string_queries, n_pats = x
            x = torch.sum(string_queries, dim=1) / n_pats.view(-1, 1)
        elif agg == "attn":
            x = self.attn(x)
        x = torch.sigmoid(self.model(x))
        return x


class SelectivityEstimatorAstridEach(nn.Module):
    def __init__(self, string_helper, configs: SelectivityEstimatorConfigs):
        super(SelectivityEstimatorAstridEach, self).__init__()
        self.embedding_dimension = configs.embedding_dimension
        self.string_helper = string_helper
        self.max_string_length = self.string_helper.max_string_length
        self.alphabet_size = self.string_helper.alphabet_size

        layer_sizes = [256, 128, 64, 32, 16]
        self.model = nn.Sequential(
            nn.Linear(self.embedding_dimension, layer_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_sizes[0]),
            nn.Dropout(0.001),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_sizes[1]),
            nn.Dropout(0.01),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_sizes[2]),
            nn.Dropout(0.01),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_sizes[3]),
            nn.Dropout(0.01),
            nn.Linear(layer_sizes[3], layer_sizes[4]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_sizes[4]),
            nn.Dropout(0.01),

            nn.Linear(layer_sizes[4], 1),
        )

    def forward(self, x, agg=None):
        x = torch.sigmoid(self.model(x))
        return x


def eval_selEst_model(configs, model, data_loader):
    model.eval()
    device = configs.device
    losses = []
    q_errs = []
    for batch_idx, (string_queries, n_pats, true_selectivities) in enumerate(data_loader):
        string_queries = string_queries.to(device)
        n_pats = n_pats.to(device)

        predicted_selectivities = model(
            (string_queries, n_pats), agg=configs.agg)
        loss = misc_utils.qerror_loss(predicted_selectivities, true_selectivities.float(),
                                      configs.min_val, configs.max_val, reduction=False)

        loss_ = loss.detach().cpu().numpy()
        losses.extend(loss_)

        preds_ = misc_utils.unnormalize_torch(
            predicted_selectivities.view(-1).detach().cpu(), configs.min_val, configs.max_val).numpy()
        cards_ = misc_utils.unnormalize_torch(
            true_selectivities.cpu(), configs.min_val, configs.max_val).numpy()

        q_err_ = ut.mean_Q_error(cards_, preds_, reduction='none')
        q_errs.extend(q_err_)

    loss_ = sum(losses) / len(losses)
    q_err_ = sum(q_errs) / len(q_errs)

    return loss_, q_err_


def eval_selEst_model_AstridEach(configs, model, data_loader):
    model.eval()
    device = configs.device
    losses = []
    q_errs = []
    for batch_idx, (string_queries, true_selectivities) in enumerate(data_loader):
        string_queries = string_queries.to(device)

        predicted_selectivities = model(string_queries)
        loss = misc_utils.qerror_loss(predicted_selectivities, true_selectivities.float(),
                                      configs.min_val, configs.max_val, reduction=False)

        loss_ = loss.detach().cpu().numpy()
        losses.extend(loss_)

        preds_ = misc_utils.unnormalize_torch(
            predicted_selectivities.view(-1).detach().cpu(), configs.min_val, configs.max_val).numpy()
        cards_ = misc_utils.unnormalize_torch(
            true_selectivities.cpu(), configs.min_val, configs.max_val).numpy()

        q_err_ = ut.mean_Q_error(cards_, preds_, reduction='none')
        q_errs.extend(q_err_)

    loss_ = sum(losses) / len(losses)
    q_err_ = sum(q_errs) / len(q_errs)

    return loss_, q_err_


def train_selEst_model(configs: SelectivityEstimatorConfigs, train_loader, valid_loader, test_loader, string_helper, est_scale, patience, sw=None):
    model = SelectivityEstimator(string_helper, configs, est_scale)
    # Comment this line during experimentation.
    # model = torch.jit.script(model)
    model = model.to(configs.device)
    model_path = configs.selectivity_model_file_name

    optimizer = optim.Adam(model.parameters(), lr=configs.lr)
    device = configs.device

    model.train()

    bs_step = 0
    best_val_score = None
    best_val_epoch = None
    test_score_at_best_val = None

    epochs_since_improvement = 0

    for epoch in tqdm(range(1, configs.num_epochs + 1), desc="Epochs", position=0, leave=False):
        running_loss = []
        for batch_idx, (string_queries, n_pats, true_selectivities) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            # for batch_idx, output in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            model.train()
            optimizer.zero_grad()

            string_queries = string_queries.to(device)
            n_pats = n_pats.to(device)

            predicted_selectivities = model(
                (string_queries, n_pats), agg=configs.agg)
            loss = misc_utils.qerror_loss(predicted_selectivities, true_selectivities.float(),
                                          configs.min_val, configs.max_val)

            loss_ = float(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())

            preds_ = misc_utils.unnormalize_torch(
                predicted_selectivities.view(-1).detach().cpu(), configs.min_val, configs.max_val).numpy()
            cards_ = misc_utils.unnormalize_torch(
                true_selectivities.cpu(), configs.min_val, configs.max_val).numpy()

            q_err_ = ut.mean_Q_error(cards_, preds_)

            bs_step += 1
            sw.add_scalars(f"Loss", {'train': loss_}, global_step=bs_step)
            sw.add_scalars(f"ACC", {'train': q_err_}, global_step=bs_step)

        valid_loss, valid_q_err = eval_selEst_model(
            configs, model, valid_loader)

        sw.add_scalars(f"Loss", {'valid': valid_loss}, global_step=bs_step)
        sw.add_scalars(f"ACC", {'valid': valid_q_err}, global_step=bs_step)

        test_loss, test_q_err = eval_selEst_model(configs, model, test_loader)

        if best_val_score is None or valid_q_err < best_val_score:
            best_val_score = valid_q_err
            test_score_at_best_val = test_q_err
            best_val_epoch = epoch
            torch.save(model.state_dict(), model_path)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        sw.add_scalars(f"Loss", {'test': test_loss}, global_step=bs_step)
        sw.add_scalars(f"ACC", {'test': test_q_err}, global_step=bs_step)
        print(f"{epoch = :3d} valid {valid_loss = :.3f} {valid_q_err= :.3f} {test_loss = :.3f} {test_q_err = :.3f}")

        if epochs_since_improvement >= patience:
            print(f"Early stopping triggered.")
            break

        # print("Epoch: {}/{} - Mean Running Loss: {:.4f}".format(epoch+1, configs.num_epochs, np.mean(running_loss)))
        # print("Summary stats of Loss: Percentile: [0.75, 0.9, 0.95, 0.99] ", [
        #       np.quantile(running_loss, q) for q in [0.75, 0.9, 0.95, 0.99]])
    model.load_state_dict(torch.load(model_path))
    print(f"{best_val_score = }, {patience = } {best_val_epoch = }, {test_score_at_best_val = }")

    return model


def train_selEst_model_AstridEach(configs: SelectivityEstimatorConfigs, train_loader, valid_loader, test_loader, string_helper, patience, sw=None, is_train=True):
    model = SelectivityEstimatorAstridEach(string_helper, configs)
    # Comment this line during experimentation.
    # model = torch.jit.script(model)
    model = model.to(configs.device)
    model_path = configs.selectivity_model_file_name
    print(f"{model_path = }")
    if not is_train:
        print(f"try to load trained estimation model")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            return model
    MODE_SIZE = configs.num_epochs == 0
    if MODE_SIZE:
        torch.save(model.state_dict(), model_path)
        return model

    optimizer = optim.Adam(model.parameters(), lr=configs.lr)
    device = configs.device

    model.train()

    bs_step = 0
    best_val_score = None
    best_val_epoch = None
    test_score_at_best_val = None

    epochs_since_improvement = 0

    for epoch in tqdm(range(1, configs.num_epochs + 1), desc="Epochs", position=0, leave=False):
        running_loss = []
        for batch_idx, (string_queries, true_selectivities) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            # for batch_idx, output in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            model.train()
            optimizer.zero_grad()

            string_queries = string_queries.to(device)

            predicted_selectivities = model(string_queries)
            loss = misc_utils.qerror_loss(predicted_selectivities, true_selectivities.float(),
                                          configs.min_val, configs.max_val)

            loss_ = float(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())

            preds_ = misc_utils.unnormalize_torch(
                predicted_selectivities.view(-1).detach().cpu(), configs.min_val, configs.max_val).numpy()
            cards_ = misc_utils.unnormalize_torch(
                true_selectivities.cpu(), configs.min_val, configs.max_val).numpy()

            q_err_ = ut.mean_Q_error(cards_, preds_)

            bs_step += 1
            sw.add_scalars(f"Loss", {'train': loss_}, global_step=bs_step)
            sw.add_scalars(f"ACC", {'train': q_err_}, global_step=bs_step)

        valid_loss, valid_q_err = eval_selEst_model_AstridEach(
            configs, model, valid_loader)

        sw.add_scalars(f"Loss", {'valid': valid_loss}, global_step=bs_step)
        sw.add_scalars(f"ACC", {'valid': valid_q_err}, global_step=bs_step)

        test_loss, test_q_err = eval_selEst_model_AstridEach(
            configs, model, test_loader)

        if best_val_score is None or valid_q_err < best_val_score:
            best_val_score = valid_q_err
            test_score_at_best_val = test_q_err
            best_val_epoch = epoch
            torch.save(model.state_dict(), model_path)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        sw.add_scalars(f"Loss", {'test': test_loss}, global_step=bs_step)
        sw.add_scalars(f"ACC", {'test': test_q_err}, global_step=bs_step)
        print(f"{epoch = :3d} valid {valid_loss = :.3f} {valid_q_err= :.3f} {test_loss = :.3f} {test_q_err = :.3f}")

        if epochs_since_improvement >= patience:
            print(f"Early stopping triggered.")
            break

        # print("Epoch: {}/{} - Mean Running Loss: {:.4f}".format(epoch+1, configs.num_epochs, np.mean(running_loss)))
        # print("Summary stats of Loss: Percentile: [0.75, 0.9, 0.95, 0.99] ", [
        #       np.quantile(running_loss, q) for q in [0.75, 0.9, 0.95, 0.99]])
    model.load_state_dict(torch.load(model_path))
    print(f"{best_val_score = }, {patience = } {best_val_epoch = }, {test_score_at_best_val = }")

    return model
