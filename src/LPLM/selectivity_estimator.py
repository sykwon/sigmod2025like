from collections import namedtuple

import time
import numpy
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import src.LPLM.misc_utils as misc_utils
import src.util as ut


class Cardinality_Estimator(nn.Module):
    """
    Parameters:
        num_layers (int): Number of GRU layers in the model.
        hidden_size (int): Number of features in the hidden state of the GRU.
        device (torch.device): Device (CPU/GPU) on which the model will be loaded and computations will be performed.
        alphabet_size (int): Size of the input alphabet (number of different symbols).

    Attributes:
        num_layers (int): Number of GRU layers in the model.
        hidden_size (int): Number of features in the hidden state of the GRU.
        device (torch.device): Device (CPU/GPU) on which the model will be loaded and computations will be performed.
        alphabet_size (int): Size of the input alphabet (number of different symbols).
        layer_sizes (list): List of integers representing the sizes of fully connected layers in the model.
        gru (nn.GRU): GRU layer of the model.
        fc (nn.Sequential): Fully connected neural network with ReLU activation.
    """

    def __init__(self, num_layers, hidden_size, device, alphabet_size):
        super(Cardinality_Estimator, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.alphabet_size = alphabet_size
        layer_sizes = [128, 64, 32, 16, 8]
        self.gru = nn.GRU(
            batch_first=True,
            input_size=self.alphabet_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
            nn.ReLU(),
            nn.Linear(layer_sizes[3], layer_sizes[4]),
            nn.ReLU(),
            nn.Linear(layer_sizes[4], 1),
        )

    def forward_selectivity(self, x):
        output, hidden_state = self.gru(x)
        output = torch.sigmoid(self.fc(output))
        return torch.squeeze(output)

    def forward(self, x):
        output = self.forward_selectivity(x)
        return output

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)


def train_model(train_data, model, device, learning_rate, num_epocs, valid_data=None, model_path=None, patience=None, datasize=None):
    best_val_score = float("inf")
    best_epoch = -1
    best_test_score = -1
    build_time = 0

    epochs_since_improvement = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    model.train()
    for epoch in range(num_epocs):
        start_time = time.time()
        loss_list = []
        for i, (name, mask, target) in tqdm(enumerate(train_data)):
            name = name.to(device)
            output = model(name)
            output = output.to(device)
            target = target.to(device)
            mask = mask.to(device)
            loss = misc_utils.binary_crossentropy(output, target, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss)

        valid_cards, valid_estimations = misc_utils.estimate_cardinality(
            valid_data, model, device, None, datasize)
        qerror_list = ut.mean_Q_error(
            valid_cards, valid_estimations, reduction="none")
        valid_q_err = np.average(qerror_list)
        print(f'G-mean: {np.round(misc_utils.g_mean(qerror_list), 4)}')
        print(f'Mean: {np.round(np.average(qerror_list), 4)}')
        print(f'Median: {np.round(np.percentile(qerror_list, 50), 4)}')
        print(f'90th: {np.round(np.percentile(qerror_list, 90), 4)}')
        print(f'99th: {np.round(np.percentile(qerror_list, 99), 4)}')

        # evaluate
        # valid_loss_list = []
        # model.eval()
        # for i, (name, mask, target) in enumerate(valid_data):
        #     name = name.to(device)
        #     output = model(name)
        #     output = output.to(device)
        #     target = target.to(device)
        #     mask = mask.to(device)
        #     loss = misc_utils.binary_crossentropy(output, target, mask)
        #     # loss.backward()
        #     # optimizer.step()
        #     # optimizer.zero_grad()
        #     valid_loss_list.append(loss)
        # model.train()
        # valid_loss = np.mean(numpy.array( torch.tensor(valid_loss_list, device = 'cpu')))

        end_time = time.time()
        build_time += end_time - start_time

        if valid_q_err < best_val_score:
            best_val_score = valid_q_err
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        print("Epoch: {}/{} - Mean Running Loss: {:.4f} - Valid q-error: {:.4f}".format(epoch+1,
              num_epocs, np.mean(numpy.array(torch.tensor(loss_list, device='cpu'))), valid_q_err))

        # if epochs_since_improvement == 0:
        #     best_test_score = valid_loss

        if epochs_since_improvement >= patience:
            print(f"Early stopping triggered.")
            break

    print(f"{best_epoch = }, {best_val_score}")
    model.load_state_dict(torch.load(model_path))

    return model
