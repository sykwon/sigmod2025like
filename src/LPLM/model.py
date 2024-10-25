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
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from src.util import compile_LIKE_query, eval_compiled_LIKE_query
from src.LPLM.misc_utils import *


import os

class LPLMestimator(Module, Estimator):
    def __init__(self, num_layers, hidden_size, device, datasetsize, char2idx, model_path):
        super(LPLMestimator, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.alphabet_size = len(char2idx)
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
        self.datasetsize = datasetsize
        self.char2idx = char2idx
        self.model_path = model_path
        

    def forward_selectivity(self, x):
        output, hidden_state = self.gru(x)
        output = torch.sigmoid(self.fc(output))
        return torch.squeeze(output)

    def forward(self, x):
        output = self.forward_selectivity(x)
        return output

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)

    def estimate(self, test_data, is_tqdm=True):
        datasettest = convertQueries2testData(test_data, self.char2idx)
        # datasettest, test_queries = addpaddingTestQuery(test_data, self.char2idx)
        # datasettest, test_queries, test_cards = addpaddingTest(card_estimator_configs.test_data_path, char2idx)  # Assuming you have a similar function for test data
        test_dataset = DataLoader(datasettest, batch_size=1)

        # estimate_cardinality(test_dataset, model, device, dataset_size):

        estimations = []
        device = self.device
        model = self
        dataset_size = self.datasetsize
        
        for name, mask in test_dataset:
            name, mask = name.to(device), mask.to(device)
            output = model(name)
            output = torch.prod(torch.pow(output, mask)) * dataset_size
            estimations.append(output.item())
        
        return estimations

    def model_size(self, *args):
        size_model = os.path.getsize(self.model_path)
        print(f"{size_model = }")
        return size_model