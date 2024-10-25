from tensorboardX import SummaryWriter
import time
import datetime
import numpy as np
from tqdm import tqdm
from abc import ABCMeta, abstractmethod


class Estimator(metaclass=ABCMeta):
    def __init__(self):
        pass

    # @abstractmethod
    # def build(self, *args):
    #     pass

    @abstractmethod
    def model_size(self, *args):
        pass

    @abstractmethod
    def estimate(self, test_data, is_tqdm=True):
        pass

    def estimate_latency_analysis(self, test_data, is_tqdm=True, **kwargs):
        y_pred = []
        # y_data = []
        latencies = []
        test_data_itr = test_data
        if is_tqdm:
            test_data_itr = tqdm(test_data)

        for test_qry in test_data_itr:
            start = time.time()
            # y_pred_s, y_data_s = self.estimate([test_qry])
            y_pred_s = self.estimate([test_qry], is_tqdm=False, **kwargs)
            end = time.time()
            latency = end - start
            y_pred.append(y_pred_s)
            # y_data.append(y_data_s)
            latencies.append([latency] * len(y_pred_s))
        y_pred = np.concatenate(y_pred)
        latencies = np.concatenate(latencies)
        # return y_pred, y_data, latencies
        return y_pred, latencies
