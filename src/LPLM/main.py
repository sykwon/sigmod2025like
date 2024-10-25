import time
from collections import namedtuple

from pympler import asizeof
from torch.utils.data import DataLoader
import torch
from src.LPLM import misc_utils
from src.LPLM import selectivity_estimator
from src.LPLM.misc_utils import estimate_cardinality
import numpy as np


# This function load the saved model
def load_estimation_model(model_file_name, model_card, device):
    model_card.load_state_dict(torch.load(model_file_name))
    return model.to(device)


# This function trains and returns the embedding model
def modelTrain(train_data, model, device, learning_rate, num_epocs, model_save_path, valid_data=None, model_path=None, patience=None, datasize=None):
    model = selectivity_estimator.train_model(
        train_data, model, device, learning_rate, num_epocs, valid_data, model_path, patience, datasize)
    torch.save(model.state_dict(), model_save_path)
    return model


def get_vocab(trainfile, max_n_char=None):

    vocab_dict = {}
    for i in open(trainfile):
        i = i.strip().split(':')[0]
        for k in i:
            if k != '%' and k not in vocab_dict:
                vocab_dict[k] = 0
    vocab = ''
    for token in vocab_dict:
        vocab += token
    return vocab + '$.#'


if __name__ == "__main__":
    vocabulary = get_vocab('train.txt')
    trainpath = 'train.txt'
    testpath = 'test.txt'
    savepath = 'estimated_cardinalities.txt'
    savemodel = 'model.pth'
    A_NLM_configs = namedtuple('A_NLM_configs', ['vocabulary', 'hidden_size', 'learning_rate', 'batch_size', 'datasize',
                                                 'num_epocs', 'train_data_path', 'test_data_path',
                                                 'save_qerror_file_path', 'device', 'save_path'])

    card_estimator_configs = A_NLM_configs(vocabulary=vocabulary, hidden_size=256,
                                           datasize=450000, learning_rate=0.0001, batch_size=128, num_epocs=64,
                                           train_data_path=trainpath,
                                           test_data_path=testpath,
                                           save_qerror_file_path=savepath,
                                           device=torch.device(
                                               "cuda" if torch.cuda.is_available() else "cpu"),
                                           save_path=savemodel)
    char2idx = {letter: i for i, letter in enumerate(
        card_estimator_configs.vocabulary)}

    model = selectivity_estimator.Cardinality_Estimator(1, card_estimator_configs.hidden_size, card_estimator_configs.device,
                                                        len(char2idx))
    train_data = misc_utils.addpaddingTrain(
        card_estimator_configs.train_data_path, char2idx)
    dataloadertrain = DataLoader(
        train_data, batch_size=card_estimator_configs.batch_size, shuffle=True)
    trained_model = modelTrain(dataloadertrain, model, card_estimator_configs.device,
                               card_estimator_configs.learning_rate, card_estimator_configs.num_epocs,
                               card_estimator_configs.save_path, card_estimator_configs.datasize)
    datasettest = misc_utils.addpaddingTest(
        card_estimator_configs.test_data_path, char2idx)
    dataloadertest = DataLoader(datasettest, batch_size=1)

    estimate_cardinality(dataloadertest, trained_model, card_estimator_configs.device,
                         card_estimator_configs.save_qerror_file_path, card_estimator_configs.datasize)
