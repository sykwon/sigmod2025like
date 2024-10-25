import torch
from src.LPLM.model import LPLMestimator
import src.util as ut
import os
import yaml
import time
import socket
import pandas as pd
from collections import namedtuple
from src.LPLM.misc_utils import *
from src.LPLM.main import modelTrain
from src.LPLM.misc_utils import estimate_cardinality
from torch.utils.data import DataLoader

is_debug = False
MODE_EST = str(os.getenv("MODE")).lower() == "est"
MODE_SIZE = str(os.getenv("MODE")).lower() == "size"


if is_debug:
    input_config_path = "configs/LPLM/debug.yml"
else:
    input_config_path = "configs/LPLM/default.yml"

default_cfgs = yaml.safe_load(open(input_config_path))
abbr_dict = ut.get_common_abbr_dict()
abbr_dict["batch_size"] = "b"
abbr_dict["learning_rate"] = "lr"
abbr_dict["patience"] = "t"
choices_dict = ut.get_common_choices_dict()

parser = ut.get_parser(default_cfgs, abbr_dict, choices_dict)
args = parser.parse_args()
args.model_name = "LPLM"

hidden_dim_dict = {
    "WIKI": 780,
    "IMDB": 390,
    "DBLP": 392,
    "GENE": 410,
    "AUTHOR": 396,
    "DBLP-AN": 256,
    "IMDb-AN": 256,
    "IMDb-MT": 256,
    "TPCH-PN": 256,
    "link_type.link": 252,
    "movie_companies.note": 207,
    "name.name": 201,
    "title.title": 194,
}
args.hidden_dim = hidden_dim_dict[args.data_name]
args.tag = os.path.join(args.tag, args.model_name)

args = ut.add_just_params(args)

data_name = args.data_name
workload = args.workload
seed = args.seed
p_test = args.p_test
p_val = args.p_valid
p_train = args.p_train
# input_dim = args.input_dim
hidden_dim = args.hidden_dim
tag = args.tag
n_epoch = args.n_epoch
bs = args.batch_size
lr = args.lr
patience = args.patience

print(args)

(
    db_path,
    model_config_path,
    model_path,
    est_path,
    est_time_path,
    stat_path,
) = ut.get_common_pathes_for_estimator(data_name, tag, workload, seed)

sw = ut.get_summary_writer(data_name, tag, workload, seed)

with open(model_config_path, "w") as f:
    yaml.safe_dump(vars(args), f)


A_NLM_Configs = namedtuple('A_NLM_Configs',
                           ['vocabulary', 'hidden_size', 'learning_rate', 'batch_size', 'datasize',
                            'num_epochs', 'train_data_path', 'test_data_path',
                            'save_qerror_file_path', 'device', 'save_path'])

dataset_path = f"data/{data_name}/{data_name}.txt"
train_data_path = f'data/{data_name}/training/{workload}/train_LPLM.txt'
valid_data_path = f'data/{data_name}/training/{workload}/valid.txt'
vocab_file_path = f'data/{data_name}/vocab.txt'
test_data_path = f'data/{data_name}/training/{workload}/test.txt'

os.makedirs(os.path.dirname(model_path), exist_ok=True)
os.makedirs(os.path.dirname(est_path), exist_ok=True)

print("data  path:", os.path.abspath(dataset_path))
print("vocab path:", os.path.abspath(vocab_file_path))
print("train path:", os.path.abspath(train_data_path))
print("test  path:", os.path.abspath(test_data_path))
print("model path:", os.path.abspath(model_path))
print("est   path:", os.path.abspath(est_path))

dataset = [x.rstrip('\n') for x in open(dataset_path).readlines()]
if True:
    # if not os.path.exists(vocab_file_path):
    vocab = get_vocab_from_list(dataset, max_n_char=200)
    with open(vocab_file_path, "w") as f:
        f.write(vocab)
else:
    vocab = open(vocab_file_path).readline()
datasetsize = len(dataset)

card_estimator_configs = A_NLM_Configs(
    vocabulary=vocab,
    hidden_size=hidden_dim,
    datasize=datasetsize,
    learning_rate=lr,
    batch_size=bs,
    num_epochs=n_epoch,
    train_data_path=train_data_path,
    test_data_path=test_data_path,
    save_qerror_file_path=est_path,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_path=model_path
)


class Custom_dict(dict):
    def __init__(self):
        super().__init__()

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        else:
            return super().__getitem__('언')


if '언' in card_estimator_configs.vocabulary:
    char2idx = Custom_dict()
    for i, letter in enumerate(card_estimator_configs.vocabulary):
        char2idx[letter] = i
else:
    char2idx = {letter: i for i, letter in enumerate(
        card_estimator_configs.vocabulary)}

model = LPLMestimator(1, card_estimator_configs.hidden_size,
                      card_estimator_configs.device, card_estimator_configs.datasize, char2idx, model_path)
if not MODE_EST:
    torch.save(model.state_dict(), model_path)
model_size = model.model_size()
model_size_hr = ut.human_readable_size(model_size)
print(f"{model_size = }, {model_size_hr = }")

if MODE_EST:
    model.load_state_dict(torch.load(model_path))
    trained_model = model
    trained_model.to(card_estimator_configs.device)
    with open(stat_path) as f:
        build_time = yaml.load(f, Loader=yaml.FullLoader)['time']
    print(f"{build_time = }")
else:
    start_time = time.time()
    train_data = addpaddingTrain(
        card_estimator_configs.train_data_path, char2idx)
    # Assuming you have a similar function for valid data
    valid_data, valid_queries, valid_cards = addpaddingTest(
        valid_data_path, char2idx)
    dataloadertrain = DataLoader(
        train_data, batch_size=card_estimator_configs.batch_size, shuffle=True)
    dataloadervalid = DataLoader(
        valid_data, batch_size=card_estimator_configs.batch_size)
    trained_model = modelTrain(dataloadertrain, model, card_estimator_configs.device,
                               card_estimator_configs.learning_rate, card_estimator_configs.num_epochs,
                               card_estimator_configs.save_path,
                               dataloadervalid,
                               model_path,
                               patience,
                               card_estimator_configs.datasize)
    end_time = time.time()
    build_time = end_time - start_time

    torch.save(trained_model.state_dict(), model_path)

datasettest, test_queries, test_cards = addpaddingTest(
    card_estimator_configs.test_data_path, char2idx)  # Assuming you have a similar function for test data
dataloadertest = DataLoader(datasettest, batch_size=1)
cardinalities_, test_estimations = estimate_cardinality(
    dataloadertest, trained_model, card_estimator_configs.device, card_estimator_configs.save_qerror_file_path, card_estimator_configs.datasize)

q_errs = ut.mean_Q_error(test_cards, test_estimations, reduction='none')
print(test_cards[:5])
print(test_estimations[:5])

model_size = model.model_size()
print(f"{model_size = }")

# test_queries, test_cards = list(zip(*test_data))

# evaluate best model
model.eval()
model.cpu()
model.device = torch.device("cpu")
start_time = time.time()
with torch.no_grad():
    test_estimations, test_estimation_times = model.estimate_latency_analysis(
        test_queries
    )
end_time = time.time()

est_time = end_time - start_time

print(f"{est_time = }")

print(f"{test_estimations[:5] = }")
print(f"{test_cards[:5] = }")

q_errs = ut.mean_Q_error(test_cards, test_estimations, reduction="none")

ut.save_estimated_cards(test_queries, test_cards, test_estimations, est_path)
ut.save_estimation_times(
    test_queries, test_cards, test_estimations, test_estimation_times, est_time_path
)

# save stat (avg, q001, ..., q100, time) # train_time, test_time
ut.save_estimator_stat(stat_path, q_errs, build_time, est_time, model_size)

# #
# train_queries, train_cards = list(zip(*train_data))

# # evaluate best model
# model.eval()
# model.cpu()
# train_estimations, train_estimation_times = model.estimate_latency_analysis(
#     train_queries
# )

# assert "estimation_time" in est_time_path
# train_est_time_path = est_time_path.replace("estimation_time", "train_estimation_time")
# print(f"{train_est_time_path}")
# ut.save_estimation_times(
#     train_queries,
#     train_cards,
#     train_estimations,
#     train_estimation_times,
#     train_est_time_path,
# )
