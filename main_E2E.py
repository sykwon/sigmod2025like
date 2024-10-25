import torch
from src.E2E.model import E2Eestimator
import src.util as ut
import os
import yaml
import time
import socket

hostname = socket.gethostname()
is_debug = False

if is_debug:
    input_config_path = "configs/E2E/debug.yml"
else:
    input_config_path = "configs/E2E/default.yml"

default_cfgs = yaml.safe_load(open(input_config_path))
abbr_dict = ut.get_common_abbr_dict()
abbr_dict["min_count"] = "mc"
abbr_dict["batch_size"] = "b"
abbr_dict["learning_rate"] = "lr"
abbr_dict["mask_prob"] = "m"
abbr_dict["patience"] = "t"
choices_dict = ut.get_common_choices_dict()

parser = ut.get_parser(default_cfgs, abbr_dict, choices_dict)
args = parser.parse_args()
args.model_name = "E2E"
if args.packed:
    args.model_name += "-AUG"
args.tag = os.path.join(args.tag, args.model_name)

data_name = args.data_name
workload = args.workload
seed = args.seed
p_test = args.p_test
p_val = args.p_valid
p_train = args.p_train
input_dim = args.input_dim
hidden_dim = args.hidden_dim
hid_dim = args.hid_dim
tag = args.tag
min_count = args.min_count
n_sample = args.n_sample
n_epoch_wv = args.n_epoch_wv
n_epoch = args.n_epoch
wv_size = args.wv_size
bs = args.batch_size
lr = args.lr
patience = args.patience
packed = args.packed

print(args)

(
    db_path,
    model_config_path,
    model_path,
    est_path,
    est_time_path,
    stat_path,
) = ut.get_common_pathes_for_estimator(data_name, tag, workload, seed)
wv_path = ut.get_word_vector_path_for_E2Eestimator(data_name, min_count, tag)

sw = ut.get_summary_writer(data_name, tag, workload, seed)

with open(model_config_path, "w") as f:
    yaml.safe_dump(vars(args), f)

char_dict, n_special_char, train_data, valid_data, test_data = ut.get_training_data(
    data_name, workload, p_train)

print(train_data[:5])
print(valid_data[:5])
print(test_data[:5])

qc_dict = None
if packed:
    qc_dict = ut.get_qc_dict(data_name, workload)
    train_data_old = train_data
    train_data = ut.add_prefix_augmented(train_data, qc_dict)

model = E2Eestimator(
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
)
print(f"{model = }")

build_time = model.build(
    train_data, valid_data, test_data, lr, n_epoch, n_epoch_wv, bs, patience, sw
)

print(f"{build_time = }")
print(f"{model = }")

model_size = model.model_size()
print(f"{model_size = }")

test_queries, test_cards = list(zip(*test_data))

# evaluate best model
model.eval()
model.cpu()
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

#
train_queries, train_cards = list(zip(*train_data))

# evaluate best model
model.eval()
model.cpu()
train_estimations, train_estimation_times = model.estimate_latency_analysis(
    train_queries
)
