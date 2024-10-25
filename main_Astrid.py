# from astrid.Astrid import
from astrid.Astrid import AstridEstimator
from astrid.AstridEmbed import (
    setup_configs,
    train_astrid_embedding_model,
    train_selectivity_estimator,
    get_selectivity_for_strings,
    data2Astrid_df,
)
from astrid.misc_utils import initialize_random_seeds, setup_vocabulary
from astrid.prepare_datasets import prepare_dataset
from astrid.string_dataset_helpers import StringSelectivityDataset
import os
from src.util import get_training_data
from torch.utils.data import DataLoader
import src.util as ut
import pandas as pd
import numpy as np
import os
import time
import yaml
import torch
from tensorboardX import SummaryWriter
from datetime import datetime
from astrid.summary_data_structures import MAX_STR_SIZE

is_debug = False
if is_debug:
    input_config_path = "configs/Astrid/debug.yml"
else:
    input_config_path = "configs/Astrid/default.yml"

default_cfgs = yaml.safe_load(open(input_config_path))
abbr_dict = ut.get_common_abbr_dict()
abbr_dict["batch_size"] = "b"
abbr_dict["learning_rate"] = "lr"
choices_dict = ut.get_common_choices_dict()

parser = ut.get_parser(default_cfgs, abbr_dict, choices_dict)
args = parser.parse_args()

args.model_name = "Astrid"
if args.packed:
    args.model_name += "_AUG"

args.tag = os.path.join(args.tag, args.model_name)

# max_n_pat = args.max_n_pat

p_test = args.p_test
p_val = args.p_val
p_train = args.p_train
max_n_pat = args.max_n_pat
emb_bs = args.emb_batch_size
emb_dim = args.emb_dim
emb_epoch = args.emb_epoch
emb_lr = args.emb_lr

est_epoch = args.est_epoch
agg = args.aggregation
lr = args.learning_rate
bs = args.batch_size

data_name = args.data_name
workload = args.workload
seed = args.seed
patience = args.patience
packed = args.packed
max_str_size = args.max_str_size
est_scale = args.est_scale
tag = args.tag

print(args)

(
    db_path,
    model_config_path,
    model_path,
    est_path,
    est_time_path,
    stat_path,
) = ut.get_common_pathes_for_estimator(data_name, tag, workload, seed)
triplet_dirpath = ut.get_triplet_dirpath_for_Astrid(data_name)
embedding_model_dirpath = ut.get_embedding_model_dirpath_for_Astrid(data_name)

sw = ut.get_summary_writer(data_name, tag, workload, seed)

with open(model_config_path, "w") as f:
    yaml.safe_dump(vars(args), f)

db = ut.read_strings(db_path)
char_dict, n_special_char, train_data, valid_data, test_data = ut.get_training_data(
    data_name, workload, p_train
)
min_val = np.log(1)
max_val = np.log(len(db))

print(f"{min_val = }, {max_val = }")

qc_dict = None
if packed:
    qc_dict = ut.get_qc_dict(data_name, workload)
    train_data_old = train_data
    train_data = ut.add_prefix_augmented(train_data, qc_dict)

model = AstridEstimator(
    data_name,
    emb_dim,
    emb_bs,
    emb_lr,
    emb_epoch,
    db_path,
    triplet_dirpath,
    embedding_model_dirpath,
    model_path,
    bs,
    lr,
    patience,
    est_epoch,
    agg,
    tag,
    seed,
    min_val,
    max_val,
    max_n_pat,
    max_str_size,
    est_scale,
)
print(f"{model = }")

build_time = model.build(train_data, valid_data, test_data, sw, seed)

print(f"{build_time = }")
print(f"{model = }")

model_size = model.model_size()
print(f"{model_size = }")

test_queries, test_cards = list(zip(*test_data))

model.selectivity_model.eval()
model.selectivity_model.cpu()
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
