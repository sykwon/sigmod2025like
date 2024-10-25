# from astrid.Astrid import
from astrid.AstridEach import AstridEachEstimator
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
MODE_SIZE = str(os.getenv("MODE")).lower() == "size"

if is_debug:
    input_config_path = "configs/AstridEach/debug.yml"
else:
    input_config_path = "configs/AstridEach/default.yml"

default_cfgs = yaml.safe_load(open(input_config_path))
abbr_dict = ut.get_common_abbr_dict()
abbr_dict["batch_size"] = "b"
abbr_dict["learning_rate"] = "lr"
choices_dict = ut.get_common_choices_dict()

parser = ut.get_parser(default_cfgs, abbr_dict, choices_dict)
args = parser.parse_args()
if MODE_SIZE:
    args.emb_epoch = 0
    args.est_epoch = 0

args.model_name = "AstridEach"
if args.packed:
    args.model_name += "_AUG"

args.tag = os.path.join(args.tag, args.model_name)

p_train = args.p_train
emb_bs = args.emb_batch_size
emb_dim = args.emb_dim
emb_epoch = args.emb_epoch
emb_lr = args.emb_lr

est_epoch = args.est_epoch
lr = args.learning_rate
bs = args.batch_size

data_name = args.data_name
workload = args.workload
seed = args.seed
patience = args.patience
packed = args.packed
max_str_size = args.max_str_size
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
triplet_dirpath = ut.get_triplet_dirpath_for_Astrid(data_name, max_str_size)
embedding_model_dirpath = ut.get_embedding_model_dirpath_for_Astrid(
    data_name, emb_dim)

if MODE_SIZE:
    embedding_model_dirpath = embedding_model_dirpath.replace('res', 'res_mem')
    model_path = model_path.replace('res', 'res_mem')
    print(f"Path changed: {embedding_model_dirpath = } {model_path = }")

sw = ut.get_summary_writer(data_name, tag, workload, seed)

with open(model_config_path, "w") as f:
    yaml.safe_dump(vars(args), f)

db = ut.read_strings(db_path)
if MODE_SIZE:
    pass
else:
    char_dict, n_special_char, train_data, valid_data, test_data = ut.get_training_data(
        data_name, workload, p_train
    )
    train_data_triple, valid_data_triple, test_data_triple = ut.get_training_data_AstridEach(
        data_name, p_train)

min_val = np.log(1)
max_val = np.log(len(db))

print(f"{min_val = }, {max_val = }")

qc_dict = None
if packed:
    qc_dict = ut.get_qc_dict(data_name, workload)
    train_data_old = train_data
    train_data = ut.add_prefix_augmented(train_data, qc_dict)

model = AstridEachEstimator(
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
    tag,
    seed,
    min_val,
    max_val,
    max_str_size,
)
print(f"{model = }")

if MODE_SIZE:
    build_time = model.build(
        None, None, None, sw, seed)
    model_size = model.model_size()
    hr_model_size = ut.human_readable_size(model_size)
    print(f"{data_name = } {workload = } {hr_model_size = } {model_size = }")
    exit()

print(f"{model_path = }")
build_time = model.build(
    train_data_triple, valid_data_triple, test_data_triple, sw, seed)

print(f"{build_time = }")
print(f"{model = }")

model_size = model.model_size()
print(f"{model_size = }")


############### test all ###############
test_queries, test_cards = list(zip(*test_data))

for fn_desc, selectivity_model in model.selectivity_model_dict.items():
    selectivity_model.eval()
    selectivity_model.cpu()
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


############### test each ###############
fn_desc_list = model.fn_desc_list
for idx, fn_desc in enumerate(fn_desc_list):
    model.selectivity_model_dict[fn_desc].eval()
    model.selectivity_model_dict[fn_desc].cpu()
    model_size = model.model_size(fn_desc)
    test_data = test_data_triple[idx]
    test_queries, test_cards = list(zip(*test_data))
    est_path_each = est_path.replace("AstridEach", f"AstridEach/{fn_desc}")
    est_time_path_each = est_time_path.replace(
        "AstridEach", f"AstridEach/{fn_desc}")
    stat_path_each = stat_path.replace("AstridEach", f"AstridEach/{fn_desc}")

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

    ut.save_estimated_cards(test_queries, test_cards,
                            test_estimations, est_path_each)
    ut.save_estimation_times(
        test_queries, test_cards, test_estimations, test_estimation_times, est_time_path_each
    )

    # save stat (avg, q001, ..., q100, time) # train_time, test_time
    ut.save_estimator_stat(stat_path_each, q_errs,
                           build_time, est_time, model_size)
