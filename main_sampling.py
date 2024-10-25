import re
import src.util as ut
import pandas as pd
import numpy as np
import os
import yaml
import socket
import time
import src.sampling as sp

is_debug = False

if is_debug:
    input_config_path = "configs/sampling/debug.yml"
else:
    input_config_path = "configs/sampling/default.yml"

default_cfgs = yaml.safe_load(open(input_config_path))
abbr_dict = ut.get_common_abbr_dict()
abbr_dict['m'] = 'm'
abbr_dict['max_q'] = 'mq'
choices_dict = ut.get_common_choices_dict()

parser = ut.get_parser(default_cfgs, abbr_dict, choices_dict)
args = parser.parse_args()

args.model_name = ut.get_sampling_model_name(args.is_adapt, args.is_greek)

max_q_dict = {
    "WIKI": 100.0,
    "IMDB": 100.0,
    "DBLP": 20.0,
    "GENE": 10.0,
    "AUTHOR": 50.0,
    "DBLP-AN": 100.0,
    "IMDb-AN": 100.0,
    "IMDb-MT": 100.0,
    "TPCH-PN": 100.0,
}

if args.model_name == "EST_B":
    args.max_q = max_q_dict[args.data_name]

args.tag = os.path.join(args.tag, f"{args.model_name}")

print(f"{default_cfgs = }")
print(f"{args = }")


data_name = args.data_name
workload = args.workload
model_name = args.model_name
seed = args.seed
is_adapt = args.is_adapt
is_greek = args.is_greek
m = args.m
max_q = args.max_q
eps = 1e-5
tag = args.tag
print(args)

db_path, model_config_path, model_path, est_path, est_time_path, stat_path = ut.get_common_pathes_for_estimator(
    data_name, tag, workload, seed)
save_count_path = ut.get_count_path_for_sampling_estimator(
    data_name, tag)

with open(model_config_path, "w") as f:
    yaml.safe_dump(vars(args), f)

db, train_data, valid_data, test_data = ut.load_training_files(
    data_name, workload)
model = sp.SamplingEstimator(
    model_path, is_adapt, is_greek, seed, m, max_q, eps)

# print(train_data[:5])
# print(valid_data[:5])
# print(test_data[:5])

build_time = model.build(db)

print(f"{build_time = }")

test_queries, test_cards = list(zip(*test_data))

start_time = time.time()
test_estimations_with_info, test_estimation_times = model.estimate_latency_analysis(
    test_queries, is_info=True)
end_time = time.time()

est_time = end_time - start_time

print(f"{est_time = }")

model_size = model.model_size()
print(f"{model_size = }")

test_estimations, test_count_info = test_estimations_with_info[:,
                                                               0], test_estimations_with_info[:, 1:].astype(np.int32)

print(f"{test_estimations[:5]}")
print(f"{test_count_info[:5]}")
print(f"{test_cards[:5] = }")

q_errs = ut.mean_Q_error(test_cards, test_estimations, reduction='none')

ut.save_estimated_cards(test_queries, test_cards, test_estimations, est_path)
ut.save_estimation_times(test_queries, test_cards,
                         test_estimations, test_estimation_times, est_time_path)
ut.save_count_infos(test_queries, test_cards,
                    test_estimations, test_count_info, save_count_path)

# save stat (avg, q001, ..., q100, time) # train_time, test_time
ut.save_estimator_stat(stat_path, q_errs, build_time, est_time, model_size)
