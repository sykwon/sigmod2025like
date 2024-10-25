from src.LBS import LBS, get_sample_valid_instances
import src.util as ut
import os
import yaml
import time

is_debug = False

input_config_path = "configs/LBS/default.yml"
default_cfgs = yaml.safe_load(open(input_config_path))
abbr_dict = ut.get_common_abbr_dict()
abbr_dict['N'] = 'N'
abbr_dict['L'] = 'L'
abbr_dict['PT'] = 'PT'
choices_dict = ut.get_common_choices_dict()

parser = ut.get_parser(default_cfgs, abbr_dict, choices_dict)
args = parser.parse_args()
args.model_name = f"LBS"
args.tag = os.path.join(args.tag, args.model_name)

N = args.N
L = args.L
PT = args.PT
max_n_sample = args.max_n_sample
data_name = args.data_name
workload = args.workload
seed = args.seed
tag = args.tag
print(args)

db_path, model_config_path, model_path, est_path, est_time_path, stat_path = ut.get_common_pathes_for_estimator(
    data_name, tag, workload, seed)

with open(model_config_path, "w") as f:
    yaml.safe_dump(vars(args), f)

db, train_data, valid_data, test_data = ut.load_training_files(
    data_name, workload)

# print(train_data[:5])
# print(valid_data[:5])
# print(test_data[:5])

model = LBS(N, L, PT, model_path, seed, is_pre_suf=True)

sample_queries = get_sample_valid_instances(valid_data, max_n_sample, seed=0)
print(f"{sample_queries[:10] = }")

build_time = model.build(db, sample_queries, is_debug)

print(f"{build_time = }")

model_size = model.model_size()
print(f"{model_size = }")

test_queries, test_cards = list(zip(*test_data))
test_estimations, test_estimation_times = model.estimate_latency_analysis(
    test_queries[:2])

start_time = time.time()
test_estimations, test_estimation_times = model.estimate_latency_analysis(
    test_queries)
end_time = time.time()

est_time = end_time - start_time

print(f"{est_time = }")

print(f"{test_estimations[:5] = }")
print(f"{test_cards[:5] = }")

q_errs = ut.mean_Q_error(test_cards, test_estimations, reduction='none')

ut.save_estimated_cards(test_queries, test_cards, test_estimations, est_path)
ut.save_estimation_times(test_queries, test_cards,
                         test_estimations, test_estimation_times, est_time_path)

# save stat (avg, q001, ..., q100, time) # train_time, test_time
ut.save_estimator_stat(stat_path, q_errs, build_time, est_time, model_size)
