import src.util as ut
from src.util import read_strings
from src.model import *
import os
import yaml
import pandas as pd
import time

is_debug = "DEBUG" in os.environ
if is_debug:
    input_config_path = "configs/DREAM/debug.yml"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
else:
    input_config_path = "configs/DREAM/default.yml"

is_est = "EST" in os.environ

pred_hs_dict = {
    "DBLP-AN": 128,
    "IMDb-AN": 128,
    "IMDb-MT": 128,
    "TPCH-PN": 128,
}

ch_es_dict = {
    "DBLP-AN": 255,
    "IMDb-AN": 255,
    "IMDb-MT": 266,
    "TPCH-PN": 255,
}


default_cfgs = yaml.safe_load(open(input_config_path))
abbr_dict = ut.get_common_abbr_dict()
abbr_dict["batch_size"] = "b"
abbr_dict["learning_rate"] = "lr"
abbr_dict["mask_prob"] = "m"
abbr_dict["patience"] = "t"
choices_dict = ut.get_common_choices_dict()

parser = ut.get_parser(default_cfgs, abbr_dict, choices_dict)
args = parser.parse_args()

args.model_name = "DREAM"
if args.packed:
    args.model_name += "-PACK"
args.tag = os.path.join(args.tag, args.model_name)
if args.data_name in pred_hs_dict:
    args.pred_hs = pred_hs_dict[args.data_name]
if args.data_name in ch_es_dict:
    args.ch_es = ch_es_dict[args.data_name]

data_name = args.data_name
workload = args.workload
n_epoch = args.n_epoch
ch_es = args.ch_es
rnn_hs = args.rnn_hs
pred_hs = args.pred_hs
p_test = args.p_test
p_val = args.p_val
p_train = args.p_train
n_pred_layer = args.n_pred_layer
bs = args.batch_size
lr = args.learning_rate
l2 = args.l2
seed = args.seed
patience = args.patience
packed = args.packed
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

if is_est:
    sw = None
    assert os.path.exists(model_config_path), model_config_path
    with open(model_config_path) as f:
        saved_args = yaml.safe_load(f)
    assert ut.compare_two_dictionaries(
        saved_args, vars(args)
    ), f"{saved_args = }\n{vars(args) = }"
else:
    sw = ut.get_summary_writer(data_name, tag, workload, seed)

    with open(model_config_path, "w") as f:
        yaml.safe_dump(vars(args), f)

db = read_strings(db_path)
n_db = len(db)
char_dict, n_special_char, train_data, valid_data, test_data = ut.get_training_data(
    data_name, workload, p_train)

# print(train_data[:5])
# print(valid_data[:5])
# print(test_data[:5])

qc_dict = None
if packed:
    qc_dict = ut.get_qc_dict(data_name, workload)

# char_set contains special characters ['[PAD]', '[UNK]', '%', '_', '[', ']', '^', '-']
print(list(char_dict.keys())[:10])
print("train:", train_data[0])
print("valid:", valid_data[0])
print("test:", test_data[0])

model = DREAMmodel()
model.set_params(
    char_dict,
    ch_es,
    rnn_hs,
    pred_hs,
    n_pred_layer=n_pred_layer,
    n_db=n_db,
    seed=seed,
    packed=packed,
    model_path=model_path,
)
print(model)
model_size = ut.get_torch_model_size(model)
model_size_hr = ut.human_readable_size(model_size)
print(f"{model_size = }, {model_size_hr = }")

if is_est:
    assert os.path.exists(model_path), model_path
    assert os.path.exists(stat_path), stat_path
    with open(stat_path) as f:
        stat_dict = yaml.safe_load(f)
    print(f"{stat_dict = }")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device = }")
    model.load(device)
    build_time = stat_dict["time"]
else:
    build_time = model.build(
        train_data, valid_data, test_data, lr, l2, n_epoch, bs, patience, sw, qc_dict
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

if is_est:
    assert os.path.exists(est_path), est_path
    df = pd.read_csv(est_path)
    test_estimations = df["est"].to_numpy()

print(f"{est_time = }")

print(f"{test_estimations[:5] = }")
print(f"{test_cards[:5] = }")

q_errs = ut.mean_Q_error(test_cards, test_estimations, reduction="none")
if is_est:
    print(f"{q_errs = }")
    print(f"{test_estimations = }")
    print(f"{df['est'].to_numpy() = }")
    for card, e1, e2 in zip(test_cards, test_estimations, df["est"].to_numpy()):
        assert abs(e1 - e2) < 1, f"{card}, {e1:.8f}, {e2:.8f}"


ut.save_estimated_cards(test_queries, test_cards, test_estimations, est_path)
ut.save_estimation_times(
    test_queries, test_cards, test_estimations, test_estimation_times, est_time_path
)

# save stat (avg, q001, ..., q100, time) # train_time, test_time
ut.save_estimator_stat(stat_path, q_errs, build_time, est_time, model_size)
