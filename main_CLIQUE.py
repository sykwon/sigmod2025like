from src.util import read_strings
import src.util as ut
import os
import yaml
import time
import torch
import pickle
from tqdm import tqdm
from src.CLIQUE import CLIQUEestimator

MODE_EST = str(os.getenv("MODE")).lower() == "est"
MODE_SIZE = str(os.getenv("MODE")).lower() == "size"

input_config_path = "configs/CLIQUE/default.yml"
default_cfgs = yaml.safe_load(open(input_config_path))
abbr_dict = ut.get_common_abbr_dict()
choices_dict = ut.get_common_choices_dict()

parser = ut.get_parser(default_cfgs, abbr_dict, choices_dict)
args = parser.parse_args()


max_entry_ratio_dict = {
    "WIKI": 4.35,
    "IMDB": 1.675,
    "DBLP": 3.12,
    "GENE": 204.8,
    "AUTHOR": 4.8,
    "DBLP-AN": 3.15,
    "IMDb-AN": 2.6,
    "IMDb-MT": 1.6,
    "TPCH-PN": 15.0,
    "link_type.link": 8.0,
    "movie_companies.note": 0.99,
    "name.name": 0.359,
    "title.title": 0.223,
}
rnn_hs_dict = {
    "DBLP-AN": 160,
    "IMDb-AN": 160,
    "IMDb-MT": 160,
    "TPCH-PN": 160,
    "link_type.link": 181,
    "movie_companies.note": 154,
    "name.name": 152,
    "title.title": 151,
}
if args.data_name in max_entry_ratio_dict:
    args.max_entry_ratio = max_entry_ratio_dict[args.data_name]
if args.data_name in rnn_hs_dict:
    args.rnn_hs = rnn_hs_dict[args.data_name]

args.model_name = f"CLIQUE"
args = ut.add_just_params(args)

if args.packed:
    args.model_name += "-PACK"

if not args.crh:
    args.model_name += "-T"

if args.pack_all:
    args.model_name += "-ALL"


args.tag = os.path.join(args.tag, args.model_name)

print(args)


N = args.N
PT = args.PT
data_name = args.data_name
workload = args.workload
seed = args.seed
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
patience = args.patience
packed = args.packed
bound_in = args.bound_in  # bool, whether the model takes bounds as input or not
dynamicPT = args.dynamicPT
max_entry_ratio = args.max_entry_ratio
batch_norm = args.batch_norm
pack_all = args.pack_all
crh = args.crh
tag = args.tag
print(args)

last_flip = True

(
    db_path,
    model_config_path,
    model_path,
    est_path,
    est_time_path,
    stat_path,
) = ut.get_common_pathes_for_estimator(data_name, tag, workload, seed)

sw = ut.get_summary_writer(data_name, tag, workload, seed)
count_path = ut.get_count_path_for_HybridEstimator(
    data_name, PT, N, dynamicPT, max_entry_ratio
)

with open(model_config_path, "w") as f:
    yaml.safe_dump(vars(args), f)

db = read_strings(db_path)
n_db = len(db)
char_dict, n_special_char = ut.gen_char_dict(db)


char_dict, n_special_char, train_data, valid_data, test_data = ut.get_training_data(
    data_name, workload, p_train)

qc_dict = None
if packed:
    qc_dict = ut.get_qc_dict(data_name, workload)


def packed_card(train_data, qc_dict, last_flip):
    output = []
    for query, card in tqdm(train_data):
        query_cano = ut.canonicalize_like_query(query, last_flip)
        cards = []
        for query_len in range(1, len(query) + 1):
            sub_query = query_cano[:query_len]
            sub_query = ut.canonicalize_like_query(sub_query)
            # sub_query = query[:query_len]
            card = qc_dict[sub_query]
            cards.append(card)
        output.append([query, cards])
    return output


if packed:
    train_data_old = train_data
    train_data = packed_card(train_data, qc_dict, last_flip)

model = CLIQUEestimator()
model.set_params(
    N,
    PT,
    seed,
    char_dict,
    ch_es,
    1,
    packed,
    rnn_hs,
    pred_hs,
    n_pred_layer,
    lr,
    l2,
    n_epoch,
    bs,
    patience,
    last_flip,
    sw,
    count_path,
    model_path,
    bound_in,
    pack_all,
    dynamicPT,
    max_entry_ratio,
    batch_norm,
)


if not MODE_EST:
    torch.save(model.state_dict(), model_path)
build_time_count = model.build_count(data_name)
model_size = model.model_size()
model_size_hr = ut.human_readable_size(model_size)
print(f"{model_size = }, {model_size_hr = }")


if MODE_EST:
    model.load_state_dict(torch.load(model_path))
    model.n_db = len(db)
    model.cuda()
    with open(stat_path) as f:
        build_time = yaml.load(f, Loader=yaml.FullLoader)['time']
    print(f"{build_time = }")
else:
    build_time = model.build(data_name, db, train_data, valid_data, test_data)

print(f"{build_time = }")

test_queries, test_cards = list(zip(*test_data))

model.eval()
model.cpu()

start_time = time.time()
with torch.no_grad():
    test_estimations, test_estimation_times = model.estimate_latency_analysis(
        test_queries, to_cuda=False,
    )
end_time = time.time()

est_time = end_time - start_time

print(f"{est_time = }")

print(f"{test_estimations[:5] = }")
print(f"{test_cards[:5] = }")

model_size = model.model_size()
print(f"{model_size = }")

q_errs = ut.mean_Q_error(test_cards, test_estimations, reduction="none")

ut.save_estimated_cards(test_queries, test_cards, test_estimations, est_path)
ut.save_estimation_times(
    test_queries, test_cards, test_estimations, test_estimation_times, est_time_path
)

# save stat (avg, q001, ..., q100, time) # train_time, test_time
count_size = os.path.getsize(model.count_path)
neural_size = os.path.getsize(model.model_path)
ut.save_estimator_stat(stat_path, q_errs, build_time,
                       est_time, model_size=model_size,
                       count_size=count_size, neural_size=neural_size)
