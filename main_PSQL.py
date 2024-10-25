from socket import gethostname
from src.util import parse_all_alg, read_strings
import socket
import os
import csv
import yaml
import sys

alg_name, args, exp_name, exp_key = parse_all_alg()

alg_name = "PSQL"
data_name = args.dname
query_key = args.query_key
workload = args.workload
trial = args.trial

db_path = f"data/{data_name}/{data_name}.txt"
qrys_path = f"data/{data_name}/query/{query_key}.txt"

res_path = f"res/{data_name}/{workload}/data/{query_key}/{alg_name}.txt"
time_path = f"res/{data_name}/{workload}/time/{query_key}/{alg_name}.txt"

db = read_strings(db_path)
qrys = read_strings(qrys_path)

os.makedirs(os.path.dirname(res_path), exist_ok=True)
os.makedirs(os.path.dirname(time_path), exist_ok=True)

res, time_dict = data_gen_alg_psql(db, qrys)

with open(res_path, "w") as f:
    csv_w = csv.writer(f, lineterminator='\n')
    csv_w.writerows(res)

with open(time_path, "w") as f:
    yaml.dump(time_dict, f)
