# from astrid.prepare_datasets import prepare_dataset
from src.LPLM.prepare_training_data import generate_like_patterns
from src.LPLM.misc_utils import LIKE_pattern_to_extendedLanguage, extendedLanguage_to_LIKE_pattern
from src.LPLM.compute_ground_truth_by_alg import load_like_patterns, compute_gt_LPLM_main
import sys
import os
import time
import shutil
import subprocess
from tqdm import tqdm
import src.gen_query as gq
import src.util as ut
import pandas as pd
import re
from astrid.prepare_datasets import gen_Astrid_workload, aggregate_Astrid_workload


def run_alg(data_name, workload, q_type, is_force):
    alg_name = 'LEADER-S'
    CMD = f'./main {alg_name} {data_name} {workload} {q_type} 0'
    res_path = f'res/{data_name}/{workload}/data/{q_type}/{alg_name}.txt'
    if not is_force and os.path.exists(res_path):
        print(f"Already exists {res_path = }")
    else:
        print(f"{CMD = }")
        subprocess.run(CMD.split())


def gather_LPLM_LIKE_queries(query_path, output_path, is_force):
    print(f"{query_path = }")
    print(f"{output_path = }")
    if not is_force and os.path.exists(output_path):
        print(f"Already exists {output_path = }")
        return
    query_list = []
    with open(query_path) as f:
        for line in f.readlines():
            query_list.append(line.rstrip('\n'))

    query_dict = {}
    for pattern in tqdm(query_list):
        language, is_end_esc = LIKE_pattern_to_extendedLanguage(pattern)
        for length in range(1, len(language) + 1):
            prefix = language[:length]
            pre_pattern = extendedLanguage_to_LIKE_pattern(
                prefix, is_end_esc and length == len(language))
            if pre_pattern not in query_dict:
                query_dict[pre_pattern] = 0
            query_dict[pre_pattern] += 1
            # print(f"{pre_pattern = }")
    with open(output_path, 'w') as f:
        for query in sorted(query_dict.keys()):
            f.write(query + '\n')


def main_workload_copy_training_data(q_types=None):
    print("########### Copy training data ###########")
    if q_types is None:
        q_types = ['train', 'valid', 'test', 'pack_simple']
    for q_type in q_types:
        res_path = f'res/{data_name}/{workload}/data/{q_type}/LEADER-S.txt'
        training_path = f'data/{data_name}/training/{workload}/{q_type}.txt'
        if not is_force and os.path.exists(training_path):
            print(f"Already exists {training_path = }")
        else:
            shutil.copy(res_path, training_path)
            print(f"{res_path = } to {training_path = }")


def main_workload_compute_cardinality(q_types=None):
    if q_types is None:
        q_types = ['train', 'valid', 'test', 'train_gather', 'pack_simple']

    print("########### Compute Cardinality ###########")
    for q_type in q_types:
        run_alg(data_name, workload, q_type, is_force)


def main_workload_compute_LPLM_data():
    print("########### Compute LPLM data ###########")
    query_dir = f"data/{data_name}/query/{workload}/"
    like_patterns_path = os.path.join(query_dir, 'train.txt')
    dataset_path = f"data/{data_name}/{data_name}.txt"
    card_path = f"res/{data_name}/{workload}/data/train_gather/LEADER-S.txt"
    file_to_save_ground = f"data/{data_name}/training/{workload}/train_LPLM.txt"
    list_of_patterns = load_like_patterns(like_patterns_path)
    datasetsize = len(open(dataset_path).readlines())
    compute_gt_LPLM_main(list_of_patterns, card_path,
                         file_to_save_ground, datasetsize, is_force)


def main_workload_copy_CEB_test_data():
    query_CEB_dir = query_dir.replace(workload, "CEB")
    test_query_CEB_path = query_CEB_dir + "test.txt"
    test_query_path = query_dir + "test.txt"
    assert os.path.exists(test_query_CEB_path)
    os.makedirs(os.path.dirname(test_query_path), exist_ok=True)
    shutil.copy(test_query_CEB_path, test_query_path)

    training_CEB_dir = training_dir.replace(workload, "CEB")
    test_training_CEB_path = training_CEB_dir + "test.txt"
    test_training_path = training_dir + "test.txt"
    assert os.path.exists(test_training_CEB_path)
    os.makedirs(os.path.dirname(test_training_path), exist_ok=True)
    shutil.copy(test_training_CEB_path, test_training_path)


def main_workload_gather_LIKE_and_simple_packed():
    print("########### Gather LIKE language ###########")
    query_path = os.path.join(query_dir, 'train.txt')
    gather_path = os.path.join(query_dir, 'train_gather.txt')
    gather_LPLM_LIKE_queries(query_path, gather_path, is_force)

    query_key = workload
    gq.gen_simple_packed(data_name, query_key, is_force)


def main_workload_check_training_data():
    print("########### Check training data ###########")
    for q_type in ['train', 'valid', 'test']:
        res_path = f'res/{data_name}/{workload}/data/{q_type}/LEADER-S.txt'
        training_path = f'data/{data_name}/training/{workload}/{q_type}.txt'
        df1 = pd.read_csv(res_path)
        df2 = pd.read_csv(training_path)
        assert df1.equals(
            df2), f"mismatch two {(res_path, training_path)=}"


if __name__ == "__main__":
    data_names = ['WIKI', 'IMDB', 'DBLP', 'GENE', 'AUTHOR',
                  'DBLP-AN', 'IMDb-AN', 'IMDb-MT', 'TPCH-PN',
                  'link_type.link', 'movie_companies.note', 'name.name', 'title.title']
    workloads = ['LPLM', 'Astrid', 'CEB', 'CLIQUE']

    if len(sys.argv) <= 1:
        print("Usage: <data_name> <workload>")
        exit()

    data_name = sys.argv[1]
    workload = sys.argv[2]
    is_force = len(sys.argv) > 3
    print(f"{data_name = }, {workload = } {is_force = }")
    assert data_name in data_names, data_name
    assert any([x in workload for x in workloads]), workload
    query_dir = f"data/{data_name}/query/{workload}/"
    training_dir = f"data/{data_name}/training/{workload}/"

    CEB_DATA = data_name in ['link_type.link',
                             'movie_companies.note', 'name.name', 'title.title']

    if "LPLM" in workload:
        max_n_wild_str = re.findall('\d+', workload)
        if len(max_n_wild_str) > 0:
            max_n_wild = int(max_n_wild_str[0])
        else:
            max_n_wild = None
        data_path = f"data/{data_name}/{data_name}.txt"
        if data_name in ['link_type.link']:
            n_train = 1000
            n_valid = 200
            n_test = 0
        else:
            n_train = 5000000
            n_valid = 1000000
            n_test = 500000
        if CEB_DATA:
            n_test = 0
        ########### Gen query workload ###########
        print("########### Gen query workload ###########")
        if not is_force and os.path.exists(query_dir):
            print(f"Already exists {query_dir = }")
        else:
            generate_like_patterns(data_path, query_dir,
                                   n_train, n_valid, n_test, max_data_range=max_n_wild)
        if CEB_DATA:
            main_workload_copy_CEB_test_data()

        ########### Gather LIKE language ###########
        main_workload_gather_LIKE_and_simple_packed()

        ########### Compute Cardinality ###########
        main_workload_compute_cardinality()

        ########### Compute LPLM data ###########
        main_workload_compute_LPLM_data()

        ########### Copy training data ###########
        main_workload_copy_training_data()

    elif workload == "Astrid":
        gen_Astrid_workload(data_name)
        aggregate_Astrid_workload(data_name)

        ########### Gather LIKE language ###########
        main_workload_gather_LIKE_and_simple_packed()

        ########### Compute Cardinality ###########
        main_workload_compute_cardinality()

        ########### Compute LPLM data ###########
        main_workload_compute_LPLM_data()

        ########### Copy training data ###########
        main_workload_copy_training_data(['pack_simple'])

        ########### Check training data ###########
        main_workload_check_training_data()

    elif workload == "CEB":
        ########### Gather LIKE language ###########
        print("########### Gather LIKE language ###########")
        query_path = os.path.join(query_dir, 'train.txt')
        gather_path = os.path.join(query_dir, 'train_gather.txt')
        gather_LPLM_LIKE_queries(query_path, gather_path, is_force)

        query_key = workload
        gq.gen_simple_packed(data_name, query_key, is_force)

        ########### Compute Cardinality ###########
        main_workload_compute_cardinality()

        ########### Compute LPLM data ###########
        main_workload_compute_LPLM_data()

        ########### Copy training data ###########
        main_workload_copy_training_data(['pack_simple'])

    elif workload == "CLIQUE":
        gq.set_global_ru_common(data_name, "CLIQUE")
        if CEB_DATA:
            gq.p_test = 0.0
            gq.p_train = 0.1
            gq.p_valid = 0.01
            gq.max_word_len = 10
            # gq.max_n_under = 0
            # gq.q_types = [gq.Qtype.PREFIX]
            refine_words = False  # movie
            gq.generate_query_uniformly(refine_words=refine_words)
        else:
            gq.generate_query_uniformly()

        if CEB_DATA:
            main_workload_copy_CEB_test_data()

        ########### Gather LIKE language ###########
        main_workload_gather_LIKE_and_simple_packed()

        ########### Compute Cardinality ###########
        main_workload_compute_cardinality()

        ########### Compute LPLM data ###########
        main_workload_compute_LPLM_data()

        ########### Copy training data ###########
        main_workload_copy_training_data()
