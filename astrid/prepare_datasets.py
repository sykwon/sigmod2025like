import xml.etree.ElementTree as ET
import re
import os
import pandas as pd
import astrid.summary_data_structures as summary_data_structures
from astrid.summary_data_structures import MAX_STR_SIZE
import src.util as ut
import csv

# Matches alphanumeric and space
regex_pattern = r'[^A-Za-z0-9 ]+'

# Download dblp50000.xml from HPI at
# https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/projekte/repeatability/DBLP/dblp50000.xml


def load_and_save_dblp():
    tree = ET.parse('dblp50000.xml')
    root = tree.getroot()

    titles = []
    authors = []

    for article in root:
        title = article.find("title").text
        titles.append(title)
        for authorNode in article.findall("author"):
            authors.append(authorNode.text)

    titles = pd.Series(titles)
    authors = pd.Series(authors)

    titles.str.replace(regex_pattern, '').to_csv(
        "datasets/dblp/dblp_titles.csv", index=False, header=False)
    authors.str.replace(regex_pattern, '').to_csv(
        "datasets/dblp/dblp_authors.csv", index=False, header=False)

# Download from https://github.com/gregrahn/join-order-benchmark


def load_imdb_movie_titles():
    df = pd.read_csv("title.csv", header=None,
                     warn_bad_lines=True, error_bad_lines=False)
    # second column is the title
    df[1].str.replace(regex_pattern, '').to_csv(
        "datasets/imdb/imdb_movie_titles.csv", index=False, header=False)

# Download from https://github.com/gregrahn/join-order-benchmark


def load_imdb_movie_actors():
    df = pd.read_csv("name.csv", header=None,
                     warn_bad_lines=True, error_bad_lines=False)
    # second column is the title
    df[1].str.replace(regex_pattern, '').to_csv(
        "datasets/imdb/imdb_movie_actors.csv", index=False, header=False)

# download from https://github.com/electrum/tpch-dbgen
# schema diagram at https://docs.deistercloud.com/content/Databases.30/TPCH%20Benchmark.90/Data%20generation%20tool.30.xml?embedded=true


def load_and_save_tpch():
    col_names = ["partkey", "name", "mfgr", "brand", "type",
                 "size", "container", "retailprice", "comment"]
    df = pd.read_csv("part.tbl", sep='|', names=col_names,
                     warn_bad_lines=True, error_bad_lines=False, index_col=False)
    df["name"].str.replace(regex_pattern, '').to_csv(
        "datasets/tpch/tpch_part_names.csv", index=False, header=False)

# This function will take a input file that contains strings one line at a time
# creates summary data structures for prefix, substring, suffix
# stores their selectivities
# and stores their triplets
# dataset_prefix is the name of the input file without .csv
# it will be used to generate outputs.
# eg. dblp_authors => dblp_authors.csv is the input file
# dblp_authors_prefix_count, dblp_authors_prefix_triplets contain the frequencies and triplets respectively


def prepare_dataset_each_type(input_file_name, triplet_dirpath, max_str_size, fn_desc):
    os.makedirs(triplet_dirpath, exist_ok=True)
    # function_desc_list = ["prefix", "suffix", "substring"]
    # input_file_name = folder_path + dataset_prefix + ".csv"

    # for index, fn_desc in enumerate(function_desc_list):

    # count_file_name = os.path.join(triplet_dirpath, f"{fn_desc}_counts.csv")
    triplet_file_name = os.path.join(
        triplet_dirpath, f"{fn_desc}_triplets.csv"
    )

    tree = summary_data_structures.create_summary_datastructure(
        input_file_name, fn_desc, max_str_size)
    # tree.print_tree()
    # summary_data_structures.store_selectivities(tree, count_file_name)
    summary_data_structures.store_triplets(tree, triplet_file_name)
    # print(f"{count_file_name = }")
    print(f"{triplet_file_name = }")
    return triplet_file_name


def prepare_dataset(folder_path, dataset_prefix):
    print("Processing ", dataset_prefix)
    functions = [summary_data_structures.get_all_prefixes, summary_data_structures.get_all_suffixes,
                 summary_data_structures.get_all_substrings]
    function_desc = ["prefix", "suffix", "substring"]
    input_file_name = folder_path + dataset_prefix + ".csv"
    for index, fn in enumerate(functions):
        count_file_name = folder_path + dataset_prefix + \
            "_" + function_desc[index] + "_counts.csv"
        triplet_file_name = folder_path + dataset_prefix + \
            "_" + function_desc[index] + "_triplets.csv"

        tree = summary_data_structures.create_summary_datastructure(
            input_file_name, fn)
        # tree.print_tree()
        summary_data_structures.store_selectivities(tree, count_file_name)
        summary_data_structures.store_triplets(tree, triplet_file_name)


def gen_Astrid_workload(data_name):
    input_file_name = f'data/{data_name}/{data_name}.txt'
    print("Processing ", data_name)
    # functions = [summary_data_structures.get_all_prefixes, summary_data_structures.get_all_suffixes,
    #              summary_data_structures.get_all_substrings]
    function_desc = ["prefix", "suffix", "substring"]
    p_train = 1.0
    p_test = 0.5
    p_valid = 0.1

    for index, fn in enumerate(function_desc):
        query_file_folder = f'data/{data_name}/query/Astrid/{fn}/'
        count_file_folder = f'data/{data_name}/training/Astrid/{fn}/'
        triplet_file_folder = f'res/{data_name}/triplets/'
        os.makedirs(query_file_folder, exist_ok=True)
        os.makedirs(count_file_folder, exist_ok=True)
        os.makedirs(triplet_file_folder, exist_ok=True)
        print(f"{query_file_folder = }")
        print(f"{count_file_folder = }")
        print(f"{triplet_file_folder = }")

        tree = summary_data_structures.create_summary_datastructure(
            input_file_name, fn, MAX_STR_SIZE)
        # tree.print_tree()
        df = summary_data_structures.get_cardinalities(tree)
        print(df)
        training_instances = df.reset_index().values
        if fn == "prefix":
            training_instances = [(x[0] + '%', x[1])
                                  for x in training_instances]
        elif fn == "suffix":
            training_instances = [('%' + x[0], x[1])
                                  for x in training_instances]
        elif fn == "substring":
            training_instances = [('%' + x[0] + '%', x[1])
                                  for x in training_instances]
        # queries = df.index

        train_data, valid_data, test_data = ut.train_valid_test_split_test_first(
            training_instances, p_test, p_valid, seed=0, p_train=p_train)
        generated_data_dict = {
            "train": sorted(train_data, key=lambda x: x[0]),
            "valid": sorted(valid_data, key=lambda x: x[0]),
            "test": sorted(test_data, key=lambda x: x[0]),
        }

        for train_type in ['train', 'valid', 'test']:
            generated_data = generated_data_dict[train_type]
            generated_queries = [x[0] for x in generated_data]
            query_file_name = query_file_folder + train_type + ".txt"
            count_file_name = count_file_folder + train_type + ".txt"
            print(f"{query_file_name = }")
            # query
            with open(query_file_name, "w") as f:
                for generate_query in generated_queries:
                    f.write(generate_query + "\n")
            # training
            print(f"{count_file_name = }")
            with open(count_file_name, "w") as f:
                csv_w = csv.writer(f)
                for instance in generated_data:
                    csv_w.writerow(instance)

        df = summary_data_structures.get_triplets(tree)
        triplet_file_name = triplet_file_folder + fn + "_triplets.csv"
        print(f"{triplet_file_name = }")
        df.to_csv(triplet_file_name, index=False, header=True)
        # print(df)


def aggregate_Astrid_workload(data_name):
    function_desc = ["prefix", "suffix", "substring"]
    for train_type in ['train', 'valid', 'test']:
        agg_queries = []
        agg_instances = []
        for index, fn in enumerate(function_desc):
            query_file_folder = f'data/{data_name}/query/Astrid/{fn}/'
            count_file_folder = f'data/{data_name}/training/Astrid/{fn}/'
            query_file_name = query_file_folder + train_type + ".txt"
            count_file_name = count_file_folder + train_type + ".txt"
            print(f"read from {query_file_name = }")
            print(f"read from {count_file_name = }")
            queries = ut.read_strings(query_file_name)
            agg_queries.extend(queries)
            instances = ut.read_training(count_file_name)
            agg_instances.extend(instances)
        query_file_folder = f'data/{data_name}/query/Astrid/'
        count_file_folder = f'data/{data_name}/training/Astrid/'
        query_file_name = query_file_folder + train_type + ".txt"
        count_file_name = count_file_folder + train_type + ".txt"

        agg_queries = sorted(agg_queries)
        agg_instances = sorted(agg_instances, key=lambda x: x[0])

        print(f"write to {query_file_name = }")
        # query
        with open(query_file_name, "w") as f:
            for generate_query in agg_queries:
                f.write(generate_query + "\n")
        # training
        print(f"write to {count_file_name = }")
        with open(count_file_name, "w") as f:
            csv_w = csv.writer(f)
            for instance in agg_instances:
                csv_w.writerow(instance)

        # csv.reader()


# The following functions generate the raw files for 4 datasets.
# Note: these are already in the github repository
# load_and_save_dblp()
# load_imdb_movie_titles()
# load_imdb_movie_actors()
# load_and_save_tpch()
# The following functions generates the frequencies and triplets
# This function might take few minutes for large datasets :)
if __name__ == "__main__":
    prepare_dataset("datasets/dblp/", "dblp_authors")
    prepare_dataset("datasets/dblp/", "dblp_titles")
    prepare_dataset("datasets/imdb/", "imdb_movie_actors")
    prepare_dataset("datasets/imdb/", "imdb_movie_titles")
    prepare_dataset("datasets/tpch/", "tpch_part_names")
