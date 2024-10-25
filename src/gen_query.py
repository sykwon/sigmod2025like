import numpy as np
import warnings
from src.util import *
import src.util as ut
from collections import Counter
import subprocess
import socket
import csv
import shutil
import warnings
import pandas as pd
from IPython.display import display


def set_global_ru_common(data_name, query_key_):
    global dataNames
    global q_types
    global seed
    global max_n_token
    global max_n_under
    global p_train
    global p_valid
    global p_test
    global max_word_len
    global query_key

    # dataNames = ["DBLP", "IMDB", "WIKI", "GENE", "AUTHOR"]
    # dataNames = ["AUTHOR"]
    # dataNames = ["IMDB"]
    # dataNames = ["DBLP"]
    # dataNames = ["WIKI", "IMDB"]
    dataNames = [data_name]
    q_types = list(Qtype.iter())
    seed = 0
    max_n_token = 2
    max_n_under = 2
    p_train = 0.1
    p_valid = 0.01
    p_test = 0.01
    max_word_len = 5
    query_key = query_key_


class QueryGenerator:
    def __init__(self, strings, q_type, n_token, max_n_under, max_word_len, seed=None, m=None, refine_words=False):
        # self.strings = strings
        words_list = [refine_data_string2words(
            x, q_type, n_token, min_word_len, max_word_len) for x in strings]
        words_list = list(
            filter(lambda x: x is not None and len(x) == n_token, words_list))
        self.words_list = list(set([tuple(x) for x in words_list]))
        # self.n = len(strings)
        self.n = len(self.words_list)
        assert q_type in list(Qtype.iter())
        self.q_type = q_type
        self.n_token = n_token
        self.max_n_under = max_n_under
        self.current = -1
        self.n_qry = 0
        self.seed = seed
        self.perms = list(range(self.n))
        self.exclusive_qrys = set()
        self.max_word_len = max_word_len
        self.m = m
        self.refine_words = refine_words

        np.random.seed(seed)
        np.random.shuffle(self.perms)

    def __iter__(self):
        return self

    def set_exclusive_qrys(self, qrys):
        self.exclusive_qrys = set(qrys)

    def __next__(self):
        query = None
        n_fails = -1
        while query is None:
            n_fails += 1
            if n_fails >= 1000:
                warnings.warn("Too many fails in this while loop")
            self.current += 1
            self.current %= self.n
            index = self.perms[self.current]
            # string = self.strings[index]
            words = self.words_list[index]
            query = general_query_pat2_gen_query_from_words(
                words, self.q_type, self.max_n_under, self.n_token)
            # query = general_query_pat2_gen_query(string, self.q_type, self.max_n_under,
            #                                      self.n_token, max_word_len=self.max_word_len, is_refine_words=self.refine_words)
            if query is not None and query in self.exclusive_qrys:
                query = None
            if query is not None and self.m is not None:
                qry_m = ut.get_num_betas(query)
                if self.m != qry_m:
                    query = None

        self.n_qry += 1
        return query


def set_of_all_token1(strings, q_type, max_word_len):
    patterns = set()
    if q_type == Qtype.SUBSTR:
        for string in strings:
            for length in range(1, max_word_len + 1):
                for i in range(len(string) - length + 1):
                    sub = string[i:i+length].strip("%")
                    if "%" in sub or "_" in sub or len(sub) == 0:
                        continue
                    pattern = "%" + sub + "%"
                    patterns.add(pattern)
    elif q_type == Qtype.PREFIX:
        for string in strings:
            for length in range(1, max_word_len + 1):
                pre = string[:length].strip("%")
                if "%" in pre or "_" in pre or len(pre) == 0:
                    continue
                pattern = pre + "%"
                patterns.add(pattern)
    elif q_type == Qtype.SUFFIX:
        for string in strings:
            for length in range(1, max_word_len + 1):
                suf = string[-length:].strip("%")
                if "%" in suf or "_" in suf or len(suf) == 0:
                    continue
                pattern = "%" + suf
                patterns.add(pattern)
    else:
        raise ValueError(q_type)

    return patterns


def training_instances_of_all_words1(strings, q_type, add_suffix=False):
    training_instances = {}
    current_queries = set()

    def add_instance_aux(query_core):
        if q_type == Qtype.SUBSTR:
            query = "%" + query_core + "%"
        elif q_type == Qtype.PREFIX:
            query = query_core + "%"
        elif q_type == Qtype.SUFFIX:
            query = "%" + query_core
        else:
            raise ValueError(query)

        if query not in current_queries:
            current_queries.add(query)

            if query not in training_instances:
                card = 0
                for string in strings:
                    if q_type == Qtype.SUBSTR:
                        if query_core in string:
                            card += 1
                    elif q_type == Qtype.PREFIX:
                        if query_core == string[:len(query_core)]:
                            card += 1
                    elif q_type == Qtype.SUFFIX:
                        if query_core == string[-len(query_core):]:
                            card += 1
                    else:
                        raise ValueError(query)

                training_instances[query] = card

    def add_instances(query_core):
        if len(query_core) == 0:
            return

        if add_suffix and q_type != Qtype.PREFIX:
            for pos in range(len(query_core)):
                query_core_suf = query_core[pos:]
                add_instance_aux(query_core_suf)
        else:
            add_instance_aux(query_core)

    for string in strings:
        current_queries.clear()

        words = string.strip().split()
        # print(words)
        if len(words) == 0:
            continue
        if q_type == Qtype.SUBSTR:
            for word in words:
                # print(word)
                add_instances(word)
        elif q_type == Qtype.PREFIX:
            word = words[0]
            add_instances(word)
        elif q_type == Qtype.SUFFIX:
            word = words[-1]
            add_instances(word)
        else:
            raise ValueError(q_type)

    training_instances = sorted(
        training_instances.items(), key=lambda x: x[1], reverse=True)
    return training_instances


def set_of_all_words1(strings, q_type):
    patterns = set()
    for string in strings:
        words = string.strip().split()
        if len(words) == 0:
            continue
        if q_type == Qtype.SUBSTR:
            for word in words:
                if len(word) > 0:
                    pattern = "%" + word + "%"
                    patterns.add(pattern)
        elif q_type == Qtype.PREFIX:
            word = words[0]
            if len(word) > 0:
                pattern = word + "%"
                patterns.add(pattern)
        elif q_type == Qtype.SUBSTR:
            word = words[-1]
            if len(word) > 0:
                pattern = "%" + word
                patterns.add(pattern)
        else:
            raise ValueError(q_type)

    return patterns


def gen_packed_queries(ptype, is_force):
    global dataNames
    global query_key
    global q_types

    def add_core_sub_queries(packed_queries, sub_query_core):
        for q_type in q_types:
            if q_type == Qtype.SUBSTR:
                sub_query = "%" + sub_query_core + "%"
            elif q_type == Qtype.PREFIX:
                sub_query = sub_query_core + "%"
            elif q_type == Qtype.SUFFIX:
                sub_query = "%" + sub_query_core
            assert sub_query.strip("%") == sub_query_core
            sub_query = canonicalize_like_query(sub_query)
            packed_queries.add(sub_query)

    for dataName in dataNames:
        print(f"{dataName = }")
        train_path = f"data/{dataName}/query/{query_key}/train.txt"
        # valid_path = f"data/{dataName}/query/{query_key}/valid.txt"
        # test_path = f"data/{dataName}/query/{query_key}/test.txt"
        if ptype == Ptype.SIMPLE:
            out_path = f"data/{dataName}/query/{query_key}/pack_simple.txt"
        elif ptype == Ptype.CORE:
            out_path = f"data/{dataName}/query/{query_key}/pack_core.txt"
        elif ptype == Ptype.SIMPLE_B:
            out_path = f"data/{dataName}/query/{query_key}/pack_simple_b.txt"
        elif ptype == Ptype.CORE_B:
            out_path = f"data/{dataName}/query/{query_key}/pack_core_b.txt"
        elif ptype == Ptype.SIMPLE_M:
            out_path = f"data/{dataName}/query/{query_key}/pack_simple_m.txt"
        elif ptype == Ptype.CORE_M:
            out_path = f"data/{dataName}/query/{query_key}/pack_core_m.txt"
        elif ptype == Ptype.NORMAL:
            out_path = f"data/{dataName}/query/{query_key}/pack_normal.txt"
        elif ptype == Ptype.NORMAL_B:
            out_path = f"data/{dataName}/query/{query_key}/pack_normal_b.txt"
        elif ptype == Ptype.NORMAL_M:
            out_path = f"data/{dataName}/query/{query_key}/pack_normal_m.txt"
        print(f"{train_path = }")
        # print(f"{valid_path = }")
        # print(f"{test_path = }")
        print(f"{out_path = }")
        if not is_force and os.path.exists(out_path):
            continue

        with open(train_path) as f:
            train_queries = read_strings(filepath=train_path)
        # with open(valid_path) as f:
        #     valid_queries = read_strings(filepath=valid_path)
        #     valid_queries_part = set(valid_queries)
        # with open(test_path) as f:
        #     test_queries = read_strings(filepath=test_path)
        #     test_queries_part = set(test_queries)
        # test_valid_queries_part = valid_queries_part.union(test_queries_part)

        packed_queries = set()

        for query in tqdm(train_queries):
            cannoicalized_query = canonicalize_like_query(
                query, is_last_flip=True)
            # print(f"{query = }")
            # print(f"{cannoicalized_query = }")
            # print(f"{core_query = }")
            if ptype == Ptype.SIMPLE:
                for length in range(1, len(cannoicalized_query)+1):
                    sub_query = cannoicalized_query[:length]
                    sub_query = canonicalize_like_query(sub_query)
                    packed_queries.add(sub_query)
            elif ptype == Ptype.SIMPLE_B:
                for length in range(1, len(cannoicalized_query)+1):
                    sub_query = cannoicalized_query[-length:]
                    sub_query = canonicalize_like_query(sub_query)
                    packed_queries.add(sub_query)
            elif ptype == Ptype.SIMPLE_M:
                for length in range(1, len(cannoicalized_query)+1):
                    sub_query = cannoicalized_query[:length]
                    sub_query = canonicalize_like_query(sub_query)
                    packed_queries.add(sub_query)
                    sub_query = cannoicalized_query[-length:]
                    sub_query = canonicalize_like_query(sub_query)
                    packed_queries.add(sub_query)

            elif ptype == Ptype.CORE:
                core_query = cannoicalized_query.strip('%')
                for length in range(1, len(core_query)+1):
                    sub_query_core = core_query[:length]
                    sub_query_core = sub_query_core.strip("%")
                    add_core_sub_queries(packed_queries, sub_query_core)
            elif ptype == Ptype.CORE_B:
                core_query = cannoicalized_query.strip('%')
                for length in range(1, len(core_query)+1):
                    sub_query_core = core_query[-length:]
                    sub_query_core = sub_query_core.strip("%")
                    add_core_sub_queries(packed_queries, sub_query_core)
            elif ptype == Ptype.CORE_M:
                core_query = cannoicalized_query.strip('%')
                for length in range(1, len(core_query)+1):
                    sub_query_core = core_query[:length]
                    sub_query_core = sub_query_core.strip("%")
                    add_core_sub_queries(packed_queries, sub_query_core)
                    sub_query_core = core_query[-length:]
                    sub_query_core = sub_query_core.strip("%")
                    add_core_sub_queries(packed_queries, sub_query_core)

            elif ptype == Ptype.NORMAL:
                cannoicalized_query = canonicalize_like_query(query)
                for length in range(1, len(cannoicalized_query)+1):
                    sub_query = cannoicalized_query[:length]
                    sub_query = canonicalize_like_query(sub_query)
                    packed_queries.add(sub_query)
            elif ptype == Ptype.NORMAL_M:
                cannoicalized_query = canonicalize_like_query(query)
                for length in range(1, len(cannoicalized_query)+1):
                    sub_query = cannoicalized_query[:length]
                    sub_query = canonicalize_like_query(sub_query)
                    packed_queries.add(sub_query)
                    sub_query = cannoicalized_query[-length:]
                    sub_query = canonicalize_like_query(sub_query)
                    packed_queries.add(sub_query)

        print(f"{len(packed_queries) = }")

        with open(out_path, "w") as f:
            for query in sorted(packed_queries):
                f.write(query + "\n")


def add_additional_query_to_train(add_short, add_word, query_key_src="ru100/2_2"):
    global query_key
    global dataNames
    global q_types

    for dataName in dataNames:
        file_path = f"data/{dataName}.txt"
        src_train_path = f"data/{dataName}/query/{query_key_src}/train.txt"
        src_valid_path = f"data/{dataName}/query/{query_key_src}/valid.txt"
        src_test_path = f"data/{dataName}/query/{query_key_src}/test.txt"

        dest_train_path = f"data/{dataName}/query/{query_key}/train.txt"
        dest_valid_path = f"data/{dataName}/query/{query_key}/valid.txt"
        dest_test_path = f"data/{dataName}/query/{query_key}/test.txt"
        with open(file_path) as f:
            db = read_strings(filepath=file_path)
        print(f"{file_path = }")
        print(f"{src_train_path = }")
        print(f"{src_valid_path = }")
        print(f"{src_test_path = }")
        print(f"{dest_train_path = }")
        print(f"{dest_valid_path = }")
        print(f"{dest_test_path = }")

        # test_queries = []
        # valid_queries = []
        # train_queries = []

        with open(src_train_path) as f:
            src_train_queries = read_strings(filepath=src_train_path)
            src_train_queries_part = set(src_train_queries)
        with open(src_valid_path) as f:
            src_valid_queries = read_strings(filepath=src_valid_path)
            src_valid_queries_part = set(src_valid_queries)
        with open(src_test_path) as f:
            src_test_queries = read_strings(filepath=src_test_path)
            src_test_queries_part = set(src_test_queries)

        for q_type in q_types:
            train_queries_part = set()
            if add_short:
                tokens = set_of_all_token1(db, q_type, max_word_len)
                tokens = tokens.difference(src_train_queries_part)
                tokens = tokens.difference(src_test_queries_part)
                tokens = tokens.difference(src_valid_queries_part)
                train_queries_part = train_queries_part.union(tokens)

            if add_word:
                tokens = set_of_all_words1(db, q_type)
                tokens = tokens.difference(src_train_queries_part)
                tokens = tokens.difference(src_test_queries_part)
                tokens = tokens.difference(src_valid_queries_part)
                train_queries_part = train_queries_part.union(tokens)

            train_queries_part = sorted(train_queries_part)
            print(
                f"{q_type = }, {len(src_test_queries) = }, {len(src_valid_queries_part) = }, {len(train_queries_part) = }")
            src_train_queries.extend(train_queries_part)

        # ### test #####
        # with open(dest_train_path) as f:
        #     dest_train_queries = read_strings(filepath=dest_train_path)
        # with open(dest_valid_path) as f:
        #     dest_valid_queries = read_strings(filepath=dest_valid_path)
        # with open(dest_test_path) as f:
        #     dest_test_queries = read_strings(filepath=dest_test_path)

        # print(f"{len(src_train_queries), len(dest_train_queries) = }")
        # print(f"{len(set(src_train_queries)), len(set(dest_train_queries)) = }")
        # print(f"{len(set(src_train_queries).difference(set(dest_train_queries))) = }")
        # print(f"{len(set(src_valid_queries).difference(set(dest_valid_queries))) = }")
        # print(f"{len(set(src_test_queries).difference(set(dest_test_queries))) = }")
        # for i, (x, y) in enumerate(zip(sorted(src_train_queries), sorted(dest_train_queries))):
        #     assert x == y, ("Train", i, x, y)
        # for i, (x, y) in enumerate(zip(sorted(src_valid_queries), sorted(dest_valid_queries))):
        #     assert x == y, ("Valid", i, x, y)
        # for i, (x, y) in enumerate(zip(sorted(src_test_queries), sorted(dest_test_queries))):
        #     assert x == y, ("Test", i, x, y)
        # ######### end test ##############

        with open(dest_train_path, 'w') as f:
            for query in src_train_queries:
                f.write(query + "\n")

        with open(dest_valid_path, 'w') as f:
            for query in src_valid_queries:
                f.write(query + "\n")

        with open(dest_test_path, 'w') as f:
            for query in src_test_queries:
                f.write(query + "\n")


def generate_query_m_n(m, n):
    global query_key
    global dataNames

    seed = 0
    max_n_token = m
    max_n_under = 2
    for dataName in dataNames:
        file_path = f"data/{dataName}.txt"
        save_dir = f"data/{dataName}/query/{query_key}"
        with open(file_path) as f:
            db = read_strings(filepath=file_path)
        print(f"[{dataName:9s}] n_str: {len(db)}")

        qry_gen_list = []
        for n_token in range(max(max_n_token - 2, 1), max_n_token+1):
            for q_type in q_types:
                print(f"{q_type, n_token = }")
                qry_gen = QueryGenerator(
                    db, q_type, n_token, max_n_under, max_word_len=max_word_len, seed=seed, m=m)
                qry_gen_list.append(qry_gen)

        queries = set()
        fail_count = 0
        while len(queries) < n:
            qry_gen = np.random.choice(qry_gen_list)
            qry = next(qry_gen)
            m_of_qry = ut.get_num_betas(qry)
            assert m == m_of_qry
            if qry not in queries:
                fail_count = 0
                queries.add(qry)
            else:
                fail_count += 1
                print(f"{fail_count = }")

            if fail_count > 5:
                break

            # for qry in (pbar := tqdm(qry_gen)):
            #     pbar.set_description(f"{len(queries_part)}")
            #     queries_part.add(qry)
            #     if len(queries_part) == n_test:
            #         break
            # queries_part = list(queries_part)
            # test_queries.extend(queries_part)
            # qry_gen.set_exclusive_qrys(queries_part)

            # for qry in (pbar := tqdm(qry_gen)):
            #     pbar.set_description(f"{len(valid_queries_part)}")
            #     valid_queries_part.add(qry)
            #     if len(valid_queries_part) == n_valid:
            #         break
            # valid_queries_part = list(valid_queries_part)
            # valid_queries.extend(valid_queries_part)

            # qry_gen.set_exclusive_qrys(queries_part + valid_queries_part)

            # fail_count = 0
            # for qry in (pbar := tqdm(qry_gen)):
            #     pbar.set_description(f"{len(train_queries_part)}")
            #     if qry in train_queries:
            #         fail_count += 1
            #     else:
            #         fail_count = 0
            #     if fail_count > n_train:
            #         break
            #     train_queries_part.add(qry)
            #     if len(train_queries_part) == n_train:
            #         break

            # # add all token1
            # if add_short and n_token == 1:
            #     tokens = set_of_all_token1(db, q_type, max_word_len)
            #     tokens = tokens.difference(queries_part)
            #     tokens = tokens.difference(valid_queries_part)
            #     train_queries_part = train_queries_part.union(tokens)

            # if add_freq and n_token == 1:
            #     tokens = set_of_all_words1(db, q_type)
            #     tokens = tokens.difference(queries_part)
            #     tokens = tokens.difference(valid_queries_part)
            #     train_queries_part = train_queries_part.union(tokens)

            # train_queries_part = list(train_queries_part)
            # print(f"{n_token = }, {q_type = }, {len(test_queries) = }, {len(valid_queries_part) = }, {len(train_queries_part) = }")
            # train_queries.extend(train_queries_part)

        queries = sorted(queries)
        # print(save_dir + ".txt")
        save_query(queries, save_dir + ".txt")


def generate_query_uniformly(add_short=False, add_freq=False, refine_words=False):
    global query_key
    global dataNames

    seed = 0
    for dataName in dataNames:
        file_path = f"data/{dataName}/{dataName}.txt"
        save_dir = f"data/{dataName}/query/{query_key}/"
        with open(file_path) as f:
            db = read_strings(filepath=file_path)

        db = list(filter(lambda x: len(x) >
                  0 and '_' not in x and '%' not in x, db))
        db = list(set(db))

        print(f"[{dataName:9s}] n_str: {len(db)}")

        n_db = max(len(db), 500)
        n_test = int(n_db * p_test)
        n_valid = int(n_db * p_valid)
        n_train = int(n_db * p_train)

        max_n_prefix = int(len(set([x[:max_word_len] for x in db])) * 0.1)
        max_n_suffix = int(len(set([x[-max_word_len:] for x in db])) * 0.1)

        print(f"{max_n_prefix = } {max_n_suffix = }")

        test_queries = []
        valid_queries = []
        train_queries = []

        print(f"{n_test = } {n_valid = } {n_train = }")
        n_test_, n_valid_, n_train_ = n_test, n_valid, n_train
        n_total_ = sum([n_test_, n_valid_, n_train_])

        for n_token in range(1, max_n_token+1):
            for q_type in q_types:
                test_queries_part = set()
                valid_queries_part = set()
                train_queries_part = set()
                qry_gen = QueryGenerator(
                    db, q_type, n_token, max_n_under, max_word_len=max_word_len, seed=seed,
                    refine_words=refine_words)
                max_gen_pat = qry_gen.n
                if n_total_ > max_gen_pat:
                    n_test = int(n_test_ * max_gen_pat / n_total_)
                    n_valid = int(n_valid_ * max_gen_pat / n_total_)
                    n_train = int(n_train_ * max_gen_pat / n_total_)

                print(f"{q_type, n_token, max_gen_pat =}")
                # if q_type == Qtype.PREFIX and n_total_ > max_n_prefix:
                #     n_test = int(n_test_ * max_n_prefix / n_total_)
                #     n_valid = int(n_valid_ * max_n_prefix / n_total_)
                #     n_train = int(n_train_ * max_n_prefix / n_total_)
                # elif q_type == Qtype.SUFFIX and n_total_ > max_n_suffix:
                #     n_test = int(n_test_ * max_n_suffix / n_total_)
                #     n_valid = int(n_valid_ * max_n_suffix / n_total_)
                #     n_train = int(n_train_ * max_n_suffix / n_total_)
                # else:
                #     n_test, n_valid, n_train = n_test_, n_valid_, n_train_

                if n_test > 0:
                    pbar = tqdm(total=n_test)
                    for qry in qry_gen:
                        # pbar.set_description(f"{len(test_queries_part)}")
                        if qry not in test_queries_part:
                            pbar.update(1)
                            test_queries_part.add(qry)
                        if len(test_queries_part) == n_test:
                            break
                    pbar.close()
                test_queries_part = list(test_queries_part)
                test_queries.extend(test_queries_part)
                qry_gen.set_exclusive_qrys(test_queries_part)

                pbar = tqdm(total=n_valid)
                for qry in qry_gen:
                    # pbar.set_description(f"{len(valid_queries_part)}")
                    if qry not in valid_queries_part:
                        pbar.update(1)
                        valid_queries_part.add(qry)
                    if len(valid_queries_part) == n_valid:
                        break
                pbar.close()
                valid_queries_part = list(valid_queries_part)
                valid_queries.extend(valid_queries_part)

                qry_gen.set_exclusive_qrys(
                    test_queries_part + valid_queries_part)

                fail_count = 0
                pbar = tqdm(total=n_train)
                for qry in qry_gen:
                    # pbar.set_description(f"{len(train_queries_part)}")
                    if qry in train_queries:
                        fail_count += 1
                    else:
                        fail_count = 0
                    if fail_count > n_train:
                        break
                    if qry not in train_queries_part:
                        pbar.update(1)
                        train_queries_part.add(qry)
                    if len(train_queries_part) == n_train:
                        break
                pbar.close()

                # add all token1
                if add_short and n_token == 1:
                    tokens = set_of_all_token1(db, q_type, max_word_len)
                    tokens = tokens.difference(test_queries_part)
                    tokens = tokens.difference(valid_queries_part)
                    train_queries_part = train_queries_part.union(tokens)

                if add_freq and n_token == 1:
                    tokens = set_of_all_words1(db, q_type)
                    tokens = tokens.difference(test_queries_part)
                    tokens = tokens.difference(valid_queries_part)
                    train_queries_part = train_queries_part.union(tokens)

                train_queries_part = list(train_queries_part)
                print(
                    f"{n_token = }, {q_type = }, {len(test_queries) = }, {len(valid_queries_part) = }, {len(train_queries_part) = }")
                train_queries.extend(train_queries_part)

        save_query(test_queries, save_dir + "test.txt")
        save_query(valid_queries, save_dir + "valid.txt")
        save_query(train_queries, save_dir + "train.txt")


def copy_training_files_key2key(query_key_in, query_key_out, alg_host="cliqueGenT_wolf1"):
    for dataName in dataNames:
        for dataType in ["valid", "test", "train"]:
            data_key_in = query_key_in + "/" + dataType
            # replace underscore
            file_path_in = f"res/{dataName}/0/data/{data_key_in}/{alg_host}.txt"

            data_key_out = query_key_out + "/" + dataType
            # replace underscore
            file_path_out = f"res/{dataName}/0/data/{data_key_out}/{alg_host}.txt"

            print(f"{file_path_in} -> {file_path_out}")

            os.makedirs(os.path.dirname(file_path_out), exist_ok=True)
            shutil.copy(file_path_in, file_path_out)


def copy_training_files(add_freq=False, add_suffix=False):
    global dataNames
    if add_suffix:
        add_freq = True
    for dataName in dataNames:
        file_path = f"data/{dataName}.txt"
        with open(file_path) as f:
            db = read_strings(filepath=file_path)

        global valid_df
        valid_df = None
        test_df = None

        for dataType in ["valid", "test", "train"]:
            data_key = query_key + "/" + dataType
            # replace underscore
            file_path = f"res/{dataName}/0/data/{data_key}/cliqueGenT_wolf1.txt"
            save_path = f"data/{dataName}/training/{data_key}.txt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"{file_path} {save_path}")

            if dataType == "valid":
                valid_df = pd.read_csv(file_path, header=None)
                # display(valid_df)

            if dataType == "test":
                test_df = pd.read_csv(file_path, header=None)
                # display(test_df)

            if dataType == "train":
                train_df = pd.read_csv(file_path, header=None)
                train_data = list(train_df.values.tolist())

                valid_set = set(valid_df[0].values.tolist())
                test_set = set(test_df[0].values.tolist())
                train_set = set(train_df[0].values.tolist())
                exclude_set = valid_set.union(test_set)
                exclude_set = exclude_set.union(train_set)

                # display(train_df)

                if add_freq:
                    for q_type in q_types:
                        instances = training_instances_of_all_words1(
                            db, q_type, add_suffix=add_suffix)
                        # print(instances[0])
                        for instance in instances:
                            query, card = instance
                            if query not in exclude_set:
                                train_data.append(instance)
                        # tokens = tokens.difference(test_queries_part)
                        # tokens = tokens.difference(valid_queries_part)
                        # train_queries_part = train_queries_part.union(tokens)
                new_train_df = pd.DataFrame(train_data)
                display(new_train_df)
                new_train_df.to_csv(save_path, header=None, index=False)

            else:
                shutil.copy(file_path, save_path)


def copy_packed_files_from_res_to_data(query_key, alg_host="cliqueGenT_wolf1"):
    global dataNames
    for dataName in dataNames:
        data_key_in = query_key
        # replace underscore
        file_path_in = f"res/{dataName}/0/data/{data_key_in}/{alg_host}.txt"

        data_key_out = query_key
        # replace underscore
        file_path_out = f"data/{dataName}/training/{data_key_out}.txt"

        print(f"{file_path_in} -> {file_path_out}")

        os.makedirs(os.path.dirname(file_path_out), exist_ok=True)
        shutil.copy(file_path_in, file_path_out)


def generate_query():
    for dataName in dataNames:
        file_path = f"data/{dataName}.txt"
        with open(file_path) as f:
            lines = read_strings(filepath=file_path)
        print(f"[{dataName:9s}] n_str: {len(lines)}", end="")

        patterns = general_query_pat2(
            lines, max_n_token, max_n_under, seed=seed)
        print(patterns[:3])
        # max_n_under = default_max_n_under
        # for n_token in range(1, max_n_token+1):
        #     patterns = general_query_pat(lines, n_token, max_n_under=max_n_under, seed=seed)
        #     assert len(patterns) == len(set(patterns))
        #     print(f", n_qry: {len(patterns)}, max_l_q: {max([len(x) for x in patterns])}", end="")
        #     print(patterns[:3])

        # replace underscore
        save_path = f"data/{dataName}/query/{query_key}.txt"
        save_query(patterns, save_path)


def gen_ru():
    set_global_ru_common()
    # generate_query()
    # save_test_query(dataNames)
# gen_ru()


def gen_ru2():
    set_global_ru_common(query_key_="ru2/2_2")
    generate_query_uniformly(add_short=True)


def gen_ru3():
    set_global_ru_common("ru3/2_2")
    generate_query_uniformly(add_freq=False, add_short=True)


def gen_ru4():
    set_global_ru_common("ru4/2_2")
    # global dataNames
    # dataNames = ["DBLP"]
    # generate_query_uniformly(add_freq=False, add_short=True)
    copy_training_files(add_freq=True, add_suffix=True)


def gen_join_m_n(m, n):
    query_key = f"join/{m}/{n}"
    set_global_ru_common(query_key)
    global dataNames
    dataNames = ["WIKI", "IMDB", "DBLP"]
    generate_query_m_n(m, n)


def gen_ru100():
    set_global_ru_common("ru100/2_2")
    generate_query_uniformly(add_short=False)
    # copy_training_files()


def gen_ru101():
    set_global_ru_common("ru101/2_2")
    copy_training_files_key2key("ru100/2_2", "ru101/2_2")
    copy_training_files(add_freq=True)


def gen_ru102():
    set_global_ru_common("ru102/2_2")
    copy_training_files_key2key("ru100/2_2", "ru102/2_2")
    copy_training_files(add_freq=True, add_suffix=True)


def gen_ru110():
    global dataNames
    set_global_ru_common("ru110/2_2")

    # dataNames = ["WIKI", "IMDB", "DBLP", "AUTHOR", "GENE"]
    # dataNames = ["AUTHOR"]
    add_additional_query_to_train(add_short=True, add_word=False)
    # generate_query_uniformly(add_short=True)
    # copy_training_files()


def gen_ru111():
    set_global_ru_common("ru111/2_2")
    copy_training_files_key2key("ru110/2_2", "ru111/2_2")
    copy_training_files(add_freq=True)


def gen_ru112():
    set_global_ru_common("ru112/2_2")
    copy_training_files_key2key("ru110/2_2", "ru112/2_2")
    copy_training_files(add_freq=True, add_suffix=True)


def gen_packed(data_name, query_key, ptype, is_force):
    # aug_type = query_key.split('/')[0]
    set_global_ru_common(data_name, query_key)
    # print(f"{aug_type = }")
    gen_packed_queries(ptype=ptype, is_force=is_force)


def gen_core_packed(query_key, is_force):
    gen_packed(query_key, Ptype.CORE, is_force)


def gen_simple_packed(data_name, query_key, is_force):
    gen_packed(data_name, query_key, Ptype.SIMPLE, is_force)


def gen_normal_packed(query_key, is_force):
    gen_packed(query_key, Ptype.NORMAL, is_force)
