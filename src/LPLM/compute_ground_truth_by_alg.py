import os
import sqlite3
import tqdm
import argparse
import csv
import pandas as pd
import time
from src.LPLM.extension import *
# from misc_utils import canonicalize_like_query, get_corelikepatterns

is_debug = False
is_tc_card = False
gather_path = None
dname = None
query_key = None


def parse_like_query(qry, split_beta=False):
    parsed = []
    curr = qry[0]
    is_wild = curr == "_" or curr == "%"
    for ch in qry[1:]:
        if ch == "_" or ch == "%":
            if is_wild:
                curr += ch
            else:
                parsed.append(curr)
                is_wild = True
                curr = ch
        else:
            if is_wild:
                parsed.append(curr)
                is_wild = False
                curr = ch
            else:
                curr += ch
    parsed.append(curr)
    if split_beta:
        parsed_bak = parsed
        parsed = []
        for token in parsed_bak:
            if "%" in token or "_" in token:
                parsed.append(token)
            else:
                parsed.extend(list(token))
    return parsed


def canonicalize_like_query(qry, is_last_flip=False):
    parsed = parse_like_query(qry)

    out_tokens = []
    n_alpha = 0
    for token in parsed:
        if "_" in token or "%" in token:
            n_alpha += 1
            if "%" in token:
                new_token = "%"
            else:
                new_token = ""
            new_token += "_" * token.count("_")
            out_tokens.append(new_token)
        else:
            out_tokens.append(token)
    if is_last_flip and n_alpha > 1 and "%" in out_tokens[-1]:
        out_tokens[-1] = "_" * (len(out_tokens[-1]) - 1) + "%"
    return "".join(out_tokens)


def return_true_cardinality(query, c):
    count = c.execute(
        'SELECT count(*) FROM pattern WHERE trans LIKE ?', (query,)).fetchall()[0][0]
    return count


def return_cardinality_load(query_list, card_dict, dataset_size):
    if len(query_list) == 1:
        cn = card_dict[canonicalize_like_query(query_list[0])]
        prob = cn / dataset_size
        return prob
    else:
        cn = card_dict[canonicalize_like_query(query_list[0])]

        cn1 = card_dict[canonicalize_like_query(query_list[1])]
        if cn1 == 0:
            print(f"Error: Like pattern actual card is {0}")
        prob = float(cn) / cn1
        return prob


def return_cardinality(query_list, c, dataset_size):
    if len(query_list) == 1:
        cn = c.execute('SELECT count(*) FROM pattern WHERE trans LIKE ?',
                       (query_list[0],)).fetchall()[0][0]
        prob = cn / dataset_size
        return prob
    else:
        cn = c.execute('SELECT count(*) FROM pattern WHERE trans LIKE ?',
                       (query_list[0],)).fetchall()[0][0]

        cn1 = c.execute('SELECT count(*) FROM pattern WHERE trans LIKE ?',
                        (query_list[1],)).fetchall()[0][0]
        if cn1 == 0:
            print(f"Error: Like pattern actual card is {0}")
        prob = float(cn) / cn1
        return prob


def find_all_possible_probabilities(like, all_con_prob_list):
    wildcard_list = ['$', '^']
    if len(like) == 0:
        return all_con_prob_list
    elif len(like) == 1:
        all_con_prob_list.append(('%' + like[-1] + '%',))
        return all_con_prob_list
    else:
        if like[-1] not in wildcard_list:
            if like[-2] not in wildcard_list:
                all_con_prob_list.append(
                    ('%' + like + '%', '%' + like[:-1] + '%'))
                return find_all_possible_probabilities(like[:-1], all_con_prob_list)
            else:
                if len(like) > 2:
                    if like[-2] == '^':
                        all_con_prob_list.append(
                            ('%' + like[:-2] + '^' + like[-1] + '%', '%' + like[:-2] + '%' + like[-1] + '%'))
                        all_con_prob_list.append(
                            ('%' + like[:-2] + '%' + like[-1] + '%', '%' + like[:-2] + '%'))

                        return find_all_possible_probabilities(like[:-2], all_con_prob_list)
                    elif like[-2] == '$':
                        all_con_prob_list.append(
                            ('%' + like + '%', '%' + like[:-2] + '%' + like[-1] + '%'))
                        all_con_prob_list.append(
                            ('%' + like[:-2] + '%' + like[-1] + '%', '%' + like[:-2] + '%'))

                        return find_all_possible_probabilities(like[:-2], all_con_prob_list)
                else:
                    all_con_prob_list.append(
                        ('^' + like[-1] + '%', '%' + like[-1] + '%'))
                    all_con_prob_list.append(('%' + like[-1] + '%',))

                    return find_all_possible_probabilities('', all_con_prob_list)
        else:
            if len(all_con_prob_list) == 0:
                all_con_prob_list.append(
                    ('%' + like[:-1] + '^', '%' + like[:-1] + '%'))
                return find_all_possible_probabilities(like[:-1], all_con_prob_list)


def language_to_query(list_languages):
    list_queries = []
    list_wildcards = ['$', '%', '_', '@']
    for con in list_languages:
        list_pairs = []
        for l in con:
            query = ''
            l_ = l.replace('%^', '_').replace(
                '^%', '_').replace('^', '_').replace('@', ' ')
            for i in range(len(l_)):
                if len(query) == 0:
                    query += l_[i]
                else:

                    if query[-1] not in list_wildcards and l_[i] not in list_wildcards:
                        query += '%' + l_[i]
                    else:
                        query += l_[i]
            list_pairs.append(query.replace('$', ''))
        list_queries.append(list_pairs)
    return list_queries


def LIKE_pattern_to_newLanguage(liste):
    transformed_pattern = ''
    for key in liste:
        if len(key) == 1:
            transformed_pattern += key
        else:
            new = ''
            count = 0
            for char in key:
                if count < 1:
                    new += char
                    count += 1
                else:
                    if new[-1] != '_' and char != '_':
                        new += '$' + char
                    else:
                        new += char

            result = ''
            for ch in new:
                if len(result) == 0:
                    result += ch
                elif result[-1] == '^' and ch == '_':
                    result = result[:-1] + '_^'
                elif ch == '_':
                    result += '^'
                else:
                    result += ch
            new = result

            transformed_pattern += new
    # transformed_pattern = transformed_pattern.replace('_', '^')
    # result = ''
    # for ch in transformed_pattern:
    #     if len(result) == 0:
    #         result += ch
    #     elif result[-1] == '^' and ch == '_':
    #         result = result[:-1] + '_^'
    #     elif ch == '_':
    #         result += '^'
    #     else:
    #         result += ch
    # transformed_pattern = result

    return transformed_pattern


def inject_type(liste, type_):
    newliste = []
    if type_ == 'prefix' or type_ == 'end_underscore':
        liste.insert(-1, [liste[-1][0][1:], liste[-1][0]])
        for i in range(len(liste)):
            if i < len(liste) - 2:
                newliste.insert(0, [liste[i][0][1:], liste[i][1][1:]])
            else:
                newliste.insert(0, liste[i])
        return newliste
    if type_ == 'suffix' or type_ == 'begin_underscore':
        liste.insert(0, [liste[0][0][:-1], liste[0][0]])
        for i in range(len(liste)):
            newliste.insert(0, liste[i])
        return newliste

    elif type_ == 'prefix_suffix':
        liste.insert(-1, [liste[-1][0][1:], liste[-1][0]])
        liste.insert(0, [liste[0][0][:-1], liste[0][0]])
        for i in range(len(liste)):
            if i < len(liste) - 2:
                newliste.insert(0, [liste[i][0][1:], liste[i][1][1:]])
            else:
                newliste.insert(0, liste[i])
        return newliste
    else:
        for i in range(len(liste)):
            newliste.insert(0, liste[i])
        return newliste


def create_db(db, dataset_path):
    if os.path.exists(db):
        # os.remove(db)
        return
    con = sqlite3.connect(db)
    c = con.cursor()
    with open(dataset_path) as f:
        dataset = list(f.readlines())

    cn = c.execute(
        "create table if not exists pattern (trans text)").fetchall()

    for text in dataset:
        text = text.rstrip('\n')
        text = text.replace('"', '""')
        query = f'insert into pattern (trans) values ("{text}");'
        cn = c.execute(query).fetchall()
    c.close()
    con.commit()
    # c.close()

    # cn = c.execute('SELECT count(*) FROM pattern WHERE trans LIKE ?', (dataset[0],)).fetchall()[0][0]


def testcase_cardinality(db, list_of_patterns, path_file_tosave):
    c = sqlite3.connect(db).cursor()
    c.execute("PRAGMA case_sensitive_like = 1;")
    os.makedirs(os.path.dirname(path_file_tosave), exist_ok=True)
    file_to_save = open(path_file_tosave, "w")

    print("start DB")
    start_time = time.time()

    csv_w = csv.writer(file_to_save, lineterminator='\n')
    for likepatterns in tqdm.tqdm(list_of_patterns):
        card = return_true_cardinality(likepatterns, c)
        csv_w.writerow([likepatterns, card])

    end_time = time.time()

    res_path = f"res/time/{dname}_time.txt"
    res_time = end_time - start_time
    os.makedirs(os.path.dirname(res_path), exist_ok=True)

    with open(res_path, "w") as f:
        f.write(f"time: {res_time}\n")


def get_card_dict(card_path):
    print(f"{card_path = }")

    card_dict = {}
    with open(card_path) as f:
        csv_r = csv.reader(f)
        for query, card in csv_r:
            card_dict[query] = int(card)
    return card_dict


def compute_gt_LPLM_main(list_of_patterns, card_path, path_file_tosave, datasetsize, is_force):
    os.makedirs(os.path.dirname(path_file_tosave), exist_ok=True)
    card_dict = get_card_dict(card_path)
    print(f"{path_file_tosave = }")
    if not is_force and os.path.exists(path_file_tosave):
        print(f"Already exists {path_file_tosave = }")
        return
    file_to_save = open(path_file_tosave, "w")
    csv_w = csv.writer(file_to_save)

    for likepatterns in tqdm.tqdm(list_of_patterns):
        language, is_end_esc = LIKE_pattern_to_extendedLanguage(likepatterns)
        # print(f"{likepatterns = }, {language = }")
        liste = []
        prev_card = datasetsize
        for length in range(1, len(language) + 1):
            prefix = language[:length]
            pre_pattern = extendedLanguage_to_LIKE_pattern(
                prefix, is_end_esc and length == len(language))
            # print(f"{prefix = }, {pre_pattern = }")
            card = card_dict[pre_pattern]
            if prev_card == 0:
                assert card == 0, f"{likepatterns, pre_pattern, card = }"
                ratio = 0
            else:
                ratio = card/prev_card
            liste.append(ratio)
            prev_card = card

        # newlike = likepatterns
        # if newlike[0] == "%" and newlike[-1] != "%" and newlike[-1] != "_":
        #     newlike = likepatterns + ":" + "suffix"

        # elif newlike[0] != "%" and newlike[-1] == "%" and newlike[0] != "_":
        #     newlike = likepatterns + ":" + "prefix"

        # elif (
        #     newlike[0] != "%"
        #     and newlike[-1] != "%"
        #     and newlike[0] != "_"
        #     and newlike[-1] != "_"
        # ):
        #     newlike = likepatterns + ":" + "prefix_suffix"

        # elif newlike[0] != "%" and newlike[-1] == "_" and newlike[0] != "_":
        #     newlike = likepatterns + ":" + "end_underscore"

        # elif newlike[-1] != "%" and newlike[0] == "_" and newlike[-1] != "_":
        #     newlike = likepatterns + ":" + "begin_underscore"
        # else:
        #     newlike = likepatterns + ":" + "substring"

        s = [str(k) for k in liste]
        # file_to_save.write(newlike + ":" + " ".join(s) + "\n")
        csv_w.writerow([likepatterns, " ".join(s)])


def load_like_patterns(filename):
    list_of_patterns = []
    with open(filename, "r") as file:
        for line in file:
            list_of_patterns.append(line.strip('\n'))
    return list_of_patterns
