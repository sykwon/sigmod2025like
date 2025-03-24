import time
import numpy as np
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import string as _string
import itertools
import re
import pickle
import torch
import sys
import math
import argparse
import socket
import platform
from packaging import version
from tensorboardX import SummaryWriter
from datetime import datetime
import warnings
import os
import csv
import yaml
import pandas as pd
from typing import Iterable
from collections import defaultdict
from copy import deepcopy
from functools import partial

esc_sc = "\u02C2"  # escape start character
esc_ec = "\u02C3"  # escape end character
esc_ac = "\u203C"  # escape any character
like_query_wildcard = r"""%_[]^-\()"""
regex_query_wildcard = r"$+*.[]|?\()"
punctuations = _string.punctuation + r"""—¨©«®°´»‒‘’‚,“”„•…″"""
min_word_len = 2
max_word_len = 10
# region query pattern gen

_aug_func_dict = None


class Qtype:
    SUBSTR = 0
    PREFIX = 1
    SUFFIX = 2

    @classmethod
    def iter(cls):
        return iter([cls.SUBSTR, cls.PREFIX, cls.SUFFIX])


class Ptype:
    SIMPLE = 0
    CORE = 1
    SIMPLE_B = 2
    CORE_B = 3
    SIMPLE_M = 4
    CORE_M = 5
    NORMAL = 6
    NORMAL_B = 7
    NORMAL_M = 8

    @classmethod
    def iter(cls):
        return iter(
            [
                cls.SIMPLE,
                cls.CORE,
                cls.SIMPLE_B,
                cls.CORE_B,
                cls.SIMPLE_M,
                cls.CORE_M,
                cls.NORMAL,
                cls.NORMAL_B,
                cls.NORMAL_M,
            ]
        )


class passingdict(dict):
    @staticmethod
    def __missing__(key):
        return key


def get_aug_func_dict(trim=False):
    global _aug_func_dict
    _aug_func_dict = {
        "q": None,
        "pr": augment_query_prefix,
        "de": lambda x: augment_query_delimiter(x, False, trim),
        "pl": lambda x: augment_query_placeholder(x, False, trim),
        "gq": None,
        "ru": None,
    }
    return _aug_func_dict


def save_augmented_queries(qrys, aug_func, exp_path):
    os.makedirs(os.path.dirname(exp_path), exist_ok=True)
    aqrys = augment_queries(qrys, aug_func)
    with open(exp_path, "w") as f:
        print(f"n_qry: {len(qrys)}, n_aqry: {len(aqrys)} path:", exp_path)
        for aqry in aqrys:
            f.write(aqry + "\n")
    print(aqrys[:3])


def augment_queries(patterns, aug_func):
    expanded_patterns = set()
    for pattern in patterns:
        for exp_pat in aug_func(pattern):
            expanded_patterns.add(exp_pat)

    return list(sorted(expanded_patterns))


def augment_query_prefix(query):
    aug_queries = []
    for i in range(len(query)):
        aug_queries.append(query[: i + 1])
    return aug_queries


def augment_query_delimiter(query, unique=True, trim=False):
    aug_queries = []
    if trim:
        assert query[0] == "%" and query[-1] != "%"
        query += "%"

    assert query[0] == "%" and query[-1] == "%"
    if unique:
        if query[0] == "%":
            aug_queries.append("%")
    for i in range(len(query)):
        prefix = query[: i + 1]
        if prefix[-1] == "%":
            aug_query = prefix
            if unique:
                continue
        else:
            aug_query = prefix + "%"
        aug_queries.append(aug_query)
    if trim:
        aug_queries = aug_queries[:-1]
    return aug_queries


def augment_query_placeholder(query, unique=True, trim=False):
    aug_queries = []
    if trim:
        assert query[0] == "%" and query[-1] != "%"
        query += "%"
    assert query[0] == "%" and query[-1] == "%"
    query_len = len(query)
    placeholder_list = ["%" if ch == "%" else "_" for ch in query]
    for i in range(query_len):
        placeholder_list[i] = query[i]
        aug_query = "".join(placeholder_list)
        aug_query = normalize_like_query(aug_query, trim)
        if unique:
            if len(aug_queries) == 0 or aug_queries[-1] != aug_query:
                aug_queries.append(aug_query)
        else:
            aug_queries.append(aug_query)
    if trim:
        aug_queries = aug_queries[:-1]

    return aug_queries


def normalize_like_query(query, trim=False):
    #         '_'
    #         / \
    #   '%'   v |
    # 0 --->   1   ---> 2
    #         ^ |
    #         \ /
    #         '%'
    assert query[-1] == "%"
    char_list = list(query)
    state = 0
    for i in reversed(range(len(query))):
        ch = query[i]
        if state == 0:
            if ch == "%":
                state = 1
            else:
                break
        elif state == 1:
            if ch == "%":
                char_list.pop(i)
            elif ch == "_":
                state = 1
            else:
                state = 2
                break

    return "".join(char_list)


def gen_all_like_patterns_with_two_subwords(sentences, max_len, suppress2=False):
    patterns = set()
    for sentence in tqdm(sentences):
        words = sentence.split(" ")
        for word in words:
            for i, j in zip(*np.triu_indices(len(word) + 1, k=1)):
                if (j - i) + 2 > max_len:
                    continue
                pattern = "%" + word[i:j] + "%"
                if len(pattern) <= max_len:
                    patterns.add(pattern)

                if not suppress2:
                    remain_word = word[j:]
                    for m, n in zip(*np.triu_indices(len(remain_word) + 1, k=1)):
                        if (j - i) + (m - n) + 3 > max_len:
                            continue
                        pattern = "%" + word[i:j] + \
                            "%" + remain_word[m:n] + "%"
                        if len(pattern) <= max_len:
                            patterns.add(pattern)

    patterns = sorted(patterns)
    return patterns


def gen_like_pattern_with_two_subwords(
    sentences, max_len, seed=None, repeat=1, **kwargs
):
    random.seed(seed)
    # gen pattern %s1%s2%
    patterns = set()
    phrases = set()
    max_len_subwords = max_len - 3  # three %
    for sentence in sentences:
        words = sentence.split()
        count_long_word = 0
        for i in range(len(words)):
            word = words[i]
            if len(word) > max_len_subwords:
                count_long_word += 1
                phrases.add(word)
        if count_long_word == 0:
            count_long_phrase = 0
            for i in range(len(words) - 1):
                phrase = " ".join(words[i: i + 1])
                if len(phrase) > max_len_subwords:
                    count_long_phrase += 1
                    phrases.add(phrase)

    for word in phrases:
        assert len(word) > max_len_subwords, word
        min_delete_len = len(word) - max_len_subwords  # min_delete_len >= 1
        for _ in range(repeat):
            (i,) = random.sample(
                range(1, max_len_subwords), 1
            )  # select i characters (word[0], word[1], ... , word[i-1])
            (j,) = random.sample(
                range(i, max_len_subwords), 1
            )  # select max_len_subwords - j

            # skip max_delete_len characters (word[i], word[i+1], ... word[i+(max_delete_len-1)])
            j += min_delete_len

            pattern = "%" + word[:i] + "%" + word[j:] + "%"
            patterns.add(pattern)
    patterns = list(sorted(patterns))
    return patterns


def pattern_gen2(sentences, pat_type="sub", max_len=None, **kwargs):
    patterns = set()
    word_tuples = set()
    for sentence in sentences:
        words = sentence.split()
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                word_tuples.add((words[i], words[j]))

    for words in word_tuples:
        word1, word2 = words
        for i in range(len(word1)):
            for j in range(i + 1, len(word1) + 1):
                for k in range(len(word2)):
                    for l in range(k + 1, len(word2) + 1):
                        if (j - i) + (l - k) > max_len:
                            continue
                        pattern = "%" + word1[i:j] + "%" + word2[k:l] + "%"
                        patterns.add(pattern)
    patterns = list(sorted(patterns))
    return patterns


def pattern_gen3(sentences, pat_type="sub", max_len=None, **kwargs):
    patterns = set()
    word_tuples = set()
    for sentence in sentences:
        words = sentence.split()
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                word_tuples.add((words[i], words[j]))

    for words in word_tuples:
        word1, word2 = words
        for i in range(len(word1)):
            for j in range(i + 1, len(word1) + 1):
                for k in range(len(word2)):
                    for l in range(k + 1, len(word2) + 1):
                        if (j - i) + (l - k) + 3 > max_len:
                            continue
                        pattern = "%" + word1[i:j] + "%" + word2[k:l] + "%"
                        patterns.add(pattern)
    patterns = list(sorted(patterns))
    return patterns


def pattern_gen(sentences, pat_type, n_pat=1, max_len=None, **kwargs):
    if n_pat == 2:
        assert pat_type == "sub"
        return pattern_gen3(sentences, "sub", max_len)
    patterns = set()

    strings = set()
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            strings.add(word)

    if pat_type == "sub":
        for string in strings:
            for i in range(len(string)):
                for j in range(i + 1, len(string) + 1):
                    if max_len is not None:
                        if (j - i) > max_len:
                            continue
                    pat = "%" + string[i:j] + "%"
                    patterns.add(pat)
    elif pat_type == "prf":
        for string in strings:
            for i in range(1, len(string) + 1):
                if max_len is not None:
                    if i > max_len:
                        continue
                pat = string[:i] + "%"
                patterns.add(pat)
    elif pat_type == "suf":
        for string in strings:
            for i in range(1, len(string) + 1):
                if max_len is not None:
                    if i > max_len:
                        continue
                pat = "%" + string[-i:]
                patterns.add(pat)
    patterns = list(sorted(patterns))
    return patterns


def trim_words(words, min_word_len, max_word_len):
    words = list(
        filter(
            lambda word: len(word) >= min_word_len
            and len(word) <= max_word_len,
            words,
        )
    )
    return words


def refine_words(words, min_word_len, max_word_len, max_n_under):
    # min_word_len restricts only for words[1:-1]
    # It returns refined words
    # The first word is a prefix of sentence
    # The last word is a suffix of sentence
    # Each word may contain underscores

    def rstrip_prefix(first_word):
        if first_word.count("_") > max_n_under:
            indices = []
            for i in range(len(first_word)):
                if first_word[i] == "_":
                    indices.append(i)
            first_word = first_word[: indices[max_n_under - 1] + 1]
        return first_word

    def lstrip_suffix(last_word):
        if last_word.count("_") > max_n_under:
            indices = []
            for i in range(len(last_word)):
                if last_word[i] == "_":
                    indices.append(i)
            last_word = last_word[indices[-max_n_under]:]
        return last_word

    global like_query_wildcard
    global regex_query_wildcard
    global punctuations

    # replace like query wildcards of words with underscore
    words = [
        "".join(["_" if ch in like_query_wildcard else ch for ch in word])
        for word in words
    ]

    # replace regex wildcards of words with underscore
    words = [
        "".join(["_" if ch in regex_query_wildcard else ch for ch in word])
        for word in words
    ]

    # replace punctuations of words with underscore
    words = [
        "".join(["_" if ch in punctuations else ch for ch in word]) for word in words
    ]

    # replace space of words with underscore
    words = ["".join(["_" if ch == " " else ch for ch in word])
             for word in words]
    assert len(words) > 0, words

    # remove too short or long word

    if len(words) > 1:
        first_word = words[0]
        if len(first_word) > max_word_len:
            first_word = first_word[:max_word_len]
        first_word = rstrip_prefix(first_word)

        words_inter = list(
            filter(
                lambda word: len(word) >= min_word_len
                and len(word) <= max_word_len
                and word.count("_") <= max_n_under,
                words[1:-1],
            )
        )
        last_word = words[-1]
        last_word = lstrip_suffix(last_word)

        if len(last_word) > max_word_len:
            last_word = last_word[-max_word_len:]

        if last_word.count("_") > max_n_under:
            last_word = last_word[last_word.rfind("_", max_n_under - 1):]
        words = [first_word]
        words.extend(words_inter)
        words.append(last_word)
    else:
        word = words[0]
        if len(word) > max_word_len or word.count("_") > max_n_under:
            if len(word) >= max_word_len * 2:
                sp = max_word_len
            else:
                sp = len(word) // 2
            first_word = word[:sp]
            first_word = rstrip_prefix(first_word)
            second_word = word[-sp:]
            second_word = lstrip_suffix(second_word)
            words = [first_word, second_word]

    return words


def filter_words(words, min_word_len=None, max_word_len=None):
    global like_query_wildcard
    global regex_query_wildcard
    global punctuations

    # remove some words having like query wildcard
    words = list(
        filter(lambda word: all(x not in like_query_wildcard for x in word), words)
    )

    # remove some words having regex wildcard
    words = list(
        filter(lambda word: all(x not in regex_query_wildcard for x in word), words)
    )

    # remove some words consisting of only punctuations
    words = list(filter(lambda word: any(
        x not in punctuations for x in word), words))

    # remove starting and ending punctuations
    words = [x.strip(punctuations) for x in words]

    # remove too short or long word
    if min_word_len is not None:
        words = list(filter(lambda word: len(word) >= min_word_len, words))

    if max_word_len is not None:
        words = list(filter(lambda word: len(word) <= max_word_len, words))

    return words


def general_query_pat(
    strings,
    n_token=1,
    max_n_under=0,
    min_word_len=2,
    max_word_len=5,
    seed=None,
    **kwargs,
):
    raise DeprecationWarning

    def gen_operator(n_op, max_n_under):
        # rejection sampling
        while True:
            op_list = []
            for i in range(n_op):
                is_percent = np.random.randint(0, 2)
                if i == 0 or i == n_op - 1:
                    n_under = np.random.randint(0, max_n_under + 1)
                else:
                    if max_n_under == 0:
                        is_percent = 1
                    if is_percent == 0:
                        n_under = np.random.randint(1, max_n_under + 1)
                    else:
                        n_under = np.random.randint(0, max_n_under + 1)

                op = "_" * n_under
                if is_percent:
                    op = "%" + op
                op_list.append(op)
            if sum([len(x) for x in op_list]) > 0:
                break
        return op_list

    np.random.seed(seed)
    qrys = []
    n_op = n_token + 1
    for string in strings:
        words = string.split()
        words = filter_words(words, min_word_len, max_word_len)

        n_word = len(words)
        if n_word < n_token:
            continue
        token_idx = sorted(np.random.choice(
            range(n_word), size=n_token, replace=False))
        tokens = [words[idx] for idx in token_idx]
        op_list = gen_operator(n_op, max_n_under)
        assert len(op_list) == n_op, op_list
        qry = op_list[0] + \
            "".join([x + y for x, y in zip(tokens, op_list[1:])])
        assert "%" in qry or "_" in qry, (qry, op_list, tokens)
        qrys.append(qry)

    qrys = sorted(set(qrys))
    return qrys


def is_wild_string(word):
    return all([(x == "_" or x == "%") for x in word])


def get_num_betas(qry):
    parsed = parse_like_query(qry)
    count = 0
    for word in parsed:
        if not is_wild_string(word):
            count += 1
    return count


def get_join_alphas(qry):
    parsed = parse_like_query(qry)
    if is_wild_string(parsed[-1]):
        parsed = parsed[:-1]
    if is_wild_string(parsed[0]):
        parsed = parsed[1:]

    # print(parsed)

    output = []
    for token in parsed:
        if is_wild_string(token):
            output.append(token)
    return output


def compile_LIKE_query(qry):
    query_pat = qry.replace("%", "(.*?)").replace("_", ".")
    qry_compiled = re.compile(query_pat)
    return qry_compiled


def eval_compiled_LIKE_query(qry_re, rec):
    return qry_re.fullmatch(rec)


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


def flip_last_canonicalized_like_query(qry):
    is_percent = False
    n_under = 0
    for i in reversed(range(len(qry))):
        ch = qry[i]
        if ch == '%':
            is_percent = True
        elif ch == '_':
            n_under += 1
        else:
            break
    if is_percent:
        return qry[:(i+1)] + ('_' * n_under + '%')
    else:
        return qry


def canonicalize_like_query(qry, is_last_flip=False):
    parsed = parse_like_query(qry)

    out_tokens = []
    for token in parsed:
        if "_" in token or "%" in token:
            if "%" in token:
                new_token = "%"
            else:
                new_token = ""
            new_token += "_" * token.count("_")
            out_tokens.append(new_token)
        else:
            out_tokens.append(token)
    if is_last_flip and "%" in out_tokens[-1]:
        out_tokens[-1] = "_" * (len(out_tokens[-1]) - 1) + "%"
    return "".join(out_tokens)


def split_string_remaining_ending_spaces(string):
    words = string.split()
    ending = string[len(string.rstrip()):]
    starting = string[: -len(string.lstrip())]
    # print(starting, len(starting))
    if len(words) > 0:
        if len(starting) > 0:
            words[0] = starting + words[0]
        if len(ending) > 0:
            words[-1] = words[-1] + ending
    else:  # one word consisting of only spaces
        words.append(ending)
    return words


def replace_under(token, max_n_under):
    n_under = token.count("_")
    max_n_under = min(len(token), max_n_under)

    # rejection sampling
    alpha = 2
    assert n_under <= max_n_under
    if max_n_under - n_under == 0:
        return token
    while True:
        n_under_sample = np.random.zipf(alpha) - 1
        if n_under_sample <= (max_n_under - n_under):
            break
    indices = []
    for i in range(len(token)):
        if token[i] != "_":
            indices.append(i)
    replace_idx_list = np.random.choice(indices, n_under_sample)
    ch_list = list(token)
    for idx in replace_idx_list:
        ch_list[idx] = "_"
    token = "".join(ch_list)
    return token


def refine_data_string2words(data_string, sample_q_type, n_token, min_word_len=2, max_word_len=5):
    # if fail return None
    words = split_string_remaining_ending_spaces(data_string)
    if len(words) == 0:
        return
    if sample_q_type == Qtype.SUBSTR:
        words = trim_words(words, min_word_len, max_word_len)
    elif sample_q_type == Qtype.PREFIX:
        prefix = words[0][:max_word_len]
        words = trim_words(words[1:], min_word_len, max_word_len)
        words.insert(0, prefix)
    elif sample_q_type == Qtype.SUFFIX:
        suffix = words[-1][-max_word_len:]
        words = trim_words(words[:-1], min_word_len, max_word_len)
        words.append(suffix)
    if sample_q_type == Qtype.PREFIX and len(words) == 1:
        return words[:1]
    elif sample_q_type == Qtype.SUFFIX and len(words) == 1:
        return words[-1:]

    if len(words) >= n_token:
        return words


def general_query_pat2_gen_query_from_words(
    words, sample_q_type, max_n_under, n_token
):
    n_word = len(words)
    if sample_q_type == Qtype.SUBSTR:
        token_idx = sorted(np.random.choice(
            range(n_word), size=n_token, replace=False))
        tokens = [words[idx] for idx in token_idx]
        tokens = [replace_under(token, max_n_under) for token in tokens]
        qry = "%" + "%".join(tokens) + "%"
        qry = canonicalize_like_query(qry, is_last_flip=True)
        if qry[0] != "%" or qry[-1] != "%":
            return

    elif sample_q_type == Qtype.PREFIX:
        token_idx = sorted(
            np.random.choice(range(1, n_word), size=n_token - 1, replace=False)
        )
        tokens = [words[0]]
        tokens.extend([words[idx] for idx in token_idx])
        tokens = [replace_under(token, max_n_under) for token in tokens]
        qry = "%".join(tokens) + "%"
        qry = canonicalize_like_query(qry, is_last_flip=True)
        if qry[-1] != "%":
            return

    elif sample_q_type == Qtype.SUFFIX:
        token_idx = sorted(
            np.random.choice(range(n_word - 1),
                             size=n_token - 1, replace=False)
        )
        tokens = [words[idx] for idx in token_idx]
        tokens.append(words[-1])
        tokens = [replace_under(token, max_n_under) for token in tokens]
        qry = "%" + "%".join(tokens)
        qry = canonicalize_like_query(qry, is_last_flip=True)
        if qry[0] != "%":
            return
    else:
        raise ValueError(sample_q_type)

    op_tokens, tokens = wild_string_tokenizer(
        canonicalize_like_query(qry, True), "%")
    if len(tokens) != n_token:
        return
    qry = canonicalize_like_query(qry)

    return qry


def general_query_pat2_gen_query(
    string, sample_q_type, max_n_under, n_token, min_word_len=2, max_word_len=5,
    is_refine_words=False
):
    # words = string.split()

    words = split_string_remaining_ending_spaces(string)

    if len(words) == 0:
        return
    words_bak = words
    if is_refine_words:
        words = refine_words(words, min_word_len, max_word_len, max_n_under)
    else:
        if sample_q_type == Qtype.SUBSTR:
            words = trim_words(words, min_word_len, max_word_len)
        elif sample_q_type == Qtype.PREFIX:
            prefix = words[0][:max_word_len]
            words = trim_words(words[1:], min_word_len, max_word_len)
            words.insert(0, prefix)
        elif sample_q_type == Qtype.SUFFIX:
            suffix = words[-1][-max_word_len:]
            words = trim_words(words[:-1], min_word_len, max_word_len)
            words.append(suffix)
    if not all([x.count("_") <= max_n_under for x in words]):
        return
    assert all([x.count("_") <= max_n_under for x in words]
               ), (string, words_bak, words)

    if sample_q_type == Qtype.SUBSTR:
        words_bak = words
        words = []
        for word in words_bak:
            if len(word) > 0:
                words.append(word)
    elif sample_q_type == Qtype.PREFIX:
        if len(words[0]) == 0:
            return
    elif sample_q_type == Qtype.SUFFIX:
        if len(words[-1]) == 0:
            return

    n_word = len(words)
    if n_word < n_token:
        return

    if sample_q_type == Qtype.SUBSTR:
        token_idx = sorted(np.random.choice(
            range(n_word), size=n_token, replace=False))
        tokens = [words[idx] for idx in token_idx]
        tokens = [replace_under(token, max_n_under) for token in tokens]
        qry = "%" + "%".join(tokens) + "%"
        qry = canonicalize_like_query(qry, is_last_flip=True)
        if qry[0] != "%" or qry[-1] != "%":
            return

    elif sample_q_type == Qtype.PREFIX:
        token_idx = sorted(
            np.random.choice(range(1, n_word), size=n_token - 1, replace=False)
        )
        tokens = [words[0]]
        tokens.extend([words[idx] for idx in token_idx])
        tokens = [replace_under(token, max_n_under) for token in tokens]
        qry = "%".join(tokens) + "%"
        qry = canonicalize_like_query(qry, is_last_flip=True)
        if qry[-1] != "%":
            return

    elif sample_q_type == Qtype.SUFFIX:
        token_idx = sorted(
            np.random.choice(range(n_word - 1),
                             size=n_token - 1, replace=False)
        )
        tokens = [words[idx] for idx in token_idx]
        tokens.append(words[-1])
        tokens = [replace_under(token, max_n_under) for token in tokens]
        qry = "%" + "%".join(tokens)
        qry = canonicalize_like_query(qry, is_last_flip=True)
        if qry[0] != "%":
            return
    else:
        raise ValueError(sample_q_type)

    op_tokens, tokens = wild_string_tokenizer(
        canonicalize_like_query(qry, True), "%")
    if len(tokens) != n_token:
        return
    qry = canonicalize_like_query(qry)

    return qry


def general_query_pat2(
    strings,
    max_n_token=2,
    max_n_under=2,
    min_word_len=2,
    max_word_len=5,
    sub_r=0.5,
    pre_r=0.25,
    suf_r=0.25,
    seed=None,
    **kwargs,
):
    np.random.seed(seed)
    assert sub_r + pre_r + suf_r == 1, (sub_r, pre_r, suf_r)
    q_ratio = [sub_r, pre_r, suf_r]
    sample_q_types = np.random.choice(
        3, (len(strings), max_n_token), p=q_ratio)

    qrys = []
    for sid, string in tqdm(enumerate(strings), total=len(strings)):
        for n_token in range(1, max_n_token + 1):
            sample_q_type = sample_q_types[sid][n_token - 1]
            qry = general_query_pat2_gen_query(
                string, sample_q_type, max_n_under, n_token
            )
            if qry is not None:
                qrys.append(qry)
    qrys = sorted(set(qrys))
    return qrys


def query_n_pat(strings, max_word_len, max_n_pat=1, seed=None, **kwargs):
    np.random.seed(seed)
    qrys = []
    for string in strings:
        words = string.split()
        words = filter_words(words, min_word_len, max_word_len)

        n_word = len(words)
        if n_word == 0:
            continue
        n_pat = np.random.randint(1, min(max_n_pat, n_word) + 1)
        pat_idx = sorted(np.random.choice(
            range(n_word), size=n_pat, replace=False))
        pats = [words[idx] for idx in pat_idx]
        qry = "%" + "%".join(pats) + "%"
        qrys.append(qry)

    qrys = sorted(set(qrys))
    return qrys


# endregion

# region (load|stat|preproc|measure)


def keras_pad_sequences(
    sequences, maxlen=None, dtype="int32", padding="pre", truncating="pre", value=0.0
):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError(
                "`sequences` must be a list of iterables. "
                "Found non-iterable: " + str(x)
            )

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(
        dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError(
            "`dtype` {} is not compatible with `value`'s type: {}\n"
            "You should set `dtype=object` for variable length strings.".format(
                dtype, type(value)
            )
        )

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]
        else:
            raise ValueError(
                'Truncating type "%s" ' "not understood" % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                "Shape of sample %s of sequence at position %s "
                "is different from expected shape %s"
                % (trunc.shape[1:], idx, sample_shape)
            )

        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def compute_n_placeholders(string):
    string = string[1:] + "a"
    res = list(np.cumsum([x != "%" for x in reversed(string)])[::-1])
    return res


def get_union_char_set(db, alpha_list=None):
    special_chars = ["[PAD]", "[UNK]", "%", "_", "[", "]", "^", "-"]
    if alpha_list is not None:
        special_chars = ["[PAD]", "[UNK]"]
        special_chars.extend(alpha_list)
        special_chars.extend(["[", "]", "^", "-"])
    char_set_db = char_set_from_db(db, sort_by_freq=True)
    for special_char in special_chars:
        # assert special_char not in char_set_db
        if special_char in char_set_db:
            char_set_db.remove(special_char)

    char_set = special_chars + list(char_set_db)
    n_special_char = len(special_chars)

    return char_set, n_special_char


def train_valid_test_split(input_list, p_test, p_valid, seed, p_train=None):
    # total: 1.0 + p_test + p_valid
    train_valid, test = train_test_split(
        input_list, test_size=p_test / (1.0 + p_test + p_valid), random_state=seed
    )
    train, valid = train_test_split(
        train_valid, test_size=p_valid / (1.0 + p_valid), random_state=seed)
    if p_train is not None:
        train = clip_string_by_its_len(train, p_train)
    return train, valid, test


def train_valid_test_split_test_first(input_list, p_test, p_valid, seed, p_train=None):
    train_valid, test = train_test_split(
        input_list, test_size=p_test, random_state=seed
    )
    train, valid = train_test_split(
        train_valid, test_size=p_valid, random_state=seed)
    if p_train is not None:
        train = clip_string_by_its_len(train, p_train)
    return train, valid, test


def clip_string_by_its_len(x, prob):
    assert isinstance(prob, float)
    if len(x) == 0 or prob == 1:
        return x
    group = {}
    for rec in x:
        query = rec
        if len(query) not in group:
            group[len(query)] = []
        group[len(query)].append(rec)

    output = []
    for length, g in group.items():
        prob2 = max(prob, 1 / len(g))
        selected, discarded = train_test_split(
            g, train_size=prob2, random_state=0)
        output.extend(selected)
    return output


# def annotate_cardinalities_to_query_steps(qry_steps, res_dict, aug=False):
#     data = []
#     for step, qry_step in enumerate(qry_steps, start=1):
#         data.append(annotate_cardinalities_to_queries(qry_step, res_dict, aug=aug, step=step))
#     return data


def annotate_cards_to_queries(queries, res_dict, aug_func=None, trim=False):
    res = []
    for query in queries:
        if aug_func is not None:
            aug_queries = aug_func(query)
            cards = []
            for aug_qry in aug_queries:
                if trim:
                    cards.append(res_dict[aug_qry])
                else:
                    cards.append(res_dict[aug_qry])

        else:
            if trim:
                cards = res_dict[query + "%"]
            else:
                cards = res_dict[query]
        res.append([query, cards])
    return res


# def annotate_cardinalities_to_queries(queries, res_dict, aug=False, step=1):
#     """
#     Parameters
#     ----------
#     """
#     res = []
#     for query in queries:
#         if aug and step == 2:
#             sb_list = query.strip("%").split("%")
#             patterns = []
#             for i in range(len(sb_list)):
#                 pattern = "%" + "%".join(sb_list[:i+1]) + "%"
#                 patterns.append(pattern)
#             cards = [res_dict[pattern] for pattern in patterns]
#             res.append([query, len(query), cards])
#         else:
#             res.append([query, len(query), res_dict[query]])
#     return res


def split_query_in_steps(queries, is_train, n_step):
    """
    Parameters
    ----------
    n_step : # of learning steps

    Examples
    --------
    >>> X = ["%Top%Gun%"]
    >>> split_query_in_steps(X, True, 2)
    >>> [[%Top%], [%Top%Gun%]]
    """
    assert n_step >= 1 and n_step <= 2
    if n_step == 1:
        return queries

    step1 = []
    step2 = []

    if is_train:
        for qry in queries:
            sb_list = qry.strip("%").split("%")
            n_sb = len(sb_list)
            assert n_sb >= 1 and n_sb <= 2
            if n_sb == 1:  # sb1
                step1.append(qry)
            elif n_sb == 2:  # sb2
                step1.append(f"%{sb_list[0]}%")  # sb21
                step2.append(qry)
    else:
        for qry in queries:
            sb_list = qry.strip("%").split("%")
            n_sb = len(sb_list)
            assert n_sb >= 1 and n_sb <= 2
            if n_sb == 1:  # sb1
                step1.append(qry)
                step2.append(qry)
            elif n_sb == 2:  # sb2
                step1.append(f"%{sb_list[0]}%")  # sb21
                step2.append(qry)

    step1 = sorted(set(step1))
    step2 = sorted(set(step2))
    assert len(step1) >= 1

    return [step1, step2]


def save_query(patterns, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        print("path:", save_path)
        for pattern in patterns:
            f.write(pattern + "\n")
    print(patterns[:3])


def get_qc_dict(data_name, workload):
    training_path = f"data/{data_name}/training/{workload}/pack_simple.txt"
    qc_dict = {}
    with open(training_path) as f:
        csv_fr = csv.reader(f, delimiter=",")
        for line in csv_fr:
            pat, card = line
            card = int(card)
            qc_dict[pat] = card
    return qc_dict


def get_cf_dict(data_name):
    file_path = f"data/{data_name}/{data_name}.txt"
    db = read_strings(file_path)
    n = len(db)
    prefix_count = defaultdict(float)
    substr_count = defaultdict(float)
    suffix_count = defaultdict(float)
    occurence_count = defaultdict(float)
    # db = db[:3]
    # print(db)
    for rec in db:
        prefix_count[rec[0]] += 1
        suffix_count[rec[-1]] += 1
        for ch in rec:
            occurence_count[ch] += 1
        for ch in set(rec):
            substr_count[ch] += 1

    def normalize(input_dict):
        for k, v in input_dict.items():
            input_dict[k] = v / n

    normalize(prefix_count)
    normalize(substr_count)
    normalize(suffix_count)
    normalize(occurence_count)

    return [prefix_count, substr_count, suffix_count, occurence_count]


def gen_char_dict(db):
    char_set, n_special_char = get_union_char_set(db)
    char_dict = char_dict_from_char_set(char_set)
    return char_dict, n_special_char


def get_training_data(
    data_name,
    workload,
    p_train=1.0,
):
    db, train_data, valid_data, test_data = load_training_files(
        data_name, workload)
    char_dict, n_special_char = gen_char_dict(db)
    train_data = clip_string_by_its_len(train_data, p_train)

    return char_dict, n_special_char, train_data, valid_data, test_data


def get_training_data_AstridEach(
    data_name,
    p_train=1.0,
):
    train_data_triple, valid_data_triple, test_data_triple = load_training_files_AstridEach(
        data_name)
    train_data_triple = [clip_string_by_its_len(
        x, p_train) for x in train_data_triple]

    return train_data_triple, valid_data_triple, test_data_triple


def string_encoding(strings, char_dict):
    output = []
    for string in strings:
        encoded = [char_dict[x] for x in string]
        output.append(encoded)
    return output


def indices_decoding(indices, char_dict):
    output = ""
    char_set = {v: k for k, v in char_dict.items()}
    for index in indices:
        output += char_set[index]
    return output


def char_frequecy_from_db(db, char_set=None):
    if char_set is None:
        char_set = char_set_from_db(db)

    char_freq = dict()
    for rid, record in enumerate(db):
        for char in record:
            if char not in char_freq:
                char_freq[char] = 0
            char_freq[char] += 1
    return char_freq


def alpha_list_from_queries(queries):
    char_set = set()
    for query in queries:
        for char in parse_like_query(query):
            if "%" in char or "_" in char:
                char_set.add(char)
    return sorted(char_set)


def char_set_from_db(db, sort_by_freq=False):
    # """

    # :param db:
    # :param max_char: if max_char is None, return all characters. Otherwise, return at most max_char characters by their
    #     frequencies
    # :return:
    # """
    # if max_char:
    #     char_dict = dict()
    #     for rid, record in enumerate(db):
    #         for char in record:
    #             if char not in char_dict:
    #                 char_dict[char] = 0
    #             char_dict[char] += 1
    #     # frequency order
    #     char_set = list(zip(*sorted(char_dict.items(), key=lambda x: x[1], reverse=True)[:max_char]))[0]
    # else:
    char_set = set()
    for record in db:
        for char in record:
            char_set.add(char)
    char_set = sorted(char_set)  # alphabetical order
    if sort_by_freq:
        char_freq = char_frequecy_from_db(db)
        # print(f"{char_set[:10] = }")
        # char_set = list(zip(*sorted(char_freq.items(), key=lambda x: x[1], reverse=True)))[0]
        char_set = sorted(char_freq.keys(),
                          key=lambda x: char_freq[x], reverse=True)
        # print(f"{char_set[:10] = }")

    return char_set


def char_dict_from_db(db):
    char_set, n_special_char = get_union_char_set(db)
    char_dict = char_dict_from_char_set(char_set)
    return char_dict


def char_dict_from_char_set(char_set):
    # """

    # Args:
    #     db: list of strings

    # Returns:
    #     dictionary whose key and value are character and index, respectively.
    #     The index 0 is kept for [PAD] token.
    #     The index 1 is also kept for [UNK] token.
    # """
    # char_set = char_set_from_db(db)
    # char_dict = dict()
    # for i, char in enumerate(char_set):
    #     char_dict[char] = i + 3  # for [PAD], [UNK], [%]
    # return char_dict
    char_dict = dict()
    for i, char in enumerate(char_set):
        char_dict[char] = i

    return char_dict


def compare_two_dictionaries(dict1, dict2):
    dict1_keys = set(dict1.keys())
    dict2_keys = set(dict2.keys())
    if len(dict1_keys) != len(dict2_keys):
        return False
    if dict1_keys != dict2_keys:
        return False
    for key in dict1_keys:
        if dict1[key] != dict2[key]:
            return False
    return True


def mean_Q_error(input, target, reduction="mean", mask=None):
    if not isinstance(input, Iterable):
        input = [input]
    if not isinstance(target, Iterable):
        target = [target]
    input = np.array(input, dtype=float)
    target = np.array(target, dtype=float)

    input[input < 1.0] = 1.0
    target[target < 1.0] = 1.0
    q_errs = np.max(np.array([input / target, target / input]), axis=0)
    if mask is not None:
        q_errs *= mask

    if reduction == "mean":
        if mask is not None:
            output = np.sum(q_errs) / np.sum(mask)
        else:
            output = float(np.mean(q_errs))
    elif reduction == "max":
        output = np.max(q_errs)
    else:
        output = q_errs
    return output


def batch2cuda(batch, to_cuda=True):
    # ys should locate at the last position
    # only tensor to cuda
    xs = []
    ys = batch[-1]
    for i in range(len(batch) - 1):
        elem = batch[i]
        if isinstance(elem, torch.Tensor) and to_cuda:
            elem = elem.cuda()
        xs.append(elem)
    xs = tuple(xs)
    if isinstance(ys, torch.Tensor) and to_cuda:
        ys = ys.cuda()
    return xs, ys


def read_training(training_path):
    queries = []
    with open(training_path) as f:
        csv_fr = csv.reader(f, delimiter=",")
        for line in csv_fr:
            pat, card = line
            card = int(card)
            # res_dict[pat] = int(card)
            queries.append((pat, card))
    return queries


def read_training_SED(training_path, delta=3):
    queries = []
    with open(training_path) as f:
        csv_fr = csv.reader(f, delimiter=",")
        for id, line in enumerate(csv_fr):
            if id == 0:
                continue
            pat = line[0]
            cards = [int(x) for x in line[1: delta + 2]]
            # res_dict[pat] = int(card)
            queries.append((pat, *cards))
    return queries


def read_strings(filepath):
    lines = []
    with open(filepath) as f:
        for line in f:
            lines.append(line.rstrip('\n'))
    return lines


def load_training_files(data_name, workload):
    # delta for samplingD
    training_files = []
    db = []
    with open(f"data/{data_name}/{data_name}.txt") as f:
        for line in f.readlines():
            db.append(line.rstrip("\n"))

    queries_all = []
    for d_type in ["train", "valid", "test"]:
        training_path = f"data/{data_name}/training/{workload}/{d_type}.txt"
        print(f"{d_type = :10s}{training_path = }")
        queries = read_training(training_path)
        queries_all.append(queries)

        training_files = [db, *queries_all]

    return training_files


def load_training_files_AstridEach(data_name):
    # delta for samplingD
    training_files = []
    for d_type in ["train", "valid", "test"]:
        data_triple = []
        for fn_desc in ["prefix", "suffix", "substring"]:
            training_path = f"data/{data_name}/training/Astrid/{fn_desc}/{d_type}.txt"
            print(f"{d_type = :10s}{training_path = }")
            queries = read_training(training_path)
            data_triple.append(queries)

        training_files.append(data_triple)

    return training_files


def pattern_with_q_type2query_string(pattern, q_type):
    if q_type == Qtype.SUBSTR:
        return "%" + pattern + "%"
    elif q_type == Qtype.PREFIX:
        return pattern + "%"
    elif q_type == Qtype.SUFFIX:
        return "%" + pattern
    else:
        raise ValueError


def find_like_query_type_idx(query):
    query = canonicalize_like_query(query, is_last_flip=True)
    wilds, norms = wild_string_tokenizer(query, wilds="%")

    n_norm = len(norms)

    if "%" in wilds[0] and "%" in wilds[-1]:
        return (n_norm - 1) * 3
    elif "%" in wilds[-1]:
        return (n_norm - 1) * 3 + 1
    elif "%" in wilds[0]:
        return (n_norm - 1) * 3 + 2
    else:
        raise TypeError(wilds, norms)


def wild_string_tokenizer(qry, wilds="_%"):
    tokens = []
    op_tokens = []
    ch = qry[0]
    is_op = any([ch == wilds for wilds in wilds])
    start = 0
    length = 0
    if not is_op:
        op_tokens.append("")

    for i, ch in enumerate(qry):
        if any([ch == wilds for wilds in wilds]):
            if is_op:
                length += 1
            else:
                token = qry[start: start + length]
                tokens.append(token)
                start = i
                length = 1
                is_op = True
        else:
            if is_op:
                token = qry[start: start + length]
                op_tokens.append(token)
                start = i
                length = 1
                is_op = False
            else:
                length += 1

    token = qry[start: start + length]
    if is_op:
        op_tokens.append(token)
    else:
        tokens.append(token)
        op_tokens.append("")

    return op_tokens, tokens


def merge_positional_lists(pl1, pl2, w):
    """
    For (id1, pos1) in pl1 and (id2, pos2) in pl2,
    if id1==id2 and (pos1 + w - 1) < pos2


    Parameters
    ----------
    pl1 : list of (id, list of pos)
        first positional inverted list
    pl2 : list of (id, list of pos)
        second positional inverted list
    w : int
        length of substring of pl1

    Return
    ------

    list of (id, list of pos)
        merged positional inverted list

    """
    ml = []  # merge list

    idx1 = 0
    idx2 = 0

    while idx1 < len(pl1) and idx2 < len(pl2):
        id1, pos_list1 = pl1[idx1]
        id2, pos_list2 = pl2[idx2]
        if id1 == id2:
            pos1 = pos_list1[0]
            for i, pos2 in enumerate(pos_list2):
                if pos1 + w - 1 < pos2:
                    ml.append((id2, pos_list2[i:]))
                    break
            idx1 += 1
            idx2 += 1
        elif id1 < id2:
            idx1 += 1
        else:
            idx2 += 1

    return ml


def human_readable_size(size, decimal_places=2):
    # for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if size < 1024.0 or unit == "PiB":
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def get_pickle_size(py_obj):
    # bytes
    return len(pickle.dumps(py_obj))


def get_torch_model_size(model):
    # bytes
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size


def find_parity_bool_array(arr, mask=None):
    arr_i = arr.reshape(-1, 1)
    arr_j = arr.reshape(1, -1)
    if mask is not None:
        return (arr_i < arr_j)[mask]
    else:
        return arr_i < arr_j


def get_non_equal_mask(arr):
    arr_i = arr.reshape(-1, 1)
    arr_j = arr.reshape(1, -1)
    mask = arr_i != arr_j
    return mask


def parity_score(c_true, c_est):
    mask = get_non_equal_mask(c_true)
    p_true = find_parity_bool_array(c_true, mask)
    p_est = find_parity_bool_array(c_est, mask)
    matched_arr = p_true == p_est

    matched = matched_arr.sum()
    total = len(matched_arr)
    return matched / total


def save_estimator_stat(
    stat_path, q_errs, build_time=None, est_time=None, model_size=None, count_size=None, neural_size=None,
):
    os.makedirs(os.path.dirname(stat_path), exist_ok=True)
    out_stat_dict = {}
    out_stat_dict["time"] = build_time
    out_stat_dict["time_est"] = est_time
    out_stat_dict["model_size"] = model_size
    out_stat_dict["avg"] = float(np.mean(q_errs))
    if count_size is not None:
        out_stat_dict["count_size"] = count_size
    if neural_size is not None:
        out_stat_dict["neural_size"] = neural_size

    for percent in range(1, 101):
        out_stat_dict[f"q{percent:03d}"] = float(
            np.percentile(q_errs, percent))
    with open(stat_path, "w") as f:
        yaml.dump(out_stat_dict, f)
    print(f"{out_stat_dict['avg'] = }, {out_stat_dict['q100'] = }")
    print(f"{stat_path = }")


def split_string_by_length(string, length):
    return [
        string[i * length: (i + 1) * length]
        for i in range(math.ceil(len(string) / length))
    ]


def bisection_method(
    func, y_target, s_init=0.1, tolerance=1e-6, max_iterations=100, verbose=False
):
    assert s_init > 0
    assert y_target > 0
    f_s = func(s_init)
    if f_s > y_target:
        s, e = 0, s_init
    elif f_s == y_target:
        return s
    else:
        # Finding the second interval endpoint b automatically for a monotonically increasing function
        e = s_init
        f_e = func(e)
        iteration = 0
        x_of_max_y = 0
        max_y = 0
        max_n_faliures = 5
        n_faliure = 0
        while (
            f_e < y_target and n_faliure < max_n_faliures and iteration < max_iterations
        ):
            s = e
            e *= 2  # Doubling b until func(e) > target
            f_e = func(e)
            if f_e > max_y:
                n_faliure = 0
                max_y = f_e
                x_of_max_y = e
            else:
                n_faliure += 1
            iteration += 1
            if verbose:
                print(f"{iteration = }, {e = }, {f_e = }, {n_faliure = }")
        if func(e) < y_target:  # faile to find func(x) > target
            return x_of_max_y

    iteration = 0
    m_prev = e
    m = (s + e) / 2
    f_m = func(m)
    if f_m == y_target:
        return m  # Found exact root
    if f_m > y_target:
        e = m
    else:
        s = m
    while (
        f_m > y_target
        or (f_m < y_target * (1 - tolerance) and (abs(m - m_prev) > 1e-6))
    ) and iteration < max_iterations:
        m_prev = m
        m = (s + e) / 2
        f_m = func(m)
        if f_m == y_target:
            return m  # Found exact root
        if f_m > y_target:
            e = m
        else:
            s = m
        if verbose:
            print(
                f"{iteration = }, {y_target = }, {f_m = }, {m = :.6f}, {m_prev = :.6f}"
            )
        iteration += 1

    return m


def get_rel_path_for_symlink(src_path, target_path):
    target_path = os.path.dirname(target_path)
    common_path = os.path.commonpath([src_path, target_path])
    src_remain = src_path[len(common_path):].strip(os.path.sep)
    target_remain = target_path[len(common_path):].strip(os.path.sep)
    # os.path.split(target_remain)
    if len(target_remain) == 0:
        back_pathes = []
    else:
        back_pathes = [".."] * len(target_remain.split(os.path.sep))
    forward_pathes = src_remain.split(os.path.sep)
    rel_path = os.path.join(*back_pathes, *forward_pathes)
    # print(f"{common_path = }, {src_remain = }, {target_remain = }")
    # print(f"{back_pathes = }, {forward_pathes = }")
    # print(f"{common_path = }, {src_remain = }, {target_remain = }, {rel_path = }")
    return rel_path


def save_rel_symlink(src_path, target_path):
    rel_path = get_rel_path_for_symlink(src_path, target_path)
    if os.path.islink(target_path):
        os.remove(target_path)
    else:
        assert not os.path.exists(target_path)
    os.symlink(rel_path, target_path)


def add_prefix_augmented(train_data, qc_dict, last_flip=False):
    queries = set()
    output = []
    for query, card in tqdm(train_data):
        query_cano = canonicalize_like_query(query, last_flip)
        for query_len in range(1, len(query) + 1):
            sub_query = query_cano[:query_len]
            sub_query = canonicalize_like_query(sub_query)
            # sub_query = query[:query_len]
            # cards.append(card)
            if sub_query not in queries:
                queries.add(sub_query)
                card = qc_dict[sub_query]
                output.append([sub_query, card])
    return output


# endregion

# region param search


def make_partial(func, target_param, **param_inits):
    if target_param in param_inits:
        del param_inits[target_param]

    partial_func = partial(func, **param_inits)
    # print(f"{func = }")
    # print(f"{param_inits = }")
    # print(f"{partial_func = }")

    return lambda x: partial_func(**{target_param: x})


def param_search_general(target_value, func, target_param, s_init=0.1, **param_inits):
    partial_func = make_partial(func, target_param, **param_inits)
    # print(partial_func)
    # print(partial_func(1.0))

    x = bisection_method(
        partial_func, target_value, tolerance=1e-2, verbose=False, s_init=s_init
    )
    return x


# def param_search_table(target_value, target_param, **param_inits):
#     pass


def param_search_table_max_entry(dname, target_value, N):
    # target_value: the size of table
    # target_param: ME

    # param_inits = {"dname": dname, "N": N, "cached_out": False}

    def table_size(max_entry_ratio):
        hashtable, PTs, build_time = getCachedPrunedExtendedNgramTableMaxEntry(
            dname, N, max_entry_ratio, cached_out=False, verbose=False
        )
        return get_pickle_size([hashtable, PTs])

    max_entry_ratio = param_search_general(
        target_value, table_size, "max_entry_ratio")

    return max_entry_ratio


def get_param_setting_dict(search_dict):
    df = pd.read_csv("configs/comp_df_params.csv", index_col=0)
    searched_idx = None
    for k, v in search_dict.items():
        if searched_idx is None:
            searched_idx = (df[k] == v)
        else:
            searched_idx &= (df[k] == v)
    row = df[searched_idx]
    # print(search_dict)
    # print(row)
    assert len(row) <= 1
    if len(row) == 1:
        row = row.iloc[0]
        return row
    else:
        return None


def set_params_setting_dict(args, param_setting_dict):
    args = vars(args)
    for i in range(1, 3):
        param = param_setting_dict['model_param_name_'+str(i)]
        val = param_setting_dict['model_param_value_'+str(i)]
        if isinstance(val, float):
            val = float(f"{val:.3f}")
        if isinstance(val, np.int_):
            val = int(val)
        # print(f"{param, val, type(param), type(val) = }")

        # print(args, param, val)
        if isinstance(param, str):
            assert isinstance(param, str), param
            param = param.replace('-', '_')
            args[param] = val
    args = argparse.Namespace(**args)
    return args


def add_just_params(args):
    if 'ratio' in args and args.ratio != 1:
        ratio = args.ratio
        args.model_name += f'-{int(ratio*100):03d}'
    else:
        ratio = 1.0

    search_dict = {
        'data': args.data_name,
        'workload': args.workload,
        'model': args.model_name,
        'ratio': ratio
    }

    param_setting_dict = get_param_setting_dict(search_dict)
    if param_setting_dict is None:
        return args
    else:
        args = set_params_setting_dict(args, param_setting_dict)
        return args
# endregion

# region args


def get_parser(default_configs: dict, abbr_dict, choices_dict):
    parser = argparse.ArgumentParser()
    for key, val in default_configs.items():
        parser_key = key.replace("_", "-")
        keyword = "--" + parser_key
        choices = choices_dict[key] if key in choices_dict else None

        keywords = [keyword]
        if key in abbr_dict:
            abbr = "-" + abbr_dict[key]
            keywords.append(abbr)

        if type(val) == bool:
            parser.add_argument(keyword, action="store_true")
            no_keyword = keyword.replace("--", "--no-")
            parser.add_argument(no_keyword, dest=key, action="store_false")
            default_bool_dict = {key: val}
            parser.set_defaults(**default_bool_dict)
        else:
            parser.add_argument(*keywords, default=val,
                                type=type(val), choices=choices)
    return parser


def check_args(args, ignore_list=None):
    args_dict = vars(args)
    if ignore_list is None:
        ignore_list = []
    for key, val in args_dict.items():
        key = f"--{key}".replace("_", "-")
        if key not in ignore_list:
            assert val is not None, f"{key} is not given"


def args2exp_key(args, ignore_list=None, model_name=None):
    args_dict = vars(args)
    exp_key = ""
    # ignore_list = [opt.replace('--', '').replace('-', '_') for opt in ignore_list]
    if ignore_list is None:
        ignore_list = []

    for key in sorted(args_dict.keys()):
        val = args_dict[key]
        key = f"--{key}".replace("_", "-")
        if key not in ignore_list:
            key = "".join([tkn[0].upper() + tkn[1:] for tkn in key.split("_")])
            key = key[0].lower() + key[1:]
            exp_key += f" {key} {val}"
    exp_key = exp_key[1:]
    if model_name is not None:
        exp_key = model_name + " " + exp_key

    return exp_key


def args2exp_name(args, ignore_list=None, model_name=None):
    args_dict = vars(args)
    exp_key = ""
    if ignore_list is None:
        ignore_list = []
    ignore_list = [opt.replace("--", "").replace("-", "_")
                   for opt in ignore_list]

    for key in sorted(args_dict.keys()):
        if key not in ignore_list:
            val = args_dict[key]
            key = "".join([tkn[0].upper() + tkn[1:] for tkn in key.split("_")])
            key = key[0].lower() + key[1:]
            exp_key += f"_{key}_{val}"
    exp_key = exp_key[1:]
    if model_name is not None:
        exp_key = model_name + "_" + exp_key

    return exp_key


def get_parser_with_ignore_opt_list_alg():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    all_subparsers = []
    ignore_opt_list = []

    psql_parser = subparsers.add_parser("psql")
    all_subparsers.append(psql_parser)

    regex_parser = subparsers.add_parser("regex")
    all_subparsers.append(regex_parser)

    inv_parser = subparsers.add_parser("inv")
    all_subparsers.append(inv_parser)

    pinv_parser = subparsers.add_parser("pinv")
    all_subparsers.append(pinv_parser)

    # common parameters
    for subparser in all_subparsers:
        subparser: argparse._ActionsContainer = subparser
        subparser.add_argument("-d", "--dname", type=str, help="data name")
        # subparser.add_argument("-mp", "--max-n-pat", type=str, help="maximum patterns in like queries")
        # subparser.add_argument("-mu", "--max-n-under", type=int, default=-1,
        #                        help="maximum underscores in each operators")
        # subparser.add_argument("-pt", "--pack-type", type=str, help="packed learning type")
        subparser.add_argument("-q", "--query-key", type=str, help="query key")
        subparser.add_argument("-t", "--trial", type=int, help="trial id")

        ow_option = "--overwrite"
        ignore_opt_list.append(ow_option)
        subparser.add_argument(
            "-o",
            ow_option,
            action="store_true",
            help="run this code even if it has already been done",
        )

    if len(sys.argv) <= 1:
        parser.print_help()
        exit()
    model_name = sys.argv[1]
    return parser, ignore_opt_list, model_name


def get_parser_with_ignore_opt_list_model():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    all_subparsers = []
    ignore_opt_list = []

    astrid_parser = subparsers.add_parser("astrid")
    all_subparsers.append(astrid_parser)

    clique_parser = subparsers.add_parser("clique")
    all_subparsers.append(clique_parser)
    clique_parser.add_argument(
        "--prfx", default=False, type=bool, help="seperate model"
    )
    clique_parser.add_argument(
        "--n-rnn", type=int, help="number of encoder layers")

    # common parameters
    for subparser in all_subparsers:
        subparser: argparse._ActionsContainer = subparser
        subparser.add_argument("--dname", type=str, help="data name")
        subparser.add_argument("--p-train", type=float,
                               help="ratio of train data")
        subparser.add_argument("--p-val", type=float,
                               help="ratio of valid data")
        subparser.add_argument("--p-test", type=float,
                               help="ratio of test data")
        subparser.add_argument("--bs", type=int, help="batch size")
        subparser.add_argument("--lr", type=float, help="learning rate")
        subparser.add_argument("--l2", type=float, help="l2 regularizer")
        subparser.add_argument("--clip-gr", type=float,
                               help="cliping gradient")
        subparser.add_argument("--seed", type=int, help="random seed")
        subparser.add_argument(
            "--pred-layer", type=int, help="number of predict layers"
        )
        subparser.add_argument(
            "--pred-hs", type=int, help="hidden size of predict layers"
        )
        subparser.add_argument(
            "--cs", type=int, help="cell size of encode layers")
        subparser.add_argument("--ch-es", type=int,
                               help="character embedding size")
        subparser.add_argument("--max-epoch", type=int, help="maximum epoch")
        subparser.add_argument("--max-l", type=int,
                               help="maximum length of query")
        subparser.add_argument(
            "--repeat",
            type=str,
            help="repeatition of sampling queries from each record",
        )
        subparser.add_argument(
            "--patience", type=int, help="patience for training a neural network"
        )
        subparser.add_argument("--pattern", type=str,
                               help="like query patterns")

        ow_option = "--overwrite"
        ignore_opt_list.append(ow_option)
        subparser.add_argument(
            ow_option,
            action="store_true",
            help="run this code even if it has already been done",
        )

    if len(sys.argv) <= 1:
        parser.print_help()
        exit()
    model_name = sys.argv[1]
    return parser, ignore_opt_list, model_name


def parse_all_model():
    parser, ignore_opt_list, model_name = get_parser_with_ignore_opt_list_model()
    args = parser.parse_args()
    check_args(args, ignore_opt_list)

    exp_name = args2exp_name(args, ignore_opt_list, model_name)
    if len(exp_name) > 255:
        warnings.warn(f"exp_name is too long. [{len(exp_name)}:{exp_name}]")
    exp_key = args2exp_key(args, ignore_opt_list, model_name)

    return model_name, args, exp_name, exp_key


def parse_all_alg():
    parser, ignore_opt_list, alg_name = get_parser_with_ignore_opt_list_alg()
    args = parser.parse_args()
    check_args(args, ignore_opt_list)

    exp_name = args2exp_name(args, ignore_opt_list, alg_name)
    if len(exp_name) > 255:
        warnings.warn(f"exp_name is too long. [{len(exp_name)}:{exp_name}]")
    exp_key = args2exp_key(args, ignore_opt_list, alg_name)

    return alg_name, args, exp_name, exp_key


def get_common_abbr_dict():
    abbr_dict = {
        "data_name": "d",
        "workload": "w",
    }
    return abbr_dict


def get_common_choices_dict():
    choices_dict = {
        "data_name": ["TOY", "DBLP", "IMDB", "WIKI", "GENE", "AUTHOR",
                      "DBLP-AN", "IMDb-AN", "IMDb-MT", "TPCH-PN",
                      "link_type.link", "movie_companies.note",
                      "name.name", "title.title",
                      ],
        "workload": ["TOY", "CLIQUE", "LPLM", "Astrid", "CEB", "LPLM20", "LPLM30"],
    }
    return choices_dict


def get_seed():
    return 0


def get_common_pathes_for_estimator(data_name, model, workload, trial):
    db_path = f"data/{data_name}/{data_name}.txt"
    model_config_path = os.path.join(
        f"res/{data_name}/{workload}/model/{model}/{trial}", f"config.yml"
    )
    model_path = os.path.join(
        f"res/{data_name}/{workload}/model/{model}/{trial}", f"model.pkl")
    est_path = os.path.join(
        f"res/{data_name}/{workload}/estimation/{model}/{trial}", f"estimation.csv")
    stat_path = os.path.join(
        f"res/{data_name}/{workload}/stat/{model}/{trial}", f"stat.yml")
    est_time_path = os.path.join(
        f"res/{data_name}/{workload}/estimation/{model}/{trial}", f"time.csv"
    )

    if "EST" in model:
        model_path = model_path.replace("pkl", "txt")
    elif "Astrid" in model or "E2E" in model or "DREAM" in model or "LPLM" in model or "CLIQUE" in model:
        model_path = model_path.replace("pkl", "pth")
    elif "LBS" in model:
        pass
    else:
        print(model)
        raise ValueError("check model extension")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(est_path), exist_ok=True)
    os.makedirs(os.path.dirname(stat_path), exist_ok=True)

    print(f"{db_path = }")
    print(f"{model_config_path = }")
    print(f"{model_path = }")
    print(f"{est_path = }")
    print(f"{est_time_path = }")
    print(f"{stat_path = }")

    return db_path, model_config_path, model_path, est_path, est_time_path, stat_path


def get_count_path_for_sampling_estimator(dname, model):
    save_count_path = os.path.join(
        f"res/{dname}/estimation/{model}", f"count.csv"
    )
    os.makedirs(os.path.dirname(save_count_path), exist_ok=True)
    print(f"{save_count_path = }")
    return save_count_path


def get_triplet_dirpath_for_Astrid(dname, max_str_size=None):
    triplet_dirpath = os.path.join(f"res/{dname}/triplets")
    if max_str_size is not None:
        triplet_dirpath += f'/ms{max_str_size}'
    os.makedirs(triplet_dirpath, exist_ok=True)
    print(f"{triplet_dirpath = }")
    return triplet_dirpath


def get_embedding_model_dirpath_for_Astrid(dname, emb_dim=None):
    embedding_dirpath = os.path.join(
        f"res/{dname}/model/triplets"
    )
    if emb_dim is not None:
        embedding_dirpath += f"/e{emb_dim}"
    os.makedirs(embedding_dirpath, exist_ok=True)
    print(f"{embedding_dirpath = }")
    return embedding_dirpath


def get_word_vector_path_for_E2Eestimator(dname, min_count, tag):
    wv_path = os.path.join(
        f"res/{dname}/model/{tag}", f"wordvectors_mc_{min_count}.kv"
    )
    os.makedirs(os.path.dirname(wv_path), exist_ok=True)
    print(f"{wv_path = }")
    return wv_path


def get_count_path_for_HybridEstimator(
    dname, PT, N, dynamicPT, max_entry_ratio, makedir=True
):
    if dynamicPT:
        assert max_entry_ratio is not None
        count_path = os.path.join(
            f"res/{dname}/Ntable", f"count_N{N}_ME{max_entry_ratio}.pkl"
        )
    else:
        count_path = os.path.join(
            f"res/{dname}/Ntable", f"count_N{N}_PT{PT}.pkl")
    if makedir:
        os.makedirs(os.path.dirname(count_path), exist_ok=True)
        print(f"{count_path = }")
    return count_path


def get_summary_writer(dname, model, workload, seed):
    sw_path = os.path.join(f"res/{dname}/{workload}/log/{model}/{seed}")
    print(f"{sw_path = }")
    sw = SummaryWriter(sw_path)
    return sw


def get_sampling_model_name(is_adapt=False, is_greek=False):
    if is_adapt:
        assert is_greek
        model_name = f"EST_B"
    else:
        if is_greek:
            model_name = f"EST_M"
        else:
            model_name = f"EST_S"
    return model_name


def save_estimated_cards(test_queries, test_cards, test_estimations, est_path):
    os.makedirs(os.path.dirname(est_path), exist_ok=True)
    header = ["query", "true", "est", "q-error"]
    q_errs = mean_Q_error(test_cards, test_estimations, reduction="none")
    data = list(zip(test_queries, test_cards, test_estimations, q_errs))

    df = pd.DataFrame(data, columns=header)
    print(f"{est_path = }")
    df.to_csv(est_path, index=False)


def save_estimation_times(
    test_queries, test_cards, test_estimations, test_estimation_times, est_time_path
):
    os.makedirs(os.path.dirname(est_time_path), exist_ok=True)
    header = ["query", "true", "est", "q-error", "est_time"]
    q_errs = mean_Q_error(test_cards, test_estimations, reduction="none")
    data = list(
        zip(test_queries, test_cards, test_estimations,
            q_errs, test_estimation_times)
    )

    df = pd.DataFrame(data, columns=header)
    df.to_csv(est_time_path, index=False)


def save_count_infos(
    test_queries, test_cards, test_estimations, count_info, save_count_path
):
    os.makedirs(os.path.dirname(save_count_path), exist_ok=True)
    q_errors = mean_Q_error(test_cards, test_estimations, reduction="none")

    stats_count = [
        [
            test_queries[i],
            test_cards[i],
            test_estimations[i],
            q_errors[i],
            count_info[i][0],
            count_info[i][1],
            count_info[i][2],
        ]
        for i in range(len(test_queries))
    ]

    os.makedirs(os.path.dirname(save_count_path), exist_ok=True)
    df = pd.DataFrame(
        stats_count, columns=["query", "true", "est", "q-error", "n", "m", "k"]
    )
    df.to_csv(save_count_path)


# endregion


# region torch


def find_k_index_in_encoded_strings(strings, value, k):
    """_summary_

    Parameters
    ----------
    strings : torch.LongTensor (2D)
        encoded strings
    value : int
        a value to find
    k : int
        k-th position from starting position

    Returns
    -------
    torch.Longtensor (1D)
        k-th index whose entry is equal to the value
    Examples
    --------
    >>> X = [[2, 3, 4, 2, 5, 2], [2, 4, 2, 5, 2, 0]]  # ["%ab%c%", "%b%c%"]
    >>> X = torch.IntTensor(X)
    >>> find_k_index_in_encoded_strings(X, 2, 2)
    >>> tensor([3, 2], dtype=torch.int32)
    """
    return (strings == value).nonzero(as_tuple=True)[1].reshape(-1, 3)[:, k - 1].cpu()


def set_seed(seed):
    """set seed for numpy and torch

    Parameters
    ----------
    seed : int or None
        seed for randomness
    """
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)


# endregion


# region Ngram table
def count_token(engram_dict, token):
    if token not in engram_dict:
        engram_dict[token] = 0
    engram_dict[token] += 1


def filter_infrequent_entry_PT(engram_dict, PT):
    output = {}
    for k, v in engram_dict.items():
        if v >= PT:
            output[k] = v
    return output


def generator_replaced_string(string):
    """
        This function considers replacement operation only.
    :param string:
    :return:
    """

    # N = len(string)
    # total = 2 ** N
    # i = 0
    # while i < total:
    #
    #     i += 1
    global esc_ac

    def idx2onoff(x):
        output = []
        curr = x
        for _ in range(n):
            output.append(curr % 2 == 1)
            curr //= 2
        return reversed(output)

    # assert esc_ac not in string
    # if "?" in string:
    #     string = string.replace("?", esc_ac)
    # assert "?" not in string
    n = len(string)
    for idx in range(2**n):
        output = []
        onoff = idx2onoff(idx)
        for i, is_change in enumerate(onoff):
            if is_change:
                # output += '?'
                output.append(esc_ac)
            else:
                # output += string[i]
                output.append(string[i])
        yield "".join(output)


def extendedNgramTable(db, N, PT):
    hashtable = {}
    hashtable[""] = len(db)  # length 0

    for rid, record in tqdm(enumerate(db), total=len(db)):
        if len(record) <= N:
            for sub_token in generator_replaced_string(record):
                count_token(hashtable, esc_sc + sub_token + esc_ec)
        length = len(record)
        for l in range(1, min(N, length) + 1):
            distinct_tokens = set()
            for s in range(length - l + 1):
                token = record[s: s + l]

                for sub_token in generator_replaced_string(token):
                    distinct_tokens.add(sub_token)
                    if s == 0:  # prefix
                        count_token(hashtable, esc_sc + sub_token)
                    if s + l == len(record):  # suffix
                        count_token(hashtable, sub_token + esc_ec)
            for token in distinct_tokens:
                count_token(hashtable, token)

    if PT > 1:
        hashtable = filter_infrequent_entry_PT(hashtable, PT)

    return hashtable


def getCachedExtendedNgramTable(output_path, db=None, N=None, verbose=True):
    time_path = output_path.replace(".pkl", ".yml")
    if verbose:
        print(f"{time_path = }")

    if not os.path.exists(output_path) or not os.path.exists(time_path):
        start_time = time.time()
        hashtable = extendedNgramTable(db, N, 1)
        end_time = time.time()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(hashtable, f)
        build_time = end_time - start_time

        out_dict = {"time": build_time}
        with open(time_path, "w") as f:
            yaml.safe_dump(out_dict, f)

    with open(output_path, "rb") as f:
        hashtable = pickle.load(f)
    with open(time_path) as f:
        out_dict = yaml.safe_load(f)
        build_time = out_dict["time"]
    return hashtable, build_time


def extendedNgramTable_per_len(engram_dict):
    engram_dict_per_len = {}

    for key, val in engram_dict.items():
        token = key.lstrip(esc_sc).rstrip(esc_ec)
        token_len = len(token)
        if token_len not in engram_dict_per_len:
            engram_dict_per_len[token_len] = {}
        # print(f"{key = }, {val = }, {token = }")
        engram_dict_per_len[len(token)][key] = val
    return engram_dict_per_len


def filter_infrequent_entry_max_entry(engram_dict, max_entry):
    sorted_counts = list(
        sorted(engram_dict.items(), key=lambda x: x[1], reverse=True))
    if max_entry < len(sorted_counts):
        # element next to the max_entry-th element
        PT = sorted_counts[max_entry][1] + 1
    else:
        PT = 1
    pruned_dict = filter_infrequent_entry_PT(engram_dict, PT)
    # print(f"{len(pruned_dict) = }, {max_entry = }, {PT = }")
    return pruned_dict, PT


def filter_infrequent_entry_max_entry_each_len(engram_dict, max_entry_ratio):
    hashtable_per_len = extendedNgramTable_per_len(engram_dict)
    if 2 in hashtable_per_len:
        default_entry = len(hashtable_per_len[2])
    else:
        default_entry = len(hashtable_per_len[1])
    max_entry = int(max_entry_ratio * default_entry)
    # print(f"{max_entry = }")

    output_dict = {}
    PTs = []
    for length, each_dict in hashtable_per_len.items():
        each_dict, PT = filter_infrequent_entry_max_entry(each_dict, max_entry)
        for k, v in each_dict.items():
            output_dict[k] = v

        PTs.append(PT)
    return output_dict, PTs


full_table_cache = {}
db_cache = {}


def getCachedPrunedExtendedNgramTableMaxEntry(
    dname, N, max_entry_ratio=None, cached_out=True, verbose=True
):
    global db_cache
    global full_table_cache
    if dname not in db_cache:
        db_path = f"data/{dname}/{dname}.txt"
        db_cache[dname] = read_strings(db_path)
    db = db_cache[dname]

    output_path = get_count_path_for_HybridEstimator(
        dname,
        PT=None,
        N=N,
        dynamicPT=True,
        max_entry_ratio=max_entry_ratio,
        makedir=False,
    )
    assert f"ME{max_entry_ratio}" in output_path, output_path

    if dname not in full_table_cache:
        full_table_path = output_path.replace(f"ME{max_entry_ratio}", "PT1")
        full_table_cache[dname] = getCachedExtendedNgramTable(
            full_table_path, db=db, N=N, verbose=verbose
        )

    full_table, full_build_time = full_table_cache[dname]

    time_path = output_path.replace(".pkl", ".yml")
    if verbose:
        print(f"{time_path = }")

    is_saved = os.path.exists(output_path) and os.path.exists(time_path)

    if is_saved:
        with open(output_path, "rb") as f:
            hashtable, PTs = pickle.load(f)
        with open(time_path) as f:
            out_dict = yaml.safe_load(f)
            build_time = out_dict["time"]
        is_saved = (PTs[1] != 2)

    if not is_saved:
        start_time = time.time()
        hashtable, PTs = filter_infrequent_entry_max_entry_each_len(
            full_table, max_entry_ratio
        )

        end_time = time.time()
        build_time = full_build_time + (end_time - start_time)

        if cached_out:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                pickle.dump([hashtable, PTs], f)

            out_dict = {"time": build_time}
            with open(time_path, "w") as f:
                yaml.safe_dump(out_dict, f)

    return hashtable, PTs, build_time


def getCachedPrunedExtendedNgramTable(
    data_name, N=None, PT=None, cached_out=True, verbose=True
):
    global db_cache
    global full_table_cache

    if data_name not in db_cache:
        db_path = f"data/{data_name}/{data_name}.txt"
        db_cache = read_strings(db_path)
    db = db_cache

    output_path = get_count_path_for_HybridEstimator(
        data_name,
        PT=PT,
        N=N,
        dynamicPT=False,
        max_entry_ratio=None,
        makedir=False,
    )

    assert f"PT{PT}" in output_path, output_path

    if data_name not in full_table_cache:
        full_table_path = output_path.replace(f"PT{PT}", "PT1")

        full_table_cache[data_name] = getCachedExtendedNgramTable(
            full_table_path, db=db, N=N, verbose=verbose
        )

    full_table, full_build_time = full_table_cache[data_name]

    time_path = output_path.replace(".pkl", ".yml")

    if verbose:
        print(f"{time_path = }")

    is_saved = os.path.exists(output_path) and os.path.exists(time_path)

    if PT > 1 and not is_saved:
        start_time = time.time()
        hashtable = filter_infrequent_entry_PT(full_table, PT)
        end_time = time.time()
        build_time = full_build_time + (end_time - start_time)

        if cached_out:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                pickle.dump(hashtable, f)
            out_dict = {"time": build_time}
            with open(time_path, "w") as f:
                yaml.safe_dump(out_dict, f)

    if is_saved:
        with open(output_path, "rb") as f:
            hashtable = pickle.load(f)
        with open(time_path) as f:
            out_dict = yaml.safe_load(f)
            build_time = out_dict["time"]
    return hashtable, build_time


# endregion
