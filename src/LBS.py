import string
from src.estimator import Estimator
from src.util import canonicalize_like_query
from tqdm import tqdm
import numpy as np
import heapq
import os
import time
import pickle

from src.util import esc_ac, esc_sc, esc_ec, generator_replaced_string

order_dicts = None
A = None
B = None


def inter_sig_list_est(hashes):
    hashes = np.array(hashes, dtype=np.int32)
    # hash_union is sig for || A1 | A2 | ... ||
    return hashes.max(axis=0)


def gen_min_hash_order_dict_list(n, A, B, prime):
    assert len(A) == len(B)
    L = len(A)
    order_dicts = []
    for i in range(L):
        # min_hash = ((np.expand_dims(val, axis=1) * A + B) % prime).min(axis=0)
        order_dict = {}
        hashes = (np.arange(n, dtype=np.uint64) * A[i] + B[i]) % prime
        heap = [x for x in hashes]
        heapq.heapify(heap)
        order_index = 0
        while len(heap) > 0:
            smallest = heapq.heappop(heap)
            order_dict[smallest] = order_index
            order_index += 1
        assert len(order_dict) == n
        order_dicts.append(order_dict)
    return order_dicts


def get_sample_valid_instances(data, max_n_sample, seed=None):
    np.random.seed(seed)
    data = list(filter(lambda x: x[1] > 0, data))
    n_val = len(data)
    n_sample = min(n_val, max_n_sample)
    idx_sample = np.random.choice(range(n_val), n_sample, replace=False)
    samples = [data[i] for i in idx_sample]
    return samples


class LBS(Estimator):
    def __init__(self, N, L, PT, model_path, seed, is_pre_suf=False):
        self.N = N
        self.L = L
        self.PT = PT
        self.seed = seed
        self.n = None
        self.empty_sig = None
        self.engram_hash = None
        self.engram_hash_pre = None
        self.engram_hash_suf = None
        self.coverage = None  # rho
        self.model_path = model_path
        self.order_dicts = None
        self.A = None
        self.B = None
        self.is_pre_suf = is_pre_suf

    def save_model(self):
        with open(self.model_path, "wb") as f:
            if self.is_pre_suf:
                model_info = [
                    self.n,
                    self.coverage,
                    self.engram_hash,
                    self.engram_hash_pre,
                    self.engram_hash_suf,
                ]
            else:
                model_info = [self.n, self.coverage, self.engram_hash]
            pickle.dump(model_info, f)

    def load_model(self):
        with open(self.model_path, "rb") as f:
            if self.is_pre_suf:
                (
                    self.n,
                    self.coverage,
                    self.engram_hash,
                    self.engram_hash_pre,
                    self.engram_hash_suf,
                ) = pickle.load(f)
            else:
                self.n, self.coverage, self.engram_hash = pickle.load(f)
            self.empty_sig = [0] + [self.n] * self.L

    def build(self, db, sample_instances, is_debug):
        N = self.N
        L = self.L
        PT = self.PT
        self.n = n = len(db)
        self.empty_sig = [0] + [self.n] * self.L

        seed = self.seed

        start_time = time.time()
        if is_debug:
            self.load_model()
        else:
            if self.is_pre_suf:
                (
                    self.engram_hash,
                    self.engram_hash_pre,
                    self.engram_hash_suf,
                ) = self.get_extended_ngram_table_hash(db, n, N, L, PT, seed)
            else:
                self.engram_hash = self.get_extended_ngram_table_hash(
                    db, n, N, L, PT, seed
                )

            if self.coverage is None:
                self.coverage = self.calculate_coverage(sample_instances)

            print("coverage:", self.coverage)
            self.save_model()
        end_time = time.time()
        build_time = end_time - start_time
        return build_time

    def model_size(self):
        size = os.path.getsize(self.model_path)
        return size

    def get_signature(self, base_string, base_type="sub"):
        engram_hash = self.engram_hash

        def get_sig(token, token_type="sub"):
            if token_type == "sub":
                engram_hash = self.engram_hash
            elif token_type == "pre":
                engram_hash = self.engram_hash_pre
            elif token_type == "suf":
                engram_hash = self.engram_hash_suf
            else:
                raise NotImplementedError

            token = token.replace("_", esc_ac)
            if token not in engram_hash:
                return self.empty_sig
            else:
                return engram_hash[token]

        n = self.n
        N = self.N
        L = self.L
        PT = self.PT
        freq = None
        hash_list = []

        if len(base_string) <= N:
            token = base_string
            return get_sig(token, token_type=base_type)

        for i in range(len(base_string) - N + 1):
            token = base_string[i: i + N]
            if base_type == "pre" and i == 0:
                sig_part = get_sig(token, token_type=base_type)
            elif base_type == "suf" and i + N == len(base_string):
                sig_part = get_sig(token, token_type=base_type)
            else:
                sig_part = get_sig(token)

            if sig_part[0] == 0:
                return self.empty_sig
            # token = token.replace('_', esc_ac)
            # if token not in engram_hash:
            #     # empty_sig = [0]
            #     # empty_sig.extend([n] * L)
            #     return self.empty_sig
            # sig_part = engram_hash[token]
            hash_list.append(sig_part[1:])
            if i == 0:
                freq = sig_part[0]
            else:
                overlap_token = token[:-1]
                # sig_over = engram_hash[overlap_token]
                sig_over = get_sig(overlap_token)
                c1 = sig_part[0]
                c2 = sig_over[0]
                freq *= c1 / c2

        freq = max(freq, 1e-6)

        hash_inter = inter_sig_list_est(hash_list)

        return [freq, *hash_inter]

    def set2min_hash_order(self, input_set, n, L, seed, prime=(1 << 31) - 1):
        order_dicts = self.order_dicts
        A = self.A
        B = self.B

        if order_dicts is None:
            assert A is None
            assert B is None
            np.random.seed(seed)
            self.A = A = np.random.randint(1, prime, L, dtype=np.uint64)
            self.B = B = np.random.randint(0, prime, L, dtype=np.uint64)
            self.order_dicts = order_dicts = gen_min_hash_order_dict_list(
                n, A, B, prime
            )

        min_hash = (
            (np.expand_dims(np.array(input_set, dtype=np.uint64), axis=1) * A + B)
            % prime
        ).min(axis=0)

        for i in range(L):
            order_dict = order_dicts[i]
            min_hash[i] = order_dict[int(min_hash[i])]
        return min_hash

    def calculate_coverage(self, sample_instances):
        fractions = []

        for query, card in tqdm(sample_instances):
            base_strings = query.strip("%").split("%")
            n_max = 0
            for base_string in base_strings:
                sig = self.get_signature(base_string)
                if sig:
                    est_card = sig[0]
                    n_max = max(n_max, est_card)

            assert card != 0, (query, card)

            fraction = n_max / card
            if fraction > 0:
                fractions.append(fraction)
        if len(fractions) > 0:
            coverage = np.average(fractions)
        else:
            coverage = 1.0
        return coverage

    def get_extended_ngram_table_hash(self, db, n, N, L, PT, seed):
        def add_token(engram_set, token, rid):
            if token not in engram_set:
                engram_set[token] = [rid]
            else:
                if engram_set[token][-1] != rid:
                    engram_set[token].append(rid)

        def hashing(engram_set, engram_hash):
            for key, elems in tqdm(engram_set.items()):
                count = len(elems)
                if count < PT:
                    continue
                hash = self.set2min_hash_order(elems, n, L, seed)
                sig = [count, *hash]
                engram_hash[key] = sig
                del elems
                del hash
            default_hash = [n]
            default_hash.extend([0] * L)
            engram_hash[""] = default_hash

        engram_hash = {}
        engram_set = {}
        if self.is_pre_suf:
            engram_hash_pre = {}
            engram_hash_suf = {}
            engram_set_pre = {}
            engram_set_suf = {}

        for rid, record in tqdm(enumerate(db), total=len(db)):
            length = len(record)
            for l in range(1, N + 1):
                for s in range(length - l + 1):
                    token = record[s: s + l]
                    add_token(engram_set, token, rid)

                    if s == 0:  # prefix
                        add_token(engram_set_pre, token, rid)
                    if s + l == len(record):  # suffix
                        add_token(engram_set_suf, token, rid)
                    # if token not in engram_set:
                    #     engram_set[token] = [rid]
                    # else:
                    #     if engram_set[token][-1] != rid:
                    #         engram_set[token].append(rid)

                    for sub_token in generator_replaced_string(token):
                        add_token(engram_set, sub_token, rid)
                        # if sub_token not in engram_set:
                        #     engram_set[sub_token] = [rid]
                        # else:
                        #     if engram_set[sub_token][-1] != rid:
                        #         engram_set[sub_token].append(rid)
                        if s == 0:  # prefix
                            add_token(engram_set_pre, sub_token, rid)
                        if s + l == len(record):  # suffix
                            add_token(engram_set_suf, sub_token, rid)

        if self.is_pre_suf:
            hashing(engram_set, engram_hash)
            hashing(engram_set_pre, engram_hash_pre)
            hashing(engram_set_suf, engram_hash_suf)
            return engram_hash, engram_hash_pre, engram_hash_suf
        else:
            hashing(engram_set, engram_hash)
            return engram_hash

        # for key, elems in tqdm(engram_set.items()):
        #     count = len(elems)
        #     if count < PT:
        #         continue
        #     hash = self.set2min_hash_order(elems, n, L, seed)
        #     sig = [count, *hash]
        #     engram_hash[key] = sig
        #     del elems
        #     del hash

    def estimate(self, test_query, is_tqdm=True):
        N = self.N
        L = self.L
        PT = self.PT

        # engram_hash = self.engram_hash
        # coverage = self.coverage

        query_estimate, mof_list = self.LBS_with_one_minima(
            test_query, is_tqdm)
        self.mof_list = mof_list
        assert len(mof_list) == len(query_estimate)

        return query_estimate

    def LBS_with_one_minima(self, queries, is_tqdm=True):
        engram_hash = self.engram_hash
        coverage = self.coverage
        L = self.L
        N = self.N

        estimations = []
        mof_bs_list = []

        query_iter = enumerate(queries)
        if is_tqdm:
            query_iter = tqdm(query_iter, total=len(queries))
        for qid, str_q in query_iter:
            str_q = canonicalize_like_query(str_q, True)

            if str_q[0] == "%" and str_q[-1] != "%":
                base_type = "suf"
            elif str_q[-1] == "%" and str_q[0] != "%":
                base_type = "pre"
            else:
                base_type = "sub"

            base_strings = str_q.strip("%").split("%")
            sig_list = []
            str_b_list = []
            for base_i, base_string in enumerate(base_strings):
                if base_i == 0 and base_type == "pre":
                    sig = self.get_signature(base_string, base_type=base_type)
                elif base_i == len(base_strings) - 1 and base_type == "suf":
                    sig = self.get_signature(base_string, base_type=base_type)
                else:
                    sig = self.get_signature(base_string)

                if sig is not None:
                    sig_list.append(sig)
                    card = sig[0]
                    str_b_list.append([card, base_string])

            assert len(sig_list)
            max_sig = max(sig_list, key=lambda x: x[0])
            min_sig = min(sig_list, key=lambda x: x[0])
            max_card, max_min_hash = max_sig[0], max_sig[1:]
            min_card, min_min_hash = min_sig[0], min_sig[1:]
            max_str_b = max(str_b_list, key=lambda x: x[0])
            hash_np = np.array([x[1:] for x in sig_list], dtype=np.int32)
            inter_min_hash = hash_np.max(axis=0)
            union_min_hash = hash_np.min(axis=0)
            # sim_est = jaccard_similarity_estimation(max_min_hash, union_min_hash)
            sim_est = (
                sum(
                    map(
                        lambda pair: pair[0] == pair[1],
                        zip(max_min_hash, union_min_hash),
                    )
                )
                / L
            )

            gamma_est = (
                sum(
                    map(
                        lambda pair: pair[0] == pair[1],
                        zip(inter_min_hash, union_min_hash),
                    )
                )
                / L
            )

            if sim_est == 0:
                sim_est = coverage
            union_estimation = max_card / sim_est

            if union_estimation / max(min_card, 1) > L or gamma_est == 0:
                gamma_est = min(
                    min_card / max_card, 1 / L
                )  # we infer this jaccard similarity less than 1 / L

            estimation = union_estimation * gamma_est
            estimation = min(estimation, min_card)

            mof_info = [max_card, max_min_hash, max_str_b, sim_est]

            mof_bs_list.append(mof_info)
            estimations.append(estimation)
        return estimations, mof_bs_list
