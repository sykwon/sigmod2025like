import math
import tqdm
import re
import numpy as np
import os
import pandas as pd
import src.util as ut
import time
import pickle
import yaml
from src.estimator import Estimator
from src.util import compile_LIKE_query, eval_compiled_LIKE_query


def eval_bitmap_for_LIKE_query(qrys, db):
    bitmaps = []
    for qry in tqdm.tqdm(qrys, total=len(qrys)):
        qry_re = compile_LIKE_query(qry)
        bitmap = []
        for rec in db:
            if qry_re.search(rec):
                bitmap.append(1)
            else:
                bitmap.append(0)
        bitmaps.append(bitmap)
    return bitmaps


def eval_LIKE_query(qrys, db, multiplier=1.0):
    cards = []
    for qry in tqdm.tqdm(qrys, total=len(qrys)):
        qry_re = compile_LIKE_query(qry)
        card = 0
        for rec in db:
            if qry_re.search(rec):
                card += 1
        cards.append([qry, card * multiplier])
    return cards


def naiveSampling(S_Q, S_D, k=1000):
    multiplier = len(S_D) / k
    sampled = np.random.choice(S_D, k, replace=False)
    computed_test_data = eval_LIKE_query(S_Q, sampled, multiplier)
    return computed_test_data


def compute_log_prob(n, m, k, l):
    def F(x):
        return math.lgamma(x + 1)

    if l < k or l > n - m + k:
        output = -math.inf
    elif k == 0:
        output = -F(n - m - l) + F(n - l) + F(n - m) - F(n)
    elif k == l:
        output = -F(m - k) + F(m) + F(n - k) - F(n)
    elif k == n - m + k:
        output = -F(k) + F(m) + F(n - m + k) - F(n)
    else:
        output = (
            -F(k)
            - F(m - k)
            + F(m)
            - F(l - k)
            + F(l)
            - F(n - m + k - l)
            + F(n - l)
            + F(n - m)
            - F(n)
        )
    return output


def compute_log_prob_with_stirling(n, m, k, l):
    assert l > k, (l, k)
    assert n - l > m - k, (n, l, m, k)
    # d = math.sqrt(2 * math.pi) * math.e ** -m * n ** (n + 0.5) * (m-k) ** (m-k + 0.5)  * k ** (k + 0.5) \
    #     / (n-m) ** (n-m+0.5) / m ** (m + 0.5)
    ln_d = (
        0.5 * math.log(2 * math.pi)
        - m
        + (n + 0.5) * math.log(n)
        + (m - k + 0.5) * math.log(m - k)
        + (k + 0.5) * math.log(k)
        - (n - m + 0.5) * math.log(n - m)
        - (m + 0.5) * math.log(m)
    )
    R = (
        (n + 0.5) * math.log(n - l)
        + 0.5 * math.log(l)
        - (n - m + k + 0.5) * math.log(n - m + k - l)
        + (k - 0.5) * math.log(l - k)
    )
    N = math.log(n - m + k - l) - math.log(n - l) + \
        math.log(l) - math.log(l - k)
    # print(f"{ln_d, R, N = }")
    return -m - ln_d + R + l * N


def compute_prob(n, m, k, l):
    return math.exp(compute_log_prob(n, m, k, l))


def compute_prob_with_stirling(n, m, k, l):
    if k == 0 or k == 1 or k == l:
        return math.exp(compute_log_prob(n, m, k, l))
    else:
        return math.exp(compute_log_prob_with_stirling(n, m, k, l))


# W_0: lambertw(x, 0) == lambertw(x)
# W_-1: lambertw(x, -1)


def compute_alpha(n, m, k, eps=1e-5):
    return bisection_method_alpha(n, m, k, eps)


def initialize_ell_for_alpha(n, m, k):
    if m < 3 * k:
        coeff = 8 / 11
    elif k > 50:
        coeff = 0.4
    elif k > 30:
        coeff = 0.3
    elif k > 20:
        coeff = 0.2
    elif k > 12:
        coeff = 0.1
    else:
        x_k_list = [None, None, 500, 80, 40, 20, 15, 10, 8, 7, 6, 6, 5]
        coeff = 1 / x_k_list[k]

    output = coeff * n / m * k
    return max(output, k + 1)


def initialize_ell_for_omega(n, m, k):
    if k > 0.95 * m:
        output = min(1.01 * n / m * k, n - 10)
    elif k > 0.9 * m:
        output = 1.04 * n / m * k
    elif k > 0.8 * m:
        output = 1.06 * n / m * k
    elif k > 0.7 * m:
        output = 1.09 * n / m * k
    elif k > 0.6 * m:
        output = 1.13 * n / m * k
    elif k > 0.4 * m:
        output = 1.17 * n / m * k
    elif k > 0.3 * m:
        output = 1.20 * n / m * k
    else:
        output = n / (m * (0.75 / (10 + 1.7 * k)))
    return min(output, n - (m - k) - 1)
    # print(f"init omega {output = } {n - m - k = }")
    # return output


def bisection_method_alpha(n, m, k, eps=1e-5):
    def f(x):
        return compute_log_prob(n, m, k, x) - math.log(eps)

    iter_max = 1000
    itr = 1
    l_min = k
    l_max = int(k * n / m)

    if f(l_min) >= 0:
        return k
    assert f(l_max) > 0

    output = -1
    while itr <= iter_max:
        if l_min + 1 >= l_max:
            output = l_max
            break
        l_mid = (l_min + l_max) // 2
        mid_val = f(l_mid)
        if mid_val == 0:
            output = l_mid
            break
        elif mid_val < 0:
            l_min = l_mid
        elif mid_val > 0:
            l_max = l_mid
        itr += 1
    return output


def bisection_method_omega(n, m, k, eps=1e-5):
    def f(x):
        return compute_log_prob(n, m, k, x) - math.log(eps)

    iter_max = 1000
    itr = 1
    l_min = int(k * n / m)
    l_max = n - m + k
    # print(f"{l_min, l_max = }")

    assert f(l_min) > 0
    if f(l_max) >= 0:
        return l_max

    output = -1
    while itr <= iter_max:
        if l_min + 1 >= l_max:
            output = l_min
            break
        l_mid = (l_min + l_max) // 2
        mid_val = f(l_mid)
        if mid_val == 0:
            output = l_mid
            break
        elif mid_val > 0:
            l_min = l_mid
        elif mid_val < 0:
            l_max = l_mid
        itr += 1
        # print(f"{l_min, l_max = }")
    return output


def newton_method(n, m, k, l_start, eps=1e-5):
    def f(x):
        return compute_log_prob_with_stirling(n, m, k, x) - math.log(eps)

    def f_delta_derivative(x):
        eps = 1e-5
        x1 = x - eps / 2
        x2 = x + eps / 2
        f1 = f(x1)
        f2 = f(x2)
        return (f2 - f1) / eps

    def f_prime(x):
        return (
            (k - m + n + 0.5) / (k - m + n - x)
            - x / (k - m + (n - x))
            + math.log(k - m + (n - x))
            + (k - 0.5) / (x - k)
            - x / (x - k)
            - math.log(x - k)
            - (n + 0.5) / (n - x)
            + x / (n - x)
            - math.log(n - x)
            + 0.5 / x
            + math.log(x)
            + 1
        )

    # print(f"{n, m, k, l_start = }")
    l = l_start
    curr_f = f(l)

    # f_prime_value = f_prime(l)
    # f_delta_derivative_value = f_delta_derivative(l)
    # print(f"{f_prime_value = }")
    # print(f"{f_delta_derivative_value = }")

    # k < l < n - (m - k)
    # is_alpha = l_start <= k * n / m

    l_min = k + 1
    l_max = n - (m - k) - 1

    while curr_f > 0.0001 or curr_f < 0:
        curr_f_prime = f_prime(l)
        curr_f_delta_derivative = f_delta_derivative(l)

        l_next = l - curr_f / curr_f_prime

        # To handle sationary point for omega
        assert l < l_max
        while l_next > l_max:
            l_next = l - (curr_f / curr_f_prime) / 10
            # print(f"{l_next, l_max = }")

        l = l_next

        # print(f"{l, curr_f, curr_f_prime, curr_f_delta_derivative, curr_f / curr_f_prime = }")
        curr_f = f(l)
        # print(f"{l, curr_f, abs(curr_f) > 0.0001 = }")
    # print(f"{l, curr_f = }")
    return l


def compute_omega(n, m, k, eps=1e-5):
    return bisection_method_omega(n, m, k, eps)
    # if k == 0:
    #     omega = -math.log(eps) * n / m
    # elif k == 1:
    #     t = m / n
    #     a = 1 - m
    #     b = math.log(eps) - math.log(n) - math.log(t) + t
    #     if m == 1:
    #         omega = n
    #     else:
    #         omega = math.floor(n * lambertw(a * math.exp(b) , -1).real / a)
    # else:
    #     l_init = initialize_ell_for_omega(n, m, k)
    #     # print(f"{l_init, n, m, k = }")
    #     omega = newton_method(n, m, k, l_init)
    # return min(omega, n - m + k)


def compute_mu(n, m, k, eps=1e-5):
    alpha = compute_alpha(n, m, k, eps)
    alpha = max(alpha, 1)
    omega = compute_omega(n, m, k, eps)
    return math.sqrt(alpha * omega)


def compute_ro(n, m, k, eps=1e-5):
    assert m >= k
    alpha = compute_alpha(n, m, k, eps)
    alpha = max(alpha, 1)
    omega = compute_omega(n, m, k, eps)
    return math.sqrt(omega / alpha)


def compute_greeks(n, m, k, eps=1e-5):
    alpha = compute_alpha(n, m, k, eps)
    omega = compute_omega(n, m, k, eps)
    alpha_clip = max(alpha, 1)

    mu = math.sqrt(alpha_clip * omega)
    ro = math.sqrt(omega / alpha_clip)

    return alpha, omega, mu, ro


def compute_zeta(n, m, max_q):
    # print(f"compute zeta {n, m, q = }", )
    k = 0
    curr_max_q = max_q + 1
    while curr_max_q > max_q and k < m:
        # print(f"compute ro {n, m, k = }", )
        k += 1
        curr_max_q = compute_ro(n, m, k)
    # print(f"in zeta: {n, m, k, max_q, q = }")
    if k == 1:
        max_q_k_0 = compute_ro(n, m, 0)
        if max_q_k_0 < max_q:
            k = 0

    if curr_max_q <= max_q:
        return k
    else:
        return


def compute_d(n, m, k):
    return (
        math.factorial(n)
        * math.factorial(m - k)
        * math.factorial(k)
        / math.factorial(n - m)
        / math.factorial(m)
    )


def reader_df_break_table(n, max_q=2.0):
    btable_path = f"res/btable/{n}_{max_q}.csv"

    if os.path.exists(btable_path):
        df = pd.read_csv(btable_path, header=None)
    else:
        os.makedirs(os.path.dirname(btable_path), exist_ok=True)

        b_table = break_table(n, max_q)
        df = pd.DataFrame(b_table)
        df.to_csv(btable_path, header=None, index=False)
    return df


def break_table(n, max_q=2.0):
    zeta_list = []
    m_prime = 1
    # for m_prime in [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]:
    for m_prime in tqdm.tqdm(range(1, n)):
        if m_prime > n:
            break
        zeta = compute_zeta(n, m_prime, max_q)
        # print(f"{n, m_prime, zeta = }")
        if zeta is not None:
            zeta_list.append([m_prime, zeta])
    zeta_list.append([n, 0])

    # compress
    output = []
    prev_zeta = zeta_list[0][1]
    output.append(zeta_list[0])
    for k, zeta in zeta_list:
        if zeta != prev_zeta:
            output.append([k, zeta])
            prev_zeta = zeta

    return output


class SamplingEstimator(Estimator):
    def __init__(
        self, model_path, is_adapt, is_greek, seed, m=None, max_q=None, eps=None
    ):
        self.model_path = model_path
        self.seed = seed
        if is_adapt:
            self.m = None
        else:
            self.m = m
        if is_greek:
            self.max_q = max_q
            self.eps = eps
        else:
            self.max_q = None
            self.eps = None
        self.b_table = None
        self.samples = None

        self.is_adapt = is_adapt
        self.is_greek = is_greek
        if self.is_adapt or self.is_greek:
            assert max_q is not None and eps is not None
        if self.is_adapt:
            assert self.m is None
        assert (max_q is None and eps is None) or (
            max_q is not None and eps is not None
        )

    def build(self, records):
        self.records = records
        n = len(records)
        self.n = n
        max_q = self.max_q
        np.random.seed(self.seed)
        if self.is_adapt:
            self.b_table = reader_df_break_table(n, max_q).to_numpy(
                dtype=np.int32
            )  # (k, zeta)s
            self.m = n
            for k, zeta in self.b_table:
                if zeta == 0:
                    self.m = k
                    break
        else:
            self.b_table = None

        start_time = time.time()
        permutes = np.random.permutation(self.n)
        permutes = permutes[: self.m]
        self.permutes = permutes
        self.samples = [self.records[idx] for idx in self.permutes]
        end_time = time.time()
        self.save_model()

        build_time = end_time - start_time
        return build_time

    def save_cfg(self):
        cfg = {
            "is_adapt": self.is_adapt,
            "is_greek": self.is_greek,
            "seed": self.seed,
            "m": self.m,
            "max_q": self.max_q,
            "eps": self.eps,
        }
        assert ".pkl" in self.model_path
        self.config_path = self.model_path.replace(".pkl", ".yml")
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.dump(cfg, f)

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "w") as f:
            for sample in self.samples:
                f.write(sample + "\n")

    def load_model(self):
        with open(self.model_path, "rb") as f:
            self.samples = pickle.load(f)

    def estimate(self, test_query, is_info=False, is_tqdm=True):
        n = self.n
        m = self.m
        max_q = self.max_q
        eps = self.eps
        b_table = self.b_table

        query_iter = enumerate(test_query)
        if is_tqdm:
            query_iter = tqdm.tqdm(query_iter, total=len(test_query))

        query_estimate = []

        for qid, query in query_iter:
            k = 0
            query_compiled = compile_LIKE_query(query)
            if not self.is_adapt:
                for rec in self.samples:
                    matched = eval_compiled_LIKE_query(query_compiled, rec)
                    if matched:
                        k += 1
                if k == 0:
                    k = 0.5
                if self.is_greek:
                    est = compute_mu(n, m, k, eps)
                else:
                    est = k * n / m
            else:
                next_b_index = 0
                break_zeta = n  # do not break
                next_m_prime = b_table[next_b_index][0]
                for m, rec in enumerate(self.samples, start=1):
                    if m == next_m_prime:
                        break_zeta = b_table[next_b_index][1]
                        next_b_index += 1
                        if next_b_index < len(b_table):
                            next_m_prime = b_table[next_b_index][0]
                        else:
                            next_m_prime = n + 1

                    matched = eval_compiled_LIKE_query(query_compiled, rec)
                    if matched:
                        k += 1
                    if k >= break_zeta:
                        break
                assert self.is_greek
                est = compute_mu(n, m, k, eps)
            if is_info:
                query_estimate.append([est, n, m, k])
            else:
                query_estimate.append(est)

        return query_estimate

    def model_size(self, *args):
        self.save_model()
        size = os.path.getsize(self.model_path)
        return size
