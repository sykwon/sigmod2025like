#ifndef FE62F0CE_5810_483F_BAF7_8A5D38FC50A3
#define FE62F0CE_5810_483F_BAF7_8A5D38FC50A3

#include <re2/re2.h>

#include <algorithm>
#include <chrono>
#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <memory>
#include <tuple>
#include <unordered_map>

#include "index.h"
#include "index2.h"
#include "plan.h"
#include "trie.h"
#include "util.h"

using namespace chrono;
using namespace indicators;
int64_t all_compare_count = 0;
int64_t compare_count = 0;
int inter_count = 0;
static bool is_share = true;

float progress;
indicators::BlockProgressBar bar{
    option::BarWidth{50},
    option::Start{" ["},
    // option::Fill{"█"},
    // option::Lead{"█"},
    // option::Remainder{"-"},
    option::End{"]"},
    option::PrefixText{"Iteration"},
    option::ForegroundColor{Color::white},
    option::ShowElapsedTime{true},
    option::ShowRemainingTime{true},
    option::FontStyles{}};

int* data_gen_alg_re2(u32string* db, int n_db, u32string* qrys_org, int n_qrys, vector<tuple<string, double>>* time_dict, vector<double>* q_times, bool is_ascii = false) {
    int* res = new int[n_qrys];

    system_clock::time_point start_time;
    system_clock::time_point end_time;

    system_clock::time_point q_time1;
    system_clock::time_point q_time2;

    double total_time;

    u32string qry;
    string qry8;
    std::shared_ptr<re2::RE2> query_pat;
    int card;
    // u32string record;

    string* db8 = new string[n_db];
    u32string* qrys = new u32string[n_qrys];

    u32string like_query;
    char32_t ch;
    for (int i = 0; i < n_qrys; ++i) {
        like_query = u32string(qrys_org[i]);
        for (int i = like_query.size() - 1; i >= 0; --i) {
            ch = like_query[i];
            if (ch != U'_') {
                if (ch == U'%') {
                    like_query[i] = U'_';
                    like_query.back() = U'%';
                }
                break;
            }
        }
        qrys[i] = like_query;
    }

    for (int i = 0; i < n_db; ++i) {
        db8[i] = utf8::utf32to8(db[i]);
    }

    string* qrys8 = nullptr;
    if (is_ascii) {
        qrys8 = new string[n_qrys];
        for (int i = 0; i < n_qrys; ++i) {
            qrys8[i] = utf8::utf32to8(qrys[i]);
        }
    }

    start_time = system_clock::now();
    for (int i = 0; i < n_qrys; i++) {
        q_time1 = system_clock::now();

        card = 0;
        if (is_ascii) {
            qry8 = qrys8[i];
            if (qry8 == "%") {
                card = n_db;
            } else {
                query_pat = gen_re2_from_like_query(qry8);
                for (int j = 0; j < n_db; j++) {
                    if (RE2::PartialMatch(db8[j], *query_pat)) {
                        card += 1;
                    }
                }
            }
        } else {
            qry = qrys[i];
            if (qry == U"%") {
                card = n_db;
            } else {
                query_pat = gen_re2_from_like_query(qry);
                for (int j = 0; j < n_db; j++) {
                    if (RE2::PartialMatch(db8[j], *query_pat)) {
                        card += 1;
                    }
                }
            }
        }

        res[i] = card;
        q_time2 = system_clock::now();
        q_times->push_back(duration<double>(q_time2 - q_time1).count());

        if (i % 100 == 0) {
            progress = (float)(i + 1) / (float)n_qrys * 100;
            bar.set_progress(progress);
        }
    }

    end_time = system_clock::now();

    // Show cursor
    bar.set_progress(100.0);
    show_console_cursor(true);
    cout << endl;

    total_time = duration<double>(end_time - start_time).count();
    time_dict->push_back(make_tuple("total", total_time));
    delete[] db8;
    if (is_ascii) {
        delete[] qrys8;
    }

    return res;
}

void data_gen_alg_LEADER_T(u32string* db, int n_db, u32string* qrys, int n_qrys, vector<tuple<string, double>>* time_dict, vector<double>& q_times, AUG_TYPE aug_type, vector<vector<int>>& output) {
    if (aug_type == AUG_TYPE::BOTH_AUG) {
        data_gen_alg_LEADER_T(db, n_db, qrys, n_qrys, time_dict, q_times, AUG_TYPE::PREFIX_AUG, output);
        data_gen_alg_LEADER_T(db, n_db, qrys, n_qrys, time_dict, q_times, AUG_TYPE::SUFFIX_AUG, output);
        return;
    }
    assert(aug_type != AUG_TYPE::BOTH_AUG);
    q_times.resize(n_qrys, 0.0);
    output.resize(n_qrys, vector<int>(0));

    system_clock::time_point start_time;
    system_clock::time_point start_mid_time;
    system_clock::time_point mid_time;
    system_clock::time_point mid_time2;
    system_clock::time_point end_time;

    system_clock::time_point q_time1;
    system_clock::time_point q_time2;

    double trie_time;
    double build_time;
    double query_time;
    double delete_time;
    double total_time;

    double q_instance_time = 0.0;

    u32string qry;
    int card;
    LEADERpTree* tree;

    if (aug_type == AUG_TYPE::SUFFIX_AUG) {
        reverseStrings(qrys, n_qrys);
        reverseStrings(db, n_db);
    }

    int max_len_q = max_length_strings(qrys, n_qrys);
    int _n_trie_node_ = n_distinct_prefix(qrys, n_qrys);

    start_time = system_clock::now();

    // trie
    Trie<char32_t>* trie_src_prfx = new Trie<char32_t>(_n_trie_node_);
    trie_src_prfx->add_strings(qrys, n_qrys);
    RadixTree<char32_t>* trie_prfx = new RadixTree<char32_t>(trie_src_prfx);
    trie_prfx->root->set_next();
    delete trie_src_prfx;

    start_mid_time = system_clock::now();
    // tree
    tree = build_SPADE_PLUS_tree(qrys, n_qrys, db, n_db);

    mid_time = system_clock::now();
    // query

    RadixTreeNode<char32_t>* curr_node = trie_prfx->root;
    if (!curr_node->is_leaf()) {
        curr_node = curr_node->children[0];
    }
    int curr_len;
    int rid;
    int count = 0;
    int len_node;
    bool is_cano_prefix;
    bool is_card;
    char32_t* rc_ptr;
    char32_t rc;

    u32string op_token;
    u32string token;

    tree->create_stacks(max_len_q);
    vector<int> cards(max_len_q + 1);

    // u32string query = U"";  // remove for efficiency
    while (curr_node) {
        curr_len = curr_node->depth - 1;
        // query = query.substr(0, curr_len);
        len_node = curr_node->n_str;
        rc_ptr = curr_node->s_str;
        for (int node_pos = 0; node_pos < len_node; ++node_pos) {
            q_time1 = system_clock::now();
            ++curr_len;
            rc = *rc_ptr;
            // query.push_back(rc);
            ++rc_ptr;  // rc_ptr indicates the next character
            ++count;
            rid = curr_node->rid(node_pos);

            // is_cano_prefix && is_card
            if (node_pos == len_node - 1) {
                is_cano_prefix = curr_node->has_wildcard_child;
            } else {
                is_cano_prefix = IS_UWILDCARD(*rc_ptr);
            }
            is_card = !(aug_type == AUG_TYPE::BASE) || rid >= 0;

            card = tree->find_card(curr_len, rc, is_cano_prefix, is_card);
            cards[curr_len] = card;
            q_time2 = system_clock::now();

            // store time
            q_instance_time += duration<double>(q_time2 - q_time1).count();
            if (rid >= 0) {
                q_times[rid] += q_instance_time;  // cumulative
                q_instance_time = 0;
                if (aug_type == AUG_TYPE::BASE) {
                    output[rid].push_back(card);
                } else {
                    for (int i = 1; i <= curr_len; ++i) {
                        output[rid].push_back(cards[i]);
                    }
                }
            }

            // verify card
            // auto card2 = tree->find_card_old(query);
            // if (!(aug_type == AUG_TYPE::BASE) || rid >= 0) {
            //     assert(card == card2);
            // }
            // cout << "Query: " << utf8::utf32to8(query) << ", Card: " << card << ", Card_old: " << card2 << endl;

            if (count % 100 == 0) {
                progress = (float)(count + 1) / (float)_n_trie_node_ * 100;
                bar.set_progress(progress);
            }
        }
        if (!curr_node->is_leaf()) {
            curr_node = curr_node->children[0];
        } else {
            curr_node = curr_node->next_node;
        }
    }

    // Show cursor
    bar.set_progress(100.0);
    show_console_cursor(true);
    cout << endl;

    mid_time2 = system_clock::now();
    delete tree;
    delete trie_prfx;

    end_time = system_clock::now();

    if (aug_type == AUG_TYPE::SUFFIX_AUG) {
        reverseStrings(qrys, n_qrys);
        reverseStrings(db, n_db);
    }

    trie_time = duration<double>(start_mid_time - start_time).count();
    build_time = duration<double>(mid_time - start_mid_time).count();
    query_time = duration<double>(mid_time2 - mid_time).count();
    delete_time = duration<double>(end_time - mid_time2).count();
    total_time = duration<double>(end_time - start_time).count();

    cumulatively_add_value(*time_dict, "trie", trie_time);
    cumulatively_add_value(*time_dict, "build", build_time);
    cumulatively_add_value(*time_dict, "query", query_time);
    cumulatively_add_value(*time_dict, "delete", delete_time);
    cumulatively_add_value(*time_dict, "total", total_time);
}

void data_gen_alg_LEADER_SR(u32string* db, int n_db, u32string* qrys, int n_qrys, vector<tuple<string, double>>* time_dict, vector<double>& q_times, AUG_TYPE aug_type, vector<vector<int>>& output) {
    // SPADE-L algorithm
    if (aug_type == AUG_TYPE::BOTH_AUG) {
        data_gen_alg_LEADER_SR(db, n_db, qrys, n_qrys, time_dict, q_times, AUG_TYPE::PREFIX_AUG, output);
        data_gen_alg_LEADER_SR(db, n_db, qrys, n_qrys, time_dict, q_times, AUG_TYPE::SUFFIX_AUG, output);
        return;
    }
    assert(aug_type != AUG_TYPE::BOTH_AUG);
    bool prefix_mode = aug_type != AUG_TYPE::BASE;
    q_times.resize(n_qrys, 0.0);
    output.resize(n_qrys, vector<int>(0));

    system_clock::time_point start_time;
    system_clock::time_point start_mid_time;
    system_clock::time_point mid_time;
    system_clock::time_point mid_time2;
    system_clock::time_point end_time;

    system_clock::time_point ivs_start;
    system_clock::time_point ivs_end;

    double sort_time = 0.0;
    double build_time = 0.0;
    double ivs_time = 0.0;
    double query_time = 0.0;
    double delete_time;
    double total_time;

    u32string* rec;
    u32string* qry;
    int q_len;
    int lcp_length;
    int saved_p_prev;
    u32string op_token;
    u32string token;
    bool is_matched;
    SPADETree* tree;
    int oid;  // origin_idx
    int qid_prime;
    bool is_sp;  // shareable prefix
    bool is_card;
    bool is_non_empty;
    int empty_pos;
    int op_last_s;
    int op_last_l;
    int token_last_s;
    int token_last_l;

    // flip S_Q S_D
    if (aug_type == AUG_TYPE::SUFFIX_AUG) {
        reverseStrings(qrys, n_qrys);
        reverseStrings(db, n_db);
    }

    int max_len_q = max_length_strings(qrys, n_qrys);
    int _n_trie_node_ = n_distinct_prefix(qrys, n_qrys);

    vector<int> sort_indexes = get_sort_wildcard_indexes(qrys, n_qrys);

    start_time = system_clock::now();
    // sort
    sort_wildcard_strings(qrys, n_qrys);
    start_mid_time = system_clock::now();

    vector<int> lcp_length_list = find_prefix_positions(qrys, n_qrys);
    vector<vector<tuple<bool, int, int, int, int, int>>> infos = find_infos(qrys, n_qrys, lcp_length_list);  // is_sp, saved_p_prev, op_last_s, op_last_l, token_last_s, token_last_l

    vector<int> cum_prfx_offset;
    if (prefix_mode) {
        cum_prfx_offset[0] = 0;
        for (int qid = 1; qid < n_qrys; ++qid) {
            u32string qry = qrys[qid - 1];
            cum_prfx_offset[qid] = cum_prfx_offset[qid - 1] + qry.size() - lcp_length_list[qid - 1];
        }
    }

    // tree
    tree = build_SPADE_tree(qrys, n_qrys);  // build trie for S_Q
    tree->posting_stack = new PositionSet*[max_len_q + 1];
    tree->reset_stacks(max_len_q);

    mid_time = system_clock::now();
    // query

    vector<int> cards;
    // vector<int> cards(max_len_q + 1);

    // unordered_map<u32string, int> cards;
    if (prefix_mode) {
        cards.resize(_n_trie_node_, 0);
    } else {
        cards.resize(n_qrys, 0);
    }

    // u32string query;

    for (int sid = 0; sid < n_db; ++sid) {
        ivs_start = system_clock::now();
        rec = db + sid;
        tree->build_ivs(*rec);  // build inverted index for I_s
        tree->reset_stacks(max_len_q);
        ivs_end = system_clock::now();
        ivs_time += duration<double>(ivs_end - ivs_start).count();

        empty_pos = max_len_q + 1;
        for (int qid = 0; qid < n_qrys; ++qid) {
            lcp_length = lcp_length_list[qid];
            if (empty_pos <= lcp_length) {
                continue;
            } else {
                empty_pos = max_len_q + 1;
            }
            vector<tuple<bool, int, int, int, int, int>>& info_list = infos[qid];

            oid = sort_indexes[qid];
            qry = qrys + qid;
            q_len = qry->size();

            for (int curr_len = lcp_length + 1; curr_len < q_len + 1; ++curr_len) {
                tuple<bool, int, int, int, int, int>& info = info_list[curr_len];
                is_sp = std::get<0>(info);
                is_card = prefix_mode || curr_len == q_len;
                if (!is_sp && !is_card) {
                    continue;
                }

                saved_p_prev = get<1>(info);
                op_last_s = get<2>(info);
                op_last_l = get<3>(info);
                token_last_s = get<4>(info);
                token_last_l = get<5>(info);
                if (op_last_l > 0) {
                    op_token = qry->substr(op_last_s - 1, op_last_l);
                } else {
                    op_token = U"";
                }
                if (token_last_l > 0) {
                    token = qry->substr(token_last_s - 1, token_last_l);
                } else {
                    token = U"";
                }

                is_non_empty = tree->computeIncNew(saved_p_prev, curr_len, op_token, token, is_sp, is_card, is_matched);

                if (!is_non_empty) {
                    empty_pos = curr_len;
                    break;
                }

                if (is_card && is_matched) {
                    if (prefix_mode) {
                        qid_prime = cum_prfx_offset[qid] + curr_len;
                    } else {
                        qid_prime = qid;
                    }
                    cards[qid_prime] += 1;
                }
            }
        }
        if (sid % 500 == 0) {
            progress = (float)(sid + 1) / (float)n_db * 100;
            bar.set_progress(progress);
        }
    }

    // Show cursor
    bar.set_progress(100.0);
    show_console_cursor(true);
    cout << endl;

    // Save output
    vector<int> collected_cards;
    if (prefix_mode) {
        collected_cards.reserve(max_len_q + 1);
    }

    for (int qid = 0; qid < n_qrys; ++qid) {
        oid = sort_indexes[qid];
        if (prefix_mode) {
            for (int curr_len = lcp_length + 1; curr_len < q_len + 1; ++curr_len) {
                qid_prime = cum_prfx_offset[qid] + curr_len;
                collected_cards[curr_len] = cards[qid_prime];
            }
            for (int curr_len = 1; curr_len < q_len + 1; ++curr_len) {
                output[oid].push_back(collected_cards[curr_len]);
            }
        } else {
            output[oid].push_back(cards[qid]);
        }
    }

    mid_time2 = system_clock::now();
    delete tree;

    end_time = system_clock::now();

    sort_with_indexes(qrys, n_qrys, sort_indexes);

    // flip S_Q S_D
    if (aug_type == AUG_TYPE::SUFFIX_AUG) {
        reverseStrings(qrys, n_qrys);
        reverseStrings(db, n_db);
    }

    sort_time = duration<double>(start_mid_time - start_time).count();
    build_time = duration<double>(mid_time - start_mid_time).count() + ivs_time;
    query_time = duration<double>(mid_time2 - mid_time).count() - ivs_time;
    delete_time = duration<double>(end_time - mid_time2).count();
    total_time = duration<double>(end_time - start_time).count();

    cumulatively_add_value(*time_dict, "sort", sort_time);
    cumulatively_add_value(*time_dict, "build", build_time);
    cumulatively_add_value(*time_dict, "query", query_time);
    cumulatively_add_value(*time_dict, "delete", delete_time);
    cumulatively_add_value(*time_dict, "total", total_time);
}

vector<int> o_type2order(int o_type) {
    assert(o_type > 0);
    vector<int> order;
    int idx;
    int n_tokens = 1;
    int o_type_cpy = o_type;
    while (o_type > 0) {
        o_type /= 10;
        n_tokens += 1;
    }
    o_type = o_type_cpy;

    while (o_type > 0) {
        idx = o_type % 10;
        order.push_back(idx);
        if (idx == 1) {
            order.push_back(0);  // prefix selection
        }
        if (idx == n_tokens - 1) {
            order.push_back(n_tokens);  // suffix selection
        }
        o_type /= 10;
    }
    reverse(order.begin(), order.end());
    return order;
}

void data_gen_alg_LEADER_S(u32string* db, int n_db, u32string* qrys, int n_qrys, vector<tuple<string, double>>* time_dict, vector<double>& q_times, AUG_TYPE aug_type, vector<vector<int>>& output, int o_type = 0) {
    // o_type indicates a join order
    // For example, consider a join "A join1 B join2 C join3 D"
    // When "o_type == 213", this algorithm performs "((A join1 (B join2 C)) join3 D)"
    // There is a special case "o_type == 0" which it works in left-to-right join only
    // The "o_type" works only when is_share is false"

    q_times.resize(n_qrys, 0.0);
    output.resize(n_qrys, vector<int>(0));

    system_clock::time_point start_time;
    system_clock::time_point start_mid_time;
    system_clock::time_point mid_time;
    system_clock::time_point mid_time2;
    system_clock::time_point end_time;

    system_clock::time_point q_time1;
    system_clock::time_point q_time2;

    double trie_time;
    double build_time;
    double query_time;
    double delete_time;
    double total_time;

    double q_instance_time = 0.0;

    u32string* qry;
    u32string* qry_next;
    int q_len;
    // int q_len_next;
    int lcp_length;
    int card;
    LEADERpTree* tree;

    // flip S_Q S_D
    if (aug_type == AUG_TYPE::SUFFIX_AUG) {
        reverseStrings(qrys, n_qrys);
        reverseStrings(db, n_db);
    }

    int max_len_q = max_length_strings(qrys, n_qrys);
    int _n_trie_node_;
    if (is_share) {
        _n_trie_node_ = 0;
        for (int i = 0; i < n_qrys; i++) {
            _n_trie_node_ += qrys[i].size();
        }
    } else {
        _n_trie_node_ = n_distinct_prefix(qrys, n_qrys);
    }

    vector<int> sort_indexes = get_sort_wildcard_indexes(qrys, n_qrys);

    start_time = system_clock::now();
    // sort
    sort_wildcard_strings(qrys, n_qrys);
    vector<int> lcp_length_list = find_prefix_positions(qrys, n_qrys);
    // vector<bool> is_last_pos_sp_list = find_is_last_pos_sp_list(qrys, n_qrys, lcp_length_list);

    start_mid_time = system_clock::now();
    // tree
    tree = build_SPADE_PLUS_tree(qrys, n_qrys, db, n_db);

    mid_time = system_clock::now();
    // query

    int oid;  // origin_idx
    int count = 0;
    bool is_sp;  // shareable prefix
    bool is_card;
    char32_t rc;
    char32_t rc_next;

    tree->create_stacks(max_len_q);
    vector<int> cards(max_len_q + 1);

    u32string query;
    if (is_share) {
        for (int qid = 0; qid < n_qrys; ++qid) {
            oid = sort_indexes[qid];
            qry = qrys + qid;
            q_len = qry->size();
            lcp_length = lcp_length_list[qid];

            for (int curr_len = lcp_length + 1; curr_len < q_len + 1; ++curr_len) {
                q_time1 = system_clock::now();
                rc = (*qry)[curr_len - 1];
                ++count;

                // is_cano_prefix && is_card
                if (curr_len == q_len) {
                    // is_sp = is_last_pos_sp_list[qid];
                    is_sp = false;
                    if (qid < n_qrys - 1) {
                        qry_next = qry + 1;
                        lcp_length = lcp_length_list[qid + 1];  // lcp length between qry and qry_next
                        // q_len_next = qry_next->size();
                        // if (q_len < q_len_next) {
                        if (q_len <= lcp_length) {
                            rc_next = (*qry_next)[q_len];
                            is_sp = IS_UWILDCARD(rc_next);
                        }
                    }
                } else {
                    rc_next = (*qry)[curr_len];
                    is_sp = IS_UWILDCARD(rc_next);
                }
                is_card = !(aug_type == AUG_TYPE::BASE) || curr_len == q_len;

                card = tree->find_card(curr_len, rc, is_sp, is_card);
                cards[curr_len] = card;
                q_time2 = system_clock::now();

                // // verify card
                // query = qry->substr(0, curr_pos);
                // auto card2 = tree->find_card_old(query);
                // if (is_card) {
                //     assert(card == card2);
                // }
                // cout << "Query: " << utf8::utf32to8(query) << ", Card: " << card << ", Card_old: " << card2 << endl;

                // store time
                q_instance_time += duration<double>(q_time2 - q_time1).count();
                if (curr_len == q_len) {
                    q_times[oid] += q_instance_time;  // cumulative
                    q_instance_time = 0;
                    if (aug_type == AUG_TYPE::BASE) {
                        output[oid].push_back(card);
                    } else {
                        for (int i = 1; i <= curr_len; ++i) {
                            output[oid].push_back(cards[i]);
                        }
                    }
                }
            }
            if (qid % 1000 == 0) {
                progress = (float)(qid + 1) / (float)n_qrys * 100;
                bar.set_progress(progress);
            }
        }

    } else if (o_type != 0) {  // join order
        vector<int> order = o_type2order(o_type);
        for (int qid = 0; qid < n_qrys; ++qid) {
            oid = sort_indexes[qid];
            qry = qrys + qid;
            q_time1 = system_clock::now();
            card = tree->find_card_with_plan(*qry, order);
            q_time2 = system_clock::now();
            q_instance_time = duration<double>(q_time2 - q_time1).count();
            q_times[oid] = q_instance_time;
            output[oid].push_back(card);
            if (qid % 10 == 0) {
                progress = (float)(qid + 1) / (float)n_qrys * 100;
                bar.set_progress(progress);
            }
        }
    } else {  // not share left-to-right only
        for (int qid = 0; qid < n_qrys; ++qid) {
            oid = sort_indexes[qid];
            qry = qrys + qid;
            q_len = qry->size();

            for (int curr_len = 1; curr_len < q_len + 1; ++curr_len) {
                q_time1 = system_clock::now();
                rc = (*qry)[curr_len - 1];
                ++count;

                // is_cano_prefix && is_card
                if (curr_len == q_len) {
                    is_sp = false;
                } else {
                    rc_next = (*qry)[curr_len];
                    is_sp = IS_UWILDCARD(rc_next);
                }
                is_card = !(aug_type == AUG_TYPE::BASE) || curr_len == q_len;

                card = tree->find_card(curr_len, rc, is_sp, is_card);
                cards[curr_len] = card;
                q_time2 = system_clock::now();

                // // verify card
                // query = qry->substr(0, curr_pos);
                // auto card2 = tree->find_card_old(query);
                // if (is_card) {
                //     assert(card == card2);
                // }
                // cout << "Query: " << utf8::utf32to8(query) << ", Card: " << card << ", Card_old: " << card2 << endl;

                // store time
                q_instance_time += duration<double>(q_time2 - q_time1).count();
                if (curr_len == q_len) {
                    q_times[oid] = q_instance_time;
                    q_instance_time = 0;
                    if (aug_type == AUG_TYPE::BASE) {
                        output[oid].push_back(card);
                    } else {
                        for (int i = 1; i <= curr_len; ++i) {
                            output[oid].push_back(cards[i]);
                        }
                    }
                }
            }
            if (qid % 1000 == 0) {
                progress = (float)(qid + 1) / (float)n_qrys * 100;
                bar.set_progress(progress);
            }
        }
    }

    // Show cursor
    bar.set_progress(100.0);
    show_console_cursor(true);
    cout << endl;

    mid_time2 = system_clock::now();
    delete tree;

    end_time = system_clock::now();

    sort_with_indexes(qrys, n_qrys, sort_indexes);

    // flip S_Q S_D
    if (aug_type == AUG_TYPE::SUFFIX_AUG) {
        reverseStrings(qrys, n_qrys);
        reverseStrings(db, n_db);
    }

    trie_time = duration<double>(start_mid_time - start_time).count();
    build_time = duration<double>(mid_time - start_mid_time).count();
    query_time = duration<double>(mid_time2 - mid_time).count();
    delete_time = duration<double>(end_time - mid_time2).count();
    total_time = duration<double>(end_time - start_time).count();

    cumulatively_add_value(*time_dict, "sort", trie_time);
    cumulatively_add_value(*time_dict, "build", build_time);
    cumulatively_add_value(*time_dict, "query", query_time);
    cumulatively_add_value(*time_dict, "delete", delete_time);
    cumulatively_add_value(*time_dict, "total", total_time);
}

int* data_gen_alg_re2_index(u32string* db, int n_db, u32string* qrys, int n_qrys, vector<tuple<string, double>>* time_dict, vector<double>* q_times) {
    int* res = new int[n_qrys];

    system_clock::time_point start_time;
    system_clock::time_point mid_time;
    system_clock::time_point mid_time2;
    system_clock::time_point end_time;

    system_clock::time_point q_time1;
    system_clock::time_point q_time2;

    double build_time;
    double query_time;
    double delete_time;
    double total_time;

    u32string qry;
    std::shared_ptr<re2::RE2> query_pat;
    int card;
    vector<int>* ids;
    u32string record;
    InvTree* tree;
    bool is_substr;
    bool view;

    start_time = system_clock::now();
    // tree

    tree = build_inv_token_tree(qrys, n_qrys, db, n_db);

    mid_time = system_clock::now();

    // query
    for (int i = 0; i < n_qrys; i++) {
        q_time1 = system_clock::now();
        card = 0;
        qry = qrys[i];
        is_substr = (qry == U"%") || ((qry[0] == U'%') && (qry.back() == U'%') && (qry.find('_') == u32string::npos) && (qry.find(U'%', 1) == qry.size() - 1));

        // cout << "query :" << utf8::utf32to8(qry) << endl;

        // find candidate
        ids = tree->find_candidate_id(qry, view);
        // cout << "ids :" << (*ids).size() << endl;
        if (ids->size() > 0) {
            if (is_substr) {
                card = (int)ids->size();
            } else {
                query_pat = gen_re2_from_like_query(qry);

                for (int id : *ids) {
                    // record = db[id];
                    if (RE2::PartialMatch(utf8::utf32to8(db[id]), *query_pat)) {
                        card += 1;
                    }
                }
            }
        }
        if (view) {
            delete ids;
        }
        res[i] = card;
        q_time2 = system_clock::now();
        q_times->push_back(duration<double>(q_time2 - q_time1).count());

        if (i % 100 == 0) {
            progress = (float)(i + 1) / (float)n_qrys * 100;
            bar.set_progress(progress);
        }
    }
    // Show cursor
    bar.set_progress(100.0);
    show_console_cursor(true);
    cout << endl;

    mid_time2 = system_clock::now();
    delete tree;

    end_time = system_clock::now();

    build_time = duration<double>(mid_time - start_time).count();
    query_time = duration<double>(mid_time2 - mid_time).count();
    delete_time = duration<double>(end_time - mid_time2).count();
    total_time = duration<double>(end_time - start_time).count();

    time_dict->push_back(make_tuple("build", build_time));
    time_dict->push_back(make_tuple("query", query_time));
    time_dict->push_back(make_tuple("delete", delete_time));
    time_dict->push_back(make_tuple("total", total_time));

    return res;
}

#endif /* FE62F0CE_5810_483F_BAF7_8A5D38FC50A3 */
