#ifndef ADBB56DD_01F2_4F2D_889A_708C381C56D5
#define ADBB56DD_01F2_4F2D_889A_708C381C56D5

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cond.h"
#include "join.h"
#include "plan.h"
#include "table.h"
#include "util.h"
#define MAX_KEY_LEN 5
#define POSTING_HASH_SIZE 100000
#define COND_B(pr, po, c) (((pr) && (po) >= (c)) || (!(pr) && (po) == (c)))
#define COND_E(pr, po, c) (((pr) && (po) <= (c)) || (!(pr) && (po) == (c)))
using namespace std;

enum AUG_TYPE {
    BASE,
    PREFIX_AUG,
    SUFFIX_AUG,
    BOTH_AUG
};

enum CHAR_STAT {
    WILDCARD,
    FIRST_REGULAR,
    REGULAR,
};

enum LEADER_DIR {
    FORWARD,
    BACKWARD,
    OPT
};

enum LEADER_SEL {
    FIRST,
    LAST
};

class PlanNode;

class InvNode {
public:
    unordered_map<char32_t, InvNode*> children;
    vector<int> inv_list;
    InvNode() {
    }
    ~InvNode() {
        for (auto itr = children.begin(); itr != children.end(); ++itr) {
            auto child = get<1>(*itr);
            delete child;
        }
    }
};

// Posting* join_two_plists_multi(const Posting* posting1, const Posting* posting2, const u32string& op_token, bool opt_next, bool opt_last) {

class LEADERpNode {
public:
    unordered_map<char32_t, LEADERpNode*> children;
    // int c_pre = 0;
    // int c_suf = 0;
    Posting* post;
    Posting* post_pre = nullptr;
    Posting* post_suf = nullptr;
    int curr_id = -1;
    int curr_idx;
    int curr_idx_pre;
    int curr_idx_suf;
    LEADERpNode() {
        this->post = new Posting(true, false);
        this->post_pre = new Posting(true, false);
        this->post_suf = new Posting(true, false);
    }
    ~LEADERpNode() {
        LEADERpNode* child;
        for (auto itr = children.begin(); itr != children.end(); ++itr) {
            child = get<1>(*itr);
            delete child;
        }
        if (this->post) {
            delete this->post;
        }
        if (this->post_pre) {
            delete this->post_pre;
        }
        if (this->post_suf) {
            delete this->post_suf;
        }
    }
};

class LEADERNode {
public:
    unordered_map<char32_t, LEADERNode*> children;
    int c_pre = 0;
    int c_suf = 0;
    Posting* post;
    int curr_id = -1;
    int curr_idx;
    int min_pos;
    int max_pos;
    LEADERNode() {
        this->post = new Posting(true, false);
    }
    ~LEADERNode() {
        LEADERNode* child;
        for (auto itr = children.begin(); itr != children.end(); ++itr) {
            child = get<1>(*itr);
            delete child;
        }
    }
};

class RegNode {
public:
    unordered_map<char32_t, RegNode*> children;
    bool isRegular = false;
    RegNode() {
    }
    ~RegNode() {
        for (auto itr = children.begin(); itr != children.end(); ++itr) {
            auto child = get<1>(*itr);
            delete child;
        }
    }
};

class RegTrie {
public:
    RegNode* root = nullptr;
    RegTrie() {
        this->root = new RegNode();
    }
    ~RegTrie() {
        delete root;
    }

    void add_strings(u32string* qset, int n_qry) {
        u32string* qry;
        for (int id = 0; id < n_qry; id++) {
            qry = &qset[id];
            add_all_regular_strings(*qry);
        }
    }

    void add_all_regular_strings(const u32string& qry) {
        u32string reg_str;
        // int qry_len = (int)qry.size();
        // char32_t ch;
        // bool is_wc;
        // int len_reg_str = 0;
        vector<u32string> op_tokens;
        vector<u32string> tokens;
        splitQuery(qry, op_tokens, tokens);
        for (auto reg_str : tokens) {
            add_regular_string(reg_str);
        }
    }

    void add_regular_string(u32string reg_str) {
        RegNode* node = this->root;
        char32_t ch;

        int reg_str_len = (int)reg_str.size();
        RegNode* child;
        for (int i = 0; i < reg_str_len; i++) {
            ch = reg_str[i];
            if (node->children.find(ch) == node->children.end()) {
                child = new RegNode();
                node->children[ch] = child;
            }
            node = node->children[ch];
            if (i == reg_str_len - 1) {
                node->isRegular = true;
            }
        }
    }

    RegNode* find_node(u32string rec) {
        RegNode* node = this->root;
        char32_t ch;
        for (int i = 0; i < (int)rec.size(); i++) {
            ch = rec[i];
            if (node->children.find(ch) == node->children.end()) {
                return nullptr;
            }
            node = node->children[ch];
        }
        return node;
    }
};
class InvTree {
public:
    static const int max_word_len = MAX_KEY_LEN;
    InvNode* root = nullptr;
    RegTrie* reg_trie;
    InvTree() {
        this->root = new InvNode();
    }
    ~InvTree() {
        delete root;
    }

    void create_trie(u32string* qset, int n_qry) {
        this->reg_trie = new RegTrie();
        this->reg_trie->add_strings(qset, n_qry);
    }

    void build_ivd(u32string* db, int n_db) {
        u32string* rec;
        for (int id = 0; id < n_db; id++) {
            rec = &db[id];
            this->build_ivs(*rec, id);
        }
    }

    void build_ivs(const u32string& rec, int rid) {
        int rec_len = (int)rec.size();

        RegNode* reg_node;
        InvNode* node;
        InvNode* child;
        char32_t ch;
        for (int i = 0; i < rec_len; i++) {
            reg_node = this->reg_trie->root;
            node = this->root;

            if (node->inv_list.size() == 0) {
                node->inv_list.push_back(rid);
            } else if (rid != node->inv_list.back()) {
                node->inv_list.push_back(rid);
            }

            for (int j = i; j < rec_len; j++) {
                ch = rec[j];
                if (reg_node->children.find(ch) == reg_node->children.end()) {
                    break;
                }
                reg_node = reg_node->children[ch];

                if (node->children.find(ch) == node->children.end()) {  // new node
                    child = new InvNode();
                    node->children[ch] = child;
                }
                node = node->children[ch];  // node represents rec[i, j] (= rec[i], ..., rec[j])
                if (node->inv_list.size() == 0) {
                    node->inv_list.push_back(rid);
                } else if (rid != node->inv_list.back()) {
                    node->inv_list.push_back(rid);
                }
            }
        }
    }

    void add_strings(u32string* db, int n_db) {
        u32string* rec;
        for (int id = 0; id < n_db; id++) {
            rec = &db[id];
            add_all_suffixes(*rec, id);
        }
    }

    void add_all_suffixes(const u32string& rec, int rid) {
        u32string suffix;
        int rec_len = (int)rec.size();
        for (int i = 0; i < rec_len; i++) {
            suffix = rec.substr(i, rec_len - i);
            add_string(suffix, rid);
        }
    }

    void add_string(u32string rec, int id) {
        InvNode* node = this->root;
        if (node->inv_list.size() == 0) {
            node->inv_list.push_back(id);
        } else if (id != node->inv_list.back()) {
            node->inv_list.push_back(id);
        }
        char32_t ch;

        int substr_len = (int)rec.size();
        if (substr_len > max_word_len) {
            substr_len = max_word_len;
        }
        InvNode* child;
        for (int i = 0; i < substr_len; i++) {
            ch = rec[i];
            if (node->children.find(ch) == node->children.end()) {
                child = new InvNode();
                node->children[ch] = child;
            }
            node = node->children[ch];

            if (node->inv_list.size() == 0) {
                node->inv_list.push_back(id);
            } else if (id != node->inv_list.back()) {
                node->inv_list.push_back(id);
            }
        }
    }

    InvNode* find_node(u32string rec) {
        InvNode* node = this->root;
        char32_t ch;
        for (int i = 0; i < (int)rec.size(); i++) {
            ch = rec[i];
            if (node->children.find(ch) == node->children.end()) {
                return nullptr;
            }
            node = node->children[ch];
        }
        return node;
    }

    vector<int>* find_candidate_id(const u32string& query, bool& view) {
        vector<int>* inv_list;
        vector<int>* candidate_set = nullptr;
        u32string str = query;
        replace(str.begin(), str.end(), '_', '%');
        vector<u32string> res = string32_split(str, '%');
        view = false;
        if (res.size() == 0) {
            candidate_set = &this->root->inv_list;
        }
        for (auto substr : res) {
            inv_list = &this->find_node(substr)->inv_list;
            if (!candidate_set) {
                candidate_set = inv_list;
            } else {
                candidate_set = merge_lists(candidate_set, inv_list, view);
                view = true;
            }
        }
        return candidate_set;
    }
};

class LEADERpTree {
public:
    // static const int max_word_len = MAX_KEY_LEN;
    LEADERpNode* root = nullptr;
    int n_db;
    Posting** posting_stack = nullptr;
    u32string* last_op_token;
    u32string* last_token;
    CHAR_STAT* status;
    int* saved_p_prev;
    int* saved_p;
    RegTrie* reg_trie;
    LEADERpNode* empty_node = nullptr;

    LEADERpTree() {
        this->root = new LEADERpNode();
        this->empty_node = new LEADERpNode();
    }

    ~LEADERpTree() {
        delete root;
        if (posting_stack) {
            delete[] posting_stack;
        }
        if (status) {
            delete[] status;
        }
        if (saved_p_prev) {
            delete[] saved_p_prev;
        }
        if (saved_p) {
            delete[] saved_p;
        }
        if (last_op_token) {
            delete[] last_op_token;
        }
        if (last_token) {
            delete[] last_token;
        }
    }

    void create_trie(u32string* qset, int n_qry) {
        this->reg_trie = new RegTrie();
        this->reg_trie->add_strings(qset, n_qry);
    }

    void build_ivd(u32string* db, int n_db) {
        u32string* rec;
        s_lens = new int[n_db];
        this->n_db = n_db;
        for (int id = 0; id < n_db; id++) {
            rec = &db[id];
            this->build_ivs(*rec, id);
            s_lens[id] = rec->size();
        }
    }

    void build_ivs(const u32string& rec, int rid) {
        int rec_len = (int)rec.size();

        RegNode* reg_node;
        LEADERpNode* node;
        LEADERpNode* child;
        char32_t ch;
        for (int i = 0; i < rec_len; i++) {
            reg_node = this->reg_trie->root;
            node = this->root;
            for (int j = i; j < rec_len; j++) {
                ch = rec[j];
                if (reg_node->children.find(ch) == reg_node->children.end()) {
                    break;
                }
                reg_node = reg_node->children[ch];

                if (node->children.find(ch) == node->children.end()) {  // new node
                    child = new LEADERpNode();
                    child->post->beta = j - i + 1;
                    node->children[ch] = child;
                }
                node = node->children[ch];   // node represents rec[i, j] (= rec[i], ..., rec[j])
                if (rid != node->curr_id) {  // set initial values
                    node->post->inv_list.push_back(rid);
                    node->curr_idx = node->post->inv_list.size();
                    node->post->inv_list.push_back(0);
                    node->post->inv_list.push_back(node->post->pos_list->size());
                    node->curr_id = rid;
                    node->post->inv_list[0] += 1;  // inv_list[0]: c_sub
                }
                node->post->pos_list->push_back(i + 1);  // add pos

                node->post->inv_list[node->curr_idx] += 1;
                if (i == 0) {                          // rec[i,j] is prefix
                    node->post_pre->inv_list[0] += 1;  // inv_list[0]: c_pre
                    node->post_pre->inv_list.push_back(rid);
                    node->post_pre->inv_list.push_back(1);
                    node->post_pre->inv_list.push_back(node->post_pre->pos_list->size());
                    node->post_pre->pos_list->push_back(1);
                    node->post_pre->beta = j - i + 1;
                }
                if (j == rec_len - 1) {                // rec[i,j] is suffix
                    node->post_suf->inv_list[0] += 1;  // inv_list[0]: c_suf
                    node->post_suf->inv_list.push_back(rid);
                    node->post_suf->inv_list.push_back(1);
                    node->post_suf->inv_list.push_back(node->post_suf->pos_list->size());
                    node->post_suf->pos_list->push_back(i + 1);
                    node->post_suf->beta = j - i + 1;
                }
            }
        }
    }

    void add_strings(u32string* db, int n_db) {
        u32string* rec;
        s_lens = new int[n_db];
        this->n_db = n_db;
        for (int id = 0; id < n_db; id++) {
            rec = &db[id];
            this->add_all_suffixes_pos(*rec, id);
            s_lens[id] = rec->size();
        }
    }
    void add_all_suffixes_pos(const u32string& rec, int rid) {
        int rec_len = (int)rec.size();
        int substr_len;
        LEADERpNode* node;
        LEADERpNode* child;
        char32_t ch;
        for (int i = 0; i < rec_len; i++) {
            // substr_len = MIN(rec_len - i, max_word_len);
            substr_len = rec_len - i;
            node = this->root;
            for (int j = i; j < i + substr_len; j++) {
                ch = rec[j];
                if (node->children.find(ch) == node->children.end()) {  // new node
                    child = new LEADERpNode();
                    child->post->beta = j - i + 1;
                    node->children[ch] = child;
                }
                node = node->children[ch];        // node represents rec[i, j] (= rec[i], ..., rec[j])
                if (i == 0 && !node->post_pre) {  // rec[i,j] is prefix
                    node->post_pre = new Posting(true, false);
                }
                if (j == rec_len - 1 && !node->post_suf) {  // rec[i,j] is suffix
                    node->post_suf = new Posting(true, false);
                }
                if (rid != node->curr_id) {  // set initial values
                    node->post->inv_list.push_back(rid);
                    node->curr_idx = node->post->inv_list.size();
                    node->post->inv_list.push_back(0);
                    node->post->inv_list.push_back(node->post->pos_list->size());
                    node->curr_id = rid;
                    node->post->inv_list[0] += 1;  // inv_list[0]: c_sub
                }
                node->post->pos_list->push_back(i + 1);  // add pos

                node->post->inv_list[node->curr_idx] += 1;
                if (i == 0) {                          // rec[i,j] is prefix
                    node->post_pre->inv_list[0] += 1;  // inv_list[0]: c_pre
                    node->post_pre->inv_list.push_back(rid);
                    node->post_pre->inv_list.push_back(1);
                    node->post_pre->inv_list.push_back(node->post_pre->pos_list->size());
                    node->post_pre->pos_list->push_back(1);
                    node->post_pre->beta = j - i + 1;
                }
                if (j == rec_len - 1) {                // rec[i,j] is suffix
                    node->post_suf->inv_list[0] += 1;  // inv_list[0]: c_suf
                    node->post_suf->inv_list.push_back(rid);
                    node->post_suf->inv_list.push_back(1);
                    node->post_suf->inv_list.push_back(node->post_suf->pos_list->size());
                    node->post_suf->pos_list->push_back(i + 1);
                    node->post_suf->beta = j - i + 1;
                }
            }
        }
    }

    LEADERpNode* find_node(const u32string& rec) {
        LEADERpNode* node = this->root;
        char32_t ch;
        for (int i = 0; i < (int)rec.size(); i++) {
            ch = rec[i];
            if (node->children.find(ch) == node->children.end()) {
                return this->empty_node;
            }
            node = node->children[ch];
        }
        return node;
    }

    int card_token1(const Posting* posting, const u32string& op_token1, const u32string& op_token2) {
        int card = 0;
        int idx = 0;
        int beta = posting->beta;
        int n = posting->inv_list[0] * 3;
        int sid;
        int n_pos;
        int p_idx;
        int p_start;
        int pos;
        const vector<int>& pos_list = *posting->pos_list;
        const vector<int>& inv_list = posting->inv_list;

        bool percent1 = false;
        bool percent2 = false;
        int under1 = 0;
        int under2 = 0;
        bool is_match;

        if (op_token1.size() > 0) {
            percent1 = op_token1[0] == U'%';
            under1 = op_token1.size() - (int)percent1;
        }
        if (op_token2.size() > 0) {
            percent2 = op_token2[0] == U'%';
            under2 = op_token2.size() - (int)percent2;
        }

        int number1 = under1 + 1;
        int number2 = beta + under2 - 1;

        while (idx < n) {
            sid = inv_list[++idx];
            n_pos = inv_list[++idx];
            p_start = inv_list[++idx];

            is_match = false;
            for (p_idx = p_start; p_idx < p_start + n_pos; ++p_idx) {
                pos = pos_list[p_idx];
                // select prefix
                if (!percent1) {           // fixed-length
                    if (pos == number1) {  // matched first position
                        // select suffix
                        if (!percent2) {  // fixed-length
                            is_match = s_lens[sid] - pos == number2;
                        } else {  // variable-length
                            is_match = s_lens[sid] - pos >= number2;
                        }
                        break;
                    }
                } else {  // variable-length
                    if (pos >= number1) {
                        break;
                    }
                }
            }
            if (percent1) {  // variable-length
                for (; p_idx < p_start + n_pos; ++p_idx) {
                    // select suffix
                    pos = pos_list[p_idx];
                    if (!percent2) {  // fixed-length
                        if (s_lens[sid] - pos == number2) {
                            is_match = true;
                            break;
                        }
                    } else {  // variable-length
                        is_match = s_lens[sid] - pos >= number2;
                        break;
                    }
                }
            }

            if (is_match) {
                card += 1;
            }
        }

        return card;
    }

    Posting* join_two_plists_old(const Posting* posting1, const Posting* posting2, const u32string& op_token, bool opt_next, bool opt_last) {
        // this function forward
        bool percent = false;
        int under = 0;
        if (op_token.size() > 0) {
            percent = op_token[0] == U'%';
            under = op_token.size() - (int)percent;
        }
        int beta = posting1->beta;
        int number = under + beta;

        Posting* output_posting;
        if (percent) {
            output_posting = new Posting(false, true);
            output_posting->pos_list = posting2->pos_list;
        } else {
            output_posting = new Posting(true, true);
        }
        output_posting->beta = posting2->beta;

        int idx1 = 1;
        int idx2 = 1;
        int n1 = posting1->inv_list[0] * 3 + 1;
        int n2 = posting2->inv_list[0] * 3 + 1;
        int sid1;
        int sid2;
        int n_pos1;
        int n_pos2;
        int p_idx1;
        int p_idx2;
        int p_start1;
        int p_start2;
        int pos1;
        int pos2;
        int curr_n_idx = -1;

        vector<int>& pos_list1 = *posting1->pos_list;
        const vector<int>& inv_list1 = posting1->inv_list;
        vector<int>& pos_list2 = *posting2->pos_list;
        const vector<int>& inv_list2 = posting2->inv_list;
        vector<int>& inv_list_out = output_posting->inv_list;
        vector<int>& pos_list_out = *output_posting->pos_list;

        while (idx1 < n1 && idx2 < n2) {  // s-predicate join
            sid1 = inv_list1[idx1];
            sid2 = inv_list2[idx2];

            if (sid1 < sid2) {
                idx1 += 3;
            } else if (sid1 > sid2) {
                idx2 += 3;
            } else {
                n_pos1 = inv_list1[++idx1];
                n_pos2 = inv_list2[++idx2];
                p_start1 = inv_list1[++idx1];
                p_start2 = inv_list2[++idx2];
                if (!percent) {  // fixed-length pattern
                    curr_n_idx = -1;
                    p_idx1 = p_start1;
                    p_idx2 = p_start2;
                    while (p_idx1 < p_start1 + n_pos1 && p_idx2 < p_start2 + n_pos2) {  // merge like
                        pos1 = pos_list1[p_idx1];
                        pos2 = pos_list2[p_idx2];
                        if (pos2 - pos1 > number) {
                            ++p_idx1;
                        } else if (pos2 - pos1 < number) {
                            ++p_idx2;
                        } else {
                            if (curr_n_idx < 0) {
                                ++inv_list_out[0];
                                if (opt_last) {
                                    break;
                                }
                                inv_list_out.push_back(sid1);
                                curr_n_idx = inv_list_out.size();
                                inv_list_out.push_back(0);
                                inv_list_out.push_back(pos_list_out.size());
                            }
                            inv_list_out[curr_n_idx] += 1;
                            pos_list_out.push_back(pos2);
                            if (opt_next) {
                                break;
                            }
                            ++p_idx1;
                            ++p_idx2;
                        }
                    }
                } else {  // variable-length pattern
                    pos1 = pos_list1[p_start1];
                    for (p_idx2 = p_start2; p_idx2 < p_start2 + n_pos2; ++p_idx2) {
                        pos2 = pos_list2[p_idx2];
                        if (pos2 - pos1 >= number) {
                            ++inv_list_out[0];
                            if (opt_last) {
                                break;
                            }
                            inv_list_out.push_back(sid1);
                            if (opt_next) {
                                inv_list_out.push_back(1);
                            } else {
                                inv_list_out.push_back(p_start2 + n_pos2 - p_idx2);
                            }
                            inv_list_out.push_back(p_idx2);
                            break;
                        }
                    }
                }
                ++idx1;
                ++idx2;
            }
        }

        // if (posting1->view) {
        //     delete posting1;
        // }
        // if (posting2->view) {
        //     delete posting2;
        // }

        return output_posting;
    }

    void find_stat(u32string& query, vector<int>& stat, int max_m) {
        vector<u32string> op_tokens;
        vector<u32string> tokens;

        u32string token;

        splitQuery(query, op_tokens, tokens);
        int n_tokens = tokens.size();

        Posting* list1 = this->find_node(tokens[0])->post;
        Posting* list2 = this->find_node(tokens[n_tokens - 1])->post;
        stat.push_back(n_tokens);
        stat.push_back(list1->inv_list[0]);
        stat.push_back(list2->inv_list[0]);

        list1 = get_prefix_plists(list1, op_tokens[0], false, true);
        list2 = get_suffix_plists(list2, op_tokens[n_tokens], false, true);
        stat.push_back(list1->inv_list[0]);
        stat.push_back(list2->inv_list[0]);
        stat.push_back(list1->pos_list->size());
        stat.push_back(list2->pos_list->size());
        delete list1;
        delete list2;

        vector<int> size_list;
        vector<int> n_tuple_list;
        int minimum = INT_MAX;
        int minimum_head = INT_MAX;
        int minimum_tail = INT_MAX;
        int size;
        int n_tuple;
        for (int i = 1; i <= n_tokens; ++i) {
            size = this->find_node(tokens[i - 1])->post->inv_list[0];
            n_tuple = this->find_node(tokens[i - 1])->post->pos_list->size();
            size_list.push_back(size);
            n_tuple_list.push_back(n_tuple);
            if (minimum > size) {
                minimum = size;
            }
            if ((i < n_tokens) && (minimum_head > size)) {
                minimum_head = size;
            }
            if ((i > 1) && (minimum_tail > size)) {
                minimum_tail = size;
            }
        }
        stat.push_back(minimum);
        stat.push_back(minimum_head);
        stat.push_back(minimum_tail);
        for (int i = 1; i <= max_m; ++i) {
            if (i <= n_tokens) {
                stat.push_back(size_list[i - 1]);
            } else {
                stat.push_back(0);
            }
        }
        for (int i = 1; i <= max_m; ++i) {
            if (i <= n_tokens) {
                stat.push_back(n_tuple_list[i - 1]);
            } else {
                stat.push_back(0);
            }
        }
    }
    void create_stacks(int max_size) {
        this->posting_stack = new Posting*[max_size + 1];
        this->status = new CHAR_STAT[max_size + 1];
        this->saved_p_prev = new int[max_size + 1];
        this->saved_p = new int[max_size + 1];
        this->last_op_token = new u32string[max_size + 1];
        this->last_token = new u32string[max_size + 1];

        for (int i = 0; i < max_size + 1; ++i) {
            this->posting_stack[i] = nullptr;
        }
        this->status[0] = CHAR_STAT::WILDCARD;
        this->saved_p_prev[0] = 0;
        this->saved_p[0] = 0;
        this->last_op_token[0] = U"";
        this->last_token[0] = U"";
    }

    void update_posting_stack(int idx, Posting* post) {
        auto ptr = this->posting_stack[idx];
        if (ptr && ptr->view) {
            delete ptr;
        }
        this->posting_stack[idx] = post;
    }

    void update_stacks(int idx, char32_t chr) {
        assert(idx > 0);
        // status
        bool is_wc = IS_UWILDCARD(chr);
        CHAR_STAT curr_status;
        if (is_wc) {
            curr_status = CHAR_STAT::WILDCARD;
        } else {
            if (this->status[idx - 1] == CHAR_STAT::WILDCARD) {
                curr_status = CHAR_STAT::FIRST_REGULAR;
            } else {
                curr_status = CHAR_STAT::REGULAR;
            }
        }
        this->status[idx] = curr_status;

        // saved_p_prev
        if (curr_status == CHAR_STAT::WILDCARD) {
            this->saved_p_prev[idx] = this->saved_p[idx - 1];
        } else {
            this->saved_p_prev[idx] = this->saved_p_prev[idx - 1];
        }

        // saved_p
        if (curr_status == CHAR_STAT::WILDCARD) {
            this->saved_p[idx] = this->saved_p[idx - 1];
        } else {
            this->saved_p[idx] = idx;
        }

        // last_op_token
        if (curr_status == CHAR_STAT::WILDCARD) {
            if (this->status[idx - 1] == CHAR_STAT::WILDCARD) {
                this->last_op_token[idx] = this->last_op_token[idx - 1];
                this->last_op_token[idx].push_back(chr);
            } else {
                this->last_op_token[idx] = chr;
            }
        } else {
            this->last_op_token[idx] = this->last_op_token[idx - 1];
        }

        // last_token
        if (curr_status == CHAR_STAT::WILDCARD) {
            if (this->status[idx - 1] == CHAR_STAT::WILDCARD) {
                this->last_token[idx] = this->last_token[idx - 1];
            } else {
                this->last_token[idx] = U"";
            }
        } else {
            this->last_token[idx] = this->last_token[idx - 1];
            this->last_token[idx].push_back(chr);
        }
    }

    int find_card(int idx, char32_t chr, bool is_sp, bool is_card) {
        // ofstream ofs;
        // ofs.open("tmp.csv", fstream::app);
        // ofs << "idx: " << idx << ", chr: " << chr << ", is_sp: " << is_sp << ", is_card: " << is_card << endl;
        // ofs << idx << ", " << chr << ", " << is_sp << ", " << is_card << endl;

        this->update_stacks(idx, chr);

        int card = 0;

        CHAR_STAT status = this->status[idx];
        int saved_p_prev = this->saved_p_prev[idx];
        Posting* prev_posting = this->posting_stack[saved_p_prev];
        Posting* save_posting = nullptr;
        Posting* posting;
        u32string& op_token = this->last_op_token[idx];
        u32string& token = this->last_token[idx];
        LEADERpNode* curr_node;

        int n_rels = 0;
        if (prev_posting) {
            ++n_rels;
        }
        if (token != U"") {
            ++n_rels;
        }

        // find R(q)
        if (n_rels == 0) {  // wildcards only (e.g., "%", "_", "%__" and "__")
            if (is_card) {
                card = 0;
                bool percent = op_token[0] == U'%';
                int under = op_token.size() - (int)percent;
                int s_len;
                for (int i = 0; i < this->n_db; ++i) {
                    s_len = s_lens[i];
                    if ((percent && s_len >= under) || (!percent && s_len == under)) {
                        card += 1;
                    }
                }
            }
            assert(save_posting == nullptr);
        } else if (n_rels == 1) {
            if (prev_posting) {  // selection suffix
                if (is_card) {
                    assert(token == U"");
                    assert(status == CHAR_STAT::WILDCARD);
                    if (op_token == U"%") {
                        card = prev_posting->inv_list[0];
                    } else {
                        card = this->card_token1(prev_posting, U"%", op_token);
                    }
                    assert(save_posting == nullptr);
                }
            } else {  // selection prefix
                // posting: [op_token token %]
                // card: [op_token token]
                assert(token != U"");
                if (is_sp || is_card) {
                    curr_node = this->find_node(token);
                }

                // find save_posting: [op_token token %]
                if (is_sp) {
                    if (op_token == U"") {  // [token %]
                        save_posting = curr_node->post_pre;
                    } else if (op_token == U"%") {  // [% token %]
                        save_posting = curr_node->post;
                    } else {  // [op_token token %]
                        save_posting = curr_node->post;
                        save_posting = get_prefix_plists(save_posting, op_token, false, false);
                    }
                }

                // find card
                if (is_card) {
                    card = 0;
                    posting = curr_node->post_suf;
                    if (posting) {
                        if (op_token == U"%") {  // [% token]
                            card = posting->inv_list[0];
                        } else {  // [op_token token]
                            card = this->card_token1(posting, op_token, U"%");
                        }
                    }
                } else {
                    card = -1;
                }
            }
        } else {
            // posting: [prev_posting op_token token %]
            // card: [prev_posting op_token token]
            assert(!token.empty());
            assert(prev_posting);
            if (is_sp || is_card) {
                curr_node = this->find_node(token);
            }

            // find save_posting: [prev_posting op_token token %]
            if (is_sp) {
                save_posting = curr_node->post;
                save_posting = join_two_plists(prev_posting, save_posting, op_token, false, false);
            }

            // find card: [prev_posting op_token token]
            if (is_card) {
                posting = curr_node->post_suf;
                if (posting) {
                    posting = join_two_plists(prev_posting, posting, op_token, false, true);
                    card = posting->inv_list[0];
                }
            } else {
                card = -1;
            }
        }

        if (is_sp) {
            this->update_posting_stack(idx, save_posting);
        }

        return card;
    }

    int find_card_with_plan(u32string& query, vector<int> order) {
        int card = 0;
        TreePlan* plan = new TreePlan();

        vector<u32string> op_tokens;
        vector<u32string> tokens;
        int n_tokens;
        int n_under;
        bool percent;
        u32string token;

        splitQuery(query, op_tokens, tokens);
        // cout << "n_tokens: " << n_tokens << endl;
        // for (auto token : tokens) {
        //     cout << utf8::utf32to8(token) << endl;
        // }
        n_tokens = tokens.size();

        if (n_tokens == 0) {
            // cout << utf8::utf32to8(query) << endl;
            token = op_tokens[0];
            percent = token[0] == U'%';
            n_under = token.size() - (int)percent;

            for (int i = 0; i < this->n_db; i++) {
                if ((percent && s_lens[i] >= n_under) || (!percent && s_lens[i] == n_under)) {
                    ++card;
                }
            }
            return card;
        } else if (n_tokens == 1) {
            if ((op_tokens[0] == U"%") && (op_tokens.back() == U"%")) {
                card = this->find_node(tokens[0])->post->inv_list[0];
            } else if ((op_tokens[0] == U"") && (op_tokens.back() == U"%")) {
                card = this->find_node(tokens[0])->post_pre->inv_list[0];
            } else if ((op_tokens[0] == U"%") && (op_tokens.back() == U"")) {
                card = this->find_node(tokens[0])->post_suf->inv_list[0];
            } else {
                card = this->card_token1(this->find_node(tokens[0])->post, op_tokens[0], op_tokens[1]);
            }
            return card;
        }

        // assert(n_tokens >= 1);

        vector<Posting*> tables;
        Posting* table;

        // CliqueGenHashTableEntry* node;
        // LEADERpNode* node;
        for (int i = 0; i < n_tokens; i++) {
            token = tokens[i];
            if (i == 0 && op_tokens[0] == U"") {  // CHECK
                table = this->find_node(token)->post_pre;
            } else if (i == n_tokens - 1 && op_tokens[n_tokens] == U"") {
                table = this->find_node(token)->post_suf;
            } else {
                table = this->find_node(token)->post;
            }
            tables.push_back(table);
        }

        // Test 1

        plan->set_Postings(tables);
        // plan->construct_by_order(query, order);
        plan->construct_by_order(op_tokens, tokens, order);
        // cout << utf8::utf32to8(query) << endl;
        // plan->print();
        card = plan->find_card();

        delete plan;

        return card;
    }

    int find_card_old(u32string& query) {
        int card = 0;
        vector<u32string> op_tokens;
        vector<u32string> tokens;

        u32string token;

        splitQuery(query, op_tokens, tokens);
        int n_tokens = tokens.size();
        bool skip_prefix = op_tokens[0] == U"%";
        bool skip_suffix = op_tokens[n_tokens] == U"%";

        Posting* list1;
        Posting* list2;
        Posting* posting;

        // find R(q)
        if (n_tokens == 0) {
            card = 0;
            u32string op_token = op_tokens[0];
            bool percent = op_token[0] == U'%';
            int under = op_token.size() - (int)percent;
            int s_len;
            for (int i = 0; i < this->n_db; ++i) {
                s_len = s_lens[i];
                if ((percent && s_len >= under) || (!percent && s_len == under)) {
                    card += 1;
                }
            }
        } else if (n_tokens == 1) {
            if (skip_prefix & skip_suffix) {
                card = this->find_node(tokens[0])->post->inv_list[0];
            } else if ((op_tokens[0] == U"") & skip_suffix) {
                posting = this->find_node(tokens[0])->post_pre;
                card = 0;
                if (posting) {
                    card = posting->inv_list[0];
                }
            } else if (skip_prefix & (op_tokens[1] == U"")) {
                posting = this->find_node(tokens[0])->post_suf;
                card = 0;
                if (posting) {
                    card = posting->inv_list[0];
                }
            } else {
                list1 = this->find_node(tokens[0])->post;
                card = this->card_token1(list1, op_tokens[0], op_tokens[1]);
            }
        } else {
            list1 = this->find_node(tokens[0])->post;
            if (!skip_prefix) {
                list1 = get_prefix_plists(list1, op_tokens[0], op_tokens[1][0] == U'%', false);
            }
            for (int i = 2; i <= n_tokens; ++i) {
                list2 = this->find_node(tokens[i - 1])->post;
                if (!skip_suffix && i == n_tokens) {
                    list2 = get_suffix_plists(list2, op_tokens[n_tokens], op_tokens[n_tokens - 1][0] == U'%', false);
                }

                posting = join_two_plists(list1, list2, op_tokens[i - 1], op_tokens[i][0] == U'%', i == n_tokens);
                if (list1->view) {
                    delete list1;
                }
                if (list2->view) {
                    delete list2;
                }

                list1 = posting;
            }
            card = list1->inv_list[0];
            if (list1->view) {
                delete list1;
            }
        }

        return card;
    }
};

struct CliqueGenHashTableEntry {
    vector<int> inv_list;
    int curr_idx = -1;
    int curr_id = -1;
    int c_sub = 0;
    int c_pre = 0;
    int c_suf = 0;
};

InvTree* build_inv_token_tree(u32string* qset, int n_qry, u32string* db, int n_db) {
    InvTree* tree = new InvTree();
    tree->create_trie(qset, n_qry);
    tree->build_ivd(db, n_db);
    return tree;
}

LEADERpTree* build_SPADE_PLUS_tree(u32string* qset, int n_qry, u32string* db, int n_db) {
    LEADERpTree* tree = new LEADERpTree();
    tree->create_trie(qset, n_qry);
    tree->build_ivd(db, n_db);
    return tree;
}

#endif /* ADBB56DD_01F2_4F2D_889A_708C381C56D5 */
