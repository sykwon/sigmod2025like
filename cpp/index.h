#ifndef ADBB56DD_01F2_4F2D_889A_708C381C56D5
#define ADBB56DD_01F2_4F2D_889A_708C381C56D5

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "plan.h"
#include "util.h"
#define MAX_KEY_LEN 5
#define POSTING_HASH_SIZE 100000
#define COND_B(pr, po, c) (((pr) && (po) >= (c)) || (!(pr) && (po) == (c)))
#define COND_E(pr, po, c) (((pr) && (po) <= (c)) || (!(pr) && (po) == (c)))
using namespace std;

static bool is_ineq_opt = true;
static bool is_mp_only = false;
static bool is_bin_srch = false;

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

class Posting {
public:
    vector<int> inv_list = {0};
    vector<int>* pos_list;
    int beta;
    bool owner = true;
    bool view = false;

    Posting(bool owner, bool view) {
        this->owner = owner;
        this->view = view;
        if (this->owner) {
            this->pos_list = new vector<int>();
        }
    }

    ~Posting() {
        if (this->owner) {
            delete pos_list;
        }
    }
};

int binary_search_for_position_set(const vector<int>& pos_list, int l, int r, int target) {
    // It returns the position larger than or equal to target which appears the first in the pos_list.
    // If all posiitons less than n_pos, it returns -1.

    if (pos_list[l] >= target) {
        return l;
    }
    if (pos_list[r] < target) {
        return -1;
    }

    // assert(l < r);

    int m = -1;
    ++l;

    while (l < r) {
        m = l + (r - l) / 2;

        if (pos_list[m] >= target) {
            r = m;
        } else {
            l = m + 1;
        }
    }
    // assert(l == r);
    return r;
}

void inequality_p_predicate_join_bin_srch(vector<int>& inv_list_out,
                                          int pos1, const vector<int>& pos_list2,
                                          int p_start2, int n_pos2, int number, bool opt_last, bool opt_next, int sid1) {
    int p_idx2 = binary_search_for_position_set(pos_list2, p_start2, p_start2 + n_pos2 - 1, number + pos1);
    if (p_idx2 >= 0) {
        ++inv_list_out[0];
        if (!opt_last) {
            inv_list_out.push_back(sid1);
            if (opt_next) {
                inv_list_out.push_back(1);
            } else {
                inv_list_out.push_back(p_start2 + n_pos2 - p_idx2);
            }
            inv_list_out.push_back(p_idx2);
        }
    }
}

void inequality_p_predicate_join(vector<int>& inv_list_out,
                                 const vector<int>& pos_list1, const vector<int>& pos_list2,
                                 int p_start1, int p_start2, int n_pos2, int number, bool opt_last, bool opt_next, int sid1) {
    int pos1;
    int pos2;
    int p_idx2;

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

void inequality_p_predicate_join_mp_only(vector<int>& inv_list_out, vector<int>& pos_list_out,
                                         const vector<int>& pos_list1, const vector<int>& pos_list2,
                                         int p_start1, int p_start2, int n_pos2, int number, int sid1) {
    int pos1;
    int pos2;
    int p_idx2;

    pos1 = pos_list1[p_start1];
    int curr_n_idx = -1;
    for (p_idx2 = p_start2; p_idx2 < p_start2 + n_pos2; ++p_idx2) {
        pos2 = pos_list2[p_idx2];
        if (pos2 - pos1 >= number) {
            if (curr_n_idx < 0) {
                ++inv_list_out[0];
                inv_list_out.push_back(sid1);
                curr_n_idx = inv_list_out.size();
                inv_list_out.push_back(0);
                inv_list_out.push_back(pos_list_out.size());
            }
            inv_list_out[curr_n_idx] += 1;
            pos_list_out.push_back(pos2);
        }
    }
}

void inequality_p_predicate_join_all_pair(vector<int>& inv_list_out, vector<int>& pos_list_out,
                                          const vector<int>& pos_list1, const vector<int>& pos_list2,
                                          int p_start1, int p_start2, int n_pos1, int n_pos2, int number, int sid1) {
    int pos1;
    int pos2;
    int p_idx1;
    int p_idx2;

    vector<int> all_pos_list;

    for (p_idx1 = p_start1; p_idx1 < p_start1 + n_pos1; ++p_idx1) {
        pos1 = pos_list1[p_start1];
        for (p_idx2 = p_start2; p_idx2 < p_start2 + n_pos2; ++p_idx2) {
            pos2 = pos_list2[p_idx2];
            if (pos2 - pos1 >= number) {
                all_pos_list.push_back(pos2);
            }
        }
    }
    int curr_n_idx = -1;
    if (all_pos_list.size() > 0) {
        sort(all_pos_list.begin(), all_pos_list.end());
        pos1 = -1;

        ++inv_list_out[0];
        inv_list_out.push_back(sid1);
        curr_n_idx = inv_list_out.size();
        inv_list_out.push_back(0);
        inv_list_out.push_back(pos_list_out.size());
        for (int i = 0; i < (int)all_pos_list.size(); ++i) {
            pos2 = all_pos_list[i];
            if (pos2 > pos1) {
                inv_list_out[curr_n_idx] += 1;
                pos_list_out.push_back(pos2);

                pos1 = pos2;
            }
        }
    }
}

void equality_p_predicate_join(vector<int>& inv_list_out, vector<int>& pos_list_out,
                               const vector<int>& pos_list1, const vector<int>& pos_list2,
                               int p_start1, int p_start2, int n_pos1, int n_pos2, int number, bool opt_last, bool opt_next, int sid1) {
    int pos1;
    int pos2;
    int curr_n_idx = -1;
    int p_idx1 = p_start1;
    int p_idx2 = p_start2;
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
}

void p_predicate_join(bool percent, int n_pos1, int n_pos2, int p_start1, int p_start2,
                      vector<int>& inv_list_out, vector<int>& pos_list_out,
                      const vector<int>& pos_list1, const vector<int>& pos_list2,
                      int number, bool opt_next, bool opt_last, int sid1) {
    if (!percent) {  // fixed-length pattern
        equality_p_predicate_join(inv_list_out, pos_list_out, pos_list1, pos_list2,
                                  p_start1, p_start2, n_pos1, n_pos2, number, opt_last, opt_next, sid1);
    } else {  // variable-length pattern
        if (is_bin_srch) {
            inequality_p_predicate_join_bin_srch(inv_list_out, pos_list1[p_start1], pos_list2,
                                                 p_start2, n_pos2, number, opt_last, opt_next, sid1);
        } else if (is_mp_only) {
            inequality_p_predicate_join_mp_only(inv_list_out, pos_list_out, pos_list1, pos_list2,
                                                p_start1, p_start2, n_pos2, number, sid1);
        } else if (is_ineq_opt) {
            inequality_p_predicate_join(inv_list_out, pos_list1, pos_list2,
                                        p_start1, p_start2, n_pos2, number, opt_last, opt_next, sid1);
        } else {
            inequality_p_predicate_join_all_pair(inv_list_out, pos_list_out, pos_list1, pos_list2,
                                                 p_start1, p_start2, n_pos1, n_pos2, number, sid1);
        }
    }
}

Posting* join_two_plists(const Posting* posting1, const Posting* posting2, const u32string& op_token, bool opt_next, bool opt_last) {
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
    // int p_idx1;
    // int p_idx2;
    int p_start1;
    int p_start2;
    // int pos1;
    // int pos2;
    // int curr_n_idx = -1;

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
            p_predicate_join(percent, n_pos1, n_pos2, p_start1, p_start2, inv_list_out, pos_list_out, pos_list1, pos_list2, number, opt_next, opt_last, sid1);
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
    int* s_lens;
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
        delete s_lens;
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
        this->s_lens = new int[n_db];
        this->n_db = n_db;
        for (int id = 0; id < n_db; id++) {
            rec = &db[id];
            this->build_ivs(*rec, id);
            this->s_lens[id] = rec->size();
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
        this->s_lens = new int[n_db];
        this->n_db = n_db;
        for (int id = 0; id < n_db; id++) {
            rec = &db[id];
            this->add_all_suffixes_pos(*rec, id);
            this->s_lens[id] = rec->size();
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
                            is_match = this->s_lens[sid] - pos == number2;
                        } else {  // variable-length
                            is_match = this->s_lens[sid] - pos >= number2;
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
                        if (this->s_lens[sid] - pos == number2) {
                            is_match = true;
                            break;
                        }
                    } else {  // variable-length
                        is_match = this->s_lens[sid] - pos >= number2;
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

    Posting* get_prefix_plists(Posting* input_posting, const u32string& op_token, bool opt_next, bool opt_last) {
        Posting* output_posting = new Posting(false, true);
        output_posting->pos_list = input_posting->pos_list;
        output_posting->beta = input_posting->beta;

        int idx = 0;
        int n = input_posting->inv_list[0] * 3;
        int sid;
        int n_pos;
        int p_idx;
        int p_start;
        int pos;
        const vector<int>& pos_list = *input_posting->pos_list;
        const vector<int>& inv_list = input_posting->inv_list;
        vector<int>& output_list = output_posting->inv_list;

        bool percent = false;
        int under = 0;

        if (op_token.size() > 0) {
            percent = op_token[0] == U'%';
            under = op_token.size() - (int)percent;
        }

        int number = under + 1;
        while (idx < n) {
            sid = inv_list[++idx];
            n_pos = inv_list[++idx];
            p_start = inv_list[++idx];

            for (p_idx = p_start; p_idx < p_start + n_pos; ++p_idx) {
                pos = pos_list[p_idx];
                if (!percent) {  // fixed-length
                    if (pos == number) {
                        ++output_list[0];
                        if (opt_last) {
                            break;
                        }
                        output_list.push_back(sid);
                        output_list.push_back(1);
                        output_list.push_back(p_idx);
                        break;
                    }
                } else {  //  variable-length
                    if (pos >= number) {
                        ++output_list[0];
                        if (opt_last) {
                            break;
                        }
                        output_list.push_back(sid);
                        if (opt_next) {
                            output_list.push_back(1);
                        } else {
                            output_list.push_back(n_pos + p_start - p_idx);
                        }
                        output_list.push_back(p_idx);
                        break;
                    }
                }
            }
        }
        if (input_posting->view) {
            delete input_posting;
        }

        return output_posting;
    }

    Posting* get_suffix_plists(Posting* input_posting, const u32string& op_token, bool opt_next, bool opt_last) {
        Posting* output_posting = new Posting(false, true);
        output_posting->pos_list = input_posting->pos_list;
        output_posting->beta = input_posting->beta;

        int idx = 0;
        int n = input_posting->inv_list[0] * 3;
        int sid;
        int n_pos;
        int p_idx;
        int p_start;
        int pos;
        int s_len;
        const vector<int>& pos_list = *input_posting->pos_list;
        const vector<int>& inv_list = input_posting->inv_list;
        vector<int>& output_list = output_posting->inv_list;

        bool percent = false;
        int under = 0;
        if (op_token.size() > 0) {
            percent = op_token[0] == U'%';
            under = op_token.size() - (int)percent;
        }
        int beta = input_posting->beta;
        int number = under + beta - 1;

        while (idx < n) {
            sid = inv_list[++idx];
            n_pos = inv_list[++idx];
            p_start = inv_list[++idx];
            s_len = s_lens[sid];

            for (p_idx = p_start + n_pos - 1; p_idx >= p_start; --p_idx) {  // backward
                pos = pos_list[p_idx];
                if (!percent) {  // fixed-length
                    if (s_len - pos == number) {
                        ++output_list[0];
                        if (opt_last) {
                            break;
                        }
                        output_list.push_back(sid);
                        output_list.push_back(1);
                        output_list.push_back(p_idx);
                        break;
                    }
                } else {  // variable-length
                    if (s_len - pos >= number) {
                        ++output_list[0];
                        if (opt_last) {
                            break;
                        }
                        output_list.push_back(sid);
                        if (opt_next) {
                            output_list.push_back(1);
                            output_list.push_back(p_idx);
                        } else {
                            output_list.push_back(p_idx - p_start + 1);
                            output_list.push_back(p_start);
                        }
                        break;
                    }
                }
            }
        }

        return output_posting;
    }

    Posting* join_two_plists_backward(Posting* posting1, Posting* posting2, const u32string& op_token, bool opt_next, bool opt_last) {
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
            output_posting->pos_list = posting1->pos_list;
        } else {
            output_posting = new Posting(true, true);
        }
        output_posting->beta = posting1->beta;

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
        vector<int>& inv_list1 = posting1->inv_list;
        vector<int>& pos_list2 = *posting2->pos_list;
        vector<int>& inv_list2 = posting2->inv_list;
        vector<int>& inv_list_out = output_posting->inv_list;
        vector<int>& pos_list_out = *output_posting->pos_list;

        while (idx1 < n1 && idx2 < n2) {
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
                    if (opt_next) {
                        p_idx1 = p_start1 + n_pos1 - 1;
                        p_idx2 = p_start2 + n_pos2 - 1;
                        while (p_idx1 >= p_start1 && p_idx2 >= p_start2) {  // merge like
                            pos1 = pos_list1[p_idx1];
                            pos2 = pos_list2[p_idx2];
                            if (pos2 - pos1 > number) {
                                --p_idx2;
                            } else if (pos2 - pos1 < number) {
                                --p_idx1;
                            } else {
                                ++inv_list_out[0];
                                if (opt_last) {
                                    break;
                                }
                                inv_list_out.push_back(sid1);
                                inv_list_out.push_back(1);
                                inv_list_out.push_back(pos_list_out.size());
                                pos_list_out.push_back(pos1);
                                break;
                            }
                        }
                    } else {
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
                                pos_list_out.push_back(pos1);
                                ++p_idx1;
                                ++p_idx2;
                            }
                        }
                    }

                } else {  // variable-length pattern
                    pos2 = pos_list2[p_start2 + n_pos2 - 1];
                    for (p_idx1 = p_start1 + n_pos1 - 1; p_idx1 >= p_start1; --p_idx1) {
                        pos1 = pos_list1[p_idx1];
                        if (pos2 - pos1 >= number) {
                            ++inv_list_out[0];
                            if (opt_last) {
                                break;
                            }
                            inv_list_out.push_back(sid1);
                            if (opt_next) {
                                inv_list_out.push_back(1);
                                inv_list_out.push_back(p_idx1);
                            } else {
                                inv_list_out.push_back(p_idx1 - p_start1 + 1);
                                inv_list_out.push_back(p_start1);
                            }
                            break;
                        }
                    }
                }
                ++idx1;
                ++idx2;
            }
        }

        if (posting1->view) {
            delete posting1;
        }
        if (posting2->view) {
            delete posting2;
        }

        return output_posting;
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

        list1 = this->get_prefix_plists(list1, op_tokens[0], false, true);
        list2 = this->get_suffix_plists(list2, op_tokens[n_tokens], false, true);
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
                        save_posting = this->get_prefix_plists(save_posting, op_token, false, false);
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
                list1 = this->get_prefix_plists(list1, op_tokens[0], op_tokens[1][0] == U'%', false);
            }
            for (int i = 2; i <= n_tokens; ++i) {
                list2 = this->find_node(tokens[i - 1])->post;
                if (!skip_suffix && i == n_tokens) {
                    list2 = this->get_suffix_plists(list2, op_tokens[n_tokens], op_tokens[n_tokens - 1][0] == U'%', false);
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

class LEADERTree {
public:
    static const int max_word_len = MAX_KEY_LEN;
    LEADERNode* root = nullptr;
    int* s_lens;
    int n_db;
    bool opt_view = false;
    bool opt_next = false;
    bool opt_last = false;
    LEADER_DIR opt_dir = LEADER_DIR::FORWARD;
    LEADER_SEL opt_sel = LEADER_SEL::LAST;

    LEADERTree() {
        this->root = new LEADERNode();
    }

    ~LEADERTree() {
        delete root;
        delete s_lens;
    }

    void add_strings(u32string* db, int n_db) {
        u32string* rec;
        this->s_lens = new int[n_db];
        this->n_db = n_db;
        for (int id = 0; id < n_db; id++) {
            rec = &db[id];
            add_all_suffixes_pos(*rec, id);
            this->s_lens[id] = rec->size();
        }
    }
    void add_all_suffixes_pos(const u32string& rec, int rid) {
        // u32string suffix;
        int rec_len = (int)rec.size();
        int substr_len;
        LEADERNode* node;
        LEADERNode* child;
        char32_t ch;
        for (int i = 0; i < rec_len; i++) {
            substr_len = MIN(rec_len - i, max_word_len);
            node = this->root;
            for (int j = i; j < i + substr_len; j++) {
                ch = rec[j];
                if (node->children.find(ch) == node->children.end()) {
                    child = new LEADERNode();
                    node->children[ch] = child;
                }
                node = node->children[ch];
                if (rid != node->curr_id) {  // set initial values
                    node->post->beta = j - i + 1;
                    node->post->inv_list.push_back(rid);
                    node->curr_idx = node->post->inv_list.size();
                    node->post->inv_list.push_back(0);
                    node->post->inv_list.push_back(node->post->pos_list->size());
                    node->curr_id = rid;
                    node->post->inv_list[0] += 1;  // inv_list[0]: c_sub
                }
                node->post->pos_list->push_back(i + 1);  // add pos
                // if (node->min_pos > i+1) {
                //     node->min_pos = i+1;
                // }
                // if (node->min_pos > i+1) {
                //     node->min_pos = i+1;
                // }
                node->post->inv_list[node->curr_idx] += 1;
                if (i == 0) {
                    node->c_pre += 1;
                }
                if (j == rec_len - 1) {
                    node->c_suf += 1;
                }
            }
        }
    }

    LEADERNode* find_node(const u32string& rec) {
        LEADERNode* node = this->root;
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
                            is_match = this->s_lens[sid] - pos == number2;
                        } else {  // variable-length
                            is_match = this->s_lens[sid] - pos >= number2;
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
                        if (this->s_lens[sid] - pos == number2) {
                            is_match = true;
                            break;
                        }
                    } else {  // variable-length
                        is_match = this->s_lens[sid] - pos >= number2;
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

    Posting* get_prefix_plists(Posting* input_posting, const u32string& op_token, bool opt_next, bool opt_last) {
        Posting* output_posting = new Posting(false, true);
        output_posting->pos_list = input_posting->pos_list;
        output_posting->beta = input_posting->beta;

        int idx = 0;
        int n = input_posting->inv_list[0] * 3;
        int sid;
        int n_pos;
        int p_idx;
        int p_start;
        int pos;
        const vector<int>& pos_list = *input_posting->pos_list;
        const vector<int>& inv_list = input_posting->inv_list;
        vector<int>& output_list = output_posting->inv_list;

        bool percent = false;
        int under = 0;

        if (op_token.size() > 0) {
            percent = op_token[0] == U'%';
            under = op_token.size() - (int)percent;
        }

        int number = under + 1;
        while (idx < n) {
            sid = inv_list[++idx];
            n_pos = inv_list[++idx];
            p_start = inv_list[++idx];

            for (p_idx = p_start; p_idx < p_start + n_pos; ++p_idx) {
                pos = pos_list[p_idx];
                if (!percent) {  // fixed-length
                    if (pos == number) {
                        ++output_list[0];
                        if (opt_last) {
                            break;
                        }
                        output_list.push_back(sid);
                        output_list.push_back(1);
                        output_list.push_back(p_idx);
                        break;
                    }
                } else {  //  variable-length
                    if (pos >= number) {
                        ++output_list[0];
                        if (opt_last) {
                            break;
                        }
                        output_list.push_back(sid);
                        if (opt_next) {
                            output_list.push_back(1);
                        } else {
                            output_list.push_back(n_pos + p_start - p_idx);
                        }
                        output_list.push_back(p_idx);
                        break;
                    }
                }
            }
        }
        if (input_posting->view) {
            delete input_posting;
        }

        return output_posting;
    }

    Posting* get_suffix_plists(Posting* input_posting, const u32string& op_token, bool opt_next, bool opt_last) {
        Posting* output_posting = new Posting(false, true);
        output_posting->pos_list = input_posting->pos_list;
        output_posting->beta = input_posting->beta;

        int idx = 0;
        int n = input_posting->inv_list[0] * 3;
        int sid;
        int n_pos;
        int p_idx;
        int p_start;
        int pos;
        int s_len;
        const vector<int>& pos_list = *input_posting->pos_list;
        const vector<int>& inv_list = input_posting->inv_list;
        vector<int>& output_list = output_posting->inv_list;

        bool percent = false;
        int under = 0;
        if (op_token.size() > 0) {
            percent = op_token[0] == U'%';
            under = op_token.size() - (int)percent;
        }
        int beta = input_posting->beta;
        int number = under + beta - 1;

        while (idx < n) {
            sid = inv_list[++idx];
            n_pos = inv_list[++idx];
            p_start = inv_list[++idx];
            s_len = s_lens[sid];

            for (p_idx = p_start + n_pos - 1; p_idx >= p_start; --p_idx) {  // backward
                pos = pos_list[p_idx];
                if (!percent) {  // fixed-length
                    if (s_len - pos == number) {
                        ++output_list[0];
                        if (opt_last) {
                            break;
                        }
                        output_list.push_back(sid);
                        output_list.push_back(1);
                        output_list.push_back(p_idx);
                        break;
                    }
                } else {  // variable-length
                    if (s_len - pos >= number) {
                        ++output_list[0];
                        if (opt_last) {
                            break;
                        }
                        output_list.push_back(sid);
                        if (opt_next) {
                            output_list.push_back(1);
                            output_list.push_back(p_idx);
                        } else {
                            output_list.push_back(p_idx - p_start + 1);
                            output_list.push_back(p_start);
                        }
                        break;
                    }
                }
            }
        }

        return output_posting;
    }

    Posting* join_two_plists_backward(Posting* posting1, Posting* posting2, const u32string& op_token, bool opt_next, bool opt_last) {
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
            output_posting->pos_list = posting1->pos_list;
        } else {
            output_posting = new Posting(true, true);
        }
        output_posting->beta = posting1->beta;

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
        vector<int>& inv_list1 = posting1->inv_list;
        vector<int>& pos_list2 = *posting2->pos_list;
        vector<int>& inv_list2 = posting2->inv_list;
        vector<int>& inv_list_out = output_posting->inv_list;
        vector<int>& pos_list_out = *output_posting->pos_list;

        while (idx1 < n1 && idx2 < n2) {
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
                    if (opt_next) {
                        p_idx1 = p_start1 + n_pos1 - 1;
                        p_idx2 = p_start2 + n_pos2 - 1;
                        while (p_idx1 >= p_start1 && p_idx2 >= p_start2) {  // merge like
                            pos1 = pos_list1[p_idx1];
                            pos2 = pos_list2[p_idx2];
                            if (pos2 - pos1 > number) {
                                --p_idx2;
                            } else if (pos2 - pos1 < number) {
                                --p_idx1;
                            } else {
                                ++inv_list_out[0];
                                if (opt_last) {
                                    break;
                                }
                                inv_list_out.push_back(sid1);
                                inv_list_out.push_back(1);
                                inv_list_out.push_back(pos_list_out.size());
                                pos_list_out.push_back(pos1);
                                break;
                            }
                        }
                    } else {
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
                                pos_list_out.push_back(pos1);
                                ++p_idx1;
                                ++p_idx2;
                            }
                        }
                    }

                } else {  // variable-length pattern
                    pos2 = pos_list2[p_start2 + n_pos2 - 1];
                    for (p_idx1 = p_start1 + n_pos1 - 1; p_idx1 >= p_start1; --p_idx1) {
                        pos1 = pos_list1[p_idx1];
                        if (pos2 - pos1 >= number) {
                            ++inv_list_out[0];
                            if (opt_last) {
                                break;
                            }
                            inv_list_out.push_back(sid1);
                            if (opt_next) {
                                inv_list_out.push_back(1);
                                inv_list_out.push_back(p_idx1);
                            } else {
                                inv_list_out.push_back(p_idx1 - p_start1 + 1);
                                inv_list_out.push_back(p_start1);
                            }
                            break;
                        }
                    }
                }
                ++idx1;
                ++idx2;
            }
        }

        if (posting1->view) {
            delete posting1;
        }
        if (posting2->view) {
            delete posting2;
        }

        return output_posting;
    }

    Posting* join_two_plists(Posting* posting1, Posting* posting2, const u32string& op_token, bool opt_next, bool opt_last) {
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
        // int p_idx1;
        // int p_idx2;
        int p_start1;
        int p_start2;
        // int pos1;
        // int pos2;
        // int curr_n_idx = -1;

        vector<int>& pos_list1 = *posting1->pos_list;
        vector<int>& inv_list1 = posting1->inv_list;
        vector<int>& pos_list2 = *posting2->pos_list;
        vector<int>& inv_list2 = posting2->inv_list;
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
                p_predicate_join(percent, n_pos1, n_pos2, p_start1, p_start2, inv_list_out, pos_list_out, pos_list1, pos_list2, number, opt_next, opt_last, sid1);
                ++idx1;
                ++idx2;
            }
        }

        if (posting1->view) {
            delete posting1;
        }
        if (posting2->view) {
            delete posting2;
        }

        return output_posting;
    }
    LEADER_DIR find_choices(u32string& query, bool is_sid) {
        vector<u32string> op_tokens;
        vector<u32string> tokens;

        u32string token;

        splitQuery(query, op_tokens, tokens);
        int n_tokens = tokens.size();
        // bool select_last;
        LEADER_DIR direct;

        float prefix_count;
        float suffix_count;
        // float prefix_count2 = this->find_node(tokens[0])->post->pos_list->size();
        // float suffix_count2 = this->find_node(tokens[n_tokens - 1])->post->pos_list->size();
        if (is_sid) {
            prefix_count = this->find_node(tokens[0])->post->inv_list[0];
            suffix_count = this->find_node(tokens[n_tokens - 1])->post->inv_list[0];
        } else {
            prefix_count = this->find_node(tokens[0])->post->pos_list->size();
            suffix_count = this->find_node(tokens[n_tokens - 1])->post->pos_list->size();
        }

        if (prefix_count <= suffix_count) {
            direct = LEADER_DIR::FORWARD;
        } else {
            direct = LEADER_DIR::BACKWARD;
        }
        return direct;
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

        list1 = this->get_prefix_plists(list1, op_tokens[0], false, true);
        list2 = this->get_suffix_plists(list2, op_tokens[n_tokens], false, true);
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

    int find_card(u32string& query) {
        int card = 0;
        vector<u32string> op_tokens;
        vector<u32string> tokens;

        u32string token;

        splitQuery(query, op_tokens, tokens);
        int n_tokens = tokens.size();
        bool skip_prefix = op_tokens[0] == U"%";
        bool skip_suffix = op_tokens[n_tokens] == U"%";
        bool select_last;
        bool direct = this->opt_dir;
        switch (this->opt_sel) {
            case LEADER_SEL::FIRST:
                select_last = false;
                break;
            case LEADER_SEL::LAST:
                select_last = true;
                break;
            // case LEADER_SEL::OPT:
            //     select_last = op_tokens[n_tokens][0] == U'%';  // variable: last fixed: first
            //     break;
            default:
                select_last = false;
                break;
        }

        Posting* list1;
        Posting* list2;

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
                card = this->find_node(tokens[0])->c_pre;
            } else if (skip_prefix & (op_tokens[1] == U"")) {
                card = this->find_node(tokens[0])->c_suf;
            } else {
                list1 = this->find_node(tokens[0])->post;
                card = this->card_token1(list1, op_tokens[0], op_tokens[1]);
            }
        } else {
            // optimization
            if (this->opt_dir == LEADER_DIR::OPT) {
                float prefix_count = this->find_node(tokens[0])->post->inv_list[0];
                float suffix_count = this->find_node(tokens[n_tokens - 1])->post->inv_list[0];
                // float prefix_count2 = this->find_node(tokens[0])->post->pos_list->size();
                // float suffix_count2 = this->find_node(tokens[n_tokens - 1])->post->pos_list->size();

                // if (op_tokens[0][0] == U"%"){
                //     prefix_count
                // }
                direct = LEADER_DIR::FORWARD;
                if (prefix_count <= suffix_count) {
                    direct = LEADER_DIR::FORWARD;
                } else {
                    direct = LEADER_DIR::BACKWARD;
                }
            }
            if (direct == LEADER_DIR::FORWARD) {
                list1 = this->find_node(tokens[0])->post;
                if (!skip_prefix) {
                    list1 = this->get_prefix_plists(list1, op_tokens[0], this->opt_next && op_tokens[1][0] == U'%', false);
                }
                if (select_last) {
                    for (int i = 2; i <= n_tokens; ++i) {
                        list2 = this->find_node(tokens[i - 1])->post;
                        list1 = join_two_plists(list1, list2, op_tokens[i - 1], this->opt_next && op_tokens[i][0] == U'%', opt_last && skip_suffix && i == n_tokens);
                    }
                    if (!skip_suffix) {
                        list1 = this->get_suffix_plists(list1, op_tokens[n_tokens], false, true);
                    }
                } else {
                    for (int i = 2; i <= n_tokens; ++i) {
                        list2 = this->find_node(tokens[i - 1])->post;
                        if (!skip_suffix && i == n_tokens) {
                            list2 = this->get_suffix_plists(list2, op_tokens[n_tokens], this->opt_next && op_tokens[n_tokens - 1][0] == U'%', false);
                        }
                        list1 = join_two_plists(list1, list2, op_tokens[i - 1], this->opt_next && op_tokens[i][0] == U'%', opt_last && i == n_tokens);
                    }
                }
                card = list1->inv_list[0];
                if (list1->view) {
                    delete list1;
                }
            } else {  // BACKWARD
                list2 = this->find_node(tokens[n_tokens - 1])->post;
                if (!skip_suffix) {
                    list2 = this->get_suffix_plists(list2, op_tokens[n_tokens], this->opt_next && op_tokens[n_tokens - 1][0] == U'%', false);
                }
                if (select_last) {
                    for (int i = n_tokens - 1; i >= 1; --i) {
                        list1 = this->find_node(tokens[i - 1])->post;
                        list2 = this->join_two_plists_backward(list1, list2, op_tokens[i], this->opt_next && op_tokens[i - 1][0] == U'%', opt_last && skip_prefix && i == 1);
                    }
                    if (!skip_prefix) {
                        list2 = this->get_prefix_plists(list2, op_tokens[0], false, true);
                    }
                } else {
                    for (int i = n_tokens - 1; i >= 1; --i) {
                        list1 = this->find_node(tokens[i - 1])->post;
                        if (!skip_prefix && i == 1) {
                            list1 = this->get_prefix_plists(list1, op_tokens[0], this->opt_next && op_tokens[1][0] == U'%', false);
                        }
                        list2 = this->join_two_plists_backward(list1, list2, op_tokens[i], this->opt_next && op_tokens[i - 1][0] == U'%', opt_last && i == 1);
                    }
                }
                card = list2->inv_list[0];
                if (list2->view) {
                    delete list2;
                }
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

class CliqueGenHashTable {
public:
    static const int max_word_len = MAX_KEY_LEN;
    unordered_map<u32string, CliqueGenHashTableEntry>* inverted_index = nullptr;
    int* s_lens;
    int n_db;

    CliqueGenHashTable() {
        this->inverted_index = new unordered_map<u32string, CliqueGenHashTableEntry>();
        this->inverted_index->reserve(POSTING_HASH_SIZE);
    }

    ~CliqueGenHashTable() {
        delete this->inverted_index;
        delete[] this->s_lens;
    }

    void add_strings(u32string* db, int n_db) {
        u32string* rec;
        this->s_lens = new int[n_db];
        this->n_db = n_db;
        for (int id = 0; id < n_db; id++) {
            rec = &db[id];
            add_all_suffixes_pos(*rec, id);
            this->s_lens[id] = rec->size();
        }
    }

    void add_all_suffixes_pos(const u32string& rec, int rid) {
        u32string suffix;
        int rec_len = (int)rec.size();
        int substr_len;
        CliqueGenHashTableEntry* entry;
        for (int i = 0; i < rec_len; i++) {
            substr_len = MIN(rec_len - i, max_word_len);
            for (int j = 1; j <= substr_len; j++) {
                suffix = rec.substr(i, j);

                if (this->inverted_index->find(suffix) == this->inverted_index->end()) {
                    // node->children[ch] = child;
                    (*this->inverted_index)[suffix] = CliqueGenHashTableEntry();
                }
                entry = &(*this->inverted_index)[suffix];
                // node = node->children[ch];
                if (rid != entry->curr_id) {
                    entry->inv_list.push_back(rid);
                    entry->curr_idx = entry->inv_list.size();
                    entry->inv_list.push_back(0);
                    entry->curr_id = rid;
                    entry->c_sub += 1;
                }
                entry->inv_list.push_back(i + 1);
                if (i == 0) {
                    entry->c_pre += 1;
                }
                if (i + j == rec_len) {
                    entry->c_suf += 1;
                }
                entry->inv_list[entry->curr_idx] += 1;
            }
        }
    }

    CliqueGenHashTableEntry* find_node(const u32string& rec) {
        return &(*this->inverted_index)[rec];
    }

    int card_token1(const vector<int>& p_list, const u32string& op_token1, const u32string& op_token2, int token_len) {
        int card = 0;
        int idx = 0;
        int n;
        int pos;

        int id;
        int percent1 = 0;
        int percent2 = 0;
        int under1 = 0;
        int under2 = 0;
        bool is_match;

        if (op_token1.size() > 0) {
            percent1 = (int)(op_token1[0] == U'%');
            under1 = op_token1.size() - percent1;
        }
        if (op_token2.size() > 0) {
            percent2 = (int)(op_token2[0] == U'%');
            under2 = op_token2.size() - percent2;
        }

        while (idx < (int)p_list.size()) {
            id = p_list[idx++];
            n = p_list[idx++];

            is_match = false;
            for (int i = 0; i < n; ++i) {
                pos = p_list[idx + i];
                // select prefix
                if (percent1 == 0) {
                    if (pos == under1 + 1) {
                        // select suffix
                        if (percent2 == 0) {
                            if (pos + token_len - 1 == this->s_lens[id] - under2) {
                                is_match = true;
                            }
                        } else {
                            if (pos + token_len - 1 <= this->s_lens[id] - under2) {
                                is_match = true;
                            } else {
                                is_match = false;
                            }
                        }
                        break;
                    }
                } else {  // o_0 includes '%'
                    // if (under1 == 0)  // o_0 = '%'
                    if (pos >= under1 + 1) {
                        // select suffix
                        if (percent2 == 0) {
                            if (pos + token_len - 1 == this->s_lens[id] - under2) {
                                is_match = true;
                                break;
                            }
                        } else {
                            if (pos + token_len - 1 <= this->s_lens[id] - under2) {
                                is_match = true;
                                break;
                            }
                        }
                    }
                }
            }
            idx += n;

            if (is_match) {
                card += 1;
            }
        }

        return card;
    }

    vector<int> get_suffix_plists(const vector<int>& input_list, const u32string& op_token, int token_len) {
        vector<int> output_list;
        int idx = 0;
        int percent = 0;
        int under = 0;
        int id;
        int n;
        int pos;
        int curr_n_idx;

        if (op_token.size() > 0) {
            percent = (int)(op_token[0] == U'%');
            under = op_token.size() - percent;
        }
        while (idx < (int)input_list.size()) {
            id = input_list[idx++];
            n = input_list[idx++];

            curr_n_idx = -1;
            for (int i = 0; i < n; ++i) {
                pos = input_list[idx + i];
                if (percent == 0) {
                    if (pos + token_len - 1 == this->s_lens[id] - under) {
                        output_list.push_back(id);
                        output_list.push_back(1);
                        output_list.push_back(pos);
                        break;
                    }
                } else {  // o_0 includes '%'
                    // if (under1 == 0)  // o_0 = '%'
                    if (pos + token_len - 1 <= this->s_lens[id] - under) {
                        if (curr_n_idx < 0) {
                            output_list.push_back(id);
                            curr_n_idx = output_list.size();
                            output_list.push_back(0);
                        }
                        output_list[curr_n_idx] += 1;
                        output_list.push_back(pos);
                    }
                }
            }
            idx += n;
        }

        return output_list;
    }

    vector<int> get_prefix_plists(const vector<int>& input_list, const u32string& op_token) {
        vector<int> output_list;
        int idx = 0;
        int percent = 0;
        int under = 0;
        int id;
        int n;
        int pos;

        if (op_token.size() > 0) {
            percent = (int)(op_token[0] == U'%');
            under = op_token.size() - percent;
        }
        while (idx < (int)input_list.size()) {
            id = input_list[idx++];
            n = input_list[idx++];

            for (int i = 0; i < n; ++i) {
                pos = input_list[idx + i];
                if (percent == 0) {
                    if (pos == under + 1) {
                        output_list.push_back(id);
                        output_list.push_back(1);
                        output_list.push_back(pos);
                        break;
                    }
                } else {  // o_0 includes '%'
                    // if (under1 == 0)  // o_0 = '%'
                    if (pos >= under + 1) {
                        output_list.push_back(id);
                        output_list.push_back(n - i);
                        while (i < n) {
                            pos = input_list[idx + i];
                            output_list.push_back(pos);
                            ++i;
                        }
                        break;
                    }
                }
            }
            idx += n;
        }

        return output_list;
    }
    int count_join_two_plists(const vector<int>& p_list1, const vector<int>& p_list2, const u32string& op_token, int token_len) {
        int card = 0;
        int idx1 = 0;
        int idx2 = 0;
        int i1;
        int i2;
        int n1;
        int n2;
        int pos1;
        int pos2;
        int id1;
        int id2;
        int curr_n_idx;

        int percent = 0;
        int under = 0;
        if (op_token.size() > 0) {
            percent = (int)(op_token[0] == U'%');
            under = op_token.size() - percent;
        }

        while (idx1 < (int)p_list1.size() && idx2 < (int)p_list2.size()) {
            id1 = p_list1[idx1];
            id2 = p_list2[idx2];

            if (id1 < id2) {
                n1 = p_list1[++idx1];
                idx1 += n1 + 1;
            } else if (id1 > id2) {
                n2 = p_list2[++idx2];
                idx2 += n2 + 1;
            } else {
                n1 = p_list1[++idx1];
                n2 = p_list2[++idx2];
                idx1 += 1;
                idx2 += 1;
                curr_n_idx = -1;
                if (percent == 0) {
                    i1 = 0;
                    i2 = 0;
                    while (i1 < n1 && i2 < n2) {
                        pos1 = p_list1[idx1 + i1];
                        pos2 = p_list2[idx2 + i2];
                        if (pos1 + token_len + under < pos2) {
                            ++i1;
                        } else if (pos1 + token_len + under > pos2) {
                            ++i2;
                        } else {
                            if (curr_n_idx < 0) {
                                card += 1;
                                curr_n_idx = 0;
                            }
                            ++i1;
                            ++i2;
                        }
                    }
                } else {
                    pos1 = p_list1[idx1];
                    for (i2 = 0; i2 < n2; ++i2) {
                        pos2 = p_list2[idx2 + i2];
                        if (pos1 + token_len + under <= pos2) {
                            // output_list.push_back(id1);
                            // output_list.push_back(1);
                            // output_list.push_back(pos1);
                            // break;
                            card += 1;
                            break;
                        }
                    }
                }
                idx1 += n1;
                idx2 += n2;
            }
        }

        return card;
    }

    int card_token2(const vector<int>& p_list1, const vector<int>& p_list2, const vector<u32string>& op_tokens, const vector<u32string>& tokens) {
        int card = 0;
        citr_vi itr1 = p_list1.begin();
        citr_vi itr2 = p_list2.begin();

        int i1;
        int i2;
        int n1;
        int n2;
        int pos1;
        int pos2;
        int id1;
        int id2;
        // int curr_n_idx;
        int s_len;
        int e2;

        bool p0 = op_tokens[0][0] == U'%';
        bool p1 = op_tokens[1][0] == U'%';
        bool p2 = op_tokens[2][0] == U'%';
        int u0 = op_tokens[0].size() - (int)p0;
        int u1 = op_tokens[1].size() - (int)p1;
        int u2 = op_tokens[2].size() - (int)p2;
        int c0 = u0 + 1;
        int c1 = u1 + tokens[0].size();
        int c2;
        int c2_sub = u2 + tokens[1].size() - 1;

        while (itr1 != p_list1.end() && itr2 != p_list2.end()) {
            id1 = *itr1;
            id2 = *itr2;
            if (id1 < id2) {
                n1 = *(++itr1);
                itr1 += n1 + 1;
            } else if (id1 > id2) {
                n2 = *(++itr2);
                itr2 += n2 + 1;
            } else {
                n1 = *(++itr1);
                n2 = *(++itr2);
                ++itr1;
                ++itr2;
                s_len = this->s_lens[id1];
                c2 = s_len - c2_sub;

                i1 = binary_search_gt(itr1, n1, c0);
                if (i1 >= 0) {
                    if (p0) {
                        if (p1) {
                            pos1 = *(itr1 + i1);
                            i2 = binary_search_gt(itr2, n2, c1 + pos1);
                            if (i2 >= 0) {
                                if (p2) {
                                    pos2 = *(itr2 + i2);
                                    if (pos2 <= c2) {
                                        card += 1;
                                    }
                                } else {
                                    i2 = binary_search(itr2 + i2, n2 - i2, c2);
                                    if (i2 >= 0) {
                                        card += 1;
                                    }
                                }
                            }
                        } else {
                            if (p2) {
                                e2 = binary_search_lt(itr2, n2, c2) + 1;
                                if (e2 > 0) {
                                    i2 = 0;
                                    for (; i1 < n1; ++i1) {
                                        pos1 = *(itr1 + i1);
                                        i2 = binary_search(itr2, e2, c1 + pos1);
                                        if (i2 >= 0) {
                                            card += 1;
                                            break;
                                        }
                                        // i2 = binary_search_gt(itr2 + i2, e2 - i2, c1 + pos1);
                                        // if (i2 < 0) {
                                        //     break;
                                        // }
                                        // pos2 = *(itr2 + i2);
                                        // if (pos2 - pos1 == c1) {
                                        //     card += 1;
                                        //     break;
                                        // }
                                        // if (i2 > 0) {
                                        //     --i2;
                                        // }
                                    }
                                }
                            } else {
                                i2 = binary_search(itr2, n2, c2);
                                if (i2 >= 0) {
                                    pos2 = *(itr2 + i2);
                                    i1 = binary_search(itr1 + i1, n1 - i1, pos2 - c1);
                                    if (i1 >= 0) {
                                        card += 1;
                                    }
                                }
                            }
                        }
                    } else {
                        pos1 = *(itr1 + i1);
                        if (pos1 == c0) {
                            i2 = binary_search_gt(itr2, n2, c1 + pos1);
                            if (i2 >= 0) {
                                if (p1) {
                                    pos2 = *(itr2 + i2);
                                    if (p2 && pos2 <= c2) {
                                        card += 1;
                                    } else {
                                        i2 = binary_search(itr2 + i2, n2 - i2, c2);
                                        if (i2 >= 0) {
                                            card += 1;
                                        }
                                    }
                                } else {
                                    pos2 = *(itr2 + i2);
                                    if (pos2 == c1 + pos1) {
                                        if (COND_E(p2, pos2, c2)) {
                                            card += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                itr1 += n1;
                itr2 += n2;
            }
        }
        return card;
    }

    vector<int> join_two_plists_B(const vector<int>& p_list1, const vector<int>& p_list2, const u32string& op_token, int token_len) {
        vector<int> output_list;
        int idx1 = 0;
        int idx2 = 0;
        int i1;
        int i2;
        int n1;
        int n2;
        int pos1;
        int pos2;
        int id1;
        int id2;
        int curr_n_idx;

        int percent = 0;
        int under = 0;
        if (op_token.size() > 0) {
            percent = (int)(op_token[0] == U'%');
            under = op_token.size() - percent;
        }

        while (idx1 < (int)p_list1.size() && idx2 < (int)p_list2.size()) {
            id1 = p_list1[idx1];
            id2 = p_list2[idx2];

            if (id1 < id2) {
                n1 = p_list1[++idx1];
                idx1 += n1 + 1;
            } else if (id1 > id2) {
                n2 = p_list2[++idx2];
                idx2 += n2 + 1;
            } else {
                n1 = p_list1[++idx1];
                n2 = p_list2[++idx2];
                idx1 += 1;
                idx2 += 1;
                curr_n_idx = -1;
                if (percent == 0) {
                    i1 = 0;
                    i2 = 0;
                    while (i1 < n1 && i2 < n2) {
                        pos1 = p_list1[idx1 + i1];
                        pos2 = p_list2[idx2 + i2];
                        if (pos1 + token_len + under < pos2) {
                            ++i1;
                        } else if (pos1 + token_len + under > pos2) {
                            ++i2;
                        } else {
                            if (curr_n_idx < 0) {
                                output_list.push_back(id1);
                                curr_n_idx = output_list.size();
                                output_list.push_back(0);
                            }
                            output_list[curr_n_idx] += 1;
                            output_list.push_back(pos1);
                            ++i1;
                            ++i2;
                        }
                    }
                } else {
                    pos2 = p_list2[idx2 + n2 - 1];
                    for (i1 = n1 - 1; i1 >= 0; --i1) {
                        pos1 = p_list1[idx1 + i1];
                        if (pos1 + token_len + under <= pos2) {
                            // output_list.push_back(id1);
                            // output_list.push_back(1);
                            // output_list.push_back(pos1);
                            // break;
                            output_list.push_back(id1);
                            output_list.push_back(i1 + 1);
                            break;
                        }
                    }
                    for (int i = 0; i <= i1; ++i) {
                        pos1 = p_list1[idx1 + i];
                        output_list.push_back(pos1);
                    }
                    // while (i1 >= 0) {
                    //     pos1 = p_list1[idx1 + i1];
                    //     output_list.push_back(pos1);
                    //     --i1;
                    // }
                }
                idx1 += n1;
                idx2 += n2;
            }
        }

        return output_list;
    }
    vector<int> join_two_plists(const vector<int>& p_list1, const vector<int>& p_list2, const u32string& op_token, int token_len) {
        vector<int> output_list;
        int idx1 = 0;
        int idx2 = 0;
        int i1;
        int i2;
        int n1;
        int n2;
        int pos1;
        int pos2;
        int id1;
        int id2;
        int curr_n_idx;

        int percent = 0;
        int under = 0;
        if (op_token.size() > 0) {
            percent = (int)(op_token[0] == U'%');
            under = op_token.size() - percent;
        }

        while (idx1 < (int)p_list1.size() && idx2 < (int)p_list2.size()) {
            id1 = p_list1[idx1];
            id2 = p_list2[idx2];

            if (id1 < id2) {
                n1 = p_list1[++idx1];
                idx1 += n1 + 1;
            } else if (id1 > id2) {
                n2 = p_list2[++idx2];
                idx2 += n2 + 1;
            } else {
                n1 = p_list1[++idx1];
                n2 = p_list2[++idx2];
                idx1 += 1;
                idx2 += 1;
                curr_n_idx = -1;
                if (percent == 0) {
                    i1 = 0;
                    i2 = 0;
                    while (i1 < n1 && i2 < n2) {
                        pos1 = p_list1[idx1 + i1];
                        pos2 = p_list2[idx2 + i2];
                        if (pos1 + token_len + under < pos2) {
                            ++i1;
                        } else if (pos1 + token_len + under > pos2) {
                            ++i2;
                        } else {
                            if (curr_n_idx < 0) {
                                output_list.push_back(id1);
                                curr_n_idx = output_list.size();
                                output_list.push_back(0);
                            }
                            output_list[curr_n_idx] += 1;
                            output_list.push_back(pos2);
                            ++i1;
                            ++i2;
                        }
                    }
                } else {
                    pos1 = p_list1[idx1];
                    for (i2 = 0; i2 < n2; ++i2) {
                        pos2 = p_list2[idx2 + i2];
                        if (pos1 + token_len + under <= pos2) {
                            // output_list.push_back(id1);
                            // output_list.push_back(1);
                            // output_list.push_back(pos1);
                            // break;
                            output_list.push_back(id1);
                            output_list.push_back(n2 - i2);
                            break;
                        }
                    }
                    while (i2 < n2) {
                        pos2 = p_list2[idx2 + i2];
                        output_list.push_back(pos2);
                        ++i2;
                    }
                }
                idx1 += n1;
                idx2 += n2;
            }
        }

        return output_list;
    }

    int distinct_sid(vector<int> p_list) {
        int card = 0;
        int n;
        int idx = 1;
        while (idx < (int)p_list.size()) {
            n = p_list[idx++];
            card += 1;
            idx += n + 1;
        }
        return card;
    }

    int find_card_P(u32string& query, int o_type) {
        int card = 0;
        TreePlan* plan = new TreePlan();

        vector<int> order;

        vector<u32string> op_tokens;
        vector<u32string> tokens;
        vector<int>* list1;
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
                if ((percent && this->s_lens[i] >= n_under) || (!percent && this->s_lens[i] == n_under)) {
                    ++card;
                }
            }
            return card;
        } else if (n_tokens == 1) {
            if ((op_tokens[0] == U"%") && (op_tokens.back() == U"%")) {
                card = this->find_node(tokens[0])->c_sub;
            } else if ((op_tokens[0] == U"") && (op_tokens.back() == U"%")) {
                card = this->find_node(tokens[0])->c_pre;
            } else if ((op_tokens[0] == U"%") && (op_tokens.back() == U"")) {
                card = this->find_node(tokens[0])->c_suf;
            } else {
                list1 = &this->find_node(tokens[0])->inv_list;
                card = this->card_token1(*list1, op_tokens[0], op_tokens[1], tokens[0].size());
            }
            return card;
        }

        // assert(n_tokens >= 1);

        vector<int>* L;

        vector<LTable*> tables;
        LTable* table;
        int* s_lens = this->s_lens;

        CliqueGenHashTableEntry* node;
        for (int i = 0; i < n_tokens; i++) {
            token = tokens[i];

            node = this->find_node(token);
            L = &node->inv_list;
            table = new LTable(L, 1, s_lens, false);
            table->set_count(node->c_sub);
            tables.push_back(table);
        }

        // Test 1
        int idx;
        while (o_type > 0) {
            idx = o_type % 10;
            order.push_back(idx);
            if (idx == 1) {
                order.push_back(0);
            }
            if (idx == n_tokens - 1) {
                order.push_back(n_tokens);
            }
            o_type /= 10;
        }
        reverse(order.begin(), order.end());
        // print_vector(order);
        // return 0;

        // switch (o_type) {
        //     case 11:
        //         for (int i = 0; i <= n_tokens - 2; i++) {
        //             order.push_back(i);
        //         }
        //         order.push_back(n_tokens);
        //         order.push_back(n_tokens - 1);
        //         break;

        //     case 12:
        //         for (int i = 1; i <= n_tokens - 1; i++) {
        //             order.push_back(i);
        //         }
        //         order.push_back(0);
        //         order.push_back(n_tokens);
        //         break;

        //     case 21:
        //         for (int i = n_tokens; i >= 2; --i) {
        //             order.push_back(i);
        //         }
        //         order.push_back(0);
        //         order.push_back(1);
        //         break;

        //     case 22:
        //         for (int i = n_tokens - 1; i >= 1; --i) {
        //             order.push_back(i);
        //         }
        //         order.push_back(n_tokens);
        //         order.push_back(0);
        //         break;

        //     case 31:
        //         for (int i = n_tokens / 2; i >= 1; --i) {
        //             if (i == 1) {
        //                 order.push_back(0);
        //             }
        //             if (i == n_tokens - 1) {
        //                 order.push_back(n_tokens);
        //             }
        //             order.push_back(i);
        //         }

        //         for (int i = n_tokens / 2 + 1; i <= n_tokens - 1; i++) {
        //             if (i == n_tokens - 1) {
        //                 order.push_back(n_tokens);
        //             }
        //             order.push_back(i);
        //         }
        //         break;

        //     case 32:
        //         for (int i = n_tokens / 2; i >= 1; --i) {
        //             order.push_back(i);
        //         }
        //         for (int i = n_tokens / 2 + 1; i <= n_tokens - 1; i++) {
        //             order.push_back(i);
        //         }
        //         order.push_back(0);
        //         order.push_back(n_tokens);
        //         break;

        //     default:
        //         break;
        // }

        plan->set_LTables(tables);
        // plan->construct_by_order(query, order);
        plan->construct_by_order(op_tokens, tokens, order);
        // cout << utf8::utf32to8(query) << endl;
        // plan->print();
        card = plan->find_card();

        delete plan;

        return card;
    }

    int find_card_B(u32string& query) {
        int card = 0;
        vector<u32string> op_tokens;
        vector<u32string> tokens;

        u32string token;
        uint32_t start = 0;
        uint32_t len = 0;

        char32_t ch = query[0];
        bool is_operator = (ch == U'_') | (ch == U'%');
        if (!is_operator) {
            op_tokens.push_back(u32string());
        }

        // splitQuery
        for (int i = 0; i < (int)query.size(); ++i) {
            ch = query[i];
            if ((ch == U'_') | (ch == U'%')) {
                if (is_operator) {
                    len += 1;
                } else {
                    token = query.substr(start, len);
                    tokens.push_back(token);
                    start = i;
                    len = 1;
                    is_operator = true;
                }
            } else {
                if (is_operator) {
                    token = query.substr(start, len);
                    op_tokens.push_back(token);
                    start = i;
                    len = 1;
                    is_operator = false;
                } else {
                    len += 1;
                }
            }
        }
        if (is_operator) {
            token = query.substr(start, len);
            op_tokens.push_back(token);
        } else {
            token = query.substr(start, len);
            tokens.push_back(token);
            op_tokens.push_back(u32string());
        }

        int n_tokens = tokens.size();

        vector<int>* list1;
        vector<int>* list2;
        vector<int> list_tmp;
        vector<int> list_tmp2;
        vector<int> list_tmp3;

        // find R(q)
        if (n_tokens == 0) {
            card = 0;
            assert(op_tokens.size() == 1);
            u32string op_token = op_tokens[0];
            bool is_variable = op_token[0] == U'%';
            int wild_len = op_token.size();
            if (is_variable) {
                wild_len -= 1;
            }
            for (int i = 0; i < this->n_db; ++i) {
                if ((is_variable && this->s_lens[i] >= wild_len) || (!is_variable && this->s_lens[i] == wild_len)) {
                    card += 1;
                }
            }
        } else if (n_tokens == 1) {
            if ((op_tokens[0] == U"%") & (op_tokens[1] == U"%")) {
                card = this->find_node(tokens[0])->c_sub;
            } else if ((op_tokens[0] == U"") & (op_tokens[1] == U"%")) {
                card = this->find_node(tokens[0])->c_pre;
            } else if ((op_tokens[0] == U"%") & (op_tokens[1] == U"")) {
                card = this->find_node(tokens[0])->c_suf;
            } else {
                list1 = &this->find_node(tokens[0])->inv_list;
                card = this->card_token1(*list1, op_tokens[0], op_tokens[1], tokens[0].size());
            }
        } else if (n_tokens == 2) {
            // L_1
            list1 = &this->find_node(tokens[0])->inv_list;
            if (op_tokens[0] != U"%") {
                list_tmp = this->get_prefix_plists(*list1, op_tokens[0]);
                list1 = &list_tmp;
            }
            // L_2
            list2 = &this->find_node(tokens[1])->inv_list;
            if (op_tokens[2] != U"%") {
                list_tmp2 = this->get_suffix_plists(*list2, op_tokens[2], tokens[1].size());
                list2 = &list_tmp2;
            }

            // list_tmp3 = this->join_two_plists(*list1, *list2, op_tokens[1], tokens[0].size());
            // list1 = &list_tmp3;
            // card = distinct_sid(*list1);
            card = this->count_join_two_plists(*list1, *list2, op_tokens[1], tokens[0].size());
            // card = count_valid_string_positional_lists(*list1, *list2, tokens[0].size());
            // card = 0;
        } else {
            // // L_1
            // list1 = &this->find_node(tokens[0])->inv_list;
            // if (op_tokens[0] != U"%") {
            //     list_tmp = this->get_prefix_plists(*list1, op_tokens[0]);
            //     list1 = &list_tmp;
            // }

            // L_2
            list2 = &this->find_node(tokens[n_tokens - 1])->inv_list;
            if (op_tokens[n_tokens] != U"%") {
                list_tmp2 = this->get_suffix_plists(*list2, op_tokens[n_tokens], tokens[n_tokens - 1].size());
                list2 = &list_tmp2;
            }

            for (int i = n_tokens - 1; i >= 1; --i) {
                list1 = &this->find_node(tokens[i - 1])->inv_list;
                if (i == 1 && op_tokens[0] != U"%") {
                    list_tmp = this->get_prefix_plists(*list1, op_tokens[0]);
                    list1 = &list_tmp;
                }
                list_tmp3 = this->join_two_plists_B(*list1, *list2, op_tokens[i], tokens[i - 1].size());
                list2 = &list_tmp3;
            }
            card = distinct_sid(*list2);
        }
        //         // cout << "card: " << card << endl;

        return card;
    }

    int find_card(u32string& query) {
        int card = 0;
        vector<u32string> op_tokens;
        vector<u32string> tokens;

        u32string token;
        uint32_t start = 0;
        uint32_t len = 0;

        char32_t ch = query[0];
        bool is_operator = (ch == U'_') | (ch == U'%');
        if (!is_operator) {
            op_tokens.push_back(u32string());
        }

        // splitQuery
        for (int i = 0; i < (int)query.size(); ++i) {
            ch = query[i];
            if ((ch == U'_') | (ch == U'%')) {
                if (is_operator) {
                    len += 1;
                } else {
                    token = query.substr(start, len);
                    tokens.push_back(token);
                    start = i;
                    len = 1;
                    is_operator = true;
                }
            } else {
                if (is_operator) {
                    token = query.substr(start, len);
                    op_tokens.push_back(token);
                    start = i;
                    len = 1;
                    is_operator = false;
                } else {
                    len += 1;
                }
            }
        }
        if (is_operator) {
            token = query.substr(start, len);
            op_tokens.push_back(token);
        } else {
            token = query.substr(start, len);
            tokens.push_back(token);
            op_tokens.push_back(u32string());
        }

        int n_tokens = tokens.size();

        vector<int>* list1;
        vector<int>* list2;
        vector<int> list_tmp;
        vector<int> list_tmp2;
        vector<int> list_tmp3;

        // find R(q)
        if (n_tokens == 0) {
            card = 0;
            assert(op_tokens.size() == 1);
            u32string op_token = op_tokens[0];
            bool is_variable = op_token[0] == U'%';
            int wild_len = op_token.size();
            if (is_variable) {
                wild_len -= 1;
            }
            for (int i = 0; i < this->n_db; ++i) {
                if ((is_variable && this->s_lens[i] >= wild_len) || (!is_variable && this->s_lens[i] == wild_len)) {
                    card += 1;
                }
            }
        } else if (n_tokens == 1) {
            if ((op_tokens[0] == U"%") && (op_tokens[1] == U"%")) {
                card = this->find_node(tokens[0])->c_sub;
            } else if ((op_tokens[0] == U"") && (op_tokens[1] == U"%")) {
                card = this->find_node(tokens[0])->c_pre;
            } else if ((op_tokens[0] == U"%") && (op_tokens[1] == U"")) {
                card = this->find_node(tokens[0])->c_suf;
            } else {
                list1 = &this->find_node(tokens[0])->inv_list;
                card = this->card_token1(*list1, op_tokens[0], op_tokens[1], tokens[0].size());
            }
        } else if (n_tokens == 2) {
            // L_1
            list1 = &this->find_node(tokens[0])->inv_list;
            if (op_tokens[0] != U"%") {
                list_tmp = this->get_prefix_plists(*list1, op_tokens[0]);
                list1 = &list_tmp;
            }
            // L_2
            list2 = &this->find_node(tokens[1])->inv_list;
            if (op_tokens[2] != U"%") {
                list_tmp2 = this->get_suffix_plists(*list2, op_tokens[2], tokens[1].size());
                list2 = &list_tmp2;
            }

            // list_tmp3 = this->join_two_plists(*list1, *list2, op_tokens[1], tokens[0].size());
            // list1 = &list_tmp3;
            // card = distinct_sid(*list1);
            card = this->count_join_two_plists(*list1, *list2, op_tokens[1], tokens[0].size());
            // card = count_valid_string_positional_lists(*list1, *list2, tokens[0].size());
            // card = 0;
        } else {
            // L_1
            list1 = &this->find_node(tokens[0])->inv_list;
            if (op_tokens[0] != U"%") {
                list_tmp = this->get_prefix_plists(*list1, op_tokens[0]);
                list1 = &list_tmp;
            }

            for (int i = 2; i <= n_tokens; ++i) {
                list2 = &this->find_node(tokens[i - 1])->inv_list;
                if (i == n_tokens && op_tokens[n_tokens] != U"%") {
                    list_tmp2 = this->get_suffix_plists(*list2, op_tokens[i], tokens[i - 1].size());
                    list2 = &list_tmp2;
                }
                list_tmp3 = this->join_two_plists(*list1, *list2, op_tokens[i - 1], tokens[i - 2].size());
                list1 = &list_tmp3;
            }
            card = distinct_sid(*list1);
        }
        // cout << "card: " << card << endl;

        return card;
    }
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
