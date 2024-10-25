#ifndef BA7734EC_6850_4C77_8798_529CEBE28576
#define BA7734EC_6850_4C77_8798_529CEBE28576

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "index.h"
#include "plan.h"
#include "util.h"

#define MAX_KEY_LEN 5
#define POSTING_HASH_SIZE 100000
#define COND_B(pr, po, c) (((pr) && (po) >= (c)) || (!(pr) && (po) == (c)))
#define COND_E(pr, po, c) (((pr) && (po) <= (c)) || (!(pr) && (po) == (c)))
using namespace std;

class PositionSet {
public:
    // vector<int> inv_list = {0};
    int p_start;
    int n_pos;
    vector<int>* pos_list;
    int beta;
    bool owner;
    bool view;

    PositionSet(bool owner = true, bool view = false) : p_start(0), n_pos(0), owner(owner), view(view) {
        // this->owner = owner;
        // this->view = view;
        // this->p_start = 0;
        // this->n_pos = 0;
        if (this->owner) {
            this->pos_list = new vector<int>();
        }
    }

    ~PositionSet() {
        if (this->owner) {
            delete pos_list;
        }
    }
};

void inequality_position_set_join(PositionSet& output_position_set, const vector<int>& pos_list1, const vector<int>& pos_list2,
                                  int p_start1, int p_start2, int n_pos2, int number, bool opt_last, bool opt_next) {
    int pos1;
    int pos2;
    int p_idx2;

    // output_position_set has pos_list2

    pos1 = pos_list1[p_start1];
    for (p_idx2 = p_start2; p_idx2 < p_start2 + n_pos2; ++p_idx2) {
        pos2 = pos_list2[p_idx2];
        if (pos2 - pos1 >= number) {
            // ++inv_list_out[0];
            if (opt_last) {
                output_position_set.n_pos = 1;
                break;
            }
            // inv_list_out.push_back(sid1);
            if (opt_next) {
                // inv_list_out.push_back(1);
                output_position_set.n_pos = 1;
            } else {
                // inv_list_out.push_back(p_start2 + n_pos2 - p_idx2);
                output_position_set.n_pos = p_start2 + n_pos2 - p_idx2;
            }
            // inv_list_out.push_back(p_idx2);
            output_position_set.p_start = p_idx2;
            break;
        }
    }
}

void inequality_position_set_join_bin_srch(PositionSet& output_position_set, int pos1, const vector<int>& pos_list2,
                                           int p_start2, int n_pos2, int number, bool opt_last, bool opt_next) {
    int p_idx2 = binary_search_for_position_set(pos_list2, p_start2, p_start2 + n_pos2 - 1, number + pos1);

    if (p_idx2 >= 0) {
        if (opt_last) {
            output_position_set.n_pos = 1;
        } else {
            if (opt_next) {
                output_position_set.n_pos = 1;
            } else {
                output_position_set.n_pos = p_start2 + n_pos2 - p_idx2;
            }
            output_position_set.p_start = p_idx2;
        }
    }
}

void inequality_position_set_join_all_pair(PositionSet& output_position_set, const vector<int>& pos_list1, const vector<int>& pos_list2, int p_start1, int p_start2, int n_pos1, int n_pos2, int number) {
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
    // int curr_n_idx = -1;
    if (all_pos_list.size() > 0) {
        sort(all_pos_list.begin(), all_pos_list.end());
        pos1 = -1;

        // ++inv_list_out[0];
        // inv_list_out.push_back(sid1);
        // curr_n_idx = inv_list_out.size();
        // inv_list_out.push_back(0);
        // inv_list_out.push_back(pos_list_out.size());
        for (int i = 0; i < (int)all_pos_list.size(); ++i) {
            pos2 = all_pos_list[i];
            if (pos2 > pos1) {
                output_position_set.n_pos += 1;
                // inv_list_out[curr_n_idx] += 1;
                output_position_set.pos_list->push_back(pos2);
                // pos_list_out.push_back(pos2);
                pos1 = pos2;
            }
        }
    }
}

void equality_position_set_join(PositionSet& output_position_set,
                                const vector<int>& pos_list1, const vector<int>& pos_list2,
                                int p_start1, int p_start2, int n_pos1, int n_pos2, int number, bool opt_last, bool opt_next) {
    int pos1;
    int pos2;
    int curr_n_idx = -1;
    int p_idx1 = p_start1;
    int p_idx2 = p_start2;
    vector<int>* pos_list_out = output_position_set.pos_list;
    while (p_idx1 < p_start1 + n_pos1 && p_idx2 < p_start2 + n_pos2) {  // merge like
        pos1 = pos_list1[p_idx1];
        pos2 = pos_list2[p_idx2];
        if (pos2 - pos1 > number) {
            ++p_idx1;
        } else if (pos2 - pos1 < number) {
            ++p_idx2;
        } else {  // matched
            if (curr_n_idx < 0) {
                // ++inv_list_out[0];
                if (opt_last) {
                    output_position_set.n_pos = 1;
                    break;
                }
                // inv_list_out.push_back(sid1);
                // curr_n_idx = inv_list_out.size();
                curr_n_idx = 0;
                // inv_list_out.push_back(0);
                // inv_list_out.push_back(pos_list_out.size());
            }
            // inv_list_out[curr_n_idx] += 1;
            // output_position_set.pos_list->push_back(pos2);
            output_position_set.n_pos += 1;
            pos_list_out->push_back(pos2);
            if (opt_next) {
                break;
            }
            ++p_idx1;
            ++p_idx2;
        }
    }
}

void position_set_join(bool percent, int n_pos1, int n_pos2, int p_start1, int p_start2, PositionSet& output_posting, const vector<int>& pos_list1, const vector<int>& pos_list2, int number, bool opt_next, bool opt_last) {
    if (!percent) {  // fixed-length pattern
        equality_position_set_join(output_posting, pos_list1, pos_list2,
                                   p_start1, p_start2, n_pos1, n_pos2, number, opt_last, opt_next);
    } else {  // variable-length pattern
        if (is_bin_srch) {
            inequality_position_set_join_bin_srch(output_posting, pos_list1[p_start1], pos_list2, p_start2, n_pos2, number, opt_last, opt_next);
        } else if (is_ineq_opt) {
            inequality_position_set_join(output_posting, pos_list1, pos_list2,
                                         p_start1, p_start2, n_pos2, number, opt_last, opt_next);
        } else {
            inequality_position_set_join_all_pair(output_posting, pos_list1, pos_list2,
                                                  p_start1, p_start2, n_pos1, n_pos2, number);
        }
    }
}

PositionSet* join_two_position_sets(const PositionSet* posting1, const PositionSet* posting2, const u32string& op_token, bool opt_next, bool opt_last) {
    // this function forward
    bool percent = false;
    int under = 0;
    if (op_token.size() > 0) {
        percent = op_token[0] == U'%';
        under = op_token.size() - (int)percent;
    }
    int beta = posting1->beta;
    int number = under + beta;

    PositionSet* output_position_set;
    if (percent) {
        output_position_set = new PositionSet(false, true);
        output_position_set->pos_list = posting2->pos_list;
    } else {
        output_position_set = new PositionSet(true, true);
    }
    output_position_set->beta = posting2->beta;

    // int idx1 = 1;
    // int idx2 = 1;
    // int n1 = posting1->inv_list[0] * 3 + 1;
    // int n2 = posting2->inv_list[0] * 3 + 1;
    // int sid1;
    // int sid2;
    int n_pos1 = posting1->n_pos;
    int n_pos2 = posting2->n_pos;
    // int p_idx1;
    // int p_idx2;
    int p_start1 = posting1->p_start;
    int p_start2 = posting2->p_start;
    // int pos1;
    // int pos2;
    // int curr_n_idx = -1;

    vector<int>& pos_list1 = *posting1->pos_list;
    // const vector<int>& inv_list1 = posting1->inv_list;
    vector<int>& pos_list2 = *posting2->pos_list;
    // const vector<int>& inv_list2 = posting2->inv_list;
    // vector<int>& inv_list_out = output_posting->inv_list;
    // vector<int>& pos_list_out = *output_posting->pos_list;

    // n_pos1 = inv_list1[++idx1];
    // n_pos2 = inv_list2[++idx2];
    // p_start1 = inv_list1[++idx1];
    // p_start2 = inv_list2[++idx2];
    position_set_join(percent, n_pos1, n_pos2, p_start1, p_start2, *output_position_set, pos_list1, pos_list2, number, opt_next, opt_last);
    // ++idx1;
    // ++idx2;

    // if (posting1->view) {
    //     delete posting1;
    // }
    // if (posting2->view) {
    //     delete posting2;
    // }

    return output_position_set;
}

PositionSet* get_prefix_position_set(PositionSet* input_posting, const u32string& op_token, bool opt_next, bool opt_last) {
    PositionSet* output_posting;
    // PositionSet* output_posting = new PositionSet(false, true);
    // output_posting->pos_list = input_posting->pos_list;
    // output_posting->beta = input_posting->beta;

    // int idx = 0;
    // int n = input_posting->inv_list[0] * 3;
    // int sid;
    int n_pos = input_posting->n_pos;
    int p_idx;
    int p_start = input_posting->p_start;
    int pos;
    const vector<int>& pos_list = *input_posting->pos_list;
    // const vector<int>& inv_list = input_posting->inv_list;
    // vector<int>& output_list = output_posting->inv_list;

    bool percent = false;
    int under = 0;

    if (op_token.size() > 0) {
        percent = op_token[0] == U'%';
        under = op_token.size() - (int)percent;
    }

    int number = under + 1;

    if (!percent) {  // fixed-length
        output_posting = new PositionSet(true, true);

    } else {
        output_posting = new PositionSet(false, true);
        output_posting->pos_list = input_posting->pos_list;
    }
    output_posting->beta = input_posting->beta;

    // while (idx < n) {
    //     sid = inv_list[++idx];
    //     n_pos = inv_list[++idx];
    //     p_start = inv_list[++idx];

    for (p_idx = p_start; p_idx < p_start + n_pos; ++p_idx) {
        pos = pos_list[p_idx];
        if (!percent) {  // fixed-length
            if (pos == number) {
                output_posting->n_pos += 1;
                // ++output_list[0];
                if (opt_last) {
                    break;
                }
                // output_list.push_back(sid);
                // output_list.push_back(1);
                // output_list.push_back(p_idx);
                output_posting->pos_list->push_back(pos);
                break;
            }
        } else {  //  variable-length
            if (pos >= number) {
                output_posting->n_pos += 1;
                // ++output_list[0];
                if (opt_last) {
                    break;
                }
                // output_list.push_back(sid);
                if (opt_next) {
                    output_posting->n_pos = 1;
                    // output_list.push_back(1);
                } else {
                    output_posting->n_pos = n_pos + p_start - p_idx;
                    // output_list.push_back(n_pos + p_start - p_idx);
                }
                output_posting->p_start = p_idx;
                // output_list.push_back(p_idx);
                break;
            }
        }
    }
    // }
    if (input_posting->view) {
        delete input_posting;
    }

    return output_posting;
}

class SPADENode {
public:
    unordered_map<char32_t, SPADENode*> children;
    int c_pre = 0;
    int c_suf = 0;
    PositionSet* post;
    PositionSet* post_pre = nullptr;
    PositionSet* post_suf = nullptr;
    // int curr_id = -1;
    int curr_idx;
    int min_pos;
    int max_pos;
    SPADENode() {
        this->post = new PositionSet(true, false);
    }
    ~SPADENode() {
        SPADENode* child;
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

class SPADETree {
public:
    RegTrie* reg_trie = nullptr;
    SPADENode* root = nullptr;
    int s_len;
    // int* s_lens;
    // int n_db;
    bool opt_view = false;
    bool opt_next = false;
    bool opt_last = false;
    LEADER_DIR opt_dir = LEADER_DIR::FORWARD;
    LEADER_SEL opt_sel = LEADER_SEL::LAST;

    PositionSet** posting_stack = nullptr;
    u32string* last_op_token;
    u32string* last_token;
    CHAR_STAT* status;
    int* saved_p_prev;
    int* saved_p;

    SPADETree() {
        this->root = new SPADENode();
    }

    ~SPADETree() {
        delete root;
        // delete s_lens;
    }

    void create_trie(u32string* qset, int n_qry) {
        this->reg_trie = new RegTrie();
        this->reg_trie->add_strings(qset, n_qry);
    }

    // void add_strings(u32string* db, int n_db) {
    //     u32string* rec;
    //     this->s_lens = new int[n_db];
    //     this->n_db = n_db;
    //     for (int id = 0; id < n_db; id++) {
    //         rec = &db[id];
    //         add_all_suffixes_pos(*rec, id);
    //         this->s_lens[id] = rec->size();
    //     }
    // }

    void build_ivs(const u32string& rec) {
        delete this->root;
        this->root = new SPADENode();

        this->s_len = (int)rec.size();
        // int substr_len;
        RegNode* reg_node;
        SPADENode* node;
        SPADENode* child;
        char32_t ch;
        for (int i = 0; i < this->s_len; i++) {
            // substr_len = MIN(rec_len - i, max_word_len);
            reg_node = this->reg_trie->root;
            node = this->root;
            for (int j = i; j < this->s_len; j++) {
                ch = rec[j];
                if (reg_node->children.find(ch) == reg_node->children.end()) {
                    break;
                }
                reg_node = reg_node->children[ch];

                if (node->children.find(ch) == node->children.end()) {  // new node
                    child = new SPADENode();
                    child->post->beta = j - i + 1;
                    child->post_pre = new PositionSet(true, false);
                    child->post_pre->beta = j - i + 1;
                    child->post_suf = new PositionSet(true, false);
                    child->post_suf->beta = j - i + 1;
                    node->children[ch] = child;
                }
                node = node->children[ch];  // node represents rec[i, j] (= rec[i], ..., rec[j])
                // if (i == 0 && !node->post_pre) {  // rec[i,j] is prefix
                //     node->post_pre = new PositionSet(true, false);
                //     node->post_pre->beta = j - i + 1;
                // }
                // if (j == rec_len - 1 && !node->post_suf) {  // rec[i,j] is suffix
                //     node->post_suf = new PositionSet(true, false);
                //     node->post_suf->beta = j - i + 1;
                // }
                // if (rid != node->curr_id) {  // set initial values
                //     node->post->inv_list.push_back(rid);
                //     node->curr_idx = node->post->inv_list.size();
                //     node->post->inv_list.push_back(0);
                //     node->post->inv_list.push_back(node->post->pos_list->size());
                //     node->curr_id = rid;
                //     node->post->inv_list[0] += 1;  // inv_list[0]: c_sub
                // }
                node->post->pos_list->push_back(i + 1);  // add pos
                node->post->n_pos += 1;

                // node->post->inv_list[node->curr_idx] += 1;
                if (i == 0) {  // rec[i,j] is prefix
                    // node->post_pre->inv_list[0] += 1;  // inv_list[0]: c_pre
                    // node->post_pre->inv_list.push_back(rid);
                    // node->post_pre->inv_list.push_back(1);
                    // node->post_pre->inv_list.push_back(node->post_pre->pos_list->size());
                    node->post_pre->pos_list->push_back(1);
                    node->post_pre->n_pos += 1;
                }
                if (j == this->s_len - 1) {  // rec[i,j] is suffix
                    // node->post_suf->inv_list[0] += 1;  // inv_list[0]: c_suf
                    // node->post_suf->inv_list.push_back(rid);
                    // node->post_suf->inv_list.push_back(1);
                    // node->post_suf->inv_list.push_back(node->post_suf->pos_list->size());
                    node->post_suf->pos_list->push_back(i + 1);
                    node->post_suf->n_pos += 1;
                }
            }
        }
    }

    void build_ivs_old(const u32string& rec) {
        delete this->reg_trie;
        this->reg_trie = new RegTrie();
        // u32string suffix;
        int rec_len = (int)rec.size();
        this->s_len = rec_len;
        // int substr_len;
        SPADENode* node;
        SPADENode* child;
        char32_t ch;
        for (int i = 0; i < rec_len; i++) {
            // substr_len = MIN(rec_len - i, max_word_len);
            node = this->root;
            for (int j = i; j < rec_len; j++) {
                ch = rec[j];
                if (node->children.find(ch) == node->children.end()) {
                    child = new SPADENode();
                    node->children[ch] = child;
                    child->post->beta = j - i + 1;
                }
                node = node->children[ch];
                // if (rid != node->curr_id) {  // set initial values
                //     node->post->inv_list.push_back(rid);
                //     node->curr_idx = node->post->inv_list.size();
                //     node->post->inv_list.push_back(0);
                //     node->post->inv_list.push_back(node->post->pos_list->size());
                //     node->curr_id = rid;
                //     node->post->inv_list[0] += 1;  // inv_list[0]: c_sub
                // }
                node->post->pos_list->push_back(i + 1);  // add pos
                // if (node->min_pos > i+1) {
                //     node->min_pos = i+1;
                // }
                // if (node->min_pos > i+1) {
                //     node->min_pos = i+1;
                // }
                // node->post->inv_list[node->curr_idx] += 1;
                if (i == 0) {
                    node->c_pre += 1;
                }
                if (j == rec_len - 1) {
                    node->c_suf += 1;
                }
            }
        }
    }

    SPADENode* find_node(const u32string& rec) {
        SPADENode* node = this->root;
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

    bool card_token1(const PositionSet* posting, const u32string& op_token1, const u32string& op_token2) {
        // int card = 0;
        // int idx = 0;
        int beta = posting->beta;
        // int n = posting->inv_list[0] * 3;
        // int sid;
        int n_pos = posting->n_pos;
        int p_idx;
        int p_start = posting->p_start;
        int pos;
        const vector<int>& pos_list = *posting->pos_list;
        // const vector<int>& inv_list = posting->inv_list;

        bool percent1 = false;
        bool percent2 = false;
        int under1 = 0;
        int under2 = 0;
        bool is_match = false;

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

        for (p_idx = p_start; p_idx < p_start + n_pos; ++p_idx) {
            pos = pos_list[p_idx];
            // select prefix
            if (!percent1) {           // fixed-length
                if (pos == number1) {  // matched first position
                    // select suffix
                    if (!percent2) {  // fixed-length
                        is_match = this->s_len - pos == number2;
                    } else {  // variable-length
                        is_match = this->s_len - pos >= number2;
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
                    if (this->s_len - pos == number2) {
                        is_match = true;
                        break;
                    }
                } else {  // variable-length
                    is_match = this->s_len - pos >= number2;
                    break;
                }
            }
        }

        // if (is_match) {
        //     card += 1;
        // }

        // return card;
        return is_match;
    }

    // PositionSet* get_suffix_plists(PositionSet* input_posting, const u32string& op_token, bool opt_next, bool opt_last) {
    //     PositionSet* output_posting = new PositionSet(false, true);
    //     output_posting->pos_list = input_posting->pos_list;
    //     output_posting->beta = input_posting->beta;

    //     int idx = 0;
    //     int n = input_posting->inv_list[0] * 3;
    //     int sid;
    //     int n_pos;
    //     int p_idx;
    //     int p_start;
    //     int pos;
    //     int s_len;
    //     const vector<int>& pos_list = *input_posting->pos_list;
    //     const vector<int>& inv_list = input_posting->inv_list;
    //     vector<int>& output_list = output_posting->inv_list;

    //     bool percent = false;
    //     int under = 0;
    //     if (op_token.size() > 0) {
    //         percent = op_token[0] == U'%';
    //         under = op_token.size() - (int)percent;
    //     }
    //     int beta = input_posting->beta;
    //     int number = under + beta - 1;

    //     while (idx < n) {
    //         sid = inv_list[++idx];
    //         n_pos = inv_list[++idx];
    //         p_start = inv_list[++idx];
    //         s_len = s_lens[sid];

    //         for (p_idx = p_start + n_pos - 1; p_idx >= p_start; --p_idx) {  // backward
    //             pos = pos_list[p_idx];
    //             if (!percent) {  // fixed-length
    //                 if (s_len - pos == number) {
    //                     ++output_list[0];
    //                     if (opt_last) {
    //                         break;
    //                     }
    //                     output_list.push_back(sid);
    //                     output_list.push_back(1);
    //                     output_list.push_back(p_idx);
    //                     break;
    //                 }
    //             } else {  // variable-length
    //                 if (s_len - pos >= number) {
    //                     ++output_list[0];
    //                     if (opt_last) {
    //                         break;
    //                     }
    //                     output_list.push_back(sid);
    //                     if (opt_next) {
    //                         output_list.push_back(1);
    //                         output_list.push_back(p_idx);
    //                     } else {
    //                         output_list.push_back(p_idx - p_start + 1);
    //                         output_list.push_back(p_start);
    //                     }
    //                     break;
    //                 }
    //             }
    //         }
    //     }

    //     return output_posting;
    // }

    // PositionSet* join_two_plists_backward(PositionSet* posting1, PositionSet* posting2, const u32string& op_token, bool opt_next, bool opt_last) {
    //     // this function forward
    //     bool percent = false;
    //     int under = 0;
    //     if (op_token.size() > 0) {
    //         percent = op_token[0] == U'%';
    //         under = op_token.size() - (int)percent;
    //     }
    //     int beta = posting1->beta;
    //     int number = under + beta;

    //     PositionSet* output_posting;
    //     if (percent) {
    //         output_posting = new PositionSet(false, true);
    //         output_posting->pos_list = posting1->pos_list;
    //     } else {
    //         output_posting = new PositionSet(true, true);
    //     }
    //     output_posting->beta = posting1->beta;

    //     int idx1 = 1;
    //     int idx2 = 1;
    //     int n1 = posting1->inv_list[0] * 3 + 1;
    //     int n2 = posting2->inv_list[0] * 3 + 1;
    //     int sid1;
    //     int sid2;
    //     int n_pos1;
    //     int n_pos2;
    //     int p_idx1;
    //     int p_idx2;
    //     int p_start1;
    //     int p_start2;
    //     int pos1;
    //     int pos2;
    //     int curr_n_idx = -1;

    //     vector<int>& pos_list1 = *posting1->pos_list;
    //     vector<int>& inv_list1 = posting1->inv_list;
    //     vector<int>& pos_list2 = *posting2->pos_list;
    //     vector<int>& inv_list2 = posting2->inv_list;
    //     vector<int>& inv_list_out = output_posting->inv_list;
    //     vector<int>& pos_list_out = *output_posting->pos_list;

    //     while (idx1 < n1 && idx2 < n2) {
    //         sid1 = inv_list1[idx1];
    //         sid2 = inv_list2[idx2];

    //         if (sid1 < sid2) {
    //             idx1 += 3;
    //         } else if (sid1 > sid2) {
    //             idx2 += 3;
    //         } else {
    //             n_pos1 = inv_list1[++idx1];
    //             n_pos2 = inv_list2[++idx2];
    //             p_start1 = inv_list1[++idx1];
    //             p_start2 = inv_list2[++idx2];
    //             if (!percent) {  // fixed-length pattern
    //                 if (opt_next) {
    //                     p_idx1 = p_start1 + n_pos1 - 1;
    //                     p_idx2 = p_start2 + n_pos2 - 1;
    //                     while (p_idx1 >= p_start1 && p_idx2 >= p_start2) {  // merge like
    //                         pos1 = pos_list1[p_idx1];
    //                         pos2 = pos_list2[p_idx2];
    //                         if (pos2 - pos1 > number) {
    //                             --p_idx2;
    //                         } else if (pos2 - pos1 < number) {
    //                             --p_idx1;
    //                         } else {
    //                             ++inv_list_out[0];
    //                             if (opt_last) {
    //                                 break;
    //                             }
    //                             inv_list_out.push_back(sid1);
    //                             inv_list_out.push_back(1);
    //                             inv_list_out.push_back(pos_list_out.size());
    //                             pos_list_out.push_back(pos1);
    //                             break;
    //                         }
    //                     }
    //                 } else {
    //                     curr_n_idx = -1;
    //                     p_idx1 = p_start1;
    //                     p_idx2 = p_start2;
    //                     while (p_idx1 < p_start1 + n_pos1 && p_idx2 < p_start2 + n_pos2) {  // merge like
    //                         pos1 = pos_list1[p_idx1];
    //                         pos2 = pos_list2[p_idx2];
    //                         if (pos2 - pos1 > number) {
    //                             ++p_idx1;
    //                         } else if (pos2 - pos1 < number) {
    //                             ++p_idx2;
    //                         } else {
    //                             if (curr_n_idx < 0) {
    //                                 ++inv_list_out[0];
    //                                 if (opt_last) {
    //                                     break;
    //                                 }
    //                                 inv_list_out.push_back(sid1);
    //                                 curr_n_idx = inv_list_out.size();
    //                                 inv_list_out.push_back(0);
    //                                 inv_list_out.push_back(pos_list_out.size());
    //                             }
    //                             inv_list_out[curr_n_idx] += 1;
    //                             pos_list_out.push_back(pos1);
    //                             ++p_idx1;
    //                             ++p_idx2;
    //                         }
    //                     }
    //                 }

    //             } else {  // variable-length pattern
    //                 pos2 = pos_list2[p_start2 + n_pos2 - 1];
    //                 for (p_idx1 = p_start1 + n_pos1 - 1; p_idx1 >= p_start1; --p_idx1) {
    //                     pos1 = pos_list1[p_idx1];
    //                     if (pos2 - pos1 >= number) {
    //                         ++inv_list_out[0];
    //                         if (opt_last) {
    //                             break;
    //                         }
    //                         inv_list_out.push_back(sid1);
    //                         if (opt_next) {
    //                             inv_list_out.push_back(1);
    //                             inv_list_out.push_back(p_idx1);
    //                         } else {
    //                             inv_list_out.push_back(p_idx1 - p_start1 + 1);
    //                             inv_list_out.push_back(p_start1);
    //                         }
    //                         break;
    //                     }
    //                 }
    //             }
    //             ++idx1;
    //             ++idx2;
    //         }
    //     }

    //     if (posting1->view) {
    //         delete posting1;
    //     }
    //     if (posting2->view) {
    //         delete posting2;
    //     }

    //     return output_posting;
    // }

    // void find_stat(u32string& query, vector<int>& stat, int max_m) {
    //     vector<u32string> op_tokens;
    //     vector<u32string> tokens;
    //
    //     u32string token;
    //
    //     splitQuery(query, op_tokens, tokens);
    //     int n_tokens = tokens.size();
    //
    //     PositionSet* list1 = this->find_node(tokens[0])->post;
    //     PositionSet* list2 = this->find_node(tokens[n_tokens - 1])->post;
    //     stat.push_back(n_tokens);
    //     stat.push_back(list1->inv_list[0]);
    //     stat.push_back(list2->inv_list[0]);
    //
    //     list1 = this->get_prefix_plists(list1, op_tokens[0], false, true);
    //     list2 = this->get_suffix_plists(list2, op_tokens[n_tokens], false, true);
    //     stat.push_back(list1->inv_list[0]);
    //     stat.push_back(list2->inv_list[0]);
    //     stat.push_back(list1->pos_list->size());
    //     stat.push_back(list2->pos_list->size());
    //     delete list1;
    //     delete list2;
    //
    //     vector<int> size_list;
    //     vector<int> n_tuple_list;
    //     int minimum = INT_MAX;
    //     int minimum_head = INT_MAX;
    //     int minimum_tail = INT_MAX;
    //     int size;
    //     int n_tuple;
    //     for (int i = 1; i <= n_tokens; ++i) {
    //         size = this->find_node(tokens[i - 1])->post->inv_list[0];
    //         n_tuple = this->find_node(tokens[i - 1])->post->pos_list->size();
    //         size_list.push_back(size);
    //         n_tuple_list.push_back(n_tuple);
    //         if (minimum > size) {
    //             minimum = size;
    //         }
    //         if ((i < n_tokens) && (minimum_head > size)) {
    //             minimum_head = size;
    //         }
    //         if ((i > 1) && (minimum_tail > size)) {
    //             minimum_tail = size;
    //         }
    //     }
    //     stat.push_back(minimum);
    //     stat.push_back(minimum_head);
    //     stat.push_back(minimum_tail);
    //     for (int i = 1; i <= max_m; ++i) {
    //         if (i <= n_tokens) {
    //             stat.push_back(size_list[i - 1]);
    //         } else {
    //             stat.push_back(0);
    //         }
    //     }
    //     for (int i = 1; i <= max_m; ++i) {
    //         if (i <= n_tokens) {
    //             stat.push_back(n_tuple_list[i - 1]);
    //         } else {
    //             stat.push_back(0);
    //         }
    //     }
    // }

    // void create_stacks(int max_size) {
    //     this->posting_stack = new PositionSet*[max_size + 1];
    //     this->status = new CHAR_STAT[max_size + 1];
    //     this->saved_p_prev = new int[max_size + 1];
    //     this->saved_p = new int[max_size + 1];
    //     this->last_op_token = new u32string[max_size + 1];
    //     this->last_token = new u32string[max_size + 1];

    //     for (int i = 0; i < max_size + 1; ++i) {
    //         this->posting_stack[i] = nullptr;
    //     }
    //     this->status[0] = CHAR_STAT::WILDCARD;
    //     this->saved_p_prev[0] = 0;
    //     this->saved_p[0] = 0;
    //     this->last_op_token[0] = U"";
    //     this->last_token[0] = U"";
    // }

    void reset_stacks(int max_size) {
        for (int i = 0; i < max_size + 1; ++i) {
            this->posting_stack[i] = nullptr;
        }
    }

    void update_posting_stack(int idx, PositionSet* post) {
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

    bool computeIncNewDebug(int saved_p_prev, int idx, const u32string& op_token, const u32string& token, bool is_sp, bool is_card, bool& is_matched) {
        is_matched = false;
        return false;
    }

    bool computeIncNew(int saved_p_prev, int idx, const u32string& op_token, const u32string& token, bool is_sp, bool is_card, bool& is_matched) {
        PositionSet* prev_posting = nullptr;
        if (saved_p_prev >= 0) {
            prev_posting = this->posting_stack[saved_p_prev];  // prev_posting: previous shareable result
        }
        PositionSet* save_posting = nullptr;
        PositionSet* posting;
        SPADENode* curr_node;
        bool is_non_empty = true;
        is_matched = false;

        int n_rels = 0;
        if (prev_posting) {
            ++n_rels;
        }
        if (token != U"") {
            ++n_rels;
        }

        if (prev_posting) {
            assert(prev_posting->n_pos > 0);
        }

        if (n_rels == 0) {
            assert(!is_sp);
            // is_non_empty = true
            if (is_card) {
                is_matched = false;
                bool percent = op_token[0] == U'%';
                int under = op_token.size() - (int)percent;
                if ((percent && this->s_len >= under) || (!percent && this->s_len == under)) {
                    is_matched = true;
                }
            }
            assert(save_posting == nullptr);
        } else if (n_rels == 1) {
            if (prev_posting) {  // selection suffix
                if (is_card) {
                    assert(token == U"");
                    if (op_token == U"%") {
                        // card = prev_posting->inv_list[0];
                        is_matched = true;
                    } else {
                        is_matched = this->card_token1(prev_posting, U"%", op_token);
                    }
                    assert(save_posting == nullptr);
                }
            } else {  // selection prefix
                // posting: [op_token token %]
                // card: [op_token token]
                assert(token != U"");
                if (is_sp || is_card) {
                    curr_node = this->find_node(token);
                    if (curr_node == nullptr) {
                        is_matched = false;
                        is_non_empty = false;
                        if (is_sp) {
                            save_posting = new PositionSet(true, true);
                            this->update_posting_stack(idx, save_posting);
                        }
                        return is_non_empty;
                    }
                }

                // find save_posting: [op_token token %]
                if (is_sp) {
                    if (op_token == U"") {  // [token %]
                        save_posting = curr_node->post_pre;
                    } else if (op_token == U"%") {  // [% token %]
                        save_posting = curr_node->post;
                    } else {  // [op_token token %]
                        save_posting = curr_node->post;
                        save_posting = get_prefix_position_set(save_posting, op_token, false, false);
                    }

                    if (save_posting->n_pos == 0) {
                        is_non_empty = false;
                        is_matched = false;
                        this->update_posting_stack(idx, save_posting);
                        return is_non_empty;
                    }
                    // else: is_non_empty = true;
                }

                // find card
                if (is_card) {
                    is_matched = false;
                    posting = curr_node->post_suf;
                    if (posting && posting->n_pos > 0) {
                        if (op_token == U"%") {  // [% token]
                            // card = posting->inv_list[0];
                            is_matched = true;
                        } else {  // [op_token token]
                            is_matched = this->card_token1(posting, op_token, U"%");
                        }
                    }
                } else {
                    is_matched = false;
                }
            }
        } else {
            // posting: [prev_posting op_token token %]
            // card: [prev_posting op_token token]
            assert(!token.empty());
            assert(prev_posting);
            if (is_sp || is_card) {
                curr_node = this->find_node(token);
                if (curr_node == nullptr) {
                    is_matched = false;
                    is_non_empty = false;
                    if (is_sp) {
                        save_posting = new PositionSet(true, true);
                        this->update_posting_stack(idx, save_posting);
                    }
                    return is_non_empty;
                }
            }

            // find save_posting: [prev_posting op_token token %]
            if (is_sp) {
                save_posting = curr_node->post;
                save_posting = join_two_position_sets(prev_posting, save_posting, op_token, false, false);

                if (save_posting->n_pos == 0) {
                    is_non_empty = false;
                    is_matched = false;
                    this->update_posting_stack(idx, save_posting);
                    return is_non_empty;
                }
            }

            // else: is_non_empty = true;

            // find card: [prev_posting op_token token]
            if (is_card) {
                posting = curr_node->post_suf;
                if (posting && posting->n_pos > 0) {
                    posting = join_two_position_sets(prev_posting, posting, op_token, false, true);
                    is_matched = posting->n_pos > 0;
                    // is_matched = posting->inv_list[0];
                }
            } else {
                is_matched = false;
            }
        }

        if (is_sp) {
            this->update_posting_stack(idx, save_posting);
        }
        return is_non_empty;
    }

    bool computeInc(int idx, char32_t chr, bool is_sp, bool is_card, bool& is_matched) {
        // It assigns is_matched, and returns is_non_empty.

        // ofstream ofs;
        // ofs.open("tmp.csv", fstream::app);
        // ofs << "idx: " << idx << ", chr: " << chr << ", is_sp: " << is_sp << ", is_card: " << is_card << endl;
        // ofs << idx << ", " << chr << ", " << is_sp << ", " << is_card << endl;

        this->update_stacks(idx, chr);

        CHAR_STAT status = this->status[idx];
        int saved_p_prev = this->saved_p_prev[idx];
        PositionSet* prev_posting = this->posting_stack[saved_p_prev];
        PositionSet* save_posting = nullptr;
        PositionSet* posting;
        u32string& op_token = this->last_op_token[idx];
        u32string& token = this->last_token[idx];
        SPADENode* curr_node;
        bool is_non_empty = true;
        is_matched = false;

        int n_rels = 0;
        if (prev_posting) {
            ++n_rels;
        }
        if (token != U"") {
            ++n_rels;
        }

        if (prev_posting && prev_posting->n_pos == 0) {
            is_matched = false;
            is_non_empty = false;
            if (is_sp) {
                save_posting = new PositionSet(true, true);
                this->update_posting_stack(idx, save_posting);
            }
            return is_non_empty;
        }

        // find S_q^s and R^s(q)
        if (n_rels == 0) {
            // is_non_empty = true
            if (is_card) {
                is_matched = false;
                bool percent = op_token[0] == U'%';
                int under = op_token.size() - (int)percent;
                if ((percent && this->s_len >= under) || (!percent && this->s_len == under)) {
                    is_matched = true;
                }
            }
            assert(save_posting == nullptr);
        } else if (n_rels == 1) {
            if (prev_posting) {  // selection suffix
                if (is_card) {
                    assert(token == U"");
                    assert(status == CHAR_STAT::WILDCARD);
                    if (op_token == U"%") {
                        // card = prev_posting->inv_list[0];
                        is_matched = true;
                    } else {
                        is_matched = this->card_token1(prev_posting, U"%", op_token);
                    }
                    assert(save_posting == nullptr);
                }
            } else {  // selection prefix
                // posting: [op_token token %]
                // card: [op_token token]
                assert(token != U"");
                if (is_sp || is_card) {
                    curr_node = this->find_node(token);
                    if (curr_node == nullptr) {
                        is_matched = false;
                        is_non_empty = false;
                        if (is_sp) {
                            save_posting = new PositionSet(true, true);
                            this->update_posting_stack(idx, save_posting);
                        }
                        return is_non_empty;
                    }
                }

                // find save_posting: [op_token token %]
                if (is_sp) {
                    if (op_token == U"") {  // [token %]
                        save_posting = curr_node->post_pre;
                    } else if (op_token == U"%") {  // [% token %]
                        save_posting = curr_node->post;
                    } else {  // [op_token token %]
                        save_posting = curr_node->post;
                        save_posting = get_prefix_position_set(save_posting, op_token, false, false);
                    }

                    if (save_posting->n_pos == 0) {
                        is_non_empty = false;
                        is_matched = false;
                        this->update_posting_stack(idx, save_posting);
                        return is_non_empty;
                    }
                    // else: is_non_empty = true;
                }

                // find card
                if (is_card) {
                    is_matched = false;
                    posting = curr_node->post_suf;
                    if (posting && posting->n_pos > 0) {
                        if (op_token == U"%") {  // [% token]
                            // card = posting->inv_list[0];
                            is_matched = true;
                        } else {  // [op_token token]
                            is_matched = this->card_token1(posting, op_token, U"%");
                        }
                    }
                } else {
                    is_matched = false;
                }
            }
        } else {
            // posting: [prev_posting op_token token %]
            // card: [prev_posting op_token token]
            assert(!token.empty());
            assert(prev_posting);
            if (is_sp || is_card) {
                curr_node = this->find_node(token);
                if (curr_node == nullptr) {
                    is_matched = false;
                    is_non_empty = false;
                    if (is_sp) {
                        save_posting = new PositionSet(true, true);
                        this->update_posting_stack(idx, save_posting);
                    }
                    return is_non_empty;
                }
            }

            // find save_posting: [prev_posting op_token token %]
            if (is_sp) {
                save_posting = curr_node->post;
                save_posting = join_two_position_sets(prev_posting, save_posting, op_token, false, false);

                if (save_posting->n_pos == 0) {
                    is_non_empty = false;
                    is_matched = false;
                    this->update_posting_stack(idx, save_posting);
                    return is_non_empty;
                }
            }

            // else: is_non_empty = true;

            // find card: [prev_posting op_token token]
            if (is_card) {
                posting = curr_node->post_suf;
                if (posting && posting->n_pos > 0) {
                    posting = join_two_position_sets(prev_posting, posting, op_token, false, true);
                    is_matched = posting->n_pos > 0;
                    // is_matched = posting->inv_list[0];
                }
            } else {
                is_matched = false;
            }
        }

        if (is_sp) {
            this->update_posting_stack(idx, save_posting);
        }
        return is_non_empty;
    }
};

SPADETree* build_SPADE_tree(u32string* qrys, int n_qry) {
    SPADETree* tree = new SPADETree();
    tree->create_trie(qrys, n_qry);
    return tree;
}

#endif /* BA7734EC_6850_4C77_8798_529CEBE28576 */
