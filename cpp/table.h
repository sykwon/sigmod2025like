#pragma once

#include "cond.h"
#include "join.h"
#include "util.h"

static int* s_lens;

class Posting {
public:
    vector<int> inv_list = {0};
    vector<int>* pos_list;
    int beta;
    bool owner = true;
    bool view = false;
    bool is_single = true;

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

Posting* join_two_plists_forward(const Posting* posting1, const Posting* posting2, const u32string& op_token, bool opt_next, bool opt_last) {
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

            if (!percent) {  // fixed-length pattern
                equality_p_predicate_join(inv_list_out, pos_list_out, pos_list1, pos_list2,
                                          p_start1, p_start2, n_pos1, n_pos2, number, opt_last, opt_next, sid1);
            } else {  // variable-length pattern
                inequality_p_predicate_join_bin_srch(inv_list_out, pos_list1[p_start1], pos_list2,
                                                     p_start2, n_pos2, number, opt_last, opt_next, sid1);
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

Posting* join_two_plists_backward(const Posting* posting1, const Posting* posting2, const u32string& op_token, bool opt_next, bool opt_last) {
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
    const vector<int>& inv_list1 = posting1->inv_list;
    vector<int>& pos_list2 = *posting2->pos_list;
    const vector<int>& inv_list2 = posting2->inv_list;
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
                equality_p_predicate_join_backward(inv_list_out, pos_list_out, pos_list1, pos_list2,
                                                   p_start1, p_start2, n_pos1, n_pos2, number, opt_last, opt_next, sid1);

            } else {  // variable-length pattern
                inequality_p_predicate_join_backward(inv_list_out, pos_list2[p_start2 + n_pos2 - 1], pos_list1, p_start1, n_pos1, number, opt_last, opt_next, sid1);
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

Posting* join_two_plists_multi(const Posting* posting1, const Posting* posting2, Condition* cond) {
    // this function forward
    int number = cond->number;
    bool percent = cond->percent;
    bool left_end = cond->left_end;
    bool right_end = cond->right_end;
    // int beta = posting1->beta;
    bool opt_next = cond->opt_next;
    bool opt_last = cond->is_last;
    Posting* output_posting;
    if (posting1->is_single && posting2->is_single) {
        int alpha = cond->alpha;
        u32string op_token(alpha + cond->percent, U'_');
        if (cond->percent) {
            op_token[0] = U'%';
        }
        if (cond->left_end) {
            output_posting = join_two_plists_forward(posting1, posting2, op_token, opt_next, opt_last);
            return output_posting;
        } else if (cond->right_end) {
            output_posting = join_two_plists_backward(posting1, posting2, op_token, opt_next, opt_last);
            return output_posting;
        }
    }

    if (posting1->is_single && posting2->is_single && percent) {
        if (left_end) {
            output_posting = new Posting(false, true);
            output_posting->pos_list = posting2->pos_list;
        } else if (right_end) {
            output_posting = new Posting(false, true);
            output_posting->pos_list = posting1->pos_list;
        } else {
            output_posting = new Posting(true, true);
        }
    } else {
        output_posting = new Posting(true, true);
    }
    output_posting->beta = posting2->beta;
    if (!left_end && !right_end) {
        output_posting->is_single = false;
    }

    int idx1 = 1;
    int idx2 = 1;
    int inv_size1 = posting1->inv_list[0] * 3 + 1;
    int inv_size2 = posting2->inv_list[0] * 3 + 1;
    int sid1;
    int sid2;
    int n_pos1;                                // size of positions or position pairs
    int n_pos2;                                // size of positions or position pairs
    int n_col1 = posting1->is_single ? 1 : 2;  // pos or (pos1, pos2)
    int n_col2 = posting2->is_single ? 1 : 2;  // pos or (pos1, pos2)
    int p_start1;
    int p_start2;
    int pos1;
    int pos2;
    int pos3;
    int pos4;
    int curr_n_idx = -1;
    // int count = 0;

    const vector<int>& inv_list1 = posting1->inv_list;
    vector<int>& pos_list1 = *posting1->pos_list;
    const vector<int>& inv_list2 = posting2->inv_list;
    vector<int>& pos_list2 = *posting2->pos_list;
    vector<int>& inv_list_out = output_posting->inv_list;
    vector<int>& pos_list_out = *output_posting->pos_list;

    // if (posting1->is_single && posting2->is_single && cond->left_end){
    //     p_predicate_join(percent, n_pos1, n_pos2, p_start1, p_start2, inv_list_out, pos_list_out, pos_list1, pos_list2, number, opt_next, opt_last, sid1);
    //     return
    // }

    while (idx1 < inv_size1 && idx2 < inv_size2) {  // s-predicate join
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
            if (posting1->is_single && posting2->is_single && cond->left_end) {  // TODO
                p_predicate_join(percent, n_pos1, n_pos2, p_start1, p_start2, inv_list_out, pos_list_out, pos_list1, pos_list2, number, opt_next, opt_last, sid1);
            } else {
                curr_n_idx = -1;
                for (int i1 = 0; i1 < n_pos1; ++i1) {
                    pos1 = pos_list1[p_start1 + i1 * n_col1];
                    pos2 = pos_list1[p_start1 + (i1 + 1) * n_col1 - 1];
                    for (int i2 = 0; i2 < n_pos2; ++i2) {
                        pos3 = pos_list2[p_start2 + i2 * n_col2];
                        cond->set_identifier(pos2, pos3);
                        if (cond->eval()) {
                            if (curr_n_idx < 0) {  //
                                output_posting->inv_list[0] += 1;
                                output_posting->inv_list.push_back(sid1);
                                curr_n_idx = output_posting->inv_list.size();
                                output_posting->inv_list.push_back(0);
                                output_posting->inv_list.push_back(output_posting->pos_list->size());
                            }
                            output_posting->inv_list[curr_n_idx] += 1;
                            pos4 = pos_list2[p_start2 + (i2 + 1) * n_col2 - 1];
                            if (!left_end) {
                                output_posting->pos_list->push_back(pos1);
                            }
                            if (!right_end) {
                                output_posting->pos_list->push_back(pos4);
                            }
                        }
                    }
                }
            }

            // p_predicate_join(percent, n_pos1, n_pos2, p_start1, p_start2, inv_list_out, pos_list_out, pos_list1, pos_list2, number, opt_next, opt_last, sid1);

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
