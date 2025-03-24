#pragma once

#include <algorithm>
#include <vector>

using namespace std;
static bool is_ineq_opt = true;
static bool is_mp_only = false;
static bool is_bin_srch = false;

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

void inequality_p_predicate_join_backward(vector<int>& inv_list_out,
                                          int pos2, const vector<int>& pos_list1,
                                          int p_start1, int n_pos1, int number, bool opt_last, bool opt_next, int sid1) {
    // int p_idx1 = binary_search_for_position_set(pos_list1, p_start1, p_start2 + n_pos2 - 1, number + pos1);
    int p_idx1;
    int pos1;
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

void equality_p_predicate_join_backward(vector<int>& inv_list_out, vector<int>& pos_list_out,
                                        const vector<int>& pos_list1, const vector<int>& pos_list2,
                                        int p_start1, int p_start2, int n_pos1, int n_pos2, int number, bool opt_last, bool opt_next, int sid1) {
    int pos1;
    int pos2;
    int curr_n_idx = -1;
    int p_idx1 = p_start1;
    int p_idx2 = p_start2;

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