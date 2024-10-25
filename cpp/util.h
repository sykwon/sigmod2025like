#ifndef AF77AD98_26A2_45E5_B66D_6D69D2BF9C26
#define AF77AD98_26A2_45E5_B66D_6D69D2BF9C26

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
// #include <regex>
#include <re2/re2.h>
#include <unicode/utypes.h>

#include <boost/regex.hpp>
#include <boost/regex/icu.hpp>
#include <boost/regex/v5/icu.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "utf8.h"

using namespace std;

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define IS_UWILDCARD(X) ((X == U'_') || (X == U'%'))

typedef std::basic_stringstream<char32_t> u32stringstream;
typedef vector<int>::const_iterator citr_vi;
typedef vector<int>::iterator itr_vi;

using namespace boost;

extern int64_t compare_count;
extern int64_t all_compare_count;
extern int inter_count;

template <typename T>
class span {
    T* ptr_;
    std::size_t len_;

public:
    span(T* ptr, std::size_t len) noexcept
        : ptr_{ptr}, len_{len} {}

    T& operator[](int i) noexcept {
        return *ptr_[i];
    }

    T const& operator[](int i) const noexcept {
        return *ptr_[i];
    }

    std::size_t size() const noexcept {
        return len_;
    }

    T* begin() noexcept {
        return ptr_;
    }

    T* end() noexcept {
        return ptr_ + len_;
    }
};

constexpr unsigned int str2inthash(const char* str, int h = 0) {
    return !str[h] ? 5381 : (str2inthash(str, h + 1) * 33) ^ str[h];
}

tuple<u32string*, int> read_strings(string path) {
    u32string* strings = nullptr;
    int n = 0;
    ifstream ifs(path.data());
    string line;
    u32string u32line;
    if (!ifs.is_open()) {
        cout << "[Not Open] " << path << endl;
    }
    assert(ifs.is_open());
    if (ifs.is_open()) {
        while (getline(ifs, line)) {
            n += 1;
        }
        strings = new u32string[n];
        ifs.close();
        ifs.open(path.data());

        n = 0;
        while (getline(ifs, line)) {
            auto end_it = utf8::find_invalid(line.begin(), line.end());
            if (end_it != line.end()) {
                cout << "Invalid UTF-8 encoding detected at line " << n << "\n";
                cout << "This part is fine: " << string(line.begin(), end_it) << "\n";
            }
            u32line = utf8::utf8to32(line);
            strings[n] = u32line;
            n += 1;
        }
    }
    return make_tuple(strings, n);
}

vector<tuple<int, vector<int>>> filter_prefix_positional_lists(const vector<tuple<int, vector<int>>>& pl) {
    vector<tuple<int, vector<int>>> res;

    int id;
    int pos_first;
    for (auto elem : pl) {
        id = get<0>(elem);
        pos_first = get<1>(elem)[0];
        vector<int> pos_list;
        if (abs(pos_first) == 1) {
            pos_list.push_back(pos_first);
            res.push_back(make_tuple(id, pos_list));
        }
    }
    return res;
}

vector<tuple<int, vector<int>>> filter_suffix_positional_lists(const vector<tuple<int, vector<int>>>& pl) {
    vector<tuple<int, vector<int>>> res;

    int id;
    int pos_last;
    for (auto elem : pl) {
        id = get<0>(elem);
        pos_last = get<1>(elem).back();
        vector<int> pos_list;
        if (pos_last < 0) {
            pos_list.push_back(pos_last);
            res.push_back(make_tuple(id, pos_list));
        }
    }
    return res;
}

vector<tuple<int, int, int>> merge_range_lists(const vector<tuple<int, int, int>>& l1, const vector<tuple<int, int, int>>& l2) {
    vector<tuple<int, int, int>> res;
    auto itr1 = l1.begin();
    auto itr2 = l2.begin();

    while (itr1 != l1.end() && itr2 != l2.end()) {
        if (get<0>(*itr1) == get<0>(*itr2)) {
            res.push_back(*itr1);
            ++itr1;
            ++itr2;
        } else if (*itr1 < *itr2) {
            ++itr1;
        } else {
            ++itr2;
        }
    }
    return res;
}

int count_valid_string_range_lists(const vector<tuple<int, int, int>>& pl1, const vector<tuple<int, int, int>>& pl2, int w) {
    auto itr1 = pl1.begin();
    auto itr2 = pl2.begin();

    int id1;
    int id2;
    int pos1;
    int pos2;
    int card = 0;

    while (itr1 != pl1.end() and itr2 != pl2.end()) {
        id1 = get<0>(*itr1);
        id2 = get<0>(*itr2);

        if (id1 == id2) {
            pos1 = abs(get<1>(*itr1));
            pos2 = abs(get<2>(*itr2));
            if (pos1 + w - 1 < pos2) {
                card += 1;
            }

            ++itr1;
            ++itr2;
        } else if (id1 < id2) {
            ++itr1;
        } else {
            ++itr2;
        }
    }
    return card;
}

int count_valid_string_positional_lists(const vector<tuple<int, vector<int>>>& pl1, const vector<tuple<int, vector<int>>>& pl2, int w) {
    auto itr1 = pl1.begin();
    auto itr2 = pl2.begin();

    int id1;
    int id2;
    int pos1;
    int pos2;
    int card = 0;

    while (itr1 != pl1.end() and itr2 != pl2.end()) {
        id1 = get<0>(*itr1);
        id2 = get<0>(*itr2);

        if (id1 == id2) {
            pos1 = abs(get<1>(*itr1)[0]);
            pos2 = abs(get<1>(*itr2).back());
            if (pos1 + w - 1 < pos2) {
                card += 1;
            }

            ++itr1;
            ++itr2;
        } else if (id1 < id2) {
            ++itr1;
        } else {
            ++itr2;
        }
    }
    return card;
}

int count_valid_string_grouped_positional_lists(const vector<tuple<int, vector<int>>>& gl1, const vector<tuple<int, vector<int>>>& gl2, int w) {
    auto gitr1 = gl1.begin();
    auto gitr2 = gl2.begin();
    auto gitr2_start = gl2.begin();

    const vector<int>* id_list1;
    const vector<int>* id_list2;
    int id1;
    int id2;
    int pos1;
    int pos2;
    int card = 0;
    ++inter_count;
    unordered_set<int> id_list;
    for (gitr1 = gl1.begin(); gitr1 != gl1.end(); ++gitr1) {
        for (gitr2 = gl2.begin(); gitr2 != gl2.end(); ++gitr2) {
            //     pos1 = get<0>(*gitr1);
            //     pos2 = get<0>(*gitr2);
            id_list1 = &get<1>(*gitr1);
            id_list2 = &get<1>(*gitr2);
            auto itr1 = id_list1->begin();
            auto itr2 = id_list2->begin();
            while (itr1 != id_list1->end() and itr2 != id_list2->end()) {
                id1 = *itr1;
                id2 = *itr2;
                if (id1 == id2) {
                    ++itr1;
                    ++itr2;
                } else if (id1 < id2) {
                    ++itr1;
                } else {
                    ++itr2;
                }
                ++all_compare_count;
            }
        }
        for (gitr2 = gitr2_start; gitr2 != gl2.end(); ++gitr2) {
            pos1 = get<0>(*gitr1);
            pos2 = get<0>(*gitr2);
            if (pos1 + w - 1 < pos2) {
                id_list1 = &get<1>(*gitr1);
                id_list2 = &get<1>(*gitr2);
                auto itr1 = id_list1->begin();
                auto itr2 = id_list2->begin();
                while (itr1 != id_list1->end() and itr2 != id_list2->end()) {
                    id1 = *itr1;
                    id2 = *itr2;
                    if (id1 == id2) {
                        ++itr1;
                        ++itr2;
                        id_list.insert(id1);
                    } else if (id1 < id2) {
                        ++itr1;
                    } else {
                        ++itr2;
                    }
                    ++compare_count;
                }
            } else {
                ++gitr2_start;
            }
        }
    }
    card = id_list.size();

    return card;
}

vector<tuple<int, vector<int>>> merge_positional_lists(const vector<tuple<int, vector<int>>>& pl1, const vector<tuple<int, vector<int>>>& pl2, int w, bool is_last) {
    vector<tuple<int, vector<int>>> ml;
    auto itr1 = pl1.begin();
    auto itr2 = pl2.begin();

    int id1;
    int id2;
    int pos1;
    int pos2;
    vector<int> pos_list2;

    while (itr1 != pl1.end() and itr2 != pl2.end()) {
        id1 = get<0>(*itr1);
        id2 = get<0>(*itr2);

        if (id1 == id2) {
            pos1 = abs(get<1>(*itr1)[0]);
            pos_list2 = get<1>(*itr2);

            if (is_last) {
                pos2 = abs(pos_list2.back());
                if (pos1 + w - 1 < pos2) {
                    ml.push_back(make_tuple(id2, vector<int>(pos_list2.end() - 1, pos_list2.end())));
                }
            } else {
                for (auto pitr2 = pos_list2.begin(); pitr2 != pos_list2.end(); pitr2++) {
                    pos2 = abs(*pitr2);
                    if (pos1 + w - 1 < pos2) {
                        ml.push_back(make_tuple(id2, vector<int>(pitr2, pos_list2.end())));
                        break;
                    }
                }
            }

            ++itr1;
            ++itr2;
        } else if (id1 < id2) {
            ++itr1;
        } else {
            ++itr2;
        }
    }
    return ml;
}

vector<int>* merge_lists(vector<int>* l1, const vector<int>* l2, bool view) {
    vector<int>* res = new vector<int>();
    auto itr1 = l1->begin();
    auto itr2 = l2->begin();

    while (itr1 != l1->end() && itr2 != l2->end()) {
        if (*itr1 == *itr2) {
            res->push_back(*itr1);
            ++itr1;
            ++itr2;
        } else if (*itr1 < *itr2) {
            ++itr1;
        } else {
            ++itr2;
        }
    }
    if (view) {
        delete l1;
    }
    return res;
}

u32regex gen_regex_from_like_query(const u32string& like_query) {
    int like_query_len = (int)like_query.size();
    int start = 0;
    int length = like_query_len;
    u32string qry = like_query;
    if (like_query.at(0) != U'%') {
        qry = U"^" + qry;
        ++length;
    } else {
        start = 1;
        --length;
    }
    if (like_query.at(like_query.length() - 1) != U'%') {
        qry = qry + U"$";
        ++length;
    } else if (length > 0) {
        --length;
    }

    qry = qry.substr(start, length);                             // delete starting and ending '%'
    qry = u32regex_replace(qry, make_u32regex(L"%"), U"(.*?)");  // replace every '%' with "(.*?)"
    qry = u32regex_replace(qry, make_u32regex(L"_"), U".");      // replace every '_' with '.'

    // cout << utf8::utf32to8(like_query) << ", ";
    // cout << utf8::utf32to8(qry) << endl;
    u32regex re = make_u32regex(qry, boost::regex::optimize);
    return re;
}

string csv_token(const u32string& x) {
    u32stringstream ss_csv;
    string output;
    if (x.find(U',') != u32string::npos || x.find(U'\"') != u32string::npos) {
        ss_csv << U'\"';
        for (char32_t c : x) {
            if (c == U'\"') {
                ss_csv << U"\"\"";
            } else {
                ss_csv << c;
            }
        }
        ss_csv << U'\"';
        output = utf8::utf32to8(ss_csv.str());
    } else {
        output = utf8::utf32to8(x);
    }
    return output;
}

std::shared_ptr<re2::RE2> gen_re2_from_like_query(const string& like_query) {
    string qry;
    stringstream ss;

    auto itr = like_query.begin();
    if (*itr != '%') {
        ss << "^";
    } else {
        ++itr;
    }

    // bool is_uni = false;
    while (itr != like_query.end() - 1) {
        if (*itr == '%') {
            ss << ".*?";
        } else if (*itr == '_') {
            ss << '.';
        } else {
            ss << (char)*itr;
        }
        ++itr;
    }
    if (*itr != '%') {
        if (*itr == '_') {
            ss << '.';
        } else {
            ss << (char)*itr;
        }
        ss << '$';
    } else {
        // ss << *itr;
    }
    qry = ss.str();
    std::shared_ptr<re2::RE2> re(new re2::RE2(qry));
    return re;
}

std::shared_ptr<re2::RE2> gen_re2_from_like_query(const u32string& like_query) {
    string qry;
    stringstream ss;

    auto itr = like_query.begin();
    if (*itr != U'%') {
        ss << "^";
    } else {
        ++itr;
    }

    // bool is_uni = false;
    while (itr != like_query.end() - 1) {
        if (*itr == U'%') {
            ss << ".*?";
        } else if (*itr == U'_') {
            ss << '.';
        } else {
            uint32_t value = *itr;
            if (value <= 127) {  // ASCII code
                ss << (char)*itr;
            } else {  // non-ASCII code
                // is_uni = true;
                ss << "\\x{" << std::hex << value << "}";
            }
        }
        ++itr;
    }
    if (*itr != U'%') {
        uint32_t value = *itr;
        if (*itr == U'_') {
            ss << '.';
        } else {
            if (value <= 127) {  // ASCII code
                ss << (char)*itr;
            } else {  // non-ASCII code
                ss << "\\x{" << std::hex << value << "}";
            }
        }
        ss << '$';
    } else {
    }
    qry = ss.str();
    std::shared_ptr<re2::RE2> re(new re2::RE2(qry));
    return re;
}

vector<u32string> string32_split(const u32string& str, char delim) {
    /*
        return every substring whose length is larger than 0
    */
    vector<u32string> result;
    u32string item;
    size_t start = 0;
    size_t end = 0;
    size_t len = 0;

    while ((end = str.find(delim, start)) != u32string::npos) {
        len = end - start;
        // cout << "start: " << start << " end: " << end << endl;
        if (len > 0) {
            item = str.substr(start, len);
            result.push_back(item);
        }
        start = end + 1;
    }
    if (start < str.size()) {
        // cout << "start: " << start << " size: " << str.size() << endl;
        result.push_back(str.substr(start));
    }

    return result;
}

int binary_search(citr_vi begin, int len, int target) {
    int left = 0;
    int right = len - 1;
    int mid;

    while (left <= right) {
        mid = left + (right - left) / 2;
        if (*(begin + mid) == target) {
            return mid;
        } else if (*(begin + mid) < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}

int binary_search_gt(citr_vi begin, int len, int target) {
    int left = 0;
    int right = len - 1;
    int result = -1;
    int mid;

    while (left <= right) {
        mid = left + (right - left) / 2;
        if (*(begin + mid) >= target) {
            result = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return result;
}

int binary_search_lt(citr_vi begin, int len, int target) {
    int left = 0;
    int right = len - 1;
    int result = -1;
    int mid;

    while (left <= right) {
        mid = left + (right - left) / 2;
        if (*(begin + mid) <= target) {
            result = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return result;
}

void splitQuery(const u32string& query, vector<u32string>& op_tokens, vector<u32string>& tokens) {  // input, output, output
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
}

void print_vector(vector<bool> input) {
    cout << "len: " << input.size() << ", vector :";
    for (auto x : input) {
        cout << x << ", ";
    }
    cout << endl;
}

void print_vector(vector<int> input) {
    cout << "len: " << input.size() << ", vector :";
    for (auto x : input) {
        cout << x << ", ";
    }
    cout << endl;
}

void print_vector(vector<string> input) {
    cout << "len: " << input.size() << ", vector :";
    for (auto x : input) {
        cout << x << ", ";
    }
    cout << endl;
}

void reverseStrings(u32string* strings, int n) {
    for (int i = 0; i < n; ++i) {
        reverse(strings[i].begin(), strings[i].end());
    }
}

int max_length_strings(u32string* strings, int n) {
    int max_len = 0;
    int len;
    for (int i = 0; i < n; ++i) {
        u32string q = strings[i];
        len = q.length();
        if (max_len < len) {
            max_len = len;
        }
    }
    return max_len;
}

int n_distinct_prefix(u32string* strings, int n) {
    int n_dist = 0;
    unordered_set<u32string> prefix_set;
    for (int i = 0; i < n; ++i) {
        u32string q = strings[i];
        for (int j = 1; j <= int(q.size()); ++j) {
            auto p = q.substr(0, j);
            if (prefix_set.find(p) == prefix_set.end()) {
                prefix_set.insert(p);
                n_dist += 1;
            }
        }
    }
    return n_dist;
}

bool wildcard_character_cmp(const char32_t x, const char32_t y) {
    // return true if x < y, return false otherwise
    if (IS_UWILDCARD(x) == IS_UWILDCARD(y)) {
        if (IS_UWILDCARD(x)) {
            return (x < y);  // '%' should be in front of '_'
        } else {
            return (x > y);
        }
    } else {
        return IS_UWILDCARD(x);
    }
}

bool wildcard_string_cmp(const u32string& s1, const u32string& s2) {
    // return true if s1 < s2, return false otherwise
    auto itr1 = s1.begin();
    auto itr2 = s2.begin();
    char32_t c1, c2;

    while (itr1 != s1.end() && itr2 != s2.end()) {
        c1 = *itr1;
        c2 = *itr2;
        if (wildcard_character_cmp(c1, c2)) {
            return true;
        }
        if (wildcard_character_cmp(c2, c1)) {
            return false;
        }
        ++itr1;
        ++itr2;
    }
    if (s1.length() < s2.length()) {
        return true;
    } else {
        return false;
    }
}

// vector<int> get_sort_indexes(const u32string* qrys, int n_qrys) {
//     vector<int> idx(n_qrys);
//     iota(idx.begin(), idx.end(), 0);

//     stable_sort(idx.begin(), idx.end(),
//                 [qrys](int i1, int i2) { return qrys[i1] < qrys[i2]; });
//     return idx;
// }

void sort_wildcard_strings(u32string* qrys, int n_qrys) {
    sort(qrys, qrys + n_qrys, wildcard_string_cmp);
}

vector<int> get_sort_wildcard_indexes(const u32string* qrys, int n_qrys) {
    vector<int> idx(n_qrys);
    iota(idx.begin(), idx.end(), 0);

    stable_sort(idx.begin(), idx.end(),
                [qrys, &idx](int i1, int i2) {
                    // bool cmp = wildcard_string_cmp(qrys[i1], qrys[i2]);
                    // for (auto x : idx) {
                    //     cout << x << ", ";
                    // }
                    // cout << utf8::utf32to8(qrys[i1]) << ", " << utf8::utf32to8(qrys[i2]) << ", ";
                    // cout << i1 << ", " << i2 << ", " << cmp << endl;
                    return wildcard_string_cmp(qrys[i1], qrys[i2]);
                });
    return idx;
}

void sort_with_indexes(u32string* qrys, int n_qrys, vector<int>& idx) {
    vector<u32string> qrys_copy;
    qrys_copy.resize(n_qrys);

    for (int i = 0; i < n_qrys; ++i) {
        qrys_copy[i] = qrys[i];
    }

    for (int i = 0; i < n_qrys; ++i) {
        qrys[idx[i]] = qrys_copy[i];
    }
}

void cumulatively_add_value(vector<tuple<string, double>>& time_dict, string key, double value) {
    string key_in_dict = "";
    double value_in_dict = 0;
    int i;
    bool is_matched = false;
    for (i = 0; i < (int)time_dict.size(); ++i) {
        auto itr = time_dict[i];
        key_in_dict = get<0>(itr);
        value_in_dict = get<1>(itr);
        if (key == key_in_dict) {
            is_matched = true;
            break;
        }
    }
    if (is_matched) {
        time_dict[i] = make_tuple(key, value_in_dict + value);
    } else {
        time_dict.push_back(make_tuple(key, value));
    }
}

vector<vector<tuple<bool, int, int, int, int, int>>> find_infos(u32string* qrys, int n_qrys, const vector<int>& lcp_length_list) {
    // is_sp, saved_p_prev, op_last_s, op_last_l, token_last_s, token_last_l
    int lcp_length;
    u32string* qry;
    u32string* qry_next;
    char32_t ch;
    int q_len;
    bool is_sp;
    bool is_wc_prev;
    bool is_wc;
    bool is_wc_next;
    int saved_p_prev, op_last_s, op_last_l, token_last_s, token_last_l;
    vector<vector<tuple<bool, int, int, int, int, int>>> infos;
    for (int qid = 0; qid < n_qrys; ++qid) {
        vector<tuple<bool, int, int, int, int, int>> info_list;
        qry = qrys + qid;
        q_len = qry->size();
        info_list.push_back(make_tuple(false, -1, -1, -1, -1, -1));
        is_wc_prev = true;
        saved_p_prev = -1;
        op_last_s = 1;
        op_last_l = 0;
        token_last_s = 1;
        token_last_l = 0;

        for (int curr_len = 1; curr_len < q_len + 1; ++curr_len) {
            ch = (*qry)[curr_len - 1];
            is_wc = IS_UWILDCARD(ch);

            is_sp = false;
            if (curr_len == q_len) {
                if (!is_wc) {
                    if (qid < n_qrys) {
                        lcp_length = lcp_length_list[qid + 1];
                        if (q_len <= lcp_length) {
                            qry_next = qry + 1;
                            is_sp = IS_UWILDCARD((*qry_next)[q_len]);
                        }
                    }
                }
            } else {
                if (!is_wc) {
                    is_wc_next = IS_UWILDCARD((*qry)[curr_len]);
                    is_sp = is_wc_next;
                }
            }

            if (is_wc) {
                if (is_wc_prev) {
                    op_last_l += 1;
                } else {
                    op_last_s = curr_len;
                    op_last_l = 1;
                }
            }

            if (is_wc) {
                token_last_s = curr_len + 1;
                token_last_l = 0;
            } else {
                token_last_l += 1;
            }

            info_list.push_back(make_tuple(is_sp, saved_p_prev, op_last_s, op_last_l, token_last_s, token_last_l));

            if (is_sp) {
                saved_p_prev = curr_len;
            }
            is_wc_prev = is_wc;
        }
        infos.push_back(info_list);
    }

    return infos;
}

vector<int> find_prefix_positions(u32string* qrys, int n_qrys) {
    // output[i]: LCP between qrys[i-1] and qrys[i]
    vector<int> prfx_pos;
    prfx_pos.reserve(n_qrys);

    int k = 0;
    u32string empty(U"");
    u32string* prev_qry = &empty;
    u32string* qry;
    char32_t ch1;
    char32_t ch2;
    int prev_q_len = 0;
    int q_len;
    int min_q_len_pair;

    for (int qid = 0; qid < n_qrys; ++qid) {
        qry = qrys + qid;
        q_len = qry->length();

        min_q_len_pair = MIN(q_len, prev_q_len);
        for (k = 0; k < min_q_len_pair; ++k) {
            ch1 = (*prev_qry)[k];
            ch2 = (*qry)[k];
            if (ch1 != ch2) {
                break;
            }
        }
        prfx_pos[qid] = k;
        prev_qry = qry;
        prev_q_len = q_len;
    }
    return prfx_pos;
}

#endif /* AF77AD98_26A2_45E5_B66D_6D69D2BF9C26 */
