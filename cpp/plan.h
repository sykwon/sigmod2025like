#ifndef CD042585_90C4_4ADE_BAD1_45304513334A
#define CD042585_90C4_4ADE_BAD1_45304513334A

#include <stdexcept>

#include "index.h"
#include "util.h"

using namespace std;

enum Symbol {
    IDENTIFIER,
    NUMBER,
    PLUS,
    MINUS,
    COMPEQ,
    EQ,
    COMPGEQ,
    GEQ,
    COMPLEQ,
    LEQ,
    GT,
    LT,
    AND
};

const char* SymbolStrings[] = {
    "IDENTIFIER",
    "NUMBER",
    "PLUS",
    "MINUS",
    "COMPEQ",
    "EQ",
    "COMPGEQ",
    "GEQ",
    "COMPLEQ",
    "LEQ",
    "GT",
    "LT",
    "AND"};

enum JoinOp {
    READ,
    SELECT,
    JOIN
};

const char* JoinOpStrings[] = {
    "Read",
    "Select",
    "Join"};

enum CondOp {
    PREFIX,
    SUFFIX,
    SUBSTR
};

const char* CondOpStrings[] = {
    "Prefix",
    "Suffix",
    "Substr"};

class LTable {
public:
    vector<int>* L;
    int n_pos;
    int count = 0;
    int* s_lens;
    bool view = false;

    LTable(vector<int>* L, int n_pos, int* s_lens, bool view) {
        this->L = L;
        this->n_pos = n_pos;
        this->s_lens = s_lens;
        this->view = view;
    }

    void set_count(int count) {
        this->count = count;
    }

    ~LTable() {
        if (this->view) {
            delete L;
        }
    }
};

class Condition {
public:
    int number;
    int identifier;
    bool percent;
    bool left_end;
    bool right_end;
    bool is_last;
    CondOp cond_op;

    Condition(int number, bool percent, bool left_end, bool right_end, bool is_last, CondOp cond_op) {
        this->number = number;
        this->percent = percent;
        this->left_end = left_end;
        this->right_end = right_end;
        this->is_last = is_last;
        this->cond_op = cond_op;
    }

    ~Condition() {
    }

    void set_identifier(int num1) {  // for prefix
        this->identifier = num1;
    }

    void set_identifier(int num1, int num2) {  // for suffix and join
        this->identifier = num2 - num1;
    }

    inline int l_val() {
        return this->number;
    }

    inline int r_val() {
        return this->identifier;
    }

    bool eval() {
        if (this->percent) {
            return this->l_val() <= this->r_val();
        } else {
            return this->l_val() == this->r_val();
        }
    }

    int eval_comp() {
        if (this->l_val() == this->r_val()) {
            return Symbol::EQ;
        } else if (this->l_val() > this->r_val()) {
            return Symbol::GT;
        } else {
            return Symbol::LT;
        }
    }

    void print(int level = 0) {
        for (int i = 0; i < level; i++) {
            cout << "\t";
        }
        cout << CondOpStrings[this->cond_op] << ": " << this->number;

        if (this->percent) {
            cout << " <= ";
        } else {
            cout << " == ";
        }

        switch (this->cond_op) {
            case CondOp::PREFIX:
                cout << " L_1.pos" << endl;
                break;
            case CondOp::SUFFIX:
                cout << " L_m.len - L_m.pos" << endl;
                break;
            case CondOp::SUBSTR:
                cout << " L_2.pos - L_1.pos" << endl;
                break;
            default:
                break;
        }
    }
};

class PlanNode {
public:
    JoinOp op;  // read, select, join
    Condition* cond = nullptr;
    LTable* table = nullptr;
    PlanNode* l_child = nullptr;
    PlanNode* r_child = nullptr;

    PlanNode(JoinOp op, Condition* cond = nullptr, PlanNode* parent = nullptr) {
        this->op = op;
        this->cond = cond;
        if (parent) {
            if (parent->l_child) {
                parent->r_child = this;
            } else {
                parent->l_child = this;
            }
        }
    }

    ~PlanNode() {
        if (this->cond) {
            delete this->cond;
        }
        if (this->l_child) {
            delete this->l_child;
        }
        if (this->r_child) {
            delete this->r_child;
        }
    }

    void set_table(LTable* table) {
        this->table = table;
    }

    void add_child(PlanNode* child) {
        if (!this->l_child) {
            this->l_child = child;
        } else if (!this->r_child) {
            this->r_child = child;
        }
    }

    void add_children(PlanNode* l_child, PlanNode* r_child) {
        assert(!this->l_child);
        assert(!this->r_child);
        this->l_child = l_child;
        this->r_child = r_child;
    }

    LTable* op_select() {  // prefix & suffix
        assert(this->l_child);
        LTable* input_table = this->l_child->execute_plan();
        Condition* cond = this->cond;
        vector<int>& input_list = *input_table->L;
        vector<int>& output_list = *(new vector<int>());
        int idx = 0;
        int id;
        int n;
        int pos1;
        int pos2;
        bool is_last = cond->is_last;
        int n_pos = input_table->n_pos;
        assert(n_pos == 1 || n_pos == 2);
        int* s_lens = input_table->s_lens;
        int s_len;
        int curr_n_idx;
        int count = 0;

        while (idx < (int)input_list.size()) {
            id = input_list[idx++];
            n = input_list[idx++];

            curr_n_idx = -1;
            if (this->cond->cond_op == CondOp::PREFIX) {
                for (int i = 0; i < n; ++i) {
                    pos1 = input_list[idx + i * n_pos];
                    cond->set_identifier(pos1);
                    if (!cond->percent) {  // fixed-length
                        if (cond->eval()) {
                            if (curr_n_idx < 0) {
                                ++count;
                                if (is_last) {
                                    break;
                                }
                                output_list.push_back(id);
                                curr_n_idx = output_list.size();
                                output_list.push_back(0);
                            }
                            pos2 = input_list[idx + (i + 1) * n_pos - 1];
                            ++output_list[curr_n_idx];
                            output_list.push_back(pos2);
                        }
                    } else {  // percent exists
                        if (cond->eval()) {
                            ++count;
                            if (!is_last) {
                                output_list.push_back(id);
                                output_list.push_back(n - i);
                                while (i < n) {
                                    pos2 = input_list[idx + (i + 1) * n_pos - 1];
                                    output_list.push_back(pos2);
                                    ++i;
                                }
                            }
                            break;
                        }
                    }
                }
            } else {
                s_len = s_lens[id];
                for (int i = 0; i < n; ++i) {
                    pos2 = input_list[idx + (i + 1) * n_pos - 1];
                    cond->set_identifier(pos2, s_len);
                    if (!cond->percent) {  // fixed-length
                        if (cond->eval()) {
                            if (curr_n_idx < 0) {
                                ++count;
                                if (is_last) {
                                    break;
                                }
                                output_list.push_back(id);
                                curr_n_idx = output_list.size();
                                output_list.push_back(0);
                            }
                            pos1 = input_list[idx + i * n_pos];
                            ++output_list[curr_n_idx];
                            output_list.push_back(pos1);
                        }
                    } else {  // percent exists
                        if (cond->eval()) {
                            if (curr_n_idx < 0) {
                                ++count;
                                if (!is_last) {
                                    output_list.push_back(id);
                                    curr_n_idx = output_list.size();
                                    output_list.push_back(0);
                                } else {
                                    break;
                                }
                            }

                            if (!is_last) {
                                output_list[curr_n_idx] += 1;
                                pos1 = input_list[idx + i * n_pos];
                                output_list.push_back(pos1);
                            }
                        }
                    }
                }
            }
            idx += n * n_pos;
        }

        n_pos = 1;

        LTable* output_table = new LTable(&output_list, n_pos, s_lens, true);
        output_table->count = count;

        delete input_table;
        return output_table;
    }

    LTable* op_join() {
        assert(this->l_child);
        assert(this->r_child);
        LTable* l_table = this->l_child->execute_plan();
        LTable* r_table = this->r_child->execute_plan();

        Condition* cond = this->cond;
        bool left_end = cond->left_end;
        bool right_end = cond->right_end;
        vector<int>& p_list1 = *l_table->L;
        vector<int>& p_list2 = *r_table->L;
        vector<int>& output_list = *(new vector<int>());
        int idx1 = 0;
        int idx2 = 0;
        int id1;
        int id2;
        int n1;
        int n2;
        int pos1;
        int pos2;
        int pos3;
        int pos4;
        int n_pos1 = l_table->n_pos;
        int n_pos2 = r_table->n_pos;
        assert(n_pos1 == 1 || n_pos1 == 2);
        assert(n_pos2 == 1 || n_pos2 == 2);
        int* s_lens = l_table->s_lens;
        int curr_n_idx;
        int count = 0;

        while (idx1 < (int)p_list1.size() && idx2 < (int)p_list2.size()) {
            id1 = p_list1[idx1];
            id2 = p_list2[idx2];

            if (id1 < id2) {
                n1 = p_list1[++idx1];
                idx1 += n1 * n_pos1 + 1;
            } else if (id1 > id2) {
                n2 = p_list2[++idx2];
                idx2 += n2 * n_pos2 + 1;
            } else {
                n1 = p_list1[++idx1];
                n2 = p_list2[++idx2];
                idx1 += 1;
                idx2 += 1;
                curr_n_idx = -1;

                for (int i1 = 0; i1 < n1; ++i1) {
                    pos1 = p_list1[idx1 + i1 * n_pos1];
                    pos2 = p_list1[idx1 + (i1 + 1) * n_pos1 - 1];
                    for (int i2 = 0; i2 < n2; ++i2) {
                        pos3 = p_list2[idx2 + i2 * n_pos2];
                        cond->set_identifier(pos2, pos3);
                        if (cond->eval()) {
                            if (curr_n_idx < 0) {
                                ++count;
                                output_list.push_back(id1);
                                curr_n_idx = output_list.size();
                                output_list.push_back(0);
                            }
                            output_list[curr_n_idx] += 1;
                            pos4 = p_list2[idx2 + (i2 + 1) * n_pos2 - 1];
                            if (!left_end) {
                                output_list.push_back(pos1);
                            }
                            if (!right_end) {
                                output_list.push_back(pos4);
                            }
                        }
                    }
                }
                idx1 += n1 * n_pos1;
                idx2 += n2 * n_pos2;
            }
        }

        int n_pos = 0;
        if (!left_end) {
            ++n_pos;
        }
        if (!right_end) {
            ++n_pos;
        }

        delete l_table;
        delete r_table;

        // cout << "left end: " << left_end << endl;
        // cout << "right end: " << right_end << endl;
        // cout << "[join Header] ";
        // print_vector(H);

        LTable* output_table = new LTable(&output_list, n_pos, s_lens, true);
        output_table->count = count;

        return output_table;
    }

    LTable* execute_plan() {
        switch (this->op) {
            case JoinOp::READ:
                assert(this->table);
                // cout << "op read" << endl;
                // cout << this->table->H.front() << endl;
                return this->table;
                break;
            case JoinOp::SELECT:
                // cout << "op select" << endl;
                return this->op_select();
                break;
            case JoinOp::JOIN:
                // cout << "op join" << endl;
                return this->op_join();
                break;
            default:
                throw runtime_error("Invalid symbol");
                break;
        }
    }

    void print_node(int level = 0) {
        for (int i = 0; i < level; ++i) {
            cout << "\t";
        }
        cout << JoinOpStrings[this->op] << endl;
        if (this->cond) {
            this->cond->print(level + 1);
        }
        if (this->l_child) {
            this->l_child->print_node(level + 1);
        }
        if (this->r_child) {
            this->r_child->print_node(level + 1);
        }
    }
};

vector<bool> order2left_ends(const vector<int>& order) {
    vector<bool> output;
    int m = order.size() - 1;
    int o1;
    int o2;
    int count;
    for (size_t j = 0; j < order.size(); j++) {
        /* code */
        o2 = order[j];
        if (o2 == m) {
            output.push_back(false);
        } else if (o2 == 0) {
            count = 0;
            for (size_t i = 0; i < j; i++) {
                o1 = order[i];
                if (o1 == 1) {
                    ++count;
                    break;
                }
            }
            output.push_back(count == 1);
        } else {
            count = 0;
            for (size_t i = 0; i < j; i++) {
                o1 = order[i];
                if (o1 < o2) {
                    ++count;
                }
            }
            output.push_back(count == o2);
        }
    }
    return output;
}

vector<bool> order2right_ends(const vector<int>& order) {
    vector<bool> output;
    int m = order.size() - 1;
    int o1;
    int o2;
    int count;
    for (size_t j = 0; j < order.size(); j++) {
        /* code */
        o2 = order[j];
        if (o2 == 0) {
            output.push_back(false);
        } else if (o2 == m) {
            count = 0;
            for (size_t i = 0; i < j; ++i) {
                o1 = order[i];
                if (o1 == m - 1) {
                    ++count;
                    break;
                }
            }
            output.push_back(count == 1);
        } else {
            count = 0;
            for (size_t i = 0; i < j; ++i) {
                o1 = order[i];
                if (o1 > o2) {
                    ++count;
                }
            }
            output.push_back(count == (m - o2));
        }
    }
    return output;
}

vector<int> lens2idx(vector<int>& lengths) {
    vector<int> output;
    output.push_back(-1);

    // cout << "length: ";
    // for (auto length : lengths) {
    //     cout << length << ", ";
    // }
    // cout << endl;

    int start = 0;
    for (int length : lengths) {
        if (length >= 0) {
            start += 1;
            for (int i = 0; i < length; i++) {
                output.push_back(start);
            }
        }
    }

    // cout << "idx_list: ";
    // for (auto idx : output) {
    //     cout << idx << ", ";
    // }
    // cout << endl;
    return output;
}

class TreePlan {
public:
    PlanNode* root = nullptr;
    vector<LTable*> tables;

    TreePlan() {}

    ~TreePlan() {
        delete root;
    }

    void set_LTables(const vector<LTable*>& tables) {
        this->tables = tables;
    }

    void construct_by_order(const vector<u32string>& op_tokens, const vector<u32string>& tokens, const vector<int>& order) {
        u32string op_token;
        u32string token;
        int alpha;
        int beta;
        bool percent;
        PlanNode* node;
        Condition* cond;
        vector<PlanNode*> L_nodes;
        vector<int> join_lens;
        vector<int> join_idx;
        int jidx1;
        int jidx2;
        int idx;
        vector<bool> left_ends = order2left_ends(order);
        vector<bool> right_ends = order2right_ends(order);
        // cout << "[left_ends] ";
        // print_vector(left_ends);
        // cout << "[right_ends] ";
        // print_vector(right_ends);
        // if (op_tokens.size() != order.size()) {
        //     cout << utf8::utf32to8(query) << endl;
        //     cout << op_tokens.size() << endl;
        //     cout << order.size() << endl;
        // }
        // assert(op_tokens.size() == order.size());
        int m = tokens.size();

        // assert(m >= 1);

        // assert((int)this->tables.size() == m);

        L_nodes.push_back(nullptr);
        join_lens.push_back(-1);
        for (int i = 1; i <= m; i++) {
            node = new PlanNode(JoinOp::READ);
            node->set_table(this->tables[i - 1]);
            L_nodes.push_back(node);
            join_lens.push_back(1);
        }
        join_idx = lens2idx(join_lens);

        for (int i = 0; i <= m; i++) {
            idx = order[i];
            op_token = op_tokens[idx];
            percent = op_token[0] == U'%';
            alpha = op_token.size() - (int)percent;

            if (idx == 0) {  // prefix select
                if (op_token != U"%") {
                    cond = new Condition(alpha + 1, percent, left_ends[i], right_ends[i], i == m, CondOp::PREFIX);
                    node = new PlanNode(JoinOp::SELECT, cond);
                    node->add_child(L_nodes[1]);
                    L_nodes[1] = node;
                }
            } else if (idx == m) {  // suffix select
                if (op_token != U"%") {
                    token = tokens[idx - 1];
                    beta = token.size();
                    cond = new Condition(alpha + beta - 1, percent, left_ends[i], right_ends[i], i == m, CondOp::SUFFIX);
                    node = new PlanNode(JoinOp::SELECT, cond);
                    node->add_child(L_nodes.back());
                    L_nodes.back() = node;
                }
            } else {  // join
                token = tokens[idx - 1];
                beta = token.size();
                cond = new Condition(alpha + beta, percent, left_ends[i], right_ends[i], i == m, CondOp::SUBSTR);
                node = new PlanNode(JoinOp::JOIN, cond);
                jidx1 = join_idx[idx];
                jidx2 = join_idx[idx + 1];
                assert(jidx1 + 1 == jidx2);
                node->add_children(L_nodes[jidx1], L_nodes[jidx2]);
                join_lens[jidx1] += join_lens[jidx2];
                join_lens.erase(join_lens.begin() + jidx2);
                join_idx = lens2idx(join_lens);
                L_nodes.erase(L_nodes.begin() + jidx2);
                L_nodes[jidx1] = node;
            }
        }
        this->root = L_nodes[1];
    }

    int find_card() {
        int card;
        LTable* res;
        assert(this->root);
        res = this->root->execute_plan();
        card = res->count;
        delete res;
        return card;
    }

    void print() {
        this->root->print_node();
    }
};

#endif /* CD042585_90C4_4ADE_BAD1_45304513334A */