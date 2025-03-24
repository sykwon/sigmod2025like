#ifndef CD042585_90C4_4ADE_BAD1_45304513334A
#define CD042585_90C4_4ADE_BAD1_45304513334A

#include <stdexcept>

#include "cond.h"
#include "index.h"
#include "table.h"
#include "util.h"

using namespace std;

enum JoinOp {
    READ,
    SELECT,
    JOIN
};

const char* JoinOpStrings[] = {
    "Read",
    "Select",
    "Join"};

class LEADERpTree;

class PlanNode {
public:
    JoinOp op;  // read, select, join
    Condition* cond = nullptr;
    Posting* table = nullptr;
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

    void set_table(Posting* table) {
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

    Posting* op_select() {
        assert(this->l_child);
        Posting* input_posting = this->l_child->execute_plan();
        Condition* cond = this->cond;
        CondOp& cond_op = cond->cond_op;
        // vector<int>& output_list = *(new vector<int>());
        Posting* output_posting = new Posting(false, true);
        output_posting->beta = input_posting->beta;
        int n_pos = input_posting->is_single ? 1 : 2;
        assert(n_pos == 1 || n_pos == 2);

        int alpha = cond->alpha;
        u32string op_token(alpha + cond->percent, U'_');

        if (cond->percent) {
            op_token[0] = U'%';
        }
        bool opt_next = cond->opt_next;
        bool opt_last = cond->is_last;

        assert(input_posting->is_single);

        if (cond_op == CondOp::PREFIX) {
            output_posting = get_prefix_plists(input_posting, op_token, opt_next, opt_last);
        } else {
            output_posting = get_suffix_plists(input_posting, op_token, opt_next, opt_last);
        }

        if (input_posting->view) {
            delete input_posting;
        }
        return output_posting;
    }

    Posting* op_join() {
        assert(this->l_child);
        assert(this->r_child);
        Posting* l_table = this->l_child->execute_plan();
        Posting* r_table = this->r_child->execute_plan();

        Condition* cond = this->cond;
        Posting* output_table;
        output_table = join_two_plists_multi(l_table, r_table, cond);

        if (l_table->view) {
            delete l_table;
        }
        if (r_table->view) {
            delete r_table;
        }

        return output_table;
    }

    Posting* execute_plan() {
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

vector<bool> order2is_nexts(const vector<int>& order, const vector<u32string>& op_tokens) {
    vector<bool> output;
    int m = order.size() - 1;
    int o1;
    int o2;
    // int count;
    u32string next_op_token;
    for (size_t i = 0; i < order.size(); i++) {
        o1 = order[i];
        next_op_token = U"";
        for (size_t j = i + 1; j < order.size(); j++) {
            o2 = order[j];
            if (o2 != 0 && o2 != m) {
                next_op_token = op_tokens[o2];
                break;
            }
        }
        if (o1 == 0) {
            output.push_back(false);
        } else if (o1 == m) {
            output.push_back(false);
        } else {
            output.push_back(next_op_token[0] == U'%');  // next_op_token contains '%'
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
    vector<Posting*> tables;

    TreePlan() {}

    ~TreePlan() {
        delete root;
    }

    void set_Postings(const vector<Posting*>& tables) {
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
        vector<bool> is_nexts = order2is_nexts(order, op_tokens);
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
                    cond = new Condition(alpha + 1, alpha, percent, left_ends[i], right_ends[i], is_nexts[i], i == m, CondOp::PREFIX);
                    node = new PlanNode(JoinOp::SELECT, cond);
                    node->add_child(L_nodes[1]);
                    L_nodes[1] = node;
                }
            } else if (idx == m) {  // suffix select
                if (op_token != U"%") {
                    token = tokens[idx - 1];
                    beta = token.size();
                    cond = new Condition(alpha + beta - 1, alpha, percent, left_ends[i], right_ends[i], is_nexts[i], i == m, CondOp::SUFFIX);
                    node = new PlanNode(JoinOp::SELECT, cond);
                    node->add_child(L_nodes.back());
                    L_nodes.back() = node;
                }
            } else {  // join
                token = tokens[idx - 1];
                beta = token.size();
                cond = new Condition(alpha + beta, alpha, percent, left_ends[i], right_ends[i], is_nexts[i], i == m, CondOp::SUBSTR);
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
        Posting* res;
        assert(this->root);
        res = this->root->execute_plan();
        card = res->inv_list[0];
        delete res;
        return card;
    }

    void print() {
        this->root->print_node();
    }
};

#endif /* CD042585_90C4_4ADE_BAD1_45304513334A */