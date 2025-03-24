#pragma once
#include <iostream>
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

enum CondOp {
    PREFIX,
    SUFFIX,
    SUBSTR
};

const char* CondOpStrings[] = {
    "Prefix",
    "Suffix",
    "Substr"};

class Condition {
public:
    int number;
    int identifier;
    int alpha;
    bool percent;
    bool left_end;   // For A cond_op B, if A contains the first table, this is true
    bool right_end;  // For A cond_op B, if B contains the last table, this is true
    bool opt_next;   /* When next op is '%' for left_end or right_end,
                        it takes first or last pos only*/
    bool is_last;
    CondOp cond_op;

    Condition(int number, int alpha, bool percent, bool left_end, bool right_end, bool is_next, bool is_last, CondOp cond_op) {
        this->number = number;
        this->alpha = alpha;
        this->percent = percent;
        this->left_end = left_end;
        this->right_end = right_end;
        this->opt_next = is_next && (left_end || right_end);  // CHECK
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