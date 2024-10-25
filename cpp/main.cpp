#include <locale.h>

#include <cstdlib>
#include <cwchar>
#include <iostream>
#include <random>

#include "alg.h"
#include "index.h"
#include "util.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (!(argc == 6)) {
        cout << "usage: <alg> <data_name> <workload> <query_key> <is_aug>" << endl;
        cout << "input: ";
        for (int i = 1; i < argc; ++i) {
            if (i > 1) cout << " ";
            cout << argv[i];
        }
        cout << endl;
        return 0;
    }
    string alg_name = argv[1];
    string data_name = argv[2];
    string workload = argv[3];
    string query_key = argv[4];
    string out_query_key = query_key;
    AUG_TYPE aug_type = static_cast<AUG_TYPE>(stoi(argv[5]));

    if (aug_type == AUG_TYPE::PREFIX_AUG) {
        out_query_key = query_key + "_P";
    }

    // input paths
    string db_path = "data/" + data_name + "/" + data_name + ".txt";
    string qrys_path = "data/" + data_name + "/query/" + workload + "/" + query_key + ".txt";

    // output paths
    string res_dir = "res/" + data_name + "/" + workload + "/data/" + out_query_key + "/";
    string time_dir = "res/" + data_name + "/" + workload + "/time/" + out_query_key + "/";
    string q_time_dir = "res/" + data_name + "/" + workload + "/q_time/" + out_query_key + "/";
    string res_path = res_dir + alg_name + ".txt";
    string time_path = time_dir + alg_name + ".txt";
    string q_time_path = q_time_dir + alg_name + ".txt";

    string mk_res_dir = "mkdir -p " + res_dir;
    string mk_time_dir = "mkdir -p " + time_dir;
    string mk_q_time_dir = "mkdir -p " + q_time_dir;

    int fail;

    tuple<u32string*, int> db_tp = read_strings(db_path);
    u32string* db = get<0>(db_tp);
    int n_db = get<1>(db_tp);

    tuple<u32string*, int> qrys_tp = read_strings(qrys_path);
    u32string* qrys = get<0>(qrys_tp);
    int n_qrys = get<1>(qrys_tp);

    int* res = nullptr;
    vector<vector<int>> res_vector;
    int n_res = n_qrys;
    vector<tuple<string, double>> time_dict;
    vector<double> q_times;

    bool is_ascii = (data_name == "DBLP") || (data_name == "GENE") || (data_name == "AUTHOR");

    switch (str2inthash(alg_name.c_str())) {
        case str2inthash("Naive"): {
            res = data_gen_alg_re2(db, n_db, qrys, n_qrys, &time_dict, &q_times, is_ascii);
            break;
        }

        case str2inthash("NaiveIndex"): {
            res = data_gen_alg_re2_index(db, n_db, qrys, n_qrys, &time_dict, &q_times);
            break;
        }

        case str2inthash("LEADER-T"): {
            is_bin_srch = true;  // BinSrch
            is_share = true;     // SH
            data_gen_alg_LEADER_T(db, n_db, qrys, n_qrys, &time_dict, q_times, aug_type, res_vector);
            break;
        }

        case str2inthash("LEADER-S"): {
            is_bin_srch = true;  // BinSrch
            is_share = true;     // SH

            data_gen_alg_LEADER_S(db, n_db, qrys, n_qrys, &time_dict, q_times, aug_type, res_vector);
            break;
        }

        case str2inthash("LEADER-SR"): {
            is_bin_srch = true;
            data_gen_alg_LEADER_SR(db, n_db, qrys, n_qrys, &time_dict, q_times, aug_type, res_vector);
            break;
        }

        default:
            cout << "Invaild alg name: " << alg_name << endl;
            return 1;

            break;
    }

    fail = std::system(mk_res_dir.c_str());
    if (fail) {
        cout << "fail to create res_dir" << endl;
        return 0;
    }
    fail = std::system(mk_time_dir.c_str());
    if (fail) {
        cout << "fail to create time_dir" << endl;
        return 0;
    }
    fail = std::system(mk_q_time_dir.c_str());
    if (fail) {
        cout << "fail to create q_time_dir" << endl;
        return 0;
    }

    ofstream ofs;

    if (res) {
        cout << "res_path: " << res_path << endl;
        ofs.open(res_path.data());
        for (int i = 0; i < n_res; i++) {
            ofs << csv_token(qrys[i]) << "," << res[i] << endl;
        }
        ofs.close();
    }

    if (res_vector.size() > 0) {
        assert(n_res == (int)res_vector.size());
        cout << "res_path: " << res_path << endl;
        ofs.open(res_path.data());
        for (int i = 0; i < (int)res_vector.size(); i++) {
            ofs << csv_token(qrys[i]);
            auto cards = res_vector[i];
            assert(cards.size() > 0);
            for (int j = 0; j < (int)cards.size(); ++j) {
                ofs << "," << cards[j];
            }
            ofs << endl;
        }
        ofs.close();
    }

    if (time_dict.size() > 0) {
        ofs.open(time_path.data());
        cout << "time_path: " << time_path << endl;
        for (auto itr = time_dict.begin(); itr != time_dict.end(); ++itr) {
            ofs << get<0>(*itr) << ": " << get<1>(*itr) << endl;
        }
        ofs.close();
    }

    if ((int)q_times.size() == n_res) {
        ofs.open(q_time_path.data());
        cout << "q_time_path: " << q_time_path << endl;
        for (int i = 0; i < n_res; i++) {
            ofs << csv_token(qrys[i]) << "," << q_times[i] << endl;
        }
        ofs.close();
    }
    if (res) {
        delete[] res;
    }
    if (db) {
        delete[] db;
    }
    if (qrys) {
        delete[] qrys;
    }
    return 0;
}
