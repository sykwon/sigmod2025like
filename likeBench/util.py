import psycopg2
import json
import xmltodict
import os
import time
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
import pandas as pd
import numpy as np

container_name="ceb-like"

def copy_selectivities_from_docker(output_path="predicates.txt"):
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = f"docker cp {container_name}:/var/lib/pgsql/13.1/data/predicates.txt {output_path}"
    os.system(cmd)


def copy_selectivities_to_docker(input_path="predicates.txt", sel_file_name='predicates_custom.txt'):
    cmd = f"docker cp {input_path} {container_name}:/var/lib/pgsql/13.1/data/{sel_file_name}"
    os.system(cmd)
    

def read_single_table_selectivities(output_path="predicates.txt"):
    selectivities = []
    with open(output_path, 'r') as f:
        for l in f.readlines():
            tokens = l.split(',')
            cond = ','.join(tokens[:-1])
            sel = tokens[-1]
            selectivities.append((cond, float(sel)))
    return selectivities


def remove_predicate_txt():
    cmd = f"docker exec {container_name} rm -f /var/lib/pgsql/13.1/data/predicates.txt"
    # print(f"{cmd = }")
    os.system(cmd)


def copy_cond_sels_and_setup(cur, conn, input_path, sel_file_name='predicates_custom.txt'):
    cur.execute("SET ml_cardest_enabled=true;")
    cur.execute(f"SET ml_cardest_fname='{sel_file_name}';")
    # cur.execute(f"SET ml_cardest_fname='predicates.txt';")
    cur.execute("SET query_no=0;") # should call it for every query
    conn.commit()
    tmp_path = 'predicates_custom.txt'
    
    cmd = f"docker exec {container_name} rm -f /var/lib/pgsql/13.1/data/{sel_file_name}"
    os.system(cmd)

    cond_sels = read_single_table_selectivities(input_path)
    with open(tmp_path, 'w') as f:
        for cond, sel in cond_sels:
            f.write(f"{sel}\n")
    
    cmd = f"docker cp {tmp_path} {container_name}:/var/lib/pgsql/13.1/data/{sel_file_name}"
    os.system(cmd)


def setup_cond_sels(cond_sels, cur, conn):
    sel_file_name = "full_select.txt"
    cur.execute("SET ml_cardest_enabled=true;")
    cur.execute(f"SET ml_cardest_fname='{sel_file_name}';")
    cur.execute("SET query_no=0;") # should call it for every query
    conn.commit()
    
    cmd = f"docker exec {container_name} rm -f /var/lib/pgsql/13.1/data/{sel_file_name}"
    os.system(cmd)
    
    with open(sel_file_name, 'w') as f:
        for cond, sel in cond_sels:
            f.write(f"{sel}\n")
    
    cmd = f"docker cp full_select.txt {container_name}:/var/lib/pgsql/13.1/data/{sel_file_name}"
    os.system(cmd)


def get_db_connecter():
    conn = psycopg2.connect(host="127.0.0.1", database='imdb', user='postgres', password='postgres', port=5432)
    conn.set_client_encoding("utf-8")
    cur = conn.cursor()
    return conn, cur


def get_table(cur):
    query = "SELECT table_name FROM information_schema.tables WHERE  table_schema = 'public' ORDER  BY 1;"
    tables = []
    cur.execute(query)
    r = cur.fetchall()
    for table in r:
        tables.append(table[0])
    return tables

def get_all_schema(cur):
    tables = get_table(cur)
    schema = {}
    for table in tables:
        query = f"select column_name from INFORMATION_SCHEMA.COLUMNS where table_name = '{table}';"
        cur.execute(query)
        columns = cur.fetchall()
        schema[table] = [c[0] for c in columns]
    return schema


def make_alias(string):
    if 'char_name' in string:
        return 'chn'
    if 'movie_info_idx' in string:
        return 'mi_idx'

    return ''.join([x[0] for x in string.split('_')])


def get_short_cut_dict(cur):
    tables = get_table(cur)
    short_cut_dict = {}
    for table in tables:
        short_cut = make_alias(table)
        # print(table, short_cut)
        assert short_cut not in short_cut_dict
        short_cut_dict[short_cut] = table
    return short_cut_dict


def table_col2tab_col(table_col, short_cut_dict):
    inv_dict = {v: k for k, v in short_cut_dict.items()}
    table, col = table_col.split('.')
    tab = inv_dict[table]
    return tab + "." + col


def tab_col2table_col(table_col, short_cut_dict):
    tab, col = table_col.split('.')
    tab = re.sub(r'\d+', '', tab)
    table = short_cut_dict[tab]
    return table + "." + col


def get_PSQL_cond_sels(query, cur, output_path="predicates.txt"):
    cur.execute("set print_single_tbl_queries=true;")
    remove_predicate_txt()
    cur.execute("EXPLAIN (format xml) " + query)
    copy_selectivities_from_docker(output_path)
    cond_sels = read_single_table_selectivities(output_path)
    return cond_sels


def get_PSQL_plan(query, cur):
    cur.execute("EXPLAIN (format xml) " + query)
    rows = cur.fetchall()
    plan = rows[0][0]
    data_dict = xmltodict.parse(plan)
    json_plan = json.dumps(data_dict["explain"]["Query"], indent=4)
    return json_plan


def parse_time(time_str):
    time_str = time_str.split(": ")[-1]
    execution_time = re.findall(r'\d+\.\d+|\d+', time_str)
    assert len(execution_time) == 1
    execution_time = execution_time[0]
    # print(f"{time_str = }")
    execution_time = float(execution_time)
    if 'ms' in time_str:
        execution_time /= 1000
    # print(f"{execution_time = }")
    return execution_time


def get_PSQL_execution_time(query, cur):
    cur.execute("EXPLAIN (ANALYZE, format xml)" + query)
    result = cur.fetchall()[0][0]
    result_dict = xmltodict.parse(result)
    json_plan = json.dumps(result_dict["explain"]["Query"], indent=4)
    # print(result_dict)
    # print(json_plan)
    total_cost = result_dict['explain']['Query']['Plan']['Total-Cost']
    total_cost = float(total_cost)
    # plan = rows[0][0]
    # json_plan = json.dumps(data_dict["explain"]["Query"], indent=4)
    # total_cost = result[0][0]
    # planning_time = result[-2][0]
    # assert "Planning Time: " in planning_time
    planning_time = result_dict['explain']['Query']['Planning-Time']
    planning_time = float(planning_time)
    # planning_time = result[-2][0]
    # assert "Planning Time: " in planning_time
    # execution_time = result[-1][0]
    # assert "Execution Time: " in execution_time
    execution_time = result_dict['explain']['Query']['Execution-Time']
    execution_time = float(execution_time)

    # planning_time = parse_time(planning_time)
    # execution_time = parse_time(execution_time)
    print(f"{total_cost = :.6f} {planning_time = :.6f} {execution_time = :.6f}")

    # plan = rows[0][0]
    # data_dict = xmltodict.parse(plan)
    # json_plan = json.dumps(data_dict["explain"]["Query"], indent=4)
    total_time = planning_time + execution_time
    return total_time, planning_time, execution_time, total_cost


def attach_con_sels(cur, queries):
    output = []
    for sql_query in tqdm(queries):
        cond_sels = get_PSQL_cond_sels(sql_query, cur)
        record = [sql_query, cond_sels]
        output.append(record)
    return output


def filter_cond_from_tab_dot_col_list(cond, tab_dot_col_list):
    # return any([x + ' ' in cond[:(len(x)+1)] for x in tab_dot_col_list])
    # return any([x + ' ' in cond for x in tab_dot_col_list])
    # return not any([x + ' ' in cond[:(len(x)+1)] for x in tab_dot_col_list]) and any([x + ' ' in cond for x in tab_dot_col_list])
    tokenized = tokenize_cond(cond)
    if tokenized is not None:
        tab_col, text = tokenized
        if len(text.strip("'")) == 0:
            return False
    return any([x == re.sub(r'\d+', '', cond.split()[0]) for x in tab_dot_col_list])


def filter_attached_sql_with_like_columns(attached_sql_queries, tab_dot_col_list):
    output = []
    for sql_query, cond_sels in attached_sql_queries:
        for cond, sel in cond_sels:
            if filter_cond_from_tab_dot_col_list(cond, tab_dot_col_list):
                output.append([sql_query, cond_sels])
                break
    return output


def get_cond_pred_dict_from_attached_sql_queries(attached_sql_queries, tab_dot_col_list=None):
    cond_sel_dict = {}
    for sql_query, cond_sels in tqdm(attached_sql_queries):
        for cond, sel in cond_sels:
            if cond in cond_sel_dict:
                assert cond_sel_dict[cond] == sel
            if tab_dot_col_list is not None:
                if filter_cond_from_tab_dot_col_list(cond, tab_dot_col_list):
                    cond_sel_dict[cond] = sel
            else:
                cond_sel_dict[cond] = sel
    return cond_sel_dict


def get_cond_pred_dict(cur, sql_file_name):
    cond_sel_dict = {}
    with open(sql_file_name) as f:
        for line in tqdm(f.readlines()):
            sql_query = line.rstrip('\n')
            cond_sels = get_PSQL_cond_sels(sql_query, cur)
            for cond, sel in cond_sels:
                if cond in cond_sel_dict:
                    assert cond_sel_dict[cond] == sel
                cond_sel_dict[cond] = sel
    return cond_sel_dict


def tokenize_cond(cond):
    op_list = ['~~', '!~~', '=', '<>']
    for op in op_list:
        op_sp = ' ' + op + ' '
        if op_sp in cond:
            tab_col, text = cond.split(op_sp)
            return tab_col, text
    

def get_table_col_dict_from_cond_list(cond_list, cur):
    short_cut_dict = get_short_cut_dict(cur)
    assert all([' and ' not in x for x in cond_list])

    table_col_dict = {}
    for cond in cond_list:
        tab_col, text = tokenize_cond(cond)
        text = text.strip("'")
        tab, col = tab_col.split('.')
        tab = re.sub(r'\d+', '', tab)
        table = short_cut_dict[tab]
        # table = tab
        # print(table, col)
        table_col = (table, col)
        if table_col not in table_col_dict:
            table_col_dict[table_col] = set()
        assert len(text) > 0, (cond, text)
        table_col_dict[table_col].add(text)

    table_col_dict = {x: sorted(y) for x, y in table_col_dict.items()}
    return table_col_dict


def get_data_strings_from_table_col(table_col, cur):
    table, col = table_col
    query = f"SELECT {col} from {table};"
    cur.execute(query)
    rows = cur.fetchall()
    rows = [x[0] for x in rows]
    return rows


def get_count_data_strings_from_table(table, cur):
    query = f"SELECT COUNT(*) from {table};"
    cur.execute(query)
    return int(cur.fetchall()[0][0])
    

def get_count_data_strings_from_table_col(table, col, cur):
    query = f"SELECT COUNT({col}) from {table};"
    cur.execute(query)
    return int(cur.fetchall()[0][0])


#### LIKE query
def compile_LIKE_query(qry):
    query_pat = qry.replace("%", "(.*?)").replace("_", ".")
    qry_compiled = re.compile(query_pat)
    return qry_compiled


def eval_compiled_LIKE_query(qry_re, rec):
    return qry_re.fullmatch(rec)


def data_gen_alg_regex(db, qrys, p_bar=True, **kwargs):
    # raise NotImplementedError
    res = []
    if p_bar:
        qrys = tqdm(qrys)

    start_time = time.time()
    for qry in qrys:
        card = 0

        # # query_pat = qry.strip('%').replace("%", "(.*?)").replace("_", ".")
        # query_pat = qry.replace("%", "(.*?)").replace("_", ".")
        # # print(query_pat)
        # compiled_pat = re.compile(query_pat)
        compiled_pat = compile_LIKE_query(qry)

        for record in db:
            # search is faster than fullmatch
            # match_obj = compiled_pat.search(record)
            match_obj = eval_compiled_LIKE_query(compiled_pat, record)
            if match_obj is not None:
                card += 1

        res.append((qry, card))

    end_time = time.time()

    total_time = end_time - start_time
    time_dict = {
        "total": total_time
    }

    return res, time_dict

if __name__ == "__main__":
    conn, cur = get_db_connecter()

    cur.execute("set print_single_tbl_queries=true;")
    remove_predicate_txt()
    query = "SELECT MIN(n.name) AS voicing_actress, MIN(t.title) AS voiced_movie FROM aka_name AS an, char_name AS chn, cast_info AS ci, company_name AS cn, info_type AS it, movie_companies AS mc, movie_info AS mi, name AS n, role_type AS rt, title AS t WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code ='[us]' AND it.info = 'release dates' AND mc.note IS NOT NULL AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND n.gender ='f' AND n.name LIKE '%Ang%' AND rt.role ='actress' AND t.production_year BETWEEN 2005 AND 2009 AND t.id = mi.movie_id AND t.id = mc.movie_id AND t.id = ci.movie_id AND mc.movie_id = ci.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND cn.id = mc.company_id AND it.id = mi.info_type_id AND n.id = ci.person_id AND rt.id = ci.role_id AND n.id = an.person_id AND ci.person_id = an.person_id AND chn.id = ci.person_role_id;"
    cur.execute("EXPLAIN (format xml) " + query)

    rows = cur.fetchall()
    plan = rows[0][0]
    data_dict = xmltodict.parse(plan)
    json_plan = json.dumps(data_dict["explain"]["Query"], indent=4)
    print(json_plan)


    sels = []
    cond_sels = read_single_table_selectivities()
    for cond, sel in cond_sels:
        print(cond, sel)
        if '%Ang%' in cond:
            sel = 0.1
        sels.append((cond, sel))
    setup_cond_sels(sels, cur, conn)

    cur.execute("EXPLAIN (format xml) " + query)

    rows = cur.fetchall()
    plan = rows[0][0]
    data_dict = xmltodict.parse(plan)
    json_plan = json.dumps(data_dict["explain"]["Query"], indent=4)
    print(json_plan)
    
def read_strings(filepath):
    lines = []
    with open(filepath) as f:
        for line in f:
            lines.append(line.rstrip('\n'))
    return lines

def parse_like_query(qry, split_beta=False):
    parsed = []
    curr = qry[0]
    is_wild = curr == "_" or curr == "%"
    for ch in qry[1:]:
        if ch == "_" or ch == "%":
            if is_wild:
                curr += ch
            else:
                parsed.append(curr)
                is_wild = True
                curr = ch
        else:
            if is_wild:
                parsed.append(curr)
                is_wild = False
                curr = ch
            else:
                curr += ch
    parsed.append(curr)
    if split_beta:
        parsed_bak = parsed
        parsed = []
        for token in parsed_bak:
            if "%" in token or "_" in token:
                parsed.append(token)
            else:
                parsed.extend(list(token))
    return parsed


def collect_predict_result(model, table_cols, short_cut_dict):
    predict_dict = {}
    for table_col in table_cols:
        tab_col = table_col2tab_col(table_col, short_cut_dict)
        db_path = f"data/{table_col}/{table_col}.txt"
        db = read_strings(db_path)
        n_db = len(db)
        n_null = db.count('')
        res_path = f"res/{table_col}/CEB/estimation/{model}/0/estimation.csv"
        if not os.path.exists(res_path):
            print(f"Not exist {res_path = }")
            continue
        df = pd.read_csv(res_path)
        for idx, row in df.iterrows():
            query = row.query
            is_exact = '%' not in query and '_' not in query
            if is_exact:
                cond = tab_col + " = '" + query + "'"
                neg_cond = tab_col + " <> '" + query + "'"
            else:
                cond = tab_col + " ~~ '" + query + "'"
                neg_cond = tab_col + " !~~ '" + query + "'"
            

            if is_exact:
                neg_est = n_db - row.est
                neg_true = n_db - row.true
            else:
                neg_est = n_db - n_null - row.est
                neg_true = n_db - n_null - row.true
            predict_dict[cond] = [row.est / n_db, row.true / n_db, row.est, row.true]
            predict_dict[neg_cond] = [neg_est / n_db, neg_true / n_db, neg_est, neg_true]
            # print(row.query, row.est, row.true)
            # cond = (tab_col, row.query)
            # predict_dict[cond] = [row.est, row.true]
        # print(model, table_col)
        # print(df)
    return predict_dict


def short_cond_to_predicate(cond, short_cut_dict):
    # cond (mc.note ~~ '%(Japan)%')
    # predicate movie_companies.note LIKE '%(Japan)%'
    op_dict = {
        " !~~ ": " NOT LIKE ",
        " ~~ ": " LIKE ",
        " = ": " = ",
        " <> ": " != ",
        " < ": " < ",
        " > ": " > ",
        " >= ": " >= ",
        " <= ": " <= ",
    }
    predicate = None
    for op, sql_op in op_dict.items():
        if op in cond:
            tab_col, text = cond.split(op)
            table_col = tab_col2table_col(tab_col, short_cut_dict)
            predicate = table_col + sql_op + text 
            break
    if predicate is None:
        raise ValueError(f"{cond = } cannot be parsed")
    return predicate


def compute_cardinality_of_predicate(predicate, cur):
    table = predicate.split('.')[0]
    query = f"SELECT count(*) FROM {table} WHERE {predicate};"
    print(f"{query = }")
    cur.execute(query)
    res = cur.fetchall()
    card = res[0][0]
    return card
    

def compute_selectivity_of_predicate(predicate, cur):
    table = predicate.split('.')[0]
    query = f"SELECT count(*) FROM {table} WHERE {predicate};"
    # print(f"{query = }")
    cur.execute(query)
    res = cur.fetchall()
    card = res[0][0]

    query = f"SELECT count(*) FROM {table};"
    # print(f"{query = }")
    cur.execute(query)
    res = cur.fetchall()
    n_db = res[0][0]
    return card / n_db

    
def remove_outliers_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    return np.mean(filtered_data)


def remove_outliers_iqr_for_tuple(data, target_idx):
    target_values = [x[target_idx] for x in data]
    q1 = np.percentile(target_values, 25)
    q3 = np.percentile(target_values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_data = [x for x in data if lower_bound <= x[target_idx] <= upper_bound]
    filtered_data = np.array(filtered_data)
    return [float(f"{x:.3f}") for x in np.mean(filtered_data, axis=0)]