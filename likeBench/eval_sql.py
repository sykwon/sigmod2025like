import numpy as np
import sys
import util as ut
import os
import xmltodict
import json
import yaml

sql_file_name = "job_LPLM.sql"
con, cur = ut.get_db_connecter()


sel_type = sys.argv[1]
is_plan = len(sys.argv) > 2
print(f"{sel_type = } {is_plan = }")

sql_queries = ut.read_strings(sql_file_name)
assert '/' in sel_type
print(sel_type)

for sql_id, sql_query in enumerate(sql_queries, start=1):
    predicates_path = os.path.join("res/selectivity", sel_type, f'predicates_{sql_id:02d}.txt')
    ut.copy_cond_sels_and_setup(cur, con, input_path=predicates_path)
    # print(sql_id, predicates_path)
    # cur.execute("set print_single_tbl_queries=true;")
    # print(sql_query)
    # cur.execute(sql_query)
    if is_plan:
        plan_path = os.path.join("res/CEB/plan", sel_type, f'predicates_{sql_id:02d}.txt')
        json_plan = ut.get_PSQL_plan(sql_query, cur)
        print(f"write {plan_path = }")
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        with open(plan_path, 'w') as f:
            f.write(json_plan)
    else:
        time_path = os.path.join("res/CEB/time", sel_type, f'predicates_{sql_id:02d}.txt')
        results = []
        for trial in range(10):
            cur.execute("SET query_no=0;") # should call it for every query
            con.commit()
            total_time, planning_time, execution_time, total_cost = ut.get_PSQL_execution_time(sql_query, cur)
            results.append([total_time, planning_time, execution_time, total_cost])

        filtered_avg = ut.remove_outliers_iqr_for_tuple(results, 0)
        total_time, planning_time, execution_time, total_cost = filtered_avg
        results_dict = {
            "total_time" : total_time,
            "planning_time": planning_time,
            "execution_time": execution_time,
            "total_cost": total_cost,
        }
        print(f"saved at {time_path = }")
        os.makedirs(os.path.dirname(time_path), exist_ok=True)
        with open(time_path, "w") as f:
            yaml.safe_dump(results_dict, f)

