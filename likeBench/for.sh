#!/bin/bash
for model in True PG LPLM CLIQUE CLIQUE-PACK-ALL;do
    for remain in True PG;do
        CMD="python eval_sql.py $model/$remain"
        echo $CMD
        eval $CMD
    done
done