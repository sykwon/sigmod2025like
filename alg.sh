#!/bin/bash

alg=${1}
data=${2}
workload=${3}
query=${4}
is_aug=${5}
python main_PSQL.py <alg> <data> <workload> <query> <is_aug>

function RUN_ALG_CMD {
    if [[ "$alg " == "LBS" ]]; then
        CMD="python main_LBS.py -d $data -w $workload"
    elif [[ "$model" == "EST_S" ]]; then 
        CMD="python main_sampling.py -d $data -w $workload"
    elif [[ "$model" == "EST_M" ]]; then 
        CMD="python main_sampling.py -d $data --is-greek -w $workload"
    elif [[ "$model" == "EST_B" ]]; then 
        CMD="python main_sampling.py -d $data --is-greek --is-adapt -w $workload"

    elif [[ "$model" == "Astrid" ]]; then 
        CMD="python main_Astrid.py -d $data -w $workload"
    elif [[ "$model" == "E2E" ]]; then 
        CMD="python main_E2E.py -d $data -w $workload"
    elif [[ "$model" == "DREAM" ]]; then 
        CMD="python main_DREAM.py -d $data -w $workload"
    elif [[ "$model" == "CLIQUE" ]]; then 
        CMD="python main_CLIQUE.py -d $data -w $workload"

    elif [[ "$model" == "Astrid-AUG" ]]; then 
        CMD="python main_Astrid.py -d $data --packed -w $workload"
    elif [[ "$model" == "E2E-AUG" ]]; then 
        CMD="python main_E2E.py -d $data --packed -w $workload"
    elif [[ "$model" == "DREAM-PACK" ]]; then 
        CMD="python main_DREAM.py -d $data --packed -w $workload"
    elif [[ "$model" == "LPLM" ]]; then 
        CMD="python main_LPLM.py -d $data -w $workload"
    elif [[ "$model" == "CLIQUE-PACK" ]]; then 
        CMD="python main_CLIQUE.py -d $data --packed -w $workload"

    elif [[ "$model" == "CLIQUE-T" ]]; then 
        CMD="python main_CLIQUE.py -d $data -w $workload --no-crh"
    elif [[ "$model" == "CLIQUE-PACK-T" ]]; then 
        CMD="python main_CLIQUE.py -d $data --packed -w $workload --no-crh"
    else
        echo "model=$model is not supported"
        return
    fi
    echo $CMD
    eval $CMD
}

if [ $# -ne 3 ]; then
    echo "Usage: ./model.sh <model> <data> <workload>"
else
    RUN_ALG_CMD
fi

