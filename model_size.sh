#!/bin/bash

function RUN_MODEL_CMD {
    if [[ "$model" == "LBS" ]]; then
        CMD="python -u main_LBS.py -d $data -w $workload"
    elif [[ "$model" == "EST_S" ]]; then 
        CMD="python -u main_sampling.py -d $data -w $workload"
    elif [[ "$model" == "EST_M" ]]; then 
        CMD="python -u main_sampling.py -d $data --is-greek -w $workload"
    elif [[ "$model" == "EST_B" ]]; then 
        CMD="python -u main_sampling.py -d $data --is-greek --is-adapt -w $workload"

    elif [[ "$model" == "Astrid" ]]; then 
        CMD="python main_Astrid.py -d $data -w $workload"
    elif [[ "$model" == "AstridEach" ]]; then 
        CMD="python -u main_AstridEach.py -d $data -w $workload"
    elif [[ "$model" == "E2E" ]]; then 
        CMD="python -u main_E2E.py -d $data -w $workload"
    elif [[ "$model" == "DREAM" ]]; then 
        CMD="python -u main_DREAM.py -d $data -w $workload"
    elif [[ "$model" == "CLIQUE" ]]; then 
        CMD="python -u main_CLIQUE.py -d $data -w $workload"

    elif [[ "$model" == "Astrid-AUG" ]]; then 
        CMD="python -u main_Astrid.py -d $data --packed -w $workload"
    elif [[ "$model" == "E2E-AUG" ]]; then 
        CMD="python -u main_E2E.py -d $data --packed -w $workload"
    elif [[ "$model" == "DREAM-PACK" ]]; then 
        CMD="python -u main_DREAM.py -d $data --packed -w $workload"
    elif [[ "$model" == "LPLM" ]]; then 
        CMD="python -u main_LPLM.py -d $data -w $workload"
    elif [[ "$model" == "CLIQUE-PACK" ]]; then 
        CMD="python -u main_CLIQUE.py -d $data --packed -w $workload"
    elif [[ "$model" == "CLIQUE-PACK-ALL" ]]; then
        CMD="python -u main_CLIQUE.py -d $data --packed -w $workload --pack-all"

    elif [[ "$model" == "CLIQUE-T" ]]; then 
        CMD="python -u main_CLIQUE.py -d $data -w $workload --no-crh"
    elif [[ "$model" == "CLIQUE-PACK-T" ]]; then 
        CMD="python -u main_CLIQUE.py -d $data --packed -w $workload --no-crh"
    else
        echo "model=$model is not supported"
        return
    fi
    echo "CUDA_VISIBLE_DEVICES=2 $CMD --ratio $r --tag R$r --seed $s"
    eval "CUDA_VISIBLE_DEVICES=2 $CMD --ratio $r --tag R$r --seed $s"
}
workload="CLIQUE"
for data in "GENE" "AUTHOR" "DBLP"; do
    for s in 0 1 2; do
	    for r in 0.2 0.4 0.6 0.8; do
		for model in "CLIQUE" "CLIQUE-PACK"; do
		    RUN_MODEL_CMD 
		done
	    done
    done
done

