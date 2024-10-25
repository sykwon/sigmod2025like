# LEADER & CLIQUE

This repository implements training data generation algorithm (LEADER) and deep cardinality estimators (CLIQUE).

## Prerequisite

```bash
./setting.sh
make
# refer to sql/run_psql.sh to create postgres
```

## Generating train data (LEADER)
./alg.sh <alg> <data> <workload> <query> <is_aug>

* <alg>: the training data generation algorithm name (LEADER-S, LEADER-T, LEADER-SR, Naive, NaiveIndex)
* <data>: the dataset name (DBLP, GENE, AUTHOR, WIKI, IMDB)
* <workload>: the query workload (CLIQUE, LPLM, Astrid)
* <query>: the query type (train, valid, test)
* <is_aug>: 0 or 1 to indicate whether the training data is augmented or not

### Examples
```bash
./main LEADER-S CLIQUE/DBLP train 0
./main LEADER-S CLIQUE/DBLP valid 0
./main LEADER-S CLIQUE/DBLP test 0
./main LEADER-S CLIQUE/DBLP train 1 # prefix-aug training data
```


## Training cardinality estimator (CLIQUE)
./model.sh <model> <data> <workload>
* <model>: the cardinality estimation model (LBS, EST_S, EST_M, EST_B, Astrid, E2E, DREAM, CLIQUE, Astrid-AUG, E2E-AUG, DREAM-PACK, LPLM, CLIQUE-PACK, CLIQUE-T, CLIQUE-PACK-T) # "-T" means CLIQUE without the contional regression header
* <data>: the dataset name (DBLP, GENE, AUTHOR, WIKI, IMDB, DBLP-AN, IMDb-AN, IMDb-MT, TPCH-PN)
* <workload>: the query workload (CLIQUE, LPLM, Astrid)

### Examples
```bash
./model.sh CLIQUE DBLP-AN LPLM
./model.sh CLIQUE-PACK DBLP-AN LPLM
./model.sh LPLM DBLP-AN LPLM
./model.sh CLIQUE DBLP CLIQUE
./model.sh CLIQUE-T DBLP CLIQUE
./model.sh CLIQUE-PACK DBLP CLIQUE # using the packed learning method
./model.sh CLIQUE-PACK DBLP-AN LPLM #
```


## End-To-End Workload training data
python main_workload.py <data> <workload>
* <data>: the dataset name (DBLP, GENE, AUTHOR, WIKI, IMDB, DBLP-AN, IMDb-AN, IMDb-MT, TPCH-PN)
* <workload>: the query workload (CLIQUE, LPLM, Astrid)


### Examples
```bash
./gen_workload.sh DBLP CLIQUE
./gen_workload.sh DBLP LPLM
./gen_workload.sh DBLP Astrid
```

### Data directory
data/<data_name>
├── query
│   └── <workload>
│       ├── test.txt
│       ├── train.txt
│       └── valid.txt
├── training
│   └── <workload>
│       ├── pack_simple.txt
│       ├── test.txt
│       ├── train_LPLM.txt
│       ├── train.txt
│       └── valid.txt
├── <data_name>.txt
└── vocab.txt


### Result directory
res/<data_name>
└── <workload>
    ├── estimation
    │   └── <model>
    │       └── <trial>
    │           ├── estimation.csv
    │           └── time.csv
    ├── model
    │   └── <model>
    │       └── <trial>
    │           ├── config.yml
    │           └── model.pth
    ├── stat
    │   └── <model>
    │       └── <trial>
    │           └── stat.yml
    ├── log
    │   └── <model>
    │       └── <trial>
    │           └── events.out.tfevents
    └── Ntable
