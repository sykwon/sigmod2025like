#!/bin/bash

unzip postgresql-13.1.zip
wget http://homepages.cwi.nl/~boncz/job/imdb.tgz

# build docker file
tar cvf postgres-13.1.tar.gz postgresql-13.1 && mv postgres-13.1.tar.gz dockerfile/
cd dockerfile
docker build -t ceb .
cd ..

mkdir imdb_data
tar -xvzf imdb.tgz -C imdb_data
python truncate.py

# run docker file
docker run --name ceb-like -p 5432:5432 -d ceb

# create database and 
./load_imdb_local.sh

# stop container
docker stop ceb-like

# start container
docker start ceb-like

# enter container
docker exec -it ceb-like /bin/bash

# check worksheet.ipynb to collect string columns and predicates