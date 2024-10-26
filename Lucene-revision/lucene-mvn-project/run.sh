#!/bin/bash
dataName=("DBLP" "GENE" "AUTHOR" "WIKI" "IMDB")
for i in "${dataName[@]}"
do
    echo "Running for dataset: $i"
    java -cp target/lucene-demo-1.0-SNAPSHOT.jar com.example.lucene.LuceneExample "$i" || { echo "Error occurred"; exit 1; }
done