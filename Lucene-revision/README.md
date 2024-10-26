### Prerequisite
```
sudo apt install openjdk-11-jdk
sudo apt install maven
```

### Command
* \<data_name>: the dataset name (DBLP, GENE, AUTHOR, WIKI, IMDB, DBLP-AN, IMDb-AN, IMDb-MT, TPCH-PN)

```
cd lucene-mvn-project
mvn clean package
java -cp target/lucene-demo-1.0-SNAPSHOT.jar com.example.lucene.LuceneExample <data_name>
```