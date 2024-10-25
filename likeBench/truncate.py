import os
from os.path import isfile, join, exists
from glob import glob

imdb_data_location = "./imdb_data"
os.chdir(imdb_data_location)
only_files = glob('./*.csv')
if not exists('edited'):
    os.makedirs('edited')
for file in only_files:
    print("processing file", file)
    with open(file, "r", encoding='utf-8') as ori:
        with open(f"edited/{file}", "w", encoding='utf-8') as dest:
            for line in ori:
                dest.write(line.replace('\\\\', '#$#$').replace('\\"', '""').replace('#$#$', '\\'))
