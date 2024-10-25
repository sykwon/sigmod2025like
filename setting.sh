#!/bin/bash

# install a virtual environment
conda create -n clique python=3.9 -y
conda activate clique
python -m pip install -r requirements.txt
conda install pytorch==1.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge -y

# install prerequisites
sudo apt-get install -y pkg-config

# install a cpp progress bar
sudo apt-get install -y cmake
git clone https://github.com/p-ranav/indicators
cd indicators
git checkout a5bc05f32a9c719535054b7fa5306ce5c8d055d8
mkdir build && cd build
cmake -DINDICATORS_SAMPLES=ON -DINDICATORS_DEMO=ON ..
make
make install # with sudo
cd ../..

# install the boost library
wget "https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz"
tar -xvzf boost_1_79_0.tar.gz
cd boost_1_79_0
./bootstrap.sh
./b2 install # with sudo
cd ..

# install the icu library
# sudo apt install clang
git clone https://github.com/unicode-org/icu.git --depth=1 --branch=release-71-1
cd icu/icu4c/source
./configure --prefix=/usr && make
make install # with sudo
cd ../../..

# install the re2 library
git clone https://github.com/google/re2.git
cd re2
git checkout 954656f47fe8fb505d4818da1e128417a79ea500
make
make test
make install # with sudo
make testinstall
cd ..
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"