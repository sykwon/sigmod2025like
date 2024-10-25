#!/bin/bash

PACKAGE='gdown'

python -c "import $PACKAGE" 2>/dev/null || pip install $PACKAGE

declare -A FILE_ID_dict

FILE_ID_dict["DBLP-AN"]="19U0EVbgBwHAz9zXw-lBB-7Pv1SWIa20l"
FILE_ID_dict["IMDb-AN"]="1Y-K7LtQlJLsi3W8FpxrEeIcAHS7Zm674"
FILE_ID_dict["IMDb-MT"]="12dVJOAA_cyncy_MeHA5IH41BoOUynkQS"
FILE_ID_dict["TPCH-PN"]="14Xz_JCV4DxSToQ3G_KnuFQHVH8bR5S5I"
FILE_ID_dict["WIKI"]="1j-JV04acH3vbCdiI8pnLqmOzx8g_0gtO"
FILE_ID_dict["IMDB"]="1r_A1RL-_qrJUReFqVN0ZjQ0JzkDwpn08"
FILE_ID_dict["DBLP"]="11S8D_8CLSXAsFycaoU1x0ub6qIapo7CE"
FILE_ID_dict["GENE"]="1CNzMwclEVyzhWJDSBt8g4r460E3wXFG7"
FILE_ID_dict["AUTHOR"]="1v5SZMep0puR3LEml7hwmMhBymBP8xlBw"

# cd data/
for data_name in DBLP-AN IMDb-AN IMDb-MT TPCH-PN WIKI IMDB DBLP GENE AUTHOR; do
    if [ -f "${data_name}.tar.gz" ]; then
      echo "${data_name}.tar.gz already exists"
    else
      CMD="gdown https://drive.google.com/uc?id=${FILE_ID_dict[$data_name]}" # ${data_name}.tar.gz
      echo $CMD
      eval $CMD
      CMD="tar -zxvf ${data_name}.tar.gz"
      echo $CMD
      eval $CMD
    fi
done
cd ..
