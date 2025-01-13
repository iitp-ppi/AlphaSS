#!/bin/bash
source scripts/vars.sh

conda env create --name=${ENV_NAME} -f environment.yml
source activate ${ENV_NAME}

echo "Attempting to install FlashAttention"
git clone https://github.com/HazyResearch/flash-attention
CUR_DIR=$PWD
cd flash-attention
git checkout 5b838a8bef
python3 setup.py install
cd $CUR_DIR

# Install DeepMind's OpenMM patch
OPENFOLD_DIR=$PWD
# for line 11, you have to insert your conda path!
pushd /home/bis/anaconda3/envs/${ENV_NAME}/lib/python3.9/site-packages \
    && patch -p0 < $OPENFOLD_DIR/lib/openmm.patch \
    && popd

# Download folding resources
wget -q -P openfold/resources \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# Certain tests need access to this file
mkdir -p tests/test_data/alphafold/common
ln -rs openfold/resources/stereo_chemical_props.txt tests/test_data/alphafold/common

echo "Downloading OpenFold parameters..."
bash scripts/download_openfold_params.sh openfold/resources

echo "Downloading AlphaFold parameters..."
bash scripts/download_alphafold_params.sh openfold/resources

# Decompress test data
gunzip tests/test_data/sample_feats.pickle.gz


cd $CUR_DIR
CC=g++ python setup.py install