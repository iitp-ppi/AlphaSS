#!/bin/bash
CONDA_INSTALL_URL=${CONDA_INSTALL_URL:-"https://repo.anaconda.com/miniconda/Miniconda3-py39_25.1.1-0-Linux-x86_64.sh"}

source scripts/vars.sh

# Install Miniconda locally
rm -rf lib/conda
rm -f /tmp/Miniconda3-py39_25.1.1-0-Linux-x86_64.sh
wget -P /tmp \
    "${CONDA_INSTALL_URL}" \
    && bash /tmp/Miniconda3-py39_25.1.1-0-Linux-x86_64.sh -b -p lib/conda \
    && rm /tmp/Miniconda3-py39_25.1.1-0-Linux-x86_64.sh

# Grab conda-only packages
export PATH=lib/conda/bin:$PATH
conda env create --name=${ENV_NAME} -f environment.yml
source activate ${ENV_NAME}

# Install DeepMind's OpenMM patch
OPENFOLD_DIR=$PWD
pushd lib/conda/envs/$ENV_NAME/lib/python3.9/site-packages/ \
    && patch -p0 < $OPENFOLD_DIR/lib/openmm.patch \
    && popd


# Download folding resources
wget -q -P openfold/resources \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# # Certain tests need access to this file
# mkdir -p tests/test_data/alphafold/common
# ln -rs openfold/resources/stereo_chemical_props.txt tests/test_data/alphafold/common

# echo "Downloading OpenFold parameters..."
# bash scripts/download_openfold_params.sh openfold/resources

# echo "Downloading AlphaFold parameters..."
# bash scripts/download_alphafold_params.sh openfold/resources

# # Decompress test data
# gunzip tests/test_data/sample_feats.pickle.gz

