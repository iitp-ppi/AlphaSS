# AlphaSS
AlphaFold2 with using SSbond embedding.

AlphaSS predicts protein structures using deep learning given a sequence and a set of experimental contacts. It extends [OpenFold](https://github.com/aqlaboratory/openfold) with disulfide bond distance restraint by explicitly incorporating them in the OpenFold architecture. The experimental distance restraints may be represented in one forms:

1. As distance distributions (distograms) 

For (1), we trained our network with the distogram representation corresponding to distance between the CB(C-beta) atoms of disulfide bond.


## Installation

Please refer to the [OpenFold GitHub](https://github.com/aqlaboratory/openfold#installation-linux) for installation instructions of the required packages. AlphaSS requires the same packages, since it builds on top of OpenFold.  

## disulfide bond data

Disulfide bond data can be included either as a PyTorch dictionary with NumPy arrays: 'disulf_disto', the software may then be run with models based on upper bound distance thresholds or using generalized distograms. Distograms have shape LxLx128 where L is the length of the protein with the following binning: numpy.arange(2.3125,42,0.3125) + a catch-all bin in the end for distances >= 42A and no group embedding. Last bin is a catch-all bin. The probabilities should sum up to 1. 

residueFrom and residueTo are the residues which bonded with the disulfide to each other (sequence numbering starts at 0). Columns 2-130 contain the probability for each bin in numpy.arange(2.3125,42,0.3125)- i.e. the probability of each bin in a distogram going from 2.3125 to 42 Angstrom. The 128th bin is a catch-all bin for distances >= 42. 

Distance distributions for AlphaSS can be automatically generated from restraint lists with the script preprocessing_SSbond_features.py.
```
     python preprocessing_SSbond_features.py --infile restraints.csv
```

Where restraints.csv is a comma-separated file containing residueFrom,residueTo,sequenceLength. For example:

     92,135,1260
     108,126,1260
    
For a restraint between residue 92 and 135 imposed as a normal distribution with a mean distance and a standard deviation.

preprocessing_distributions.py will generate a restraint list with distance distributions binned in 128-bin distograms that can be given to AlphaSS.

## MSA subsampling

MSAs can be subsampled to a given Neff with --neff. 

## Usage

AlphaSS expects a FASTA file containing a single sequence, the disulfide bond pair informations features (pickle) with --disulf_info_path, and precomputed features (pickle) with --features. 

```
python3 predict_with_SS.py 8C3Y_A.fasta 
```

If there are not precoumpred alignments, 

 and databases for template/ MSA search, [see also OpenFold Inference](https://github.com/aqlaboratory/openfold#inference).

```
python predict_with_SS.py 7K3N_A.fasta photoL.csv --checkpoint_path resources/AlphaLink_params/finetuning_model_5_ptm_CACA_10A.pt --uniref90_database_path uniref90.fasta --mgnify_database_path mgy_clusters.fa --pdb70_database_path pdb70/pdb70 --uniclust30_database_path uniclust30_2018_08/uniclust30_2018_08 --jackhmmer_binary_path $CONDA_PREFIX/bin/jackhmmer --hhblits_binary_path $CONDA_PREFIX/bin/hhblits --hhsearch_binary_path $CONDA_PREFIX/bin/hhsearch --kalign_binary_path $CONDA_PREFIX/bin/kalign
```

MSA generation can be skipped if there are precomputed alignments:

```
python predict_with_SS.py 8C3Y_A.fasta photoL.csv --use_precomputed_alignments msa/ --checkpoint_path resources/AlphaLink_params/finetuning_model_5_ptm_CACA_10A.pt  --uniref90_database_path uniref90.fasta --mgnify_database_path mgy_clusters.fa --pdb70_database_path pdb70/pdb70 --uniclust30_database_path uniclust30_2018_08/uniclust30_2018_08 
```


## Network weights

Can be downloaded here: 

     It will be updated soon.
They need to be unpacked (gunzip).


## Reproducibility instructions

We eliminated all non-determinism (MSA masking), since with low Neff targets, different MSA masking can have a big effect.
