# AlphaSS
**AlphaFold2 with using disulfide (SS) bond embedding and SS bond loss.**

AlphaSS predicts protein structures using deep learning given a sequence and a set of experimental contacts. It extends [OpenFold](https://github.com/aqlaboratory/openfold) with disulfide bond distance restraint by explicitly incorporating them in the OpenFold architecture. The experimental distance restraints may be represented in two forms:

1. As distance distributions (distograms) 
2. The SG atom distance between the cysteine residues. (float)

For (1), we trained our network with the distogram representation corresponding to distance between the CB(C-beta) atoms of disulfide bond.

For (2), we trained our network with the SS loss, the difference of the distance ground-truth and prediction structure between SG(Sulfur) atoms.


## Installation

Please refer to the [OpenFold GitHub](https://github.com/aqlaboratory/openfold#installation-linux) for installation instructions of the required packages. AlphaSS requires the same packages, since it builds on top of OpenFold.  

### easy way to setup
1. git clone this repository
2. make the conda enviroments from

     scripts/install_third_party_dependencies.sh


## Disulfide bond data

Disulfide bond data can be included either as a PyTorch dictionary with NumPy arrays: **'disulf_disto'**, the software may then be run with models based on upper bound distance thresholds or using generalized distograms and the **'disulf_dist'**, the distance between the SG atoms of two cysteine residues. 

Distograms have shape LxLx128 where L is the length of the protein with the following binning: numpy.arange(2.3125,42,0.3125) + a catch-all bin in the end for distances >= 42A and no group embedding. Last bin is a catch-all bin. The probabilities should sum up to 1. 

residueFrom and residueTo are the residues which bonded with the disulfide to each other (sequence numbering starts at 0). Columns 2-130 contain the probability for each bin in numpy.arange(2.3125,42,0.3125)- i.e. the probability of each bin in a distogram going from 2.3125 to 42 Angstrom. The 128th bin is a catch-all bin for distances >= 42. 

Distance distributions for AlphaSS can be automatically generated from restraint lists with the script preprocessing_SSbond_features.py.
```
     python preprocessing_SSbond_features.py --infile restraints.csv --outfile SS_feature_path (.pkl)
```

Where restraints.csv is a comma-separated file containing residueFrom,residueTo,sequenceLength. For example:

     92,135,1260
     108,126,1260
    
For a restraint between residue 92 and 135 imposed as a normal distribution with a mean distance and a standard deviation.

preprocessing_distributions.py will generate a restraint list with distance distributions binned in 128-bin distograms and the SG atom distance informations that can be given to AlphaSS.

## MSA subsampling

MSAs can be subsampled to a given Neff with --neff. 

## Usage

AlphaSS expects a FASTA file containing a single sequence, the disulfide bond pair informations features (pickle) with --disulf_info_path, and precomputed features (pickle) with --features. 

```
     python3 predict_with_SS.py "fasta_path" --model_name AlphaSS_ft_model_SSloss1 --disulf_info_path "disulf_info_pkl_path" --features "AF_feature_path" --output_dir "output_dir_path" --output_name "output_name" --model_device "model_device" --neff "0"
```

### "--neff" condition
- 0 = full MSA
- 1 = no MSA


## Network weights

Can be downloaded here: 

[AlphaSS_ft_weight link](https://drive.google.com/file/d/1_dP7H6KenIG7rgcAvNQEH6W8cuCjwWGF/view?usp=sharing)
They need to make the folder to use the weight.

Please download the weight from the google drive link, and put it on following path.

     ./openfold/resource/AlphaSS_params/ 


## Reproducibility instructions

We eliminated all non-determinism (MSA masking), since with low Neff targets, different MSA masking can have a big effect.
