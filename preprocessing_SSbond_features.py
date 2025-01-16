import numpy as np
import sys
import pickle
import matplotlib
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(prog='AlphaSS preprocessing',
                    description='takes a restraint list and returns 128-bin distogram per restraint',
                    epilog='usage: python preprocessing_distributions.py --infile restraints.csv')

parser.add_argument("--infile", metavar="restraints.csv",
                    required=True,
                    type=str,
                    help= str("the input is a comma-separated file formatted " +
          "as follows:\n"+
          "residueFrom,residueTo,sequenceLength\n"+
          "residue numbering starts at 0.\n"+
          "Distribution types are setted as 'normal'\n"+
          "For custom distributions see the numpy random distributions list "+
          "to generate 128-bin distributions.\n"+
          "For upper-bound restraints, use normal AlphaLink restraint input.\n\n"+
          "example line in input file:\n"+
          "92,135,1260\n"+
          "to impose a restraint between residue 92 and residue 135 with a gaussian "+
          "probability distribution centered around 3.85 Angstrom and a standard "+
          "deviation of 0.26 Angstrom\n"))

parser.add_argument("--outfile", metavar="disulfide_info.pkl",
                    required=False,
                    type=str,
                    default="disulfide_info.pkl",
                    help="output file name")

args = parser.parse_args()

matplotlib.use('Agg')

np.random.seed(4242022)

CB_dist, CB_std = 3.85, 0.26
disulf_info = {}

restraints = np.genfromtxt(args.infile,
                           names=["From", "To", "Length"],
                           delimiter=",",
                           dtype=None,
                           encoding=None)

if len(restraints.shape) == 1:
    restraints = np.array([restraints])

seq_length = restraints["Length"][0][0]
distogram = np.zeros((seq_length, seq_length, 128))
pair_info = []

for i, line in enumerate(restraints):
    #convert to 0-based residue index
    res_from_0 = line["From"] #- 1
    res_to_0 = line["To"] #- 1
    pair_info.append([res_from_0, res_to_0])
    
    sample = np.random.normal(CB_dist, CB_std, size=10000)

    n, bins, p = plt.hist(sample, bins=np.arange(2.3125, 42.625, 0.3125),
                          density=True)
    n /= np.sum(n)
    n = n.tolist()
    distogram[res_from_0, res_to_0] = distogram[res_to_0, res_from_0] = np.array(list(n))

pair_info = np.array(pair_info)

disulf_info['disulf_disto'], disulf_info['pair_info'] = distogram, pair_info

with open(args.outfile,'wb') as f:
    pickle.dump(disulf_info,f)
