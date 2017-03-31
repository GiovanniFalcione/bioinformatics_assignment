from Bio import SeqIO
from Bio.Alphabet  import generic_protein
from sklearn.preprocessing import minmax_scale, scale
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

"""
Feature engineering to extrapolate useful information out of
proteins' primary structure. The features compute by this script can be used 
to determine the sub-cellular location of proteins (see bioinfo_project.py)

author: Giovanni Falcione
bioinformatics assignment, Msc Machine Learning 16/17, UCL
"""

# read data and put it into an ordered list
sequences = []
labels = []
blind_id = []

for seq_record in SeqIO.parse("cyto.fasta.txt", "fasta", alphabet=generic_protein):
    labels.append('cyto')
    sequences.append(seq_record.seq)

for seq_record in SeqIO.parse("mito.fasta.txt", "fasta", alphabet=generic_protein):
    labels.append('mito')
    sequences.append(seq_record.seq)

for seq_record in SeqIO.parse("nucleus.fasta.txt", "fasta", alphabet=generic_protein):
    labels.append('nucleus')
    sequences.append(seq_record.seq)

for seq_record in SeqIO.parse("secreted.fasta.txt", "fasta", alphabet=generic_protein):
    labels.append('secreted')
    sequences.append(seq_record.seq)

for seq_record in SeqIO.parse("blind.fasta.txt", "fasta", alphabet=generic_protein):

    blind_id.append(seq_record.id)
    sequences.append(seq_record.seq)

# create dictionary of amino acids
dict  = np.unique(sequences[0])

# create the dictionary of di-peptides
dict_di = []
for amino1 in dict:
    for amino2 in dict:
        dict_di.append(amino1+amino2)


# in the sequences there are some X, B and others that I need to clean up
for seq in sequences:
    if 'X' in seq:
        sequences.remove(seq)
    if 'B' in seq:
        sequences.remove(seq)

# for some reasons some X escape the first loop
for seq in sequences:
    if 'X' in seq:
        sequences.remove(seq)

n_train = len(sequences)

# compute amino frequency, di-peptide frequency and length of sequences
freq = np.zeros((n_train, 20))
di_freq = np.zeros((n_train,400))
len_seq = np.zeros((n_train,1))
mol_weight = np.zeros((n_train,1))
iso_point = np.zeros((n_train,1))
arom = np.zeros((n_train,1))
second_struct = np.zeros((n_train,3))
flex = np.zeros((n_train,1))

idxs = 0
for seq in sequences:
    idxa = 0
    idxd = 0
    len_seq[idxs] = len(seq)

    for amino in dict:
        freq[idxs,idxa] = seq.count(amino)/len_seq[idxs]
        idxa +=1

    for di_amino in dict_di:
        di_freq[idxs, idxd] = seq.count(di_amino)/(len_seq[idxs] - 1.)
        idxd +=1

    idxs +=1

# get molecular weight, isoeletric point, aromaticity 
# and secondary structure percentage for each protein
idxs = 0
for seq in sequences:
    seq_pt = ProteinAnalysis(str(seq))
    mol_weight[idxs] = seq_pt.molecular_weight()
    iso_point[idxs] = seq_pt.isoelectric_point()
    arom[idxs] = seq_pt.aromaticity()
    second_struct[idxs,:] = seq_pt.secondary_structure_fraction()
    idxs +=1

# scale data
len_seq = minmax_scale(len_seq)
mol_weight = scale(mol_weight)
iso_point = scale(iso_point)


# compute amino frequency in first 50 positions
first20 = [item[:20] for item in sequences]
freq_first20 = np.zeros((n_train, 20))

idxs = 0
for seq in first20:
    idxa = 0
    for amino in dict:
        freq_first20[idxs,idxa] = seq.count(amino)/20.
        idxa+=1
    idxs +=1


# compute amino frequency in last 50 positions
last20 = [item[-20:] for item in sequences]
freq_last20 = np.zeros((n_train, 20))

idxs = 0
for seq in last20:
    idxa = 0
    for amino in dict:
        freq_last20[idxs,idxa] = seq.count(amino)/20.
        idxa+=1
    idxs +=1

# merge all features into a big feature matrix and save it
features = scale(np.hstack((freq, di_freq, freq_last20, freq_first20, len_seq, mol_weight, iso_point, arom, second_struct)))

blind_feats = features[-20:, :]

np.save('feature_matrix.npy', features)
np.save('labels.npy', labels)

np.save('blind_prots.npy', blind_feats)
np.save('blind_ids.npy', blind_id)
