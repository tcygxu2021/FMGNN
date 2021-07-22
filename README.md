# FMGNN
This code is an implementation of our paper "FMGNN: A method to predict compound-protein interaction with pharmacophore features and physicochemical properti
es of amino acids" in Pytorch with RDKit. We provide three CPI datasets: human, C.elegans and Stitch, in which the ratio of positive and negative samples is 
1:1.

Dependencies:
python >= 3.6
pytorch 1.4.0+cu100
RDKit 2020.03.3

Usage:

1.  code/preprocess.py creates the substructure graphs of compounds with pharmacophore features and the n-grams of protein sequences from the original data. 
The original data are listed in dataset/human or celegans/original/data.txt. The output files are saved in the dataset/human or celegans/input directory.
2.  code/model.py trains the model using the above preprocessed data. The output model is saved in the output/model directory. The output result is saved in 
the output/result directory, including the time, loss_value, AUC, precise, recall and F1 values. 

Author
Chunyan Tang
Cheng Zhong
Mian Wang

