# DeepPhy: A phylogeny-aware machine learning approach to enhance predictive accuracy in human microbiome data analysis

 DeepPhylo, a method that employs phylogeny-aware amplicon embeddings to integrate abundance and phylogenetic information, thereby improving both the unsupervised discriminatory power and supervised predictive accuracy of microbiome data analysis. 

Compared to the existing methods, DeepPhylo demonstrated superiority in informing biologically relevant insights across four real-world microbiome use cases, including clustering of skin microbiomes, prediction of host chronological age and gender, and inflammatory bowel disease (IBD) diagnosis across 15 datasets.

This repository contains script which were used to train the DeepPhylo model with the scripts for evaluating the model's performance.

## Dependencies
* The code was developed and tested using python 3.9.
* To install python dependencies run:
  `pip install -r requirements.txt`

## Installation
You can either clone the public repository:

```bash
# clone project
git clone https://github.com/CNwangbin/DeepPhylo
# First, install dependencies
pip install -r requirements.txt
```

Once you have a copy of the source, you can install it with:

```bash
python setup.py install
```

## Running
* Download all the data files and place them into data folder


## Scripts
The scripts here are using to run the model.

* twin_DeepPhylo.py - This script is used to train model using the method of DeepPhylo in twin classification.
* twin_MDeep.py - This script is used to train model using the method of MDeep in twin classification.
* twin_MLP.py - This script is used to train model using the method of MLP in twin classification.
* usa_DeepPhylo.py - This script is used to train model using the method of DeepPhylo in age regression.
* usa_MDeep.py - This script is used to train model using the method of MDeep in age regression.
* usa_MLP.py - This script is used to train model using the method of MLP in age regression.
- to train and test a model using DeepPhylo in twin classification run sh: 
```bash
python twin_DeepPhylo.py --epochs 200 --hidden_size 64 --lr 1e-4 --bs 64 --hs 64 --k 7
```
- to train and test a model using MDeep in twin classification run sh: 
```bash
python twin_MDeep.py --epochs 200 --lr 1e-4 --bs 64
```
- to train and test a model using MLP in twin classification run sh:
```bash
python twin_MLP.py --epochs 500 --lr 1e-4 --bs 64
```
- to train and test a model using DeepPhylo in age regression run sh: 
```bash
python usa_DeepPhylo.py --epochs 200 --hidden_size 64 --lr 1e-4 --bs 64 --hs 64 --k 7
```
- to train and test a model using MDeep in age regression run sh:  
```bash
python usa_MDeep.py --epochs 500 --lr 5e-3 --bs 32
```
- to train and test a model using MLP in age regression run sh:
```bash
python usa_MLP.py  --epochs 500 --lr 5e-3 --bs 32
```
## Citation

If you use DeepPhylo for your research, or incorporate our learning algorithms in your work, please cite:



## New version specifications
Current dependencies can be found in the requirements.txt file.
The used Python version is 3.9.12.
