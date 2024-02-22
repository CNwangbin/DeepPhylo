# DeepPhylo: A phylogeny-aware machine learning approach to enhance predictive accuracy in human microbiome data analysis

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

```bash
python twin_DeepPhy.py --epochs 200 -hs 64 -kec 7 -l 1e-4 -bs 64 -kep 4 -act relu --model-type deepphylo_binary
```

- to train and test a model using DeepPhylo in age regression run sh: 
```bash
python usa_DeepPhy.py --epochs 200 -hs 64 -kec 7 -bs 64 -kep 4 --lr 1e-4 -act relu
```

- to train and test a model using unsupervised method in IBD data run sh: 
```bash
python twin_DeepPhy.py --epochs 200 --hidden_size 64 --lr 1e-4 -bs 64 -hs 64 -kec 7 --model-type unsupervised
```
* mystem_rpca.ipynb - Jupyter notebook to run unsupervised method on microbiome datasets
## Citation

If you use DeepPhylo for your research, or incorporate our learning algorithms in your work, please cite:



## New version specifications
Current dependencies can be found in the requirements.txt file.
The used Python version is 3.9.12.
