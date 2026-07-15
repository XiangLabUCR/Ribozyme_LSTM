# Baseline LSTM model

A two layer LSTM model and implementation to perform regression modeling of synthetic ribozyme libraries, mapping sequence+predicted RNA secondary structures to a z-score (standardized) corresponding to log10(RNA/DNA) ratio or log10(GFP/mCherry). 

## To install
```
conda env create -f rna_regression_baselines.yml \
conda activate rna_regression_baselines
```

## To add to jupyter notebook kernel
```
python -m ipykernel install --user \
    --name rna_regression_baselines \
    --display-name "rna_regression_baselines"
```

## Datasets
One sequence text file containing a list of strings with letters, and a label text file containing z-score values. 

## To run
```
python -m src.train
```
## To evaluate
```
python -m src.evaluate
```
