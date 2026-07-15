# To install
conda env create -f rna_regression_baselines.yml
conda activate rna_regression_baselines

# To add to jupyter notebook kernel
python -m ipykernel install --user \
    --name rna_regression_baselines \
    --display-name "rna_regression_baselines"

