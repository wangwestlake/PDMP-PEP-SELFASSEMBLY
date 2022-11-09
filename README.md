# PDMP-PEP-SELFASSEMBLY
TRN trained models of AP values of penta-, deca-, and mixed pentapeptides

## How to Use

To download the trained TRN model for penta- and deca-peptides _'model_TRN.pt'_, click:

https://drive.google.com/u/0/uc?id=18uxYSpFMQlAxiNkq_DKLut5cPJO7Ilu6&export=download

To download the trained TRN model for mixed pentapeptides _'model_mixed_penta.pt'_, click:

https://drive.google.com/u/0/uc?id=1Glvp-CfLj0_nNFO-RadjJEtLOzLMeezJ&export=download

The model files have to be downloaded and placed in the root directory.

To predict AP value for peptides, try:
`python AP_predict.py`

Then the predicted AP values for peptides included in 'deca_peptides.csv' will be output in file 'AP_predict.csv'.
