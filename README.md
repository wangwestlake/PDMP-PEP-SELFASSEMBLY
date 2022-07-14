# PDMP-PEP-SELFASSEMBLY
TRN trained models of AP values of penta-, deca-, and mixed pentapeptides

## User Manual

The trained TRN model 'model_TRN.pt' can be accessed from the following link:

https://drive.google.com/u/0/uc?id=18uxYSpFMQlAxiNkq_DKLut5cPJO7Ilu6&export=download

The model file need to be downloaded and placed in the root directory.

To predict AP value for peptides, try:
`python AP_predict.py`

Then the predicted AP value for peptided included in 'deca_peptides.csv' will be output in file 'AP_predict'.
