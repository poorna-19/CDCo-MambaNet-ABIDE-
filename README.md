# CDCo-MambaNet ABIDE Adaptation

This repository contains the ABIDE preprocessing work prepared for the CDCo-MambaNet project.

## Completed preprocessing

- ABIDE phenotype downloaded and filtered
- 284 subjects selected: 140 ASD and 144 Control
- ABIDE ROI fMRI signals downloaded from the Preprocessed Connectomes Project source
- fMRI standardized to 116 AAL ROIs and 128 time points
- 116 x 116 functional connectivity matrices generated
- ABIDE sMRI T1 and brain volumes downloaded
- N4 bias-field correction applied
- MNI-based registration and final resampling performed
- sMRI standardized to 96 x 96 x 96
- Multimodal dataset created and validated

## Training-ready dataset

The final file produced by the notebook is:

`ABIDE_DATA/processed/ABIDE_multimodal_284.npz`

It contains:

- `smri`: (284, 96, 96, 96)
- `timeseires`: (284, 116, 128)
- `corr`: (284, 116, 116)
- `label`: (284,)
- `subject_id`: (284,)

The selected subject metadata is:

`ABIDE_DATA/metadata/selected_284.csv`

## Notebook

`notebooks/ABIDE_preprocessing.ipynb` contains the executed preprocessing workflow and validation cells.

## Preprocessing script

`preprocessing/abide_preprocessing.py` contains the same preprocessing workflow in script form for reuse before model training.

## Model source

The repository retains the original CDCo-MambaNet source structure. The ABIDE preprocessing is prepared for integration with the model training pipeline.

## Data location in Kaggle

`/kaggle/working/ABIDE_DATA/processed/ABIDE_multimodal_284.npz`

`/kaggle/working/ABIDE_DATA/metadata/selected_284.csv`

Raw neuroimaging data is not included in this repository. It should be obtained from the official ABIDE/Preprocessed Connectomes Project sources according to their terms.
