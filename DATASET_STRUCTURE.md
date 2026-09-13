# ABIDE Dataset Structure

```text
ABIDE_DATA/
├── metadata/
│   ├── phenotype.csv
│   └── selected_284.csv
├── raw/
│   ├── fmri/
│   └── smri/
└── processed/
    ├── fmri_128/
    ├── corr/
    ├── smri_n4/
    ├── smri_final/
    ├── ABIDE_fMRI_284.npz
    └── ABIDE_multimodal_284.npz
```

Final multimodal arrays:

- sMRI: 284 x 96 x 96 x 96
- fMRI time series: 284 x 116 x 128
- correlation: 284 x 116 x 116
- labels: 284
- subject IDs: 284
