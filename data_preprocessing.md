import os
import glob
import requests
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from scipy.signal import resample
from nilearn import datasets
from tqdm.auto import tqdm

BASE = os.environ.get("ABIDE_BASE", "/kaggle/working/ABIDE_DATA")

folders = [
    f"{BASE}/metadata",
    f"{BASE}/raw/fmri",
    f"{BASE}/raw/smri",
    f"{BASE}/processed",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

phenotype_url = (
    "https://raw.githubusercontent.com/"
    "preprocessed-connectomes-project/abide/master/"
    "Phenotypic_V1_0b_preprocessed1.csv"
)

phenotype_file = f"{BASE}/metadata/phenotype.csv"

if not os.path.exists(phenotype_file):
    r = requests.get(phenotype_url, timeout=60)
    r.raise_for_status()
    with open(phenotype_file, "wb") as f:
        f.write(r.content)

pheno = pd.read_csv(phenotype_file)

asd = pheno[pheno["DX_GROUP"] == 1].copy()
control = pheno[pheno["DX_GROUP"] == 2].copy()

asd = asd.sort_values("FILE_ID").head(140)
control = control.sort_values("FILE_ID").head(144)

selected = pd.concat([asd, control], ignore_index=True)
selected["label"] = selected["DX_GROUP"].map({1: 1, 2: 0})

selected_file = f"{BASE}/metadata/selected_284.csv"
selected.to_csv(selected_file, index=False)

FMRI_DIR = f"{BASE}/raw/fmri"
os.makedirs(FMRI_DIR, exist_ok=True)

FMRI_BASE_URL = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/"
    "ABIDE_Initiative/Outputs/dparsf/filt_noglobal/rois_aal/"
)

failed = []

for file_id in tqdm(selected["FILE_ID"], total=len(selected), desc="Downloading ABIDE fMRI"):
    filename = f"{file_id}_rois_aal.1D"
    output_file = os.path.join(FMRI_DIR, filename)

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        continue

    try:
        response = requests.get(FMRI_BASE_URL + filename, timeout=120)
        response.raise_for_status()
        with open(output_file, "wb") as f:
            f.write(response.content)
    except Exception as e:
        failed.append((file_id, str(e)))

if failed:
    print("fMRI download failures:", len(failed))
    print(failed[:10])

fmri_out_dir = f"{BASE}/processed/fmri_128"
os.makedirs(fmri_out_dir, exist_ok=True)

failed = []

for fname in sorted(f for f in os.listdir(FMRI_DIR) if f.endswith(".1D")):
    try:
        data = np.loadtxt(os.path.join(FMRI_DIR, fname))

        if data.ndim != 2 or data.shape[1] != 116:
            raise ValueError(f"Unexpected shape: {data.shape}")

        data_128 = resample(data, 128, axis=0)
        data_128 = data_128.T.astype(np.float32)

        out_name = fname.replace(".1D", ".npy")
        np.save(os.path.join(fmri_out_dir, out_name), data_128)
    except Exception as e:
        failed.append((fname, str(e)))

if failed:
    print("fMRI preprocessing failures:", len(failed))
    print(failed[:10])

corr_dir = f"{BASE}/processed/corr"
os.makedirs(corr_dir, exist_ok=True)

failed = []

for fname in sorted(f for f in os.listdir(fmri_out_dir) if f.endswith(".npy")):
    try:
        ts = np.load(os.path.join(fmri_out_dir, fname))

        if ts.shape != (116, 128):
            raise ValueError(f"Unexpected shape: {ts.shape}")

        corr = np.corrcoef(ts)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        out_name = fname.replace(".npy", "_corr.npy")
        np.save(os.path.join(corr_dir, out_name), corr)
    except Exception as e:
        failed.append((fname, str(e)))

if failed:
    print("Correlation failures:", len(failed))
    print(failed[:10])

fmri_file = f"{BASE}/processed/ABIDE_fMRI_284.npz"
metadata = pd.read_csv(selected_file)

timeseries = []
correlations = []
labels = []
subject_ids = []

for _, row in metadata.iterrows():
    file_id = str(row["FILE_ID"])
    fmri_path = os.path.join(fmri_out_dir, file_id + "_rois_aal.npy")
    corr_path = os.path.join(corr_dir, file_id + "_rois_aal_corr.npy")

    if not os.path.exists(fmri_path):
        raise FileNotFoundError(f"Missing fMRI: {fmri_path}")
    if not os.path.exists(corr_path):
        raise FileNotFoundError(f"Missing correlation: {corr_path}")

    ts = np.load(fmri_path)
    corr = np.load(corr_path)

    if ts.shape != (116, 128):
        raise ValueError(f"{file_id}: fMRI shape {ts.shape}")
    if corr.shape != (116, 116):
        raise ValueError(f"{file_id}: correlation shape {corr.shape}")

    timeseries.append(ts)
    correlations.append(corr)
    labels.append(int(row["label"]))
    subject_ids.append(file_id)

timeseries = np.stack(timeseries).astype(np.float32)
correlations = np.stack(correlations).astype(np.float32)
labels = np.array(labels, dtype=np.int64)
subject_ids = np.array(subject_ids)

np.savez_compressed(
    fmri_file,
    timeseires=timeseries,
    corr=correlations,
    label=labels,
    subject_id=subject_ids,
)

T1_DIR = f"{BASE}/raw/smri/T1"
BRAIN_DIR = f"{BASE}/raw/smri/brain"
os.makedirs(T1_DIR, exist_ok=True)
os.makedirs(BRAIN_DIR, exist_ok=True)

SMRI_BASE_URL = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/"
    "ABIDE_Initiative/Outputs/freesurfer/5.1"
)

failed = []

for file_id in tqdm(selected["FILE_ID"], total=len(selected), desc="Downloading ABIDE sMRI"):
    t1_file = os.path.join(T1_DIR, f"{file_id}_T1.mgz")
    brain_file = os.path.join(BRAIN_DIR, f"{file_id}_brain.mgz")

    try:
        if not os.path.exists(t1_file):
            r = requests.get(f"{SMRI_BASE_URL}/{file_id}/mri/T1.mgz", timeout=180)
            r.raise_for_status()
            with open(t1_file, "wb") as f:
                f.write(r.content)

        if not os.path.exists(brain_file):
            r = requests.get(f"{SMRI_BASE_URL}/{file_id}/mri/brain.mgz", timeout=180)
            r.raise_for_status()
            with open(brain_file, "wb") as f:
                f.write(r.content)
    except Exception as e:
        failed.append((file_id, str(e)))

if failed:
    print("sMRI download failures:", len(failed))
    print(failed[:10])

smri_n4_dir = f"{BASE}/processed/smri_n4"
os.makedirs(smri_n4_dir, exist_ok=True)

for t1_path in sorted(glob.glob(f"{T1_DIR}/*_T1.mgz")):
    file_id = os.path.basename(t1_path).replace("_T1.mgz", "")
    output_path = f"{smri_n4_dir}/{file_id}_smri.nii.gz"

    if os.path.exists(output_path):
        continue

    t1 = nib.load(t1_path)
    brain = nib.load(f"{BRAIN_DIR}/{file_id}_brain.mgz")

    image = sitk.GetImageFromArray(t1.get_fdata().astype(np.float32))
    mask = sitk.GetImageFromArray((brain.get_fdata() > 0).astype(np.uint8))

    spacing = tuple(float(x) for x in t1.header.get_zooms()[:3])
    image.SetSpacing(spacing)
    mask.SetSpacing(spacing)

    size = image.GetSize()
    new_size = [128, 128, 128]
    new_spacing = [size[i] * spacing[i] / new_size[i] for i in range(3)]

    image = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0.0,
        sitk.sitkFloat32,
    )

    mask = sitk.Resample(
        mask,
        new_size,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        mask.GetOrigin(),
        new_spacing,
        mask.GetDirection(),
        0,
        sitk.sitkUInt8,
    )

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([10, 5, 3])
    corrected = corrector.Execute(image, mask)
    sitk.WriteImage(corrected, output_path)

template = datasets.load_mni152_template(resolution=2)
template_path = f"{BASE}/mni152_template.nii.gz"
template.to_filename(template_path)

smri_final_dir = f"{BASE}/processed/smri_final"
os.makedirs(smri_final_dir, exist_ok=True)

fixed = sitk.ReadImage(template_path, sitk.sitkFloat32)

for path in sorted(glob.glob(f"{smri_n4_dir}/*.nii.gz")):
    file_id = os.path.basename(path).replace("_smri.nii.gz", "")
    output_path = f"{smri_final_dir}/{file_id}.npy"

    if os.path.exists(output_path):
        continue

    moving = sitk.ReadImage(path, sitk.sitkFloat32)

    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.01)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=50,
        convergenceMinimumValue=1e-5,
        convergenceWindowSize=5,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial, inPlace=False)

    transform = registration.Execute(fixed, moving)

    registered = sitk.Resample(
        moving,
        fixed,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    registered = sitk.Resample(
        registered,
        [96, 96, 96],
        sitk.Transform(),
        sitk.sitkLinear,
        registered.GetOrigin(),
        [
            registered.GetSize()[0] * registered.GetSpacing()[0] / 96,
            registered.GetSize()[1] * registered.GetSpacing()[1] / 96,
            registered.GetSize()[2] * registered.GetSpacing()[2] / 96,
        ],
        registered.GetDirection(),
        0.0,
        sitk.sitkFloat32,
    )

    data = sitk.GetArrayFromImage(registered)
    data = np.nan_to_num(data).astype(np.float32)
    np.save(output_path, data)

multimodal_file = f"{BASE}/processed/ABIDE_multimodal_284.npz"

fmri = np.load(fmri_file, allow_pickle=True)
smri = []
timeseires = []
corr = []
labels = []
subjects = []

for _, row in selected.iterrows():
    file_id = str(row["FILE_ID"])
    smri_path = f"{smri_final_dir}/{file_id}.npy"

    if not os.path.exists(smri_path):
        continue

    index = np.where(fmri["subject_id"] == file_id)[0]
    if len(index) == 0:
        continue

    index = index[0]
    smri.append(np.load(smri_path).astype(np.float32))
    timeseires.append(fmri["timeseires"][index].astype(np.float32))
    corr.append(fmri["corr"][index].astype(np.float32))
    labels.append(int(fmri["label"][index]))
    subjects.append(file_id)

smri = np.asarray(smri, dtype=np.float32)
timeseires = np.asarray(timeseires, dtype=np.float32)
corr = np.asarray(corr, dtype=np.float32)
labels = np.asarray(labels, dtype=np.int64)
subjects = np.asarray(subjects)

np.savez_compressed(
    multimodal_file,
    smri=smri,
    timeseires=timeseires,
    corr=corr,
    label=labels,
    subject_id=subjects,
)

if smri.shape != (284, 96, 96, 96):
    raise ValueError(f"sMRI shape is {smri.shape}")
if timeseires.shape != (284, 116, 128):
    raise ValueError(f"fMRI shape is {timeseires.shape}")
if corr.shape != (284, 116, 116):
    raise ValueError(f"Correlation shape is {corr.shape}")
if labels.shape != (284,):
    raise ValueError(f"Labels shape is {labels.shape}")
if subjects.shape != (284,):
    raise ValueError(f"Subjects shape is {subjects.shape}")
if np.sum(labels == 1) != 140:
    raise ValueError("ASD count is not 140")
if np.sum(labels == 0) != 144:
    raise ValueError("Control count is not 144")
if not np.isfinite(smri).all():
    raise ValueError("sMRI contains NaN/Inf")
if not np.isfinite(timeseires).all():
    raise ValueError("fMRI contains NaN/Inf")
if not np.isfinite(corr).all():
    raise ValueError("Correlation contains NaN/Inf")

print("ABIDE preprocessing complete")
print("sMRI:", smri.shape)
print("fMRI:", timeseires.shape)
print("Correlation:", corr.shape)
print("Labels:", labels.shape)
print("Subjects:", subjects.shape)
print("ASD:", np.sum(labels == 1))
print("Control:", np.sum(labels == 0))
print("Saved:", multimodal_file)
