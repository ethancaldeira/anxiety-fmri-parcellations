# Custom Functional Parcellations for Trait Anxiety (GSP)

Individual and group-level functional brain parcellations of the GSP
resting-state fMRI, built with spatially constrained normalised-cut spectral
clustering (Craddock et al., 2012). Each subject gets its own 200-label
parcellation, and group parcellations are built for low, mid and high trait
anxiety (STAI-T tertiles) from training subjects only.

This repository builds the custom parcellations for **Chapter 6** of a
thesis on anxiety prediction from MRI:

- Ch4: structural MRI ([anxiety-smri-classification](https://github.com/ethancaldeira/anxiety-smri-classification))
- Ch5: rs-fMRI with standard atlases ([anxiety-fmri-classification](https://github.com/ethancaldeira/anxiety-fmri-classification))
- **Ch6 (this repo)**: individual and group parcellations; the models are
  trained in [anxiety-fmri-classification](https://github.com/ethancaldeira/anxiety-fmri-classification),
  which also generates the Chapter 6 stochastic parcellations
  (`src/generate_sp_masks.py`, 30 per scale at 50, 100, 150 and 200 ROIs)
- Ch7: multimodal sMRI + fMRI fusion ([anxiety-multimodal-fusion](https://github.com/ethancaldeira/anxiety-multimodal-fusion);
  structural and fMRI arms in [anxiety-smri-classification](https://github.com/ethancaldeira/anxiety-smri-classification)
  and [anxiety-fmri-classification](https://github.com/ethancaldeira/anxiety-fmri-classification))

---

## Repository layout

```
anxiety-fmri-parcellations/
├── make_gsp_split.py                  # writes data/gsp_split.json from the participants table
├── run_individual_parcellation.py     # per-subject parcellations
├── run_group_parcellation_split.py    # STAI-T group parcellations from the training split
├── compute_parcellation_features.py   # features from the individual parcellations
├── requirements.txt
├── LICENSE
├── data/                              # local only (git-ignored): gsp_split.json, caches
├── masks/
│   └── grey_matter_mask.nii.gz        # MNI152 grey-matter mask, 3 mm
└── src/                               # clustering library (adapted from pyClusterROI)
    ├── make_local_connectivity_tcorr.py   # local temporal-correlation graph
    ├── make_local_connectivity_scorr.py   # local spatial-correlation graph
    ├── binfile_parcellation.py            # subject-level normalised cut
    ├── group_mean_binfile_parcellation.py # group-mean method
    ├── group_binfile_parcellation.py      # two-level method
    └── python_ncut_lib.py                 # normalised cut and discretisation
```

---

## Installation

Tested with Python 3.12 on Ubuntu (CPU only; the thesis runs used a
c5.4xlarge instance).

```bash
git clone https://github.com/ethancaldeira/anxiety-fmri-parcellations.git
cd anxiety-fmri-parcellations
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used for the thesis
parcellations. All scripts read from and write to S3, so AWS credentials
with access to the bucket are required. The bucket is passed with
`--bucket <bucket>` or set once for all three scripts:

```bash
export PARCELLATION_BUCKET=<bucket>
```

Local scratch files go to `--work-dir` (default: `parcellation_work` and
`parcellation_group_split_work` under the system temporary directory).

---

## Data

| | |
|---|---|
| Cohort | GSP, 877 subjects with preprocessed rs-fMRI (879 participants with STAI-T; C-PAC functional preprocessing did not complete for Sub0105_S1 and Sub1382_S1) |
| Phenotype | STAI-T (`STAI_tAnxiety`) |
| Preprocessed BOLD | C-PAC 1.8.7 `preproc` preconfig, `space-MNI152NLin6ASym_reg-1_desc-preproc_bold` (3 mm), produced by [`anxiety-fmri-classification/preprocessing`](https://github.com/ethancaldeira/anxiety-fmri-classification/tree/main/preprocessing) |
| Grey-matter mask | `masks/grey_matter_mask.nii.gz`, 61 × 73 × 61, 49,786 voxels: the MNI152 3 mm grey-matter mask of Khosla et al. (2019), a byte-identical copy of `misc/grey_matter_mask.nii.gz` from [Ensemble3DCNN_connectomes](https://github.com/mk2299/Ensemble3DCNN_connectomes) (the same mask as in anxiety-fmri-classification) |
| Split | fixed: `data/gsp_split.json` (595 / 106 / 176), written by `make_gsp_split.py`; the same split as in anxiety-fmri-classification |

The Brain Genomics Superstruct Project Open Access release
(Holmes et al., 2015) is available from
[neuroinformatics.harvard.edu/gsp](http://neuroinformatics.harvard.edu/gsp/)
under the GSP Open Access Data Use Terms.

- **Participants file:** a TSV with the columns `participant_id` (GSP subject
  ID, e.g. `SubNNNN_S1`) and `STAI_tAnxiety`, built from the GSP phenotype
  table (`--split binary` in the group script also needs an
  `Anxiety_Binary` column). It is passed to the scripts with
  `--participants` and is not distributed here. The individual and feature
  scripts read only the subject IDs, so a plain list with one ID per line
  works for them too.
- **Split file:** `data/gsp_split.json` is not distributed; generate it from
  the participants file (see below). `make_gsp_split.py` uses the same recipe
  as `anxiety-fmri-classification/scripts/make_gsp_inputs.py`: STAI-T ≥ 36 as the
  stratification label, 20 % test, then 15 % of the remainder as validation,
  `random_state = 42`. `--check` compares the result with an existing
  split file instead of writing it.
- **fMRI:** read from `GSP_PREPROC/output/pipeline_cpac_preproc/sub-NNNN/ses-1/func/`
  in the S3 bucket (`--bucket` or `PARCELLATION_BUCKET`). Subjects without a
  preprocessed BOLD are reported as failed and skipped.

---

## Method

### Individual parcellations

For each subject, `run_individual_parcellation.py`:

1. downloads the preprocessed BOLD;
2. builds a sparse graph over grey-matter voxels, connecting each voxel to
   its 26 spatial neighbours, weighted by the temporal correlation of their
   timeseries (**tcorr**); only edges with r ≥ 0.5 are kept, so a voxel with
   no neighbour at or above the threshold is not connected to any other
   voxel (it keeps only a self-loop);
3. partitions the graph into K = 200 labels with normalised-cut spectral
   clustering;
4. uploads the voxel labels (`.npy`), the parcellation as a NIfTI, and,
   with `--save-connectivity`, the voxel graph used by the group-mean method.

`--connectivity scorr` uses the spatial correlation of whole-brain
connectivity maps instead of tcorr.

The clustering has a random initialisation (the start vector of the
eigendecomposition and the first rotation of the discretisation), so repeated runs
give slightly different parcellations; `--seed <int>` fixes it in both run
scripts. The exact group parcellations used in the thesis are committed in
anxiety-fmri-classification as
`src/templates/atlases/group_gm_{low,mid,high}.nii.gz`.

### Group parcellations

`run_group_parcellation_split.py` builds one parcellation per STAI-T group
from the subjects in one set of the split file (training set by default):

1. the training subjects are split into tertiles of their STAI-T scores
   (cut-offs 32 and 40, giving 211 low, 200 mid and 184 high);
2. **group-mean** (default): the subjects' voxel graphs are averaged within
   each group, and the averaged graph is partitioned into K = 200 regions
   with normalised cut;
3. each group parcellation is written as a NIfTI and uploaded.

`--method two-level` instead clusters the coincidence matrix of the
subjects' individual parcellations (how often two voxels fall in the same
region). `--split` also supports `median`, `binary` (needs the
`Anxiety_Binary` column) and `custom` (`--custom-threshold`) stratification.

Because the masks are derived from training subjects only, the test
subjects never influence the region boundaries. In anxiety-fmri-classification,
every subject is passed through all three group parcellations, one model is
trained per parcellation, and their predicted probabilities are averaged.

---

## Reproducing the Chapter 6 parcellations

```bash
export PARCELLATION_BUCKET=<bucket>

# 0. Train/val/test split -> data/gsp_split.json
python make_gsp_split.py --participants /path/to/GSP_participants.tsv

# 1. Individual parcellations (tcorr, K = 200) and the voxel graphs, all subjects
python run_individual_parcellation.py \
    --participants /path/to/GSP_participants.tsv --save-connectivity

# 2. Group parcellations: group-mean, train-only STAI-T tertiles, K = 200
python run_group_parcellation_split.py \
    --participants /path/to/GSP_participants.tsv

# 3. Features from the individual parcellations
python compute_parcellation_features.py --subjects /path/to/GSP_participants.tsv --mode roi
python compute_parcellation_features.py --subjects /path/to/GSP_participants.tsv --mode voxel
```

The individual and feature scripts skip subjects whose output is already on
S3, so they can be resumed.
`--dry-run` lists the work without running it, and `--subjects <file>`
restricts the individual pipeline to a list of IDs. `--seed <int>` makes the
clustering in steps 1 and 2 reproducible.
`run_group_parcellation_split.py --no-upload` keeps the group parcellations
in `--work-dir` (as `group_<low|mid|high>_anxiety_group-mean_200_train.nii.gz`)
instead of uploading them.

The individual pipeline takes about 45 s per subject; the three group
parcellations take a few minutes in total.

### Using the parcellations

The group parcellations are single atlases shared by all subjects. Copy them
into `anxiety-fmri-classification/src/templates/atlases/` as
`group_gm_low.nii.gz`, `group_gm_mid.nii.gz` and `group_gm_high.nii.gz`
(the names `run_experiments.py run_group_ensemble` expects; the masks used in
the thesis are already there). Its `src/compute_features.py` computes their
features like any other atlas, and `run_group_ensemble` trains the
three-model ensemble.

The individual parcellations differ for each subject, so their features are
computed here by `compute_parcellation_features.py`, under the atlas name
`individual_ROI_200`, and trained with
`anxiety-fmri-classification/run_experiments.py run_custom_atlases`. Write them
to the bucket that anxiety-fmri-classification reads as `FMRI_S3_BUCKET`
(`--bucket` or `PARCELLATION_BUCKET` set to the same name).

---

## Outputs

S3 layout, with `<K>` = 200:

```
GSP_Parcellation/
├── individual/ROI_<K>/
│   ├── npy/<subject>.npy                  # voxel labels, 0-based, in mask order
│   └── masks/<subject>.nii.gz             # parcellation, labels 1..K
├── connectivity/ROI_<K>/<subject>.npy     # voxel graph (--save-connectivity)
└── group_split/<set>/<group>_anxiety/<method>/ROI_<K>/
    ├── group_parcellation.nii.gz          # labels 1..K
    └── group_parcellation.npy

GSP_PREPROC/
├── connectivity/individual_ROI_200/<subject>.npy   # ROI-ROI, Fisher z
└── fingerprints/individual_ROI_200/<subject>.npy   # voxel fingerprints, 31 × 37 × 31 × N
```

The features use the same definitions as anxiety-fmri-classification: voxel
fingerprints are the Pearson correlation of every grey-matter voxel with each
region's mean timeseries, average-pooled 2×; ROI-ROI matrices are the
Fisher-z correlations of the region mean timeseries.

---

## Data use and acknowledgements

GSP data are used under the GSP Open Access Data Use Terms. Data were
provided by the Brain Genomics Superstruct Project of Harvard University and
the Massachusetts General Hospital (Principal Investigators: Randy Buckner,
Joshua Roffman, and Jordan Smoller), with support from the Center for Brain
Science Neuroinformatics Research Group, the Athinoula A. Martinos Center for
Biomedical Imaging, and the Center for Human Genetic Research. 20 individual
investigators at Harvard and MGH generously contributed data to the overall
project.

The clustering code in `src/` is adapted from
[pyClusterROI](https://github.com/ccraddock/cluster_roi), Copyright (C) 2010
R. Cameron Craddock, and modified by Ethan Caldeira (2026).

- Holmes, A. J. et al. (2015). Brain Genomics Superstruct Project initial
  data release with structural, functional, and behavioral measures.
  *Scientific Data*, 2, 150031.
- Craddock, R. C., James, G. A., Holtzheimer, P. E., Hu, X. P., & Mayberg,
  H. S. (2012). A whole brain fMRI atlas generated via spatially constrained
  spectral clustering. *Human Brain Mapping*, 33(8), 1914–1928.
- Khosla, M., Jamison, K., Kuceyeski, A., & Sabuncu, M. R. (2019). Ensemble
  learning with 3D convolutional neural networks for functional
  connectome-based prediction. *NeuroImage*, 199, 651–662.
- Shi, J., & Malik, J. (2000). Normalized cuts and image segmentation.
  *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(8),
  888–905.

## License

Released under the GNU General Public License v3.0 or later (see
[`LICENSE`](LICENSE)), as required by the pyClusterROI code it is derived
from. No GSP data or subject lists are distributed; they are available from
the GSP under its Open Access Data Use Terms.
