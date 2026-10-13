#!/usr/bin/env python3
"""
compute_parcellation_features.py — Compute connectivity features from each
subject's individual parcellation (run_individual_parcellation.py output).

The S3 bucket is given with --bucket or the PARCELLATION_BUCKET environment
variable. Masks are read from:
  s3://<bucket>/GSP_Parcellation/individual/ROI_200/masks/{subject_id}.nii.gz

Feature modes:
  --mode voxel : Voxel-wise correlation fingerprints (31x37x31xN_ROIs)
  --mode roi   : ROI-ROI connectivity matrices (N_ROIs x N_ROIs)

S3 output paths (atlas name 'individual_ROI_200'):
  Voxel: s3://<bucket>/GSP_PREPROC/fingerprints/individual_ROI_200/{subject_id}.npy
  ROI:   s3://<bucket>/GSP_PREPROC/connectivity/individual_ROI_200/{subject_id}.npy

Group parcellations (run_group_parcellation_split.py) are single atlases
shared by all subjects; their features are computed in anxiety-fmri-classification
like any other atlas.

Usage:
  # Voxel-wise features
  python compute_parcellation_features.py --subjects subjects.txt --mode voxel

  # ROI-ROI connectivity
  python compute_parcellation_features.py --subjects subjects.txt --mode roi

  # Dry run to check what needs computing
  python compute_parcellation_features.py --subjects subjects.txt --dry-run

Dependencies: numpy, nibabel, boto3, pandas
"""

import os
import re
import gc
import sys
import time
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
import boto3
from botocore.exceptions import ClientError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MASK_PATH = os.path.join(SCRIPT_DIR, "masks", "grey_matter_mask.nii.gz")
DEFAULT_CACHE_DIR = os.path.join(SCRIPT_DIR, "data", "fmri_cache")
DEFAULT_BUCKET = os.environ.get("PARCELLATION_BUCKET")

S3_FMRI_PREFIX = "GSP_PREPROC/output/pipeline_cpac_preproc"
S3_FP_PREFIX = "GSP_PREPROC/fingerprints"
S3_CONN_PREFIX = "GSP_PREPROC/connectivity"

# Individual parcellation masks (run_individual_parcellation.py)
S3_INDIVIDUAL_MASK_PREFIX = "GSP_Parcellation/individual/ROI_200/masks"
ATLAS_NAME = "individual_ROI_200"

MNI_DIMS = (61, 73, 61)
SUPERVOXEL_DIMS = (31, 37, 31)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_bids_id(raw_id):
    """Convert GSP participant ID to BIDS format (SubNNNN_S1 -> sub-NNNN)."""
    match = re.match(r"Sub(\d+)_S\d+", raw_id)
    if match:
        return f"sub-{int(match.group(1)):04d}"
    return raw_id


def corr2_coeff(A, B):
    """Row-wise Pearson correlation between matrices A (n_voxels x T) and B (n_rois x T)."""
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def avg_pool_3d(arr, pool_size=2):
    """
    Average pooling on (x, y, z, channels) array — pure numpy.
    Replicates keras AveragePooling3D(pool_size=2, padding='same').
    (61,73,61,C) -> (31,37,31,C)
    """
    x, y, z, c = arr.shape
    px = (pool_size - x % pool_size) % pool_size
    py = (pool_size - y % pool_size) % pool_size
    pz = (pool_size - z % pool_size) % pool_size
    if px or py or pz:
        arr = np.pad(arr, ((0, px), (0, py), (0, pz), (0, 0)))
    nx, ny, nz, _ = arr.shape
    return arr.reshape(
        nx // pool_size, pool_size,
        ny // pool_size, pool_size,
        nz // pool_size, pool_size, c
    ).mean(axis=(1, 3, 5)).astype(np.float32)


def s3_key_exists(s3_client, bucket, key):
    """Check if an S3 object exists."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def load_subjects(path):
    """
    Load subject IDs from a table with a participant_id column (tab- or
    comma-separated, whatever the file extension) or from a plain list with
    one ID per line. Same rules as run_individual_parcellation.read_subject_ids.
    """
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return []

    header = [c.strip() for c in re.split(r'[\t,]', lines[0])]
    if 'participant_id' in header:
        sep = '\t' if '\t' in lines[0] else ','
        df = pd.read_csv(path, sep=sep, dtype={'participant_id': str},
                         usecols=['participant_id'])
        return df['participant_id'].dropna().astype(str).str.strip().tolist()
    return lines


# ---------------------------------------------------------------------------
# S3 Downloads
# ---------------------------------------------------------------------------

def download_fmri(subject_id, cache_dir, s3_client, bucket):
    """Download preprocessed fMRI from S3. Returns local path or None."""
    bids_id = to_bids_id(subject_id)
    func_prefix = f"{S3_FMRI_PREFIX}/{bids_id}/ses-1/func/"

    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=func_prefix)
        if 'Contents' not in response:
            return None

        bold_key = None
        for obj in response['Contents']:
            key = obj['Key']
            if (key.endswith('.nii.gz')
                    and 'space-MNI152NLin6ASym' in key
                    and 'desc-preproc_bold' in key
                    and 'reg-1' in key):
                bold_key = key
                break

        if bold_key is None:
            return None

        filename = os.path.basename(bold_key)
        local_dir = os.path.join(cache_dir, bids_id)
        local_path = os.path.join(local_dir, filename)

        if os.path.exists(local_path):
            return local_path

        os.makedirs(local_dir, exist_ok=True)
        s3_client.download_file(bucket, bold_key, local_path)
        return local_path

    except ClientError as e:
        print(f"    S3 error downloading fMRI: {e}")
        return None


def find_local_fmri(subject_id, data_dir):
    """Find preprocessed fMRI in a local directory. Returns path or None."""
    bids_id = to_bids_id(subject_id)
    func_dir = os.path.join(data_dir, bids_id, "ses-1", "func")
    if not os.path.isdir(func_dir):
        return None
    for f in os.listdir(func_dir):
        if '_bold' in f and f.endswith('.nii.gz'):
            return os.path.join(func_dir, f)
    return None


def download_mask_individual(subject_id, cache_dir, s3_client, bucket):
    """
    Download a subject's individual parcellation mask from S3.

    S3 path: GSP_Parcellation/individual/ROI_200/masks/{subject_id}.nii.gz

    Returns local path or None.
    """
    s3_key = f"{S3_INDIVIDUAL_MASK_PREFIX}/{subject_id}.nii.gz"
    local_dir = os.path.join(cache_dir, "individual_masks")
    local_path = os.path.join(local_dir, f"{subject_id}.nii.gz")

    if os.path.exists(local_path):
        return local_path

    os.makedirs(local_dir, exist_ok=True)
    try:
        s3_client.download_file(bucket, s3_key, local_path)
        return local_path
    except ClientError as e:
        print(f"    Mask not found on S3: {s3_key} ({e})")
        return None


# ---------------------------------------------------------------------------
# Feature Computation
# ---------------------------------------------------------------------------

def compute_fingerprint(img_data, atlas_data, mask_data, nodes, n_rois):
    """
    Compute voxel-wise connectivity fingerprint for one subject.
    Returns downsampled array of shape (31, 37, 31, n_rois).
    """
    # ROI mean timeseries
    roi_means = []
    for n in nodes:
        if n > 0:
            voxels = img_data[atlas_data == n]
            if voxels.shape[0] == 0:
                roi_means.append(np.zeros(img_data.shape[-1]))
            else:
                roi_means.append(np.mean(voxels, axis=0))
    roi_means = np.asarray(roi_means)  # (n_rois, timepoints)

    # Masked voxel timeseries
    masked = img_data[mask_data > 0].reshape(-1, img_data.shape[-1])

    # Correlate every grey-matter voxel with every ROI
    n_voxels = MNI_DIMS[0] * MNI_DIMS[1] * MNI_DIMS[2]
    fp = np.zeros((n_voxels, n_rois), dtype=np.float32)
    fp[(mask_data > 0).reshape(-1)] = corr2_coeff(masked, roi_means).astype(np.float32)
    fp[np.isnan(fp)] = 0

    # Reshape to 3D + downsample
    fp = fp.reshape(*MNI_DIMS, n_rois)
    fp = avg_pool_3d(fp)

    return fp


def compute_roi_connectivity(img_data, atlas_data, nodes, n_rois, fisher_z=True):
    """
    Compute ROI-ROI connectivity matrix for one subject.
    Returns (n_rois, n_rois) connectivity matrix as float32.
    """
    roi_timeseries = []
    for n in nodes:
        if n > 0:
            voxels = img_data[atlas_data == n]
            if voxels.shape[0] == 0:
                roi_timeseries.append(np.zeros(img_data.shape[-1]))
            else:
                roi_timeseries.append(np.mean(voxels, axis=0))

    roi_timeseries = np.array(roi_timeseries)  # (n_rois, timepoints)

    conn = np.corrcoef(roi_timeseries)
    conn[np.isnan(conn)] = 0.0
    np.clip(conn, -1.0, 1.0, out=conn)

    if fisher_z:
        np.clip(conn, -0.999999, 0.999999, out=conn)
        conn = np.arctanh(conn)
        np.fill_diagonal(conn, 0.0)

    return conn.astype(np.float32)


# ---------------------------------------------------------------------------
# Main Processing Loop
# ---------------------------------------------------------------------------

def process_subjects(subject_ids, args, s3_client):
    """
    Main processing loop: for each subject, download mask + fMRI,
    compute features, and upload to S3.
    """
    mode = args.mode
    fisher_z = not args.no_fisher_z
    atlas_name = ATLAS_NAME

    s3_prefix = S3_CONN_PREFIX if mode == 'roi' else S3_FP_PREFIX

    # Load grey matter mask (shared across all subjects for voxel mode)
    mask_data = None
    if mode == 'voxel':
        print(f"Loading grey matter mask: {args.mask_file}")
        mask_data = nib.load(args.mask_file).get_fdata()

    print(f"\n{'='*60}")
    print(f"  Atlas name:   {atlas_name}")
    print(f"  Mode:         {mode}")
    if mode == 'roi':
        print(f"  Fisher z:     {fisher_z}")
    print(f"  Subjects:     {len(subject_ids)}")
    print(f"  Source:       {'local (' + args.data_dir + ')' if args.data_dir else 'S3'}")
    print(f"{'='*60}\n")

    start_time = time.time()
    done, skipped, failed = 0, 0, 0

    for i, sid in enumerate(subject_ids):
        s3_key = f"{s3_prefix}/{atlas_name}/{sid}.npy"
        tag = f"[{i+1}/{len(subject_ids)}] {sid}"

        # Skip if already on S3
        if not args.force and s3_key_exists(s3_client, args.bucket, s3_key):
            skipped += 1
            if skipped <= 3 or skipped % 100 == 0:
                print(f"  {tag} — exists on S3, skipping")
            continue

        if args.dry_run:
            print(f"  {tag} — needs computing")
            continue

        # --- Get this subject's individual parcellation ---
        mask_path = download_mask_individual(sid, args.cache_dir, s3_client, args.bucket)
        if mask_path is None:
            print(f"  {tag} — no individual mask found, skipping")
            failed += 1
            continue
        try:
            atlas_data = nib.load(mask_path).get_fdata()
        except Exception as e:
            print(f"  {tag} — error loading mask: {e}")
            failed += 1
            continue

        nodes = sorted(np.unique(atlas_data).astype(int).tolist())
        n_rois = len(nodes) - 1

        if n_rois == 0:
            print(f"  {tag} — mask has no ROIs, skipping")
            failed += 1
            continue

        # --- Get fMRI data ---
        if args.data_dir:
            fmri_path = find_local_fmri(sid, args.data_dir)
        else:
            fmri_path = download_fmri(sid, args.cache_dir, s3_client, args.bucket)

        if fmri_path is None:
            print(f"  {tag} — no fMRI found, skipping")
            failed += 1
            continue

        try:
            img_data = nib.load(fmri_path).get_fdata()
        except Exception as e:
            print(f"  {tag} — load error: {e}")
            failed += 1
            continue

        # --- Compute features ---
        t0 = time.time()
        try:
            if mode == 'roi':
                fp = compute_roi_connectivity(img_data, atlas_data, nodes, n_rois,
                                              fisher_z=fisher_z)
            else:
                fp = compute_fingerprint(img_data, atlas_data, mask_data, nodes, n_rois)
        except Exception as e:
            print(f"  {tag} — compute error: {e}")
            failed += 1
            del img_data
            gc.collect()
            continue

        # --- Save locally and upload to S3 ---
        local_tmp_dir = os.path.join(args.cache_dir, "tmp")
        os.makedirs(local_tmp_dir, exist_ok=True)
        local_tmp = os.path.join(local_tmp_dir, f"{sid}_{atlas_name}.npy")
        np.save(local_tmp, fp)

        try:
            s3_client.upload_file(local_tmp, args.bucket, s3_key)
            elapsed = time.time() - t0
            done += 1
            print(f"  {tag} — done {fp.shape} ({n_rois} ROIs) in {elapsed:.1f}s")
        except ClientError as e:
            print(f"  {tag} — upload error: {e}")
            failed += 1
        finally:
            if os.path.exists(local_tmp):
                os.remove(local_tmp)

        del img_data, fp
        gc.collect()

    total_elapsed = time.time() - start_time
    print(f"\n  {atlas_name}: {done} computed, {skipped} existed, {failed} failed "
          f"({total_elapsed:.0f}s)")

    return atlas_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compute connectivity features from individual GSP parcellations.'
    )
    parser.add_argument('--subjects', required=True,
                        help='Subject list: CSV/TSV with a participant_id column, or one ID per line')
    parser.add_argument('--mode', default='voxel', choices=['voxel', 'roi'],
                        help='Feature mode: "voxel" for voxel-wise fingerprints, '
                             '"roi" for ROI-ROI connectivity matrices (default: voxel)')
    parser.add_argument('--mask-file', '--mask_file', dest='mask_file', default=DEFAULT_MASK_PATH,
                        help='Grey matter mask NIfTI file (used in voxel mode; default: '
                             'masks/grey_matter_mask.nii.gz in this repository)')
    parser.add_argument('--cache-dir', '--cache_dir', dest='cache_dir', default=DEFAULT_CACHE_DIR,
                        help='Local cache for S3 downloads (fMRI + masks; default: '
                             'data/fmri_cache in this repository)')
    parser.add_argument('--data-dir', '--data_dir', dest='data_dir', default=None,
                        help='Local fMRI directory laid out as sub-NNNN/ses-1/func/ '
                             '(skip the S3 fMRI download if provided)')
    parser.add_argument('--bucket', default=DEFAULT_BUCKET,
                        help='S3 bucket holding the parcellations and the features '
                             '(default: $PARCELLATION_BUCKET)')
    parser.add_argument('--no-fisher-z', '--no_fisher_z', dest='no_fisher_z', action='store_true',
                        help='Skip Fisher z-transform on ROI connectivity (roi mode only)')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if feature file already exists on S3')
    parser.add_argument('--dry-run', '--dry_run', dest='dry_run', action='store_true',
                        help='Check what needs computing without doing it')
    args = parser.parse_args()

    if not args.bucket:
        sys.exit("ERROR: no S3 bucket given. Pass --bucket or set PARCELLATION_BUCKET.")

    subject_ids = load_subjects(args.subjects)
    print(f"Loaded {len(subject_ids)} subjects from {args.subjects}")

    s3_client = boto3.client('s3')

    atlas_name = process_subjects(subject_ids, args, s3_client)
    print(f"\nDone. Features written under atlas name: {atlas_name}")

if __name__ == '__main__':
    main()
