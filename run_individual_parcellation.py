"""
Individual Parcellation Pipeline (GSP Dataset)
================================================
Streams C-PAC preprocessed fMRI scans from S3, computes spatially constrained
local connectivity (temporal correlation of neighbouring voxels by default), runs
normalized-cut spectral clustering per subject, and uploads individual
parcellation results to s3://<bucket>/GSP_Parcellation/individual/.

The S3 bucket is given with --bucket or the PARCELLATION_BUCKET environment
variable. Only subject IDs are read from the participants file.

Usage:
    python run_individual_parcellation.py --participants /path/to/GSP_participants.tsv [--subjects subjects.txt] [--clusters 200]

    # Use spatial correlation of whole-brain FC maps instead
    python run_individual_parcellation.py --participants /path/to/GSP_participants.tsv --connectivity scorr
"""

import os
import re
import sys
import argparse
import tempfile
import boto3
import pandas as pd
import nibabel as nib
import numpy as np

from src.make_local_connectivity_scorr import make_local_connectivity_scorr
from src.make_local_connectivity_tcorr import make_local_connectivity_tcorr
from src.binfile_parcellation import binfile_parcellate

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MASK_PATH = os.path.join(SCRIPT_DIR, "masks", "grey_matter_mask.nii.gz")
DEFAULT_BUCKET = os.environ.get("PARCELLATION_BUCKET")
DEFAULT_NUM_CLUSTERS = 200
DEFAULT_CORRELATION_THRESHOLD = 0.5

# GSP preprocessed fMRI path on S3 (C-PAC output)
S3_PREPROC_PREFIX = "GSP_PREPROC/output/pipeline_cpac_preproc"
S3_OUTPUT_PREFIX = "GSP_Parcellation/individual"
S3_CONNECTIVITY_PREFIX = "GSP_Parcellation/connectivity"

DEFAULT_WORK_DIR = os.path.join(tempfile.gettempdir(), "parcellation_work")

# Connectivity method lookup
CONNECTIVITY_METHODS = {
    "scorr": make_local_connectivity_scorr,
    "tcorr": make_local_connectivity_tcorr,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run individual parcellation pipeline (GSP)")
    parser.add_argument("--participants", type=str, default=None,
                        help="Subjects to process: a TSV/CSV table with a participant_id column "
                             "(e.g. GSP_participants.tsv) or a plain list with one ID per line. "
                             "Only the IDs are read.")
    parser.add_argument("--subjects", type=str, default=None,
                        help="Text file with one subject ID per line (e.g. SubNNNN_S1). With "
                             "--participants, restricts processing to these IDs; on its own, "
                             "it is the list of subjects to process.")
    parser.add_argument("--mask", type=str, default=DEFAULT_MASK_PATH,
                        help="Path to grey matter mask NIfTI (default: masks/grey_matter_mask.nii.gz "
                             "in this repository)")
    parser.add_argument("--clusters", type=int, default=DEFAULT_NUM_CLUSTERS,
                        help=f"Number of clusters/parcels (default: {DEFAULT_NUM_CLUSTERS})")
    parser.add_argument("--thresh", type=float, default=DEFAULT_CORRELATION_THRESHOLD,
                        help=f"Correlation threshold: neighbour edges with r below it are dropped "
                             f"(default: {DEFAULT_CORRELATION_THRESHOLD})")
    parser.add_argument("--connectivity", type=str, default="tcorr",
                        choices=["scorr", "tcorr"],
                        help="Connectivity method: 'tcorr' (temporal correlation of neighbouring "
                             "time series — local similarity, default) or 'scorr' (spatial "
                             "correlation of whole-brain FC maps — functionally informed)")
    parser.add_argument("--save-connectivity", action="store_true",
                        help="Also upload the connectivity matrix to S3 (needed for group-mean method)")
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET,
                        help="S3 bucket holding the inputs and outputs "
                             "(default: $PARCELLATION_BUCKET)")
    parser.add_argument("--work-dir", type=str, default=DEFAULT_WORK_DIR,
                        help=f"Local scratch directory for downloads and intermediate files "
                             f"(default: {DEFAULT_WORK_DIR})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the random initialisation of the clustering, applied to "
                             "each subject (default: unseeded)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List subjects that would be processed without running")
    args = parser.parse_args()
    if not args.participants and not args.subjects:
        parser.error("give --participants and/or --subjects")
    return args


# ── Subject ID Conversion ────────────────────────────────────────────────────
def participant_to_bids(participant_id):
    """
    Convert GSP participant_id to BIDS format for S3 lookup.
    e.g. 'SubNNNN_S1' -> 'sub-NNNN'
    """
    match = re.match(r"Sub(\d+)_S\d+", participant_id)
    if not match:
        return None
    return f"sub-{match.group(1)}"


def construct_s3_preproc_key(bids_id):
    """
    Build the S3 key for a subject's preprocessed BOLD scan.
    e.g. GSP_PREPROC/output/pipeline_cpac_preproc/sub-NNNN/ses-1/func/
         sub-NNNN_ses-1_task-rest_space-MNI152NLin6ASym_reg-1_desc-preproc_bold.nii.gz
    """
    filename = f"{bids_id}_ses-1_task-rest_space-MNI152NLin6ASym_reg-1_desc-preproc_bold.nii.gz"
    return f"{S3_PREPROC_PREFIX}/{bids_id}/ses-1/func/{filename}"


# ── Stage 1: Load Subject IDs ────────────────────────────────────────────────
def read_subject_ids(path):
    """
    Read subject IDs from a table with a participant_id column (TSV or CSV)
    or from a plain list with one ID per line.
    """
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return []

    header = [c.strip() for c in re.split(r"[\t,]", lines[0])]
    if "participant_id" in header:
        sep = "\t" if "\t" in lines[0] else ","
        df = pd.read_csv(path, sep=sep, dtype={"participant_id": str},
                         usecols=["participant_id"])
        return df["participant_id"].dropna().astype(str).str.strip().tolist()
    return lines


def load_subjects(participants_path=None, subjects_file=None):
    """Subject IDs from the participants file, optionally restricted to a subject list."""
    if participants_path:
        subject_ids = read_subject_ids(participants_path)
        if subjects_file:
            keep = set(read_subject_ids(subjects_file))
            subject_ids = [s for s in subject_ids if s in keep]
    else:
        subject_ids = read_subject_ids(subjects_file)

    subjects = []
    for participant_id in subject_ids:
        bids_id = participant_to_bids(participant_id)

        if not bids_id:
            print(f"Skipping {participant_id} (could not parse subject ID)")
            continue

        subjects.append({
            "participant_id": participant_id,
            "bids_id": bids_id,
        })

    print(f"Loaded {len(subjects)} GSP subject IDs.")
    return subjects


# ── Stage 2: Stream, Cluster, Upload ─────────────────────────────────────────
def process_subject(entry, s3, mask_path, num_clusters, thresh, bucket,
                    connectivity_fn, conn_method, work_dir, save_connectivity=False,
                    seed=None):
    """Download preprocessed fMRI, compute connectivity, cluster, upload result."""
    participant_id = entry["participant_id"]
    bids_id = entry["bids_id"]

    s3_key = construct_s3_preproc_key(bids_id)
    local_filename = f"{bids_id}_ses-1_desc-preproc_bold.nii.gz"
    local_nifti = os.path.join(work_dir, local_filename)
    conn_file = os.path.join(work_dir, f"conn_{participant_id}.npy")
    cluster_prefix = os.path.join(work_dir, f"{participant_id}_cluster")
    nifti_file = os.path.join(work_dir, f"{participant_id}_parcellation.nii.gz")

    try:
        # Download
        print(f"\n{'='*60}")
        print(f"Subject: {participant_id} -> {bids_id}")
        print(f"Downloading s3://{bucket}/{s3_key}")
        s3.download_file(bucket, s3_key, local_nifti)

        img = nib.load(local_nifti)
        print(f"Scan shape: {img.shape}")

        # Compute local connectivity
        print(f"Computing local connectivity (method={conn_method}, threshold={thresh})")
        connectivity_fn(local_nifti, mask_path, conn_file, thresh)

        # Optionally upload connectivity matrix (needed for group-mean method)
        if save_connectivity:
            conn_s3_key = f"{S3_CONNECTIVITY_PREFIX}/ROI_{num_clusters}/{participant_id}.npy"
            print(f"Uploading connectivity matrix to s3://{bucket}/{conn_s3_key}")
            s3.upload_file(conn_file, bucket, conn_s3_key)

        # Run individual clustering
        print(f"Running spectral clustering (K={num_clusters})")
        binfile_parcellate(conn_file, cluster_prefix, num_clusters, seed=seed)

        # Load result
        label_file = f"{cluster_prefix}_{num_clusters}.npy"
        labels = np.load(label_file)
        print(f"Cluster labels shape: {labels.shape}, unique: {len(np.unique(labels))}")

        # Upload .npy labels
        s3_out_key = f"{S3_OUTPUT_PREFIX}/ROI_{num_clusters}/npy/{participant_id}.npy"
        print(f"Uploading to s3://{bucket}/{s3_out_key}")
        s3.upload_file(label_file, bucket, s3_out_key)

        # Convert to NIfTI and upload
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata().astype(bool)
        label_volume = np.zeros(mask_data.shape, dtype=np.uint16)
        label_volume[mask_data] = labels.astype(np.uint16) + 1  # 1-based labels in NIfTI
        nib.save(nib.Nifti1Image(label_volume, mask_img.affine), nifti_file)

        s3_nifti_key = f"{S3_OUTPUT_PREFIX}/ROI_{num_clusters}/masks/{participant_id}.nii.gz"
        print(f"Uploading NIfTI to s3://{bucket}/{s3_nifti_key}")
        s3.upload_file(nifti_file, bucket, s3_nifti_key)
        print(f"Done with subject {participant_id}")

        return True

    except Exception as e:
        print(f"FAILED for subject {participant_id}: {e}")
        return False

    finally:
        # Clean up temp files
        for f in [local_nifti, conn_file, f"{cluster_prefix}_{num_clusters}.npy", nifti_file]:
            if os.path.exists(f):
                os.remove(f)


def check_already_processed(s3, bucket, participant_id, num_clusters):
    """Check if this subject's parcellation already exists on S3."""
    s3_key = f"{S3_OUTPUT_PREFIX}/ROI_{num_clusters}/npy/{participant_id}.npy"
    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
        return True
    except s3.exceptions.ClientError:
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Validate paths
    for label, path in (("Participants", args.participants), ("Subjects", args.subjects)):
        if path and not os.path.exists(path):
            print(f"ERROR: {label} file not found: {path}")
            return
    if not os.path.exists(args.mask):
        print(f"ERROR: Mask file not found: {args.mask}")
        return

    # Select connectivity method
    connectivity_fn = CONNECTIVITY_METHODS[args.connectivity]
    print(f"Connectivity method: {args.connectivity}")
    if args.connectivity == "scorr":
        print("  -> Spatial correlation of whole-brain FC maps (functionally informed)")
    else:
        print("  -> Temporal correlation of neighbouring time series (local similarity)")

    subjects = load_subjects(args.participants, args.subjects)

    if args.dry_run:
        print(f"\nDry run - would process {len(subjects)} subjects:")
        for entry in subjects[:10]:
            print(f"  {entry['participant_id']} -> {entry['bids_id']}")
        if len(subjects) > 10:
            print(f"  ... and {len(subjects) - 10} more")
        return

    if not args.bucket:
        sys.exit("ERROR: no S3 bucket given. Pass --bucket or set PARCELLATION_BUCKET.")

    os.makedirs(args.work_dir, exist_ok=True)
    s3 = boto3.client("s3")

    # Filter out already-processed subjects
    to_process = []
    for entry in subjects:
        if check_already_processed(s3, args.bucket, entry["participant_id"], args.clusters):
            print(f"Skipping {entry['participant_id']} (already exists on S3)")
        else:
            to_process.append(entry)

    print(f"\n{len(to_process)} subjects to process ({len(subjects) - len(to_process)} already done)")

    succeeded = 0
    failed = 0

    for i, entry in enumerate(to_process):
        print(f"\n[{i+1}/{len(to_process)}]")
        if process_subject(entry, s3, args.mask, args.clusters, args.thresh, args.bucket,
                           connectivity_fn, args.connectivity, args.work_dir,
                           save_connectivity=args.save_connectivity, seed=args.seed):
            succeeded += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Pipeline complete: {succeeded} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
