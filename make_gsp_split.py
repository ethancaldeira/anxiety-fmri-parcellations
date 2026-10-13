#!/usr/bin/env python3
"""
Generate the fixed GSP train/val/test split (data/gsp_split.json).

The split is the one used in anxiety-fmri-classification, built with the same
recipe as its scripts/make_gsp_inputs.py:

  1. read the GSP participants table (``participant_id``, ``STAI_tAnxiety``);
  2. keep the subjects with a preprocessed BOLD: either the IDs listed in
     ``--preprocessed``, or every subject not named in ``--exclude`` (by
     default the two GSP subjects for which C-PAC functional preprocessing
     did not complete, Sub0105_S1 and Sub1382_S1);
  3. drop subjects without a STAI-T score and sort by participant_id;
  4. label each subject STAI-T >= 36 (the cohort median);
  5. stratified on that label with scikit-learn's ``train_test_split`` and
     ``random_state=42``: hold out 20 % as the test set, then 15 % of the
     remainder as the validation set.

For the 877 GSP subjects this gives 595 train / 106 val / 176 test.

Usage:
    python make_gsp_split.py --participants /path/to/GSP_participants.tsv

    # explicit list of subjects with a preprocessed BOLD (SubNNNN_S1 or sub-NNNN)
    python make_gsp_split.py --participants /path/to/GSP_participants.tsv \\
        --preprocessed preprocessed_subjects.txt

    # write somewhere else
    python make_gsp_split.py --participants /path/to/GSP_participants.tsv \\
        --out /some/where/gsp_split.json

    # compare with an existing split file instead of writing
    python make_gsp_split.py --participants /path/to/GSP_participants.tsv --check
"""

import argparse
import json
import os
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "data", "gsp_split.json")

# Subjects for which C-PAC functional preprocessing did not complete.
DEFAULT_EXCLUDE = ["Sub0105_S1", "Sub1382_S1"]

STAI_COLUMN = "STAI_tAnxiety"
STAI_CUTOFF = 36
TEST_SIZE = 0.20
VAL_SIZE = 0.15          # fraction of the non-test subjects
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the fixed GSP train/val/test split JSON")
    parser.add_argument("--participants", required=True,
                        help="GSP participants table (TSV or CSV) with "
                             f"'participant_id' and '{STAI_COLUMN}' columns")
    parser.add_argument("--preprocessed", default=None,
                        help="Optional text file listing the subjects with a preprocessed "
                             "BOLD (one ID per line, SubNNNN_S1 or sub-NNNN). "
                             "Overrides --exclude.")
    parser.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE,
                        help="Subjects without a preprocessed BOLD "
                             f"(default: {' '.join(DEFAULT_EXCLUDE)})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Output JSON path (default: data/gsp_split.json in this repository)")
    parser.add_argument("--check", action="store_true",
                        help="Compare with the existing file at --out instead of writing; "
                             "exit status 1 if any set differs")
    return parser.parse_args()


def to_participant_id(raw):
    """Accept 'SubNNNN_S1' or the BIDS form 'sub-NNNN'; return 'SubNNNN_S1'."""
    raw = raw.strip()
    m = re.fullmatch(r"sub-(\d+)", raw)
    return f"Sub{int(m.group(1)):04d}_S1" if m else raw


def load_subjects(participants_path, preprocessed=None, exclude=DEFAULT_EXCLUDE):
    """Participants with a preprocessed BOLD and a STAI-T score, sorted by ID."""
    sep = "\t" if participants_path.endswith(".tsv") else ","
    df = pd.read_csv(participants_path, sep=sep, dtype={"participant_id": str})
    missing = {"participant_id", STAI_COLUMN} - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: {participants_path} is missing column(s): {sorted(missing)}")
    df = df[["participant_id", STAI_COLUMN]].copy()

    if preprocessed:
        with open(preprocessed) as f:
            keep = {to_participant_id(line) for line in f if line.strip()}
        unknown = keep - set(df["participant_id"])
        if unknown:
            print(f"  [warn] {len(unknown)} ID(s) in {preprocessed} are not in the "
                  f"participants table and are ignored")
        df = df[df["participant_id"].isin(keep)]
    else:
        df = df[~df["participant_id"].isin(set(exclude or []))]

    n_before = len(df)
    df = df.dropna(subset=[STAI_COLUMN])
    if len(df) < n_before:
        print(f"  dropped {n_before - len(df)} subject(s) without {STAI_COLUMN}")
    return df.sort_values("participant_id").reset_index(drop=True)


def make_split(subjects):
    """Return {'train': [...], 'val': [...], 'test': [...]} of participant IDs."""
    ids = subjects["participant_id"].values
    labels = (subjects[STAI_COLUMN] >= STAI_CUTOFF).astype(int).values

    ids_trval, ids_test, y_trval, _ = train_test_split(
        ids, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE)
    ids_train, ids_val, _, _ = train_test_split(
        ids_trval, y_trval, test_size=VAL_SIZE, stratify=y_trval, random_state=RANDOM_STATE)

    return {
        "train": [str(s) for s in ids_train],
        "val": [str(s) for s in ids_val],
        "test": [str(s) for s in ids_test],
    }


def check_split(split, path):
    """Compare a split with the JSON file at path; return 0 if every set matches, else 1."""
    if not os.path.isfile(path):
        print(f"  {path}: not found")
        return 1
    with open(path) as f:
        old = json.load(f)
    rc = 0
    for part in ("train", "val", "test"):
        old_ids = old.get(part, [])
        if old_ids == split[part]:
            status = "identical"
        elif set(old_ids) == set(split[part]):
            status = "same subjects, different order"
        else:
            status = "DIFFERS"
            rc = 1
        print(f"  {path} [{part}]: {status}")
    return rc


def main():
    args = parse_args()
    subjects = load_subjects(args.participants, args.preprocessed, args.exclude)
    split = make_split(subjects)

    if args.check:
        sys.exit(check_split(split, args.out))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)

    n = {k: len(v) for k, v in split.items()}
    print(f"Wrote {args.out}: train {n['train']}, val {n['val']}, test {n['test']} "
          f"(total {sum(n.values())})")


if __name__ == "__main__":
    main()
