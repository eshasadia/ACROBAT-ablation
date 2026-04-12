"""
run_ablation.py
===============
Runs all ablation experiments defined in acrobat_submission_configs.py.

Each entry in ABLATION_CONFIGS maps a human-readable label to a config factory
function.  The script executes every experiment sequentially and prints a
summary of which ones succeeded or failed.

Usage
-----
    python run_ablation.py                        # run all experiments
    python run_ablation.py --methods sift orb     # run a subset by label prefix
    python run_ablation.py --list                 # list available experiment labels
"""

import argparse
import traceback

import create_acrobat_submission as cas
import acrobat_submission_configs as configs

# ---------------------------------------------------------------------------
# Registry of all ablation experiments
# ---------------------------------------------------------------------------

# Each tuple is (label, config_factory).
# Add new experiments here — no other file needs to change.
ABLATION_CONFIGS = [
    # --- Affine-only baselines ---
    ("affine",                          configs.affine_config),
    ("affine_iterative",                configs.affine_iterative_config),
    ("sift_ransac",                     configs.sift_ransac_config),
    ("orb_ransac",                      configs.orb_ransac_config),
    ("akaze_ransac",                    configs.akaze_ransac_config),
    ("brisk_ransac",                    configs.brisk_ransac_config),
    ("ecc_affine",                      configs.ecc_affine_config),
    ("ecc_euclidean",                   configs.ecc_euclidean_config),
    ("superpoint_superglue",            configs.superpoint_superglue_config),
    ("superpoint_ransac",               configs.superpoint_ransac_config),
    # --- Affine + nonrigid baselines ---
    ("affine_iterative_nonrigid",       configs.affine_nonrigid_config),
    ("sift_ransac_nonrigid",            configs.sift_ransac_nonrigid_config),
    ("orb_ransac_nonrigid",             configs.orb_ransac_nonrigid_config),
    ("akaze_ransac_nonrigid",           configs.akaze_ransac_nonrigid_config),
    ("brisk_ransac_nonrigid",           configs.brisk_ransac_nonrigid_config),
    ("ecc_nonrigid",                    configs.ecc_nonrigid_config),
    ("superpoint_superglue_nonrigid",   configs.superpoint_superglue_nonrigid_config),
    ("superpoint_ransac_nonrigid",      configs.superpoint_ransac_nonrigid_config),
]


def list_experiments():
    print("Available ablation experiments:")
    for label, _ in ABLATION_CONFIGS:
        print(f"  {label}")


def run_ablation(methods=None):
    """
    Run ablation experiments.

    Parameters
    ----------
    methods : list of str or None
        If given, only run experiments whose label starts with any of the
        provided prefixes.  If None, run all experiments.
    """
    selected = [
        (label, factory)
        for label, factory in ABLATION_CONFIGS
        if methods is None or any(label.startswith(m) for m in methods)
    ]

    if not selected:
        print("No experiments matched the provided method filter.")
        return

    results = {}
    for label, factory in selected:
        print(f"\n{'='*60}")
        print(f"Running experiment: {label}")
        print(f"{'='*60}")
        try:
            config = factory()
            cas.create_acrobat_submission(**config)
            results[label] = "SUCCESS"
        except Exception:
            traceback.print_exc()
            results[label] = "FAILED"

    print(f"\n{'='*60}")
    print("Ablation summary:")
    print(f"{'='*60}")
    for label, status in results.items():
        marker = "✓" if status == "SUCCESS" else "✗"
        print(f"  {marker} {label}: {status}")


def main():
    parser = argparse.ArgumentParser(description="Run ACROBAT ablation experiments.")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Run only experiments whose label starts with one of these prefixes.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiment labels and exit.",
    )
    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    run_ablation(methods=args.methods)


if __name__ == "__main__":
    main()
