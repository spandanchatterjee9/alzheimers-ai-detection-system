"""
Master CLI Preprocessing Execution Pipeline for Alzheimer's Disease Detection System.

Usage:
    # Full end-to-end dataset preprocessing (ALL 413 patients):
    python scripts/run_preprocessing.py

    # Options:
    python scripts/run_preprocessing.py --force --num-workers 8
    python scripts/run_preprocessing.py --max-patients 10
    python scripts/run_preprocessing.py --test-inference
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Add project root directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.scanner import OasisDatasetScanner
from preprocessing.dataset_builder import DatasetBuilder
from preprocessing.inference_preprocessor import InferencePreprocessor
from training.split_generator import PatientSplitGenerator
from scripts.verify_dataset import main as run_verification


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/preprocessing.log", encoding="utf-8")
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Alzheimer's Detection Master Preprocessing Pipeline")
    parser.add_argument("--scan", action="store_true", help="Scan dataset and generate manifests")
    parser.add_argument("--split", action="store_true", help="Generate patient-wise 80/10/10 stratified splits")
    parser.add_argument("--build", action="store_true", help="Execute 3D volume & 2D slice preprocessing across all 413 subjects")
    parser.add_argument("--force", action="store_true", help="Force reprocessing existing cached volumes")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel worker processes (default 4)")
    parser.add_argument("--max-patients", type=int, default=None, help="Limit number of patients to process (for testing)")
    parser.add_argument("--test-inference", action="store_true", help="Test inference preprocessor on sample volume")
    parser.add_argument("--verify", action="store_true", help="Run dataset verification check")

    args = parser.parse_args()
    setup_logging()

    # Default action if no flags provided: Run FULL end-to-end pipeline
    run_all = not any([args.scan, args.split, args.build, args.test_inference, args.verify])

    start_time = time.time()

    if run_all or args.scan:
        logging.info("Step 1/3: Scanning OASIS-1 dataset across all discs...")
        scanner = OasisDatasetScanner()
        manifest_df = scanner.scan()
        print(f"Scan complete: {len(manifest_df)} total patients indexed.")

    if run_all or args.split:
        logging.info("Step 2/3: Generating patient-wise stratified splits (80% Train / 10% Val / 10% Test)...")
        generator = PatientSplitGenerator()
        train_df, val_df, test_df = generator.generate_splits()
        print(f"Splits complete: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}.")

    if run_all or args.build:
        logging.info("Step 3/3: Executing parallel Dataset Builder across ALL patients (Labelled & Unlabelled)...")
        builder = DatasetBuilder()
        builder.num_workers = args.num_workers
        summary = builder.build_dataset(max_patients=args.max_patients, force=args.force)
        print(f"\nBuild complete: Processed {summary['processed_success']}, Cached {summary['skipped_cached']}, Failed {summary['failed_count']}.")

    if args.test_inference:
        logging.info("Testing Single-Volume Web Inference Preprocessor...")
        preprocessor = InferencePreprocessor()
        test_hdr = r"oasis_cross-sectional_disc1/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0001_MR1_mpr_n4_anon_sbj_111.hdr"
        res = preprocessor.preprocess_file(test_hdr)
        print(f"Inference Preprocessing Successful! 3D Tensor: {res['volume_tensor'].shape}")

    if run_all or args.verify:
        logging.info("Running Dataset Verification Engine...")
        run_verification()

    elapsed = round(time.time() - start_time, 2)
    logging.info(f"Pipeline finished in {elapsed} seconds.")


if __name__ == "__main__":
    main()
