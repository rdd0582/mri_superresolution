#!/usr/bin/env python
import os
import sys
import time
import json
import argparse
import platform
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import subprocess

# Import project modules (adjust as needed)
from utils.dataset import MRISuperResDataset
from utils.losses import SSIM
from scripts.infer import load_model, preprocess_image, postprocess_tensor, calculate_metrics
from scripts.test_comparison import upscale_with_interpolation, calculate_metrics as calc_metrics
from utils.visualise_res import analyze_resolutions, visualize_resolution_histogram

def report_hardware():
    import psutil
    info = {
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "cuda": torch.version.cuda if torch.cuda.is_available() else "None"
    }
    print("Hardware Info:", info)
    return info

def report_dataset_stats(hr_dir, lr_dir, output_dir):
    dataset = MRISuperResDataset(full_res_dir=hr_dir, low_res_dir=lr_dir, augmentation=False)
    n_images = len(dataset)
    subjects = dataset.get_unique_subjects()
    print(f"Test images: {n_images}, Subjects: {len(subjects)}")
    # Analyze resolutions
    resolutions = []
    for meta in dataset.metadata:
        img = np.array(postprocess_tensor(preprocess_image(meta['full_res_path'])[1]))
        resolutions.append((img.shape[1], img.shape[0]))
    df_res = analyze_resolutions(resolutions)
    hist_path = os.path.join(output_dir, "resolution_histogram.png")
    visualize_resolution_histogram(df_res, hist_path)
    return {"n_images": n_images, "subjects": subjects, "resolution_histogram": hist_path}

def report_implementation_details(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    params = {}
    if "hyperparameters" in checkpoint:
        params = checkpoint["hyperparameters"]
    else:
        # Try to extract common hyperparameters
        for k in ["epochs", "batch_size", "learning_rate", "ssim_weight", "perceptual_weight", "base_filters"]:
            if k in checkpoint:
                params[k] = checkpoint[k]
    print("Implementation details:", params)
    return params

def run_benchmarks(test_pairs, model, device):
    results = []
    for lr_path, hr_path in tqdm(test_pairs, desc="Evaluating"):
        # Load images
        hr_img = np.array(postprocess_tensor(preprocess_image(hr_path)[1])).astype(np.float32) / 255.0
        lr_img = np.array(postprocess_tensor(preprocess_image(lr_path)[1])).astype(np.float32) / 255.0

        # Bicubic
        t0 = time.time()
        bicubic = upscale_with_interpolation(str(lr_path), "bicubic")
        t1 = time.time()
        metrics_bicubic = calc_metrics(hr_img, bicubic)
        metrics_bicubic["method"] = "bicubic"
        metrics_bicubic["time"] = t1 - t0

        # Bilinear
        t0 = time.time()
        bilinear = upscale_with_interpolation(str(lr_path), "bilinear")
        t1 = time.time()
        metrics_bilinear = calc_metrics(hr_img, bilinear)
        metrics_bilinear["method"] = "bilinear"
        metrics_bilinear["time"] = t1 - t0

        # Sharp Bilinear
        t0 = time.time()
        sharp_bilinear = upscale_with_interpolation(str(lr_path), "sharp_bilinear")
        t1 = time.time()
        metrics_sharp = calc_metrics(hr_img, sharp_bilinear)
        metrics_sharp["method"] = "sharp_bilinear"
        metrics_sharp["time"] = t1 - t0

        # UNetSuperRes
        t0 = time.time()
        _, lr_tensor = preprocess_image(str(lr_path))
        with torch.no_grad():
            sr_tensor = model(lr_tensor.to(device))
        sr_img = sr_tensor.squeeze().cpu().numpy()
        t1 = time.time()
        metrics_sr = calc_metrics(hr_img, sr_img)
        metrics_sr["method"] = "unet"
        metrics_sr["time"] = t1 - t0

        # Collect all
        for m in [metrics_bicubic, metrics_bilinear, metrics_sharp, metrics_sr]:
            m["image"] = os.path.basename(str(lr_path))
            results.append(m)
    return pd.DataFrame(results)

def qualitative_comparison(test_pairs, model, device, output_dir):
    from scripts.test_comparison import visualize_results
    os.makedirs(output_dir, exist_ok=True)
    for i, (lr_path, hr_path) in enumerate(test_pairs[:5]):
        hr_img = np.array(postprocess_tensor(preprocess_image(hr_path)[1])).astype(np.float32) / 255.0
        lr_img = np.array(postprocess_tensor(preprocess_image(lr_path)[1])).astype(np.float32) / 255.0
        bicubic = upscale_with_interpolation(str(lr_path), "bicubic")
        _, lr_tensor = preprocess_image(str(lr_path))
        with torch.no_grad():
            sr_tensor = model(lr_tensor.to(device))
        sr_img = sr_tensor.squeeze().cpu().numpy()
        upscaled_images = {
            "bicubic": bicubic,
            "unet": sr_img
        }
        metrics = {
            "bicubic": calc_metrics(hr_img, bicubic),
            "unet": calc_metrics(hr_img, sr_img)
        }
        visualize_results(hr_img, lr_img, upscaled_images, metrics, os.path.join(output_dir, f"qualitative_{i}.png"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_dir", type=str, help="Directory with test HR images")
    parser.add_argument("--lr_dir", type=str, help="Directory with test LR images")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--checkpoint", type=str, help="Path to a single model checkpoint")
    group.add_argument("--ablation_checkpoints_dir", type=str, help="Directory containing multiple checkpoints for ablation study. Base filters will be fixed to 32.")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Where to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--base_filters", type=int, default=32, help="Base number of filters for UNet (ignored if --ablation_checkpoints_dir is used)")
    # New: ablation training configs
    parser.add_argument("--ablation_train_configs", type=str, help="Path to JSON file with ablation configs (list of dicts with loss weights)")
    parser.add_argument("--train_epochs", type=int, default=100, help="Epochs for ablation training")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for ablation training")
    parser.add_argument("--train_num_workers", type=int, default=4, help="Num workers for ablation training")
    parser.add_argument("--train_learning_rate", type=float, default=1e-4, help="Learning rate for ablation training")
    args = parser.parse_args()

    # If ablation_train_configs is provided, automate training and evaluation
    if args.ablation_train_configs:
        import json
        with open(args.ablation_train_configs, 'r') as f:
            ablation_configs = json.load(f)
        checkpoints_dir = './ablation_checkpoints'
        os.makedirs(checkpoints_dir, exist_ok=True)
        all_results_dfs = []
        ablation_summary = {}
        for config in ablation_configs:
            ssim = config.get('ssim_weight', 0.0)
            perc = config.get('perceptual_weight', 0.0)
            # Unique name for this config
            config_name = f"ssim_{ssim}_perc_{perc}"
            ckpt_dir = os.path.join(checkpoints_dir, config_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            # Train the model
            train_cmd = [
                sys.executable, 'scripts/train.py',
                '--full_res_dir', './training_data',
                '--low_res_dir', './training_data_1.5T',
                '--model_type', 'unet',
                '--base_filters', '32',
                '--checkpoint_dir', ckpt_dir,
                '--epochs', str(args.train_epochs),
                '--batch_size', str(args.train_batch_size),
                '--learning_rate', str(args.train_learning_rate),
                '--num_workers', str(args.train_num_workers),
                '--ssim_weight', str(ssim),
                '--perceptual_weight', str(perc)
            ]
            print(f"\nTraining config: {config_name}")
            subprocess.run(train_cmd, check=True)
            # Find the best checkpoint
            best_ckpt = os.path.join(ckpt_dir, 'best_model_unet.pth')
            if not os.path.exists(best_ckpt):
                # Fallback to final model
                best_ckpt = os.path.join(ckpt_dir, 'final_model_unet.pth')
            if not os.path.exists(best_ckpt):
                print(f"No checkpoint found for {config_name}, skipping evaluation.")
                continue
            # Evaluate
            print(f"Evaluating config: {config_name}")
            # Set up output dir for this config
            eval_output_dir = os.path.join(args.output_dir, config_name)
            os.makedirs(eval_output_dir, exist_ok=True)
            # Hardware/dataset/impl reporting
            hw = report_hardware()
            ds = report_dataset_stats('./test_results/hr', './test_results/lr', eval_output_dir)
            impl = report_implementation_details(best_ckpt)
            with open(os.path.join(eval_output_dir, "report.json"), "w") as f:
                json.dump({"hardware": hw, "dataset": ds, "implementation": impl}, f, indent=2)
            # Load model
            model = load_model("unet", best_ckpt, args.device, base_filters=32)
            model.eval()
            # Gather test pairs
            dataset = MRISuperResDataset(full_res_dir='./test_results/hr', low_res_dir='./test_results/lr', augmentation=False)
            test_pairs = [(meta['low_res_path'], meta['full_res_path']) for meta in dataset.metadata]
            # Quantitative evaluation
            df = run_benchmarks(test_pairs, model, args.device)
            df['checkpoint'] = config_name
            for key, value in impl.items():
                col_name = str(key).replace('.', '_').replace(' ', '_')
                if isinstance(value, (list, dict)):
                    df[col_name] = str(value)
                else:
                    df[col_name] = value
            all_results_dfs.append(df)
            metrics_path = os.path.join(eval_output_dir, "metrics.csv")
            df.to_csv(metrics_path, index=False)
            # Qualitative analysis
            qualitative_output_dir = os.path.join(eval_output_dir, "qualitative")
            qualitative_comparison(test_pairs, model, args.device, qualitative_output_dir)
            ablation_summary[config_name] = impl
            del model
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        # Aggregate results
        if all_results_dfs:
            final_df = pd.concat(all_results_dfs, ignore_index=True)
            metrics_path = os.path.join(args.output_dir, "metrics_ablation.csv")
            final_df.to_csv(metrics_path, index=False)
            print(f"\nSaved aggregated ablation metrics to {metrics_path}")
            summary_path = os.path.join(args.output_dir, "ablation_summary.json")
            with open(summary_path, "w") as f:
                json.dump(ablation_summary, f, indent=2)
            print(f"Saved ablation summary to {summary_path}")
        else:
            print("No results generated during ablation study.")
        print("\nAblation training and evaluation complete. See", args.output_dir)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    if args.ablation_checkpoints_dir:
        print("Running ablation study. Base filters fixed to 32.")
        base_filters = 32
    else:
        base_filters = args.base_filters

    # 1. Hardware, dataset, implementation reporting (initial report)
    hw = report_hardware()
    ds = report_dataset_stats(args.hr_dir, args.lr_dir, args.output_dir)
    # Implementation details reported per checkpoint in ablation mode
    if args.checkpoint:
        impl = report_implementation_details(args.checkpoint)
        with open(os.path.join(args.output_dir, "report.json"), "w") as f:
            json.dump({"hardware": hw, "dataset": ds, "implementation": impl}, f, indent=2)
    else: # Ablation mode - save initial report without specific implementation
         with open(os.path.join(args.output_dir, "report_base.json"), "w") as f:
            json.dump({"hardware": hw, "dataset": ds}, f, indent=2)


    # 2. Load model (handled within loop for ablation)

    # 3. Gather test pairs
    dataset = MRISuperResDataset(full_res_dir=args.hr_dir, low_res_dir=args.lr_dir, augmentation=False)
    test_pairs = [(meta['low_res_path'], meta['full_res_path']) for meta in dataset.metadata]

    if args.checkpoint:
        # --- Single Checkpoint Evaluation ---
        print(f"Evaluating single checkpoint: {args.checkpoint}")
        model = load_model("unet", args.checkpoint, args.device, base_filters=base_filters)
        model.eval()

        # 4. Quantitative evaluation
        df = run_benchmarks(test_pairs, model, args.device)
        # Add checkpoint info if needed, though less critical for single run
        df['checkpoint'] = os.path.basename(args.checkpoint)
        metrics_path = os.path.join(args.output_dir, "metrics.csv")
        df.to_csv(metrics_path, index=False)
        print(f"Saved metrics to {metrics_path}")

        # 5. Qualitative analysis
        qualitative_output_dir = os.path.join(args.output_dir, "qualitative")
        qualitative_comparison(test_pairs, model, args.device, qualitative_output_dir)
        print(f"Saved qualitative results to {qualitative_output_dir}")

    elif args.ablation_checkpoints_dir:
        # --- Ablation Study ---
        print(f"Starting ablation study from directory: {args.ablation_checkpoints_dir}")
        checkpoint_files = sorted([p for p in Path(args.ablation_checkpoints_dir).glob('*.ckpt')] + \
                                [p for p in Path(args.ablation_checkpoints_dir).glob('*.pth')])

        if not checkpoint_files:
            print(f"Error: No checkpoint files (.ckpt or .pth) found in {args.ablation_checkpoints_dir}")
            sys.exit(1)

        all_results_dfs = []
        ablation_summary = {}

        for ckpt_path in checkpoint_files:
            ckpt_name = ckpt_path.name
            print(f"\n--- Evaluating Checkpoint: {ckpt_name} ---")

            # Report implementation details for this checkpoint
            impl_details = report_implementation_details(str(ckpt_path))
            ablation_summary[ckpt_name] = impl_details

            # Load model (fixed base_filters=32)
            model = load_model("unet", str(ckpt_path), args.device, base_filters=32) # Fixed base_filters
            model.eval()

            # Run benchmarks
            df_ckpt = run_benchmarks(test_pairs, model, args.device)

            # Add config details to the dataframe
            df_ckpt['checkpoint'] = ckpt_name
            for key, value in impl_details.items():
                 # Ensure column names are valid (e.g., handle potential dots or spaces if keys are complex)
                 col_name = str(key).replace('.', '_').replace(' ', '_')
                 # Handle non-scalar values if necessary (e.g., lists - maybe convert to string)
                 if isinstance(value, (list, dict)):
                     df_ckpt[col_name] = str(value)
                 else:
                     df_ckpt[col_name] = value

            all_results_dfs.append(df_ckpt)

            # Run qualitative comparison in a config-specific directory
            # Create a safe dirname from checkpoint name or key params
            qual_subdir_name = ckpt_name.replace('.ckpt', '').replace('.pth', '') # Basic naming
            # Or use hyperparameters for naming if available and simple:
            # qual_subdir_name = f"ssim_{impl_details.get('ssim_weight', 'N/A')}_perc_{impl_details.get('perceptual_weight', 'N/A')}"

            qualitative_output_dir = os.path.join(args.output_dir, "qualitative", qual_subdir_name)
            qualitative_comparison(test_pairs, model, args.device, qualitative_output_dir)
            print(f"Saved qualitative results for {ckpt_name} to {qualitative_output_dir}")

            # Clear GPU memory if applicable
            del model
            if args.device == 'cuda':
                torch.cuda.empty_cache()


        # Aggregate results
        if all_results_dfs:
            final_df = pd.concat(all_results_dfs, ignore_index=True)
            metrics_path = os.path.join(args.output_dir, "metrics_ablation.csv")
            final_df.to_csv(metrics_path, index=False)
            print(f"\nSaved aggregated ablation metrics to {metrics_path}")

            # Save the summary of evaluated checkpoints and their configs
            summary_path = os.path.join(args.output_dir, "ablation_summary.json")
            with open(summary_path, "w") as f:
                 json.dump(ablation_summary, f, indent=2)
            print(f"Saved ablation summary to {summary_path}")

        else:
            print("No results generated during ablation study.")


    print("\nEvaluation complete. See", args.output_dir)

if __name__ == "__main__":
    main() 