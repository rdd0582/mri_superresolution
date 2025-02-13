import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="MRI Superresolution Project Launcher")
    parser.add_argument('--action', type=str, choices=['train', 'infer', 'extract_full', 'downsample'], required=True,
                        help="Action to perform")
    args, unknown = parser.parse_known_args()

    if args.action == 'train':
        subprocess.run(['python', 'scripts/train.py'] + unknown)
    elif args.action == 'infer':
        subprocess.run(['python', 'scripts/infer.py'] + unknown)
    elif args.action == 'extract_full':
        subprocess.run(['python', 'scripts/extract_full_res.py'] + unknown)
    elif args.action == 'downsample':
        subprocess.run(['python', 'scripts/downsample_extract.py'] + unknown)

if __name__ == '__main__':
    main()
