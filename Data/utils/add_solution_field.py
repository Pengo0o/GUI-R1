#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add solution field to GUI-R1 JSON dataset.

This script adds a 'solution' field to existing JSON files, which combines
gt_action, gt_bbox, and gt_input_text into a single dictionary.
This is required for the reward function in GRPO training.
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm


def add_solution_field(input_path, output_path=None, inplace=False):
    """
    Add solution field to JSON dataset.

    Args:
        input_path: Path to the input JSON file
        output_path: Path to the output JSON file (if None and not inplace, adds _with_solution suffix)
        inplace: If True, modify the file in place (overwrite input file)
    """
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    # Process each sample
    print("Adding solution field to each sample...")
    for item in tqdm(data):
        # Extract ground truth fields
        gt_action = item.get('gt_action', '')
        gt_bbox = item.get('gt_bbox', [-100, -100])
        gt_input_text = item.get('gt_input_text', 'no input text')

        # Create solution dictionary
        item['solution'] = {
            'action': gt_action,
            'bbox': gt_bbox,
            'input_text': gt_input_text,
        }

    # Determine output path
    if inplace:
        save_path = input_path
    elif output_path:
        save_path = output_path
    else:
        # Add _with_solution suffix to filename
        input_file = Path(input_path)
        save_path = input_file.parent / f"{input_file.stem}_with_solution{input_file.suffix}"

    # Save modified data
    print(f"Saving to {save_path}...")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ Successfully processed {len(data)} samples")
    print(f"✓ Output saved to: {save_path}")

    # Print sample
    if len(data) > 0:
        print("\nSample output:")
        print(f"  Keys: {list(data[0].keys())}")
        print(f"  Solution: {data[0]['solution']}")


def main():
    parser = argparse.ArgumentParser(
        description='Add solution field to GUI-R1 JSON dataset'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output JSON file (optional, default: adds _with_solution suffix)'
    )
    parser.add_argument(
        '--inplace',
        action='store_true',
        help='Modify the file in place (overwrite input file)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch process train.json and val.json in a directory'
    )

    args = parser.parse_args()

    print("="*80)
    print("Adding solution field to GUI-R1 dataset")
    print("="*80)

    if args.batch:
        # Batch mode: process both train.json and val.json
        input_dir = Path(args.input)
        if not input_dir.is_dir():
            print(f"Error: {args.input} is not a directory")
            return

        for filename in ['train.json', 'val.json']:
            file_path = input_dir / filename
            if file_path.exists():
                print(f"\nProcessing {filename}...")
                add_solution_field(
                    str(file_path),
                    output_path=args.output,
                    inplace=args.inplace
                )
            else:
                print(f"Warning: {file_path} not found, skipping")
    else:
        # Single file mode
        add_solution_field(
            args.input,
            output_path=args.output,
            inplace=args.inplace
        )

    print("\n" + "="*80)
    print("Processing completed!")
    print("="*80)


if __name__ == '__main__':
    main()
