#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert GUI-R1 parquet files to ms-swift compatible JSON format.

This script converts train.parquet and test.parquet from the GUI-R1 dataset
into JSON files that can be used with ms-swift framework.
Images are saved to a separate folder and referenced by path in the JSON.
"""

import argparse
import json
import os
from pathlib import Path

import datasets
from PIL import Image
from tqdm import tqdm


def convert_parquet_to_json(parquet_path, output_path, split_name, output_dir):
    """
    Convert a parquet file to JSON format compatible with ms-swift.

    Args:
        parquet_path: Path to the input parquet file
        output_path: Path to the output JSON file
        split_name: Name of the split (train/val) for logging
        output_dir: Base output directory for saving images
    """
    print(f"Loading {split_name} dataset from {parquet_path}...")
    dataset = datasets.Dataset.from_parquet(parquet_path)

    print(f"Dataset loaded. Total samples: {len(dataset)}")
    print(f"Dataset features: {dataset.features}")

    # Create images directory
    images_dir = os.path.join(output_dir, 'images', split_name)
    os.makedirs(images_dir, exist_ok=True)
    print(f"Images will be saved to: {images_dir}")

    # Convert dataset to list of dictionaries
    data_list = []

    print(f"Converting {split_name} dataset...")
    for idx, example in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
        # Keep all original fields that will be processed by GuiR1Preprocessor
        # The preprocess method expects: instruction, task_type, history, image,
        # gt_action, gt_input_text, gt_bbox

        data_item = {
            'instruction': example.get('instruction', ''),
            'task_type': example.get('task_type', 'low'),
            'history': example.get('history', ''),
            'gt_action': example.get('gt_action', ''),
            'gt_input_text': example.get('gt_input_text', 'no input text'),
            'gt_bbox': example.get('gt_bbox', [-100, -100]),
        }

        # Handle image field - save to file and store path
        if 'image' in example and example['image'] is not None:
            image_data = example['image']

            # Generate image filename
            image_filename = f"{split_name}_{idx:06d}.png"
            image_path = os.path.join(images_dir, image_filename)

            # Save image
            try:
                if isinstance(image_data, dict) and 'bytes' in image_data:
                    # Image is stored as {'bytes': b'...'}
                    import io
                    pil_image = Image.open(io.BytesIO(image_data['bytes']))
                    pil_image.save(image_path)
                elif isinstance(image_data, Image.Image):
                    # If it's already a PIL Image
                    image_data.save(image_path)
                else:
                    # Try to convert to PIL Image
                    pil_image = Image.fromarray(image_data) if hasattr(image_data, 'shape') else image_data
                    pil_image.save(image_path)

                # Store relative path in JSON (relative to output_dir)
                relative_path = os.path.relpath(image_path, output_dir)
                data_item['image'] = relative_path
            except Exception as e:
                print(f"Warning: Could not save image at index {idx}: {e}")
                # Don't add image path if save failed

        # Optional fields
        if 'id' in example:
            data_item['id'] = example['id']
        if 'verify_bbox' in example:
            data_item['verify_bbox'] = example.get('verify_bbox')
        if 'success_rate' in example:
            data_item['success_rate'] = example.get('success_rate')
        if 'scale' in example:
            data_item['scale'] = example.get('scale')

        data_list.append(data_item)

    # Save to JSON
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

    print(f" Successfully converted {len(data_list)} samples to {output_path}")
    print(f" Saved {len(data_list)} images to {images_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert GUI-R1 parquet files to ms-swift JSON format'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/opt/data/private/hyp/Gui-Agent/Data/GUI-R1',
        help='Directory containing train.parquet and test.parquet'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/opt/data/private/hyp/Gui-Agent/Data/GUI-R1-swift',
        help='Directory to save converted JSON files'
    )
    parser.add_argument(
        '--train_file',
        type=str,
        default='train.parquet',
        help='Name of the training parquet file'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default='test.parquet',
        help='Name of the test parquet file'
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("GUI-R1 Dataset Conversion to ms-swift Format")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    # Convert train set
    train_parquet = os.path.join(args.input_dir, args.train_file)
    train_json = os.path.join(args.output_dir, 'train.json')

    if os.path.exists(train_parquet):
        convert_parquet_to_json(train_parquet, train_json, 'train', args.output_dir)
    else:
        print(f"Warning: {train_parquet} not found, skipping train set conversion")

    print()

    # Convert test set (rename to val for ms-swift convention)
    test_parquet = os.path.join(args.input_dir, args.test_file)
    val_json = os.path.join(args.output_dir, 'val.json')

    if os.path.exists(test_parquet):
        convert_parquet_to_json(test_parquet, val_json, 'val', args.output_dir)
    else:
        print(f"Warning: {test_parquet} not found, skipping test set conversion")

    print()
    print("="*80)
    print("Conversion completed!")
    print(f"Output files saved in: {args.output_dir}")
    print("  - train.json (training data)")
    print("  - val.json (validation data)")
    print("  - images/train/ (training images)")
    print("  - images/val/ (validation images)")
    print("="*80)
    print()
    print("You can now use this dataset with ms-swift by specifying:")
    print(f"  --dataset_path {args.output_dir}")


if __name__ == '__main__':
    main()