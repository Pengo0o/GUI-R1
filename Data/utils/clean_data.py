#!/usr/bin/env python3
"""
Data cleaning script for GUI-R1 dataset.
Cleans train.parquet and test.parquet and saves to GUI-R1-3k directory.
Extracts images and saves them to separate folders.
"""

import pandas as pd
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm

def save_images_and_update_df(df, output_dir, split_name):
    """
    Extract images from dataframe and save to disk.
    Update dataframe to store image paths instead of bytes.

    Args:
        df: DataFrame with image column containing bytes
        output_dir: Output directory path
        split_name: 'train' or 'test'

    Returns:
        Updated dataframe with image paths
    """
    # Create image directory
    img_dir = output_dir / split_name
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting and saving images to {img_dir}...")

    # Create a copy to modify
    df_copy = df.copy()
    image_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Saving {split_name} images"):
        # Get image bytes
        img_data = row['image']

        # Determine image format and extension
        if isinstance(img_data, dict) and 'bytes' in img_data:
            img_bytes = img_data['bytes']
        else:
            img_bytes = img_data

        # Create image filename based on id
        img_id = row['id']
        img_filename = f"{img_id}.jpg"
        img_path = img_dir / img_filename

        # Save image
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.save(img_path, 'JPEG')
            # Store relative path
            image_paths.append(f"{split_name}/{img_filename}")
        except Exception as e:
            print(f"\nWarning: Failed to save image for id {img_id}: {e}")
            image_paths.append(None)

    # Update dataframe with image paths
    df_copy['image'] = image_paths

    return df_copy

def clean_and_save_data():
    # Define paths
    input_dir = Path('/opt/data/private/hyp/Gui-Agent/Data/GUI-R1')
    output_dir = Path('/opt/data/private/hyp/Gui-Agent/Data/GUI-R1-3k')

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Process train.parquet
    print("\n" + "="*50)
    print("Processing train.parquet...")
    print("="*50)

    train_df = pd.read_parquet(input_dir / 'train.parquet')
    print(f"Original train shape: {train_df.shape}")
    print(f"Columns: {train_df.columns.tolist()}")
    print(f"\nNull values:\n{train_df.isnull().sum()}")

    # Clean train data
    train_cleaned = train_df.copy()

    # Remove duplicates based on 'id' column
    original_len = len(train_cleaned)
    train_cleaned = train_cleaned.drop_duplicates(subset=['id'], keep='first')
    print(f"\nRemoved {original_len - len(train_cleaned)} duplicate rows based on 'id'")

    # Remove rows with null values in critical columns
    critical_cols = ['image', 'instruction', 'id']
    train_cleaned = train_cleaned.dropna(subset=critical_cols)
    print(f"Rows after removing nulls in critical columns: {len(train_cleaned)}")

    # Extract and save images
    # train_cleaned = save_images_and_update_df(train_cleaned, output_dir, 'train')

    # Save cleaned train data
    output_train_path = output_dir / 'train.parquet'
    # train_cleaned.to_parquet(output_train_path, index=False)
    print(f"\nSaved cleaned train data to: {output_train_path}")
    print(f"Final train shape: {train_cleaned.shape}")

    # Process test.parquet
    print("\n" + "="*50)
    print("Processing test.parquet...")
    print("="*50)

    test_df = pd.read_parquet(input_dir / 'test.parquet')
    print(f"Original test shape: {test_df.shape}")
    print(f"Columns: {test_df.columns.tolist()}")
    print(f"\nNull values:\n{test_df.isnull().sum()}")

    # Clean test data
    test_cleaned = test_df.copy()

    # Remove duplicates based on 'id' column
    original_len = len(test_cleaned)
    test_cleaned = test_cleaned.drop_duplicates(subset=['id'], keep='first')
    print(f"\nRemoved {original_len - len(test_cleaned)} duplicate rows based on 'id'")

    # Remove rows with null values in critical columns
    test_cleaned = test_cleaned.dropna(subset=critical_cols)
    print(f"Rows after removing nulls in critical columns: {len(test_cleaned)}")

    # Extract and save images
    # test_cleaned = save_images_and_update_df(test_cleaned, output_dir, 'test')

    # Save cleaned test data
    output_test_path = output_dir / 'test.parquet'
    # test_cleaned.to_parquet(output_test_path, index=False)
    print(f"\nSaved cleaned test data to: {output_test_path}")
    print(f"Final test shape: {test_cleaned.shape}")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Train: {train_df.shape[0]} -> {train_cleaned.shape[0]} rows")
    print(f"Test:  {test_df.shape[0]} -> {test_cleaned.shape[0]} rows")
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")
    print(f"  - {output_train_path}")
    print(f"  - {output_test_path}")
    print(f"  - Images saved in {output_dir}/train/")
    print(f"  - Images saved in {output_dir}/test/")

if __name__ == "__main__":
    clean_and_save_data()