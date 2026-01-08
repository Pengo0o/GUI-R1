#!/usr/bin/env python3
"""
Convert GUI-R1-3k train.parquet to the required JSON format
Following the exact logic from GUI-R1/verl/utils/dataset.py
"""
import pyarrow.parquet as pq
import json
import os
import math
from PIL import Image

def process_image(image_path, max_pixels, min_pixels):
    """
    Process image following the same logic as dataset.py:52-70
    Resize based on max_pixels and min_pixels constraints
    """
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path

    # Resize if exceeds max_pixels
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    # Resize if below min_pixels
    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image

def construct_user_prompt(text, history, task_type):
    """Construct user prompt based on task type (dataset.py:131-151)"""
    if task_type == 'high':
        prompt_str = (
            f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{text}', with the action history being '{history}'.\n"
            "Please provide the action to perform (enumerate from ['complete', 'close/delete', 'press_home', 'click', 'press_back', 'type', 'select', 'scroll', 'enter']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
            "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
            "<think> ... </think> <answer>[{'action': enum['complete', 'close/delete', 'press_home', 'click', 'press_back', 'type', 'select', 'scroll', 'enter'], 'point': [x, y], 'input_text': 'no input text [default]'}]</answer>\n"
            "Note:\n specific input text (no default) is necessary for actions enum['type', 'select', 'scroll'] \n Example:\n"
            "[{'action': enum['complete', 'close/delete', 'press_home', 'press_back', 'enter'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
            "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
            "[{'action': enum['type', 'select'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]\n"
            "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
        )
    else:  # task_type == 'low' or other
        prompt_str = (
            f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{text}', with the action history being '{history}'.\n"
            "Please provide the action to perform (enumerate from ['click']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
            "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
            "<think> ... </think> <answer>[{'action': enum[ 'click'], 'point': [x, y], 'input_text': 'no input text'}]</answer>\n"
            "Example:\n"
            "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
        )
    return prompt_str

def construct_assistant_response(gt_action, gt_bbox, gt_input_text, scalex, scaley):
    """
    Construct assistant response following dataset.py:155-163 logic
    Scale normalized bbox coordinates by image dimensions
    """
    # Clone bbox to avoid modifying original
    bbox = list(gt_bbox)

    # Scale coordinates (dataset.py:157-161)
    bbox[0] *= scalex
    bbox[1] *= scaley
    if len(bbox) > 2:
        bbox[2] *= scalex
        bbox[3] *= scaley

    # Construct point from first two coordinates
    point = [int(bbox[0]), int(bbox[1])]

    # Construct the response
    response = {
        'action': gt_action,
        'point': point,
        'input_text': gt_input_text
    }

    # Format as JSON
    return json.dumps([response])

def convert_parquet_to_json(input_path, output_path, max_pixels=4194304, min_pixels=262144, base_image_dir=None):
    """
    Convert parquet file to the required JSON format
    Following the exact logic from dataset.py
    """
    # Read parquet file
    table = pq.read_table(input_path)
    data_dict = table.to_pydict()

    # Get number of rows
    num_rows = table.shape[0]

    result = []

    for i in range(num_rows):
        # Extract row data
        image_path = data_dict['image'][i]
        gt_bbox = data_dict['gt_bbox'][i]
        instruction = data_dict['instruction'][i]
        gt_action = data_dict['gt_action'][i]
        gt_input_text = data_dict['gt_input_text'][i]
        history = data_dict['history'][i] if data_dict['history'][i] is not None else 'None'
        task_type = data_dict['task_type'][i]

        # Construct full image path
        if base_image_dir:
            full_image_path = os.path.join(base_image_dir, image_path)
        else:
            # Use relative path from parquet file location
            parquet_dir = os.path.dirname(input_path)
            full_image_path = os.path.join(parquet_dir, image_path)

        # Process image (following dataset.py:153)
        try:
            processed_image = process_image(full_image_path, max_pixels, min_pixels)
            scalex, scaley = processed_image.size  # dataset.py:155
        except Exception as e:
            print(f"Warning: Could not process image {full_image_path}: {e}")
            continue

        # Construct user prompt
        user_content = construct_user_prompt(instruction, history, task_type)

        # Construct assistant response with scaled coordinates
        assistant_content = construct_assistant_response(gt_action, gt_bbox, gt_input_text, scalex, scaley)

        # Create entry in required format
        entry = {
            "messages": [
                {
                    "content": user_content,
                    "role": "user"
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ],
            "images": [full_image_path]
        }

        result.append(entry)

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{num_rows} entries...")

    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\nConverted {len(result)} entries from {input_path} to {output_path}")
    return result

if __name__ == "__main__":
    input_file = "/opt/data/private/hyp/Gui-Agent/Data/GUI-R1-3k/train.parquet"
    output_file = "/opt/data/private/hyp/Gui-Agent/Data/GUI-R1-3k/train_converted.json"

    # Use the default values from config.py
    # max_pixels = 4194304 (default)
    # min_pixels = 262144 (default)
    # Or use the values from examples: max_pixels = 1258291

    convert_parquet_to_json(
        input_file,
        output_file,
        max_pixels=4194304,  # Adjust if needed
        min_pixels=262144    # Adjust if needed
    )

    print("\nSample output (first entry):")
    with open(output_file, 'r') as f:
        data = json.load(f)
        print(json.dumps(data[0], indent=2, ensure_ascii=False))
