# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GUI-R1 dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


def build_system_prompt(task_type):
    """
    Build the system prompt based on task type (high or low).
    System prompt contains general instructions and format requirements.
    """
    if task_type == 'high':
        system_prompt = (
            "You are GUI-R1, a reasoning GUI Agent Assistant. "
            "Please provide the action to perform (enumerate from ['complete', 'close/delete', 'press_home', 'click', 'press_back', 'type', 'select', 'scroll', 'enter']), "
            "the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action. "
            "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows: "
            "<think> ... </think> <answer>[{'action': enum['complete', 'close/delete', 'press_home', 'click', 'press_back', 'type', 'select', 'scroll', 'enter'], 'point': [x, y], 'input_text': 'no input text [default]'}]</answer> "
            "Note: specific input text (no default) is necessary for actions enum['type', 'select', 'scroll']. "
            "Examples: "
            "[{'action': enum['complete', 'close/delete', 'press_home', 'press_back', 'enter'], 'point': [-100, -100], 'input_text': 'no input text'}] "
            "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}] "
            "[{'action': enum['type', 'select'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}] "
            "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
        )
    else:
        system_prompt = (
            "You are GUI-R1, a reasoning GUI Agent Assistant. "
            "Please provide the action to perform (enumerate from ['click']), "
            "the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action. "
            "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows: "
            "<think> ... </think> <answer>[{'action': enum['click'], 'point': [x, y], 'input_text': 'no input text'}]</answer> "
            "Example: "
            "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]"
        )
    return system_prompt


def build_user_prompt(instruction, history):
    """
    Build the user prompt with specific instruction and history.
    """
    user_prompt = f"In this UI screenshot <image>, I want you to continue executing the command '{instruction}', with the action history being '{history}'."
    return user_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_dataset_path",
                        default="/opt/data/private/hyp/Gui-Agent/Data/GUI-R1/train.parquet",
                        help="The local path to the training dataset parquet file.")
    parser.add_argument("--test_dataset_path",
                        default="/opt/data/private/hyp/Gui-Agent/Data/GUI-R1/test.parquet",
                        help="The local path to the test dataset parquet file.")
    parser.add_argument(
        "--local_save_dir", default="/opt/data/private/hyp/Gui-Agent/Data/GUI-R1-Verl", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()

    data_source = "GUI-R1"

    # Load datasets from parquet files
    train_dataset = datasets.Dataset.from_parquet(args.train_dataset_path)
    test_dataset = datasets.Dataset.from_parquet(args.test_dataset_path)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            instruction_raw = example.pop("instruction")

            # Get additional fields
            task_type = example.pop("task_type", "unknown")
            history = example.pop("history", "")
            image = example.pop("image", None)

            # Extract ground truth information
            gt_action = example.pop("gt_action")
            gt_input_text = example.pop("gt_input_text")
            gt_bbox = example.pop("gt_bbox")

            # Build system and user prompts
            system_prompt = build_system_prompt(task_type)
            user_prompt = build_user_prompt(instruction_raw, history)

            # Build ground_truth as simple string (matching gsm8k style)
            ground_truth = f"action: {gt_action}, point: {gt_bbox}, input_text: {gt_input_text}"

            example_id = example.pop("id", idx)

            # Remove optional fields that may exist
            example.pop("verify_bbox", None)
            example.pop("success_rate", None)
            example.pop("scale", None)

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                "ability": "gui_grounding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "id": example_id,
                    "instruction": instruction_raw,
                    "task_type": task_type,
                    "history": history,
                    "image": image,
                    "gt_action": gt_action,
                    "gt_input_text": gt_input_text,
                    "gt_bbox": gt_bbox,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    # Expand user path
    local_save_dir = os.path.expanduser(local_save_dir)

    # Create directory if it doesn't exist
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
