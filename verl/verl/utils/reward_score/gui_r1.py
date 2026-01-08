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
Reward scoring functions for GUI-R1 dataset.
Adapted from GUI-R1 project's r1gui.py
"""

import ast
import json
import re


def calculate_f1_score(predicted_str, ground_truth_str):
    """Calculate F1 score between predicted and ground truth strings."""
    predicted_str = predicted_str.replace("[", "").replace("]", "")
    ground_truth_str = ground_truth_str.replace("[", "").replace("]", "")
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    if len(predicted_tokens) == 1 and len(ground_truth_tokens) == 1:
        predicted_token = list(predicted_tokens)[0]
        ground_truth_token = list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return 1

    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def extract_action(content):
    """Extract action from the content within <answer> tags."""
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r"'action':\s*'(\w+)'"
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
    return "no action"


def extract_input_text(content):
    """Extract input text from the content within <answer> tags."""
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r"'input_text':\s*'(.*?)'"
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
    return "no input text"


def extract_coord(content):
    """Extract coordinates from the content within <answer> tags."""
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    try:
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            coord_match = re.search(bbox_pattern, content_answer)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        else:
            coord_pattern = r'\{.*\((\d+),\s*(\d+))\s*.*\}'
            coord_match = re.search(coord_pattern, content)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        return [0, 0, 0, 0], False
    except:
        return [0, 0, 0, 0], False


def format_reward(predict_str: str) -> float:
    """
    Check if predict_str follows the format: <think></think><answer></answer>
    and verify the content in <answer> matches the expected structure.
    """
    # Check outer structure with <think> and <answer>
    outer_pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    if not re.fullmatch(outer_pattern, predict_str):
        return 0.0

    # Extract content from <answer>
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    if not answer_match:
        return 0.0

    # Parse the content in <answer> as JSON format
    answer_content = answer_match.group(1).strip()
    try:
        actions = ast.literal_eval(answer_content)  # Parse <answer> content safely

        # Verify actions is a list
        if not isinstance(actions, list):
            return 0.0

        # Verify format of each action
        for action in actions:
            if not isinstance(action, dict):
                return 0.0
            # Check if action dict contains required keys
            if "action" not in action or "point" not in action or "input_text" not in action:
                return 0.0
            # Verify values meet requirements
            if not isinstance(action["action"], str):
                return 0.0
            if not (isinstance(action["point"][0], int) and isinstance(action["point"][1], int)):
                return 0.0
            if not isinstance(action["input_text"], str):
                return 0.0
            if action["action"] in ['type', 'select', 'open_app'] and action["input_text"] in ['no input text']:
                return 0.0
            if action["action"] in ['scroll'] and action["input_text"] not in ['left', 'right', 'up', 'down']:
                return 0.0

        # If all checks pass, return 1.0
        return 1.0
    except: 
        return 0.0


def accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """
    Compare actions and parameters between predict_str and ground_truth.

    Args:
        predict_str: The predicted string with <think> and <answer> tags
        ground_truth: JSON string or dict containing action, gt_bbox, and input_text
    """
    try:
        # Parse ground_truth
        if isinstance(ground_truth, str):
            # Handle both JSON format and simple string format
            if ground_truth.startswith('{'):
                # JSON format: {"action": "click", "gt_bbox": [x, y], "input_text": "text"}
                ground_truth = json.loads(ground_truth)
                gt_action = ground_truth['action'].lower()
                gt_bbox = ground_truth['gt_bbox']
                gt_input_text = ground_truth['input_text']
            else:
                # Simple string format: "action: click, point: [x, y], input_text: text"
                # Parse the simple format
                action_match = re.search(r"action:\s*(\w+)", ground_truth)
                point_match = re.search(r"point:\s*\[([\d\.,\s]+)\]", ground_truth)
                text_match = re.search(r"input_text:\s*(.+?)(?:,|$)", ground_truth)

                if not action_match:
                    return 0.0

                gt_action = action_match.group(1).lower()
                gt_bbox = ast.literal_eval(point_match.group(0).replace("point:", "").strip()) if point_match else []
                gt_input_text = text_match.group(1).strip() if text_match else "no input text"
        else:
            # Dict format
            gt_action = ground_truth['action'].lower()
            gt_bbox = ground_truth['gt_bbox']
            gt_input_text = ground_truth['input_text']

        # Extract predicted values
        pred_action = extract_action(predict_str).lower()
        pred_input_text = extract_input_text(predict_str)
        pred_bbox, _ = extract_coord(predict_str)

        # Check if actions match
        if pred_action != gt_action:
            return 0.0

        # Check based on action type
        if gt_action in ["click"]:
            if len(gt_bbox) == 2:
                # Point-based click - check if within radius
                if (pred_bbox[0] - gt_bbox[0]) ** 2 + (pred_bbox[1] - gt_bbox[1]) ** 2 < 140 ** 2:
                    return 1.0
                else:
                    return 0.0
            elif len(gt_bbox) == 4:
                # Bounding box click - check if within box
                if (gt_bbox[0] < pred_bbox[0] < gt_bbox[2]) and (gt_bbox[1] < pred_bbox[1] < gt_bbox[3]):
                    return 1.0
                else:
                    return 0.0
            else:
                return 0.0
        elif gt_action in ['type', 'select', 'scroll']:
            # For text-based actions, use F1 score
            if calculate_f1_score(pred_input_text, gt_input_text) >= 0.5:
                return 1.0
            else:
                return 0.0
        else:
            # For other actions (complete, close/delete, press_home, press_back, enter)
            return 1.0

    except Exception as e:
        return 0.0


def compute_score(data_source:str, solution_str: str, ground_truth: str, extra_info: None,
                  format_weight: float = 0.2, accuracy_weight: float = 0.8) -> float:
    """
    The main scoring function for GUI-R1.

    This function combines format checking and accuracy checking to compute the final score.

    Args:
        solution_str: The solution text containing <think> and <answer> tags
        ground_truth: The ground truth in JSON format or simple string format
        format_weight: Weight for format score (default: 0.2)
        accuracy_weight: Weight for accuracy score (default: 0.8)

    Returns:
        float: The combined score (weighted sum of format and accuracy)
    """
    format_score = format_reward(solution_str)
    accuracy_score = accuracy_reward(solution_str, ground_truth)

    # Combined score
    overall_score = accuracy_weight * accuracy_score + format_weight * format_score

    return overall_score


# def compute_score_detailed(solution_str: str, ground_truth: str,
#                            format_weight: float = 0.2, accuracy_weight: float = 0.8) -> dict:
#     """
#     Compute detailed scores including format, accuracy, and overall.

#     Args:
#         solution_str: The solution text containing <think> and <answer> tags
#         ground_truth: The ground truth in JSON format or simple string format
#         format_weight: Weight for format score (default: 0.2)
#         accuracy_weight: Weight for accuracy score (default: 0.8)

#     Returns:
#         dict: Dictionary containing 'format', 'accuracy', and 'overall' scores
#     """
#     format_score = format_reward(solution_str)
#     accuracy_score = accuracy_reward(solution_str, ground_truth)

#     return {
#         "format": format_score,
#         "accuracy": accuracy_score,
#         "overall": accuracy_weight * accuracy_score + format_weight * format_score,
#     }



