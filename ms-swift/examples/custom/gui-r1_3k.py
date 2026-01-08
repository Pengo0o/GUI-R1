# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional
from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset,SubsetDataset
import os


# class CustomPreprocessor(ResponsePreprocessor):
#     prompt = """Task: Based on the given two sentences, provide a similarity score between 0.0 and 5.0.
# Sentence 1: {text1}
# Sentence 2: {text2}
# Similarity score: """

#     def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
#         return super().preprocess({
#             'query': self.prompt.format(text1=row['text1'], text2=row['text2']),
#             'response': f"{row['label']:.1f}"
#         })


# register_dataset(
#     DatasetMeta(
#         ms_dataset_id='swift/stsb',
#         hf_dataset_id='SetFit/stsb',
#         preprocess_func=CustomPreprocessor(),
#     ))

# if __name__ == '__main__':
#     dataset = load_dataset(['swift/stsb'])[0]
#     print(f'dataset: {dataset}')
#     print(f'dataset[0]: {dataset[0]}')


# class ClevrPreprocessor(ResponsePreprocessor):

#     def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
#         query = row.get('query', '')
#         query = f"""{query} Output the thinking process in <think> </think> and
#  final answer (number) in <answer> </answer> tags."""
#         row.update({'query': query})
#         return super().preprocess(row)


# register_dataset(
#     DatasetMeta(
#         ms_dataset_id='AI-ModelScope/clevr_cogen_a_train',
#         subsets=[
#             SubsetDataset(
#                 name='default',
#                 subset='default',
#                 split=['train'],
#             ),
#         ],
#         preprocess_func=ClevrPreprocessor(),
#         tags=['qa', 'math']))


class GuiR1Preprocessor(ResponsePreprocessor):
    """
    Preprocessor for GUI-R1 dataset.
    Adapts the dataset format from verl to ms-swift framework.

    Note: The dataset must contain a 'solution' field for reward function to work.
    Use add_solution_field.py utility to add it if missing.
    """

    @staticmethod
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

    @staticmethod
    def build_user_prompt(instruction, history):
        """
        Build the user prompt with specific instruction and history.
        """
        user_prompt = f"In this UI screenshot, I want you to continue executing the command '{instruction}', with the action history being '{history}'."
        return user_prompt

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Preprocess each row of the GUI-R1 dataset.

        Expected input fields:
        - instruction: The task instruction
        - task_type: Type of task ('high' or 'low')
        - history: Action history
        - image: Screenshot image
        - gt_action: Ground truth action
        - gt_input_text: Ground truth input text
        - gt_bbox: Ground truth bounding box [x, y]
        """
        # Extract fields from the original dataset
        instruction = row.get('instruction', '')
        task_type = row.get('task_type', 'low')  # default to 'low' if not specified
        history = row.get('history', '')
        image = row.get('images', None)
        image = os.path.join("/opt/data/private/hyp/Gui-Agent/Data/GUI-R1-swift",image)

        # Extract ground truth information
        gt_action = row.get('gt_action', '')
        gt_input_text = row.get('gt_input_text', 'no input text')
        gt_bbox = row.get('gt_bbox', [-100, -100])

        # Build system and user prompts
        system_prompt = self.build_system_prompt(task_type)
        user_prompt = self.build_user_prompt(instruction, history)

        # Build response in the expected format
        # Format: <think> thinking process </think> <answer>[{'action': ..., 'point': ..., 'input_text': ...}]</answer>
        response = f"<answer>[{{'action': '{gt_action}', 'point': {gt_bbox}, 'input_text': '{gt_input_text}'}}]</answer>"

        row.update({'system':system_prompt})
        row.update({'query':user_prompt})
        row.update({'response':response})
        # Remove history field to avoid errors (original data has string "None")
        row.pop('history', None)

        # Include image if available
        if image is not None:
            row.update({'images': [image]})

        processed_row = super().preprocess(row)
        processed_row.update({'solution':row['__#solution']})
        return  processed_row




# Register the GUI-R1 dataset
# Option 1: Using local parquet file
register_dataset(
    DatasetMeta(
        dataset_name='gui-r1',
        dataset_path='/opt/data/private/hyp/Gui-Agent/Data/GUI-R1-swift/train.json',
        preprocess_func=GuiR1Preprocessor(),
        tags=['gui', 'vision', 'reasoning']
    ))

register_dataset(
    DatasetMeta(
        dataset_name='gui-r1-val',
        dataset_path='/opt/data/private/hyp/Gui-Agent/Data/GUI-R1-swift/val.json',
        preprocess_func=GuiR1Preprocessor(),
        tags=['gui', 'vision', 'reasoning']
    ))

class ClevrPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = row.get('query', '')
        query = f"""{query} Output the thinking process in <think> </think> and
 final answer (number) in <answer> </answer> tags."""
        row.update({'query': query})
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/clevr_cogen_a_train',
        subsets=[
            SubsetDataset(
                name='default',
                subset='default',
                split=['train'],
            ),
        ],
        preprocess_func=ClevrPreprocessor(),
        tags=['qa', 'math']))


if __name__ == '__main__':
    # Test the dataset loading
    dataset = load_dataset(['AI-ModelScope/clevr_cogen_a_train'])[0]
    print(f'dataset: {dataset}')
    print(f'dataset length: {len(dataset)}')
    if len(dataset) > 0:
        print(f'dataset[0]: {dataset[0]}')
