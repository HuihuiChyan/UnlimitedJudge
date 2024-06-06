def create_prompt_mtbench(model_type, data_type):
    assert data_type == "mt-bench"
    if model_type == "judgelm":
        instruction_single = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.

[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]

Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

### Response:"""
        instruction_multi = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.

<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>

Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

### Response:"""
    elif model_type == "pandalm":
        instruction_single = """Below are two responses for a given task. The task is defined by the Instruction. Evaluate the responses and generate a reference answer for the task.

[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]

### Evaluation:\n"""
        instruction_multi = """Below are two responses for a given task. The task is defined by the Instruction. Evaluate the responses and generate a reference answer for the task.

<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>

### Evaluation:\n"""
    elif model_type == "auto-j":
        instruction_single = """[INST] You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided. [/INST]"""
        instruction_multi = """[INST] You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided. [/INST]"""
    elif model_type == "prometheus":
        instruction_single = """[INST] <<SYS>>
You are a fair evaluator language model.
<</SYS>>

###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction and response to evaluate:
[User Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]

###Score Rubrics:
{rubric}

###Feedback: [/INST]"""
        instruction_multi = """[INST] <<SYS>>
You are a fair evaluator language model.
<</SYS>>

###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction and response to evaluate:
<|The Start of Assistant's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant:\n{answer_2}\n\n<|The End of Assistant's Conversation with User|>

###Score Rubrics:
{rubric}

###Feedback: [/INST]"""
    instruction = {"single": instruction_single, "multi": instruction_multi}
    return instruction