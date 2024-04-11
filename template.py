instructions_pre = {}
instructions_pre["judgelm"] = """You are a helpful and precise assistant for checking the quality of the answer.
[Question]
{question_body}

The Start of Assistant 1's Answer]
{answer1_body}

[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{answer2_body}

[The End of Assistant 2's Answer]

[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

### Response:"""

instructions_pre["judgelm_with_reference"] = """You are a helpful and precise assistant for checking the quality of the answer.
[Question]
{question_body}

[Reference Answer]
{reference}

The Start of Assistant 1's Answer]
{answer1_body}

[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{answer2_body}

[The End of Assistant 2's Answer]

[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

### Response:"""

instructions_pre["judgelm_for_prometheus"] = """You are a helpful and precise assistant for checking the quality of the answer.
[Question]
{question_body}

[The Start of Assistant 1's Answer]
{reference}

[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{answer}

[The End of Assistant 2's Answer]

[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
{rubric} Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

### Response:10 """

instructions_pre["pandalm"]= """Below are two responses for a given task. The task is defined by the Instruction. Evaluate the responses and generate a reference answer for the task.

### Instruction:
{question_body}

### Response 1:
{answer1_body}

### Response 2:
{answer2_body}

### Evaluation:\n"""

instructions_pre["pandalm_with_reference"]= """Below are two responses for a given task. The task is defined by the Instruction. Evaluate the responses based on the reference answer for the task.

### Instruction:
{question_body}

### Reference:
{question_body}

### Response 1:
{answer1_body}

### Response 2:
{answer2_body}

### Evaluation:\n"""


instructions_pre["autoj"] = """[INST] You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
***
[Query]: {question_body}
***
[Response 1]: {answer1_body}
***
[Response 2]: {answer2_body}
***
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided. [/INST]"""

instructions_pre["autoj_with_reference"] = """[INST] You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
***
[Query]: {question_body}
***
[Reference]: {reference}
***
[Response 1]: {answer1_body}
***
[Response 2]: {answer2_body}
***
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided. [/INST]"""

instructions_pre["autoj_for_prometheus"] = """[INST] Write critiques for a submitted response on a given user's query, and grade the response:
  
# [BEGIN DATA]
# ***
# [Query]: {question_body}
# ***
# [Response]: {answer}
# ***
# [END DATA]

# Write critiques for this response. After that, you should give a final rating for the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". [/INST]"""

instructions_pre["prometheus"] = """[INST] <<SYS>>
You are a fair evaluator language model.
<</SYS>>

###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{question_body}

###Response to evaluate:
{answer}

###Score Rubrics:
{rubric}

###Feedback: [/INST]"""