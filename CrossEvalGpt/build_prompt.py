def create_prompt(model_type, data_type, demo_augmentation=False, cot_augmentation=False):
    if data_type == "mt-bench":
        if cot_augmentation:
            instruction_single = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider the following factors:
1. Helpfulness: How useful was the response in providing clear and practical advice or solutions?
2. Relevence: Did the response directly answer the question and stay on topic?
3. Accuracy: Was the information provided accurate and well-researched?
4. Depth: How thorough and detailed was the response in terms of analysis and explanation?
5. Creativity: Did the response offer novel or unique perspectives or solutions?
6. Level of Detail: Was the response detailed enough to provide necessary context and background information?
Begin your evaluation by grade each response based on each factor. Each assistant receives a score beween 1 to 5 for each factor. Do not provide any explanations for the scores. After providing your scores, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 

[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]

Scores for Assistant A:
"""

# Your Evaluation: 

# 1. Helpfulness: {score11},{score12}
# 2. Relevence: {score21},{score22}
# 3. Accuracy: {score31},{score32}
# 4. Depth: {score41},{score42}
# 5. Creativity: {score51},{score52}
# 6. Level of Detail: {score61},{score62}

# Taking all of these together, the final verdict is:
            instruction_multi = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider the following factors:
1. Helpfulness: How useful was the response in providing clear and practical advice or solutions?
2. Relevence: Did the response directly answer the question and stay on topic?
3. Accuracy: Was the information provided accurate and well-researched?
4. Depth: How thorough and detailed was the response in terms of analysis and explanation?
5. Creativity: Did the response offer novel or unique perspectives or solutions?
6. Level of Detail: Was the response detailed enough to provide necessary context and background information?
Begin your evaluation by grade each response based on each factor. Each assistant receives a score beween 1 to 5 for each factor. Do not provide any explanations for the scores. After providing your scores, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 

<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>

Scores for Assistant A:
"""

# Your Evaluation: 

# 1. Helpfulness: {score11},{score12}
# 2. Relevence: {score21},{score22}
# 3. Accuracy: {score31},{score32}
# 4. Depth: {score41},{score42}
# 5. Creativity: {score51},{score52}
# 6. Level of Detail: {score61},{score62}

# Taking all of these together, the final verdict is:
            instruction = {"single": instruction_single, "multi": instruction_multi}
        else:
#             instruction_single = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.

# [User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]

# Your Evaluation:"""
            instruction_single = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Directly output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.

[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]

Your Evaluation:"""
            instruction_multi = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. You should choose the assistant that follows the user's instructions and answers the user's questions better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. You should focus on who provides a better answer to the second user question. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Directly output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.

<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>

Your Evaluation:"""
            instruction = {"single": instruction_single, "multi": instruction_multi}
    elif data_type in ["pandalm", "judgelm", "manual", "natural", "neighbor", "gptinst", "gptout"]:
        instruction = """You are a helpful and precise assistant for checking the quality of the answer.
[Question]
{question_body}

[The Start of Assistant 1's Answer]
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
    elif data_type == "auto-j":
        instruction = """You are assessing two submitted responses on a given user’s query based on the criteria you have known and judging which response is better or they are tied (including both good and both bad). Here is the data:
        
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

1. Review the two response and the given criteria to identify **only** the criterion(s) that can significantly distinguish the two responses. Ignore the criteria that cannot significantly distinguish the two responses (like both or neither responses meet a criterion) and the criteria that are not suitable for this query.
2. Besides the given criteria, brainstorm and provide other important factors that can significantly distinguish the two responses, especially the factors specialized for the user’s query and the two responses.
3. Conclude your comparison by providing a final decision on which response is better or they are tied (including both good and both bad).
Begin your final decision statement with "So, the final decision is Response 1/Response 2/Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you’ve provided."""
    elif "prometheus" in data_type:
        if demo_augmentation:
            instruction = """You are a fair evaluator language model.

###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{demo_question_body}

###Response to evaluate:
{demo_answer_body}

###Reference Answer (Score 5):
{demo_reference}

###Score Rubrics:
{demo_rubric}

###Feedback: 
{demo_feedback}

###The instruction to evaluate:
{question_body}

###Response to evaluate:
{answer_body}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
{rubric}

###Feedback:"""        
        else:
            instruction = """You are a fair evaluator language model.

###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{question_body}

###Response to evaluate:
{answer_body}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
{rubric}

###Feedback: """
    return instruction

def create_prompt_aspect():
    instruction_single = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. The following are some scores derived for different aspects. Please refer to the scores to compare the assistants' performance. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.

[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]

Scores for Assistant A:
{scores_a}

Scores for Assistant B:
{scores_b}

Your Evaluation:
"""
    instruction_multi = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. The following are some scores derived for different aspects. Please refer to the scores to compare the assistants' performance. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.

<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>

Scores for Assistant A:
{scores_a}

Scores for Assistant B:
{scores_b}

Your Evaluation:
"""

    instruction_aspect_single = """You are a fair and accurate evaluator.

###Task Description:
A conversation between a user and an AI assistant, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the assistant response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction and response to evaluate:
[User Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant A's Answer]

###Score Rubrics:
{rubric}

###Feedback: """

    instruction_aspect_multi = """You are a fair and accurate evaluator.

###Task Description:
A conversation between a user and an AI assistant, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the assistant response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction and response to evaluate:
<|The Start of Assistant's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant:\n{answer_2}\n\n<|The End of Assistant's Conversation with User|>

###Score Rubrics:
{rubric}

###Feedback: """

    instruction_score = """1. Helpfulness: {Helpfulness}
2. Relevence: {Relevence}
3. Accuracy: {Accuracy}
4. Depth: {Depth}
5. Creativity: {Creativity}
6. Detailedness: {Detailedness}"""
    instruction = {"single": instruction_single,
                   "multi": instruction_multi,
                   "aspect_single": instruction_aspect_single,
                   "aspect_multi": instruction_aspect_multi,
                   "scores": instruction_score}
    return instruction