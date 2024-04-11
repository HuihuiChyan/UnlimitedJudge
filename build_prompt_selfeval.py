def create_prompt_predefined(model_type, data_type):
    if model_type == "auto-j":
        instruction_single = """[INST] Write critiques for a submitted response on a given user's query, and grade the response:
  
# [BEGIN DATA]
# ***
# [Query]: {question}
# ***
# [Response]: {answer}
# ***
# [END DATA]

# Write critiques for this response. After that, you should give a final rating for the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". [/INST]"""
        instruction_multi = """[INST] Write critiques for submitted responses on a given user's queries, and grade the responses:
  
# [BEGIN DATA]
# ***
# [Query1]: {question_1}
# ***
# [Response1]: {answer_1}
# ***
# [Query2]: {question_2}
# ***
# [Response2]: {answer_2}
# ***
# [END DATA]

# Write critiques for the responses. After that, you should give a final rating for the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". [/INST]"""
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
An instruction (might include an Input inside it), two responses to evaluate, and a score rubric representing a evaluation criteria are given.
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