# SelfEval

This is the official repository for paper **An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Models are Task-specific Classifiers**.

If you have any quesions, you can contact me with Wechat huanghui20200708.

## ⚡️ Usage
### Preparation
Please refer to the following command to prepare your environment.

```shell
pip install -r requirements.txt
```
Please download pre-trained LLMs and put them under ``models``. Specifically, our study are based on the following four finetuned models:

* [JudgeLM-7B](https://huggingface.co/BAAI/JudgeLM-7B-v1.0)

* [PandaLM-7B](https://huggingface.co/WeOpenML/PandaLM-7B-v1)

* [Prometheus-7b-v1.0](https://huggingface.co/kaist-ai/prometheus-7b-v1.0)

* [Auto-J-13b](https://huggingface.co/GAIR/autoj-13b)

Our study are based on the following data, and we have downloaded the respective testsets and put them under ``data``. 

* [JudgeLM-test](https://huggingface.co/datasets/BAAI/JudgeLM-100K/)

* [PandaLM-test](https://github.com/WeOpenML/PandaLM/blob/main/data/testset-v1.json)

* [Auto-J-test](https://github.com/GAIR-NLP/auto-j/blob/main/data/test/testdata_pairwise.jsonl)

* [Prometheus-test](https://github.com/kaistAI/prometheus/blob/main/evaluation/benchmark/data)

* [MT-bench-test](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)

* [LLMBar-test](https://github.com/princeton-nlp/LLMBar/tree/main/Dataset/LLMBar)

## Evaluate judges on different benchmarks

Run the following script to evaluate one judge model on one testset.

```shell
MODEL_PATH=./models/JudgeLM-7B
MODEL_TYPE=judgelm
DATA_TYPE=judgelm
python3 -u evaluate_judge.py \
    --model-name-or-path $MODEL_PATH \
    --model-type $MODEL_TYPE \
    --data-type $DATA_TYPE \
    --eval-batch-size 16 \
    --max-new-token 1024
```

Run the following script to evaluate the finetuned judges on different testsets.

```shell
MODEL_PATH=./models/llama2-7b-chat-finetuned
MODEL_TYPE=llama
DATA_TYPE=judgelm
CLASS_TYPE=generation
python -u evaluate_finetuned.py \
    --model-name-or-path $MODEL_NAME \
    --model-type $MODEL_TYPE \
    --data-type $DATA_TYPE \
    --class-type $CLASS_TYPE
```

Run the following script to evaluate GPT-3.5/4 on different testsets.

```shell
DATA_TYPE=judgelm
python -u evaluate_selfeval.py \
    --model-type "gpt-4" \
    --data-type $DATA_TYPE
```

## Fine-tune your own judge
You can train your own judge based on open-source judge data and foundation models.

We support different architecutes: LLaMA, BERT

We also support different judge schemes: Generation, Regression, Classification

Please refer to ``train.sh`` for more details

# 💬 Citation
If you find our work is helpful, please cite as:

```
@misc{huang2024empirical,
      title={An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Models are Task-specific Classifiers}, 
      author={Hui Huang and Yingqi Qu and Jing Liu and Muyun Yang and Tiejun Zhao},
      year={2024},
      eprint={2403.02839},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```