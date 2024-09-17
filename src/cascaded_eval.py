import json
import numpy as np
import random
import argparse
from evaluate_judge import calculate_metrics, build_dataset, build_params


def load_results(file_path):
    """从文件加载分数结果"""
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results


def get_average_scores(scores):
    """计算正向和逆向的平均分数"""
    # 分数列表的中间索引
    mid_index = len(scores) // 2
    # 前一半是正向，后一半是逆向
    forward_scores = scores[:mid_index]
    backward_scores = scores[mid_index:]
    # 计算平均分数
    average_scores = [(forward_scores[i] + backward_scores[i]
                       ) / 2 for i in range(mid_index)]
    return average_scores

def select_top_half_indices(average_scores, total_length):
    """获取得分最高的前 50% 数据的索引，包括正向和逆向"""
    sorted_indices = np.argsort(-np.array(average_scores))
    top_half_indices = sorted_indices[:len(sorted_indices) // 2]
    # 添加对应的逆向数据索引
    indices_with_reverse = np.concatenate(
        [top_half_indices, top_half_indices + total_length // 2])
    return indices_with_reverse

def compute_accuracy_rate_weak_weak(relia_scores1, relia_scores2, judge_output1, judge_output2, answers, dataset_type):

    def get_sort_indices(relia_scores):
        sorted_indices = list(np.argsort(-np.array(relia_scores)))
        fucked_indices = [sorted_indices.index(i) for i in range(len(sorted_indices))]

        return fucked_indices

    def get_top_half_indices(relia_scores, dataset_type, ratio=0.5):
        sorted_indices = np.argsort(-np.array(relia_scores))
        top_half_indices = sorted_indices[:int(len(sorted_indices) * ratio)]

        if dataset_type == "auto-j":
            top_half_indices = np.concatenate(
                [top_half_indices, top_half_indices + len(sorted_indices)])

        return list(top_half_indices)

    """根据分数结果计算准确率"""
    if dataset_type == "auto-j":
        relia_scores1 = get_average_scores(relia_scores1)
        relia_scores2 = get_average_scores(relia_scores2)

    sorted_indices1 = get_sort_indices(relia_scores1)
    sorted_indices2 = get_sort_indices(relia_scores2)
    
    judge_outputs = []
    judge_answers = []
    for rank1, rank2, output1, output2, answer in zip(sorted_indices1, sorted_indices2, judge_output1, judge_output2, answers):
        if rank1 < rank2:
            judge_outputs.append(output1)
            judge_answers.append(answer)
        elif rank2 < rank1:
            judge_outputs.append(output1)
            judge_answers.append(answer)
        else:
            judge_outputs.append(random.choice([output1, output2]))
            judge_answers.append(answer)

    accuracy_rate = calculate_metrics(judge_answers, judge_outputs, dataset_type)

    return accuracy_rate

def compute_accuracy_rate_weak_strong(relia_scores1, judge_output1, judge_output_gpt, answers, dataset_type, ratio):

    def get_top_half_indices(relia_scores, dataset_type, ratio=0.9):
        sorted_indices = np.argsort(-np.array(relia_scores))
        top_half_indices = sorted_indices[:int(len(sorted_indices) * ratio)]

        if dataset_type == "auto-j":
            top_half_indices = np.concatenate(
                [top_half_indices, top_half_indices + len(sorted_indices)])

        return list(top_half_indices)

    """根据分数结果计算准确率"""
    if dataset_type == "auto-j":
        relia_scores1 = get_average_scores(relia_scores1)
    
    top_half_indice1 = set(get_top_half_indices(relia_scores1, dataset_type, ratio))
    
    judge_outputs = []
    judge_answers = []
    for i, output1, output2, answer in zip(np.arange(len(judge_output1)).tolist(), judge_output1, judge_output_gpt, answers):
        if i in top_half_indice1:
            judge_outputs.append(output1)
            judge_answers.append(answer)
        else:
            judge_outputs.append(output1)
            judge_answers.append(answer)

    accuracy_rate = calculate_metrics(judge_answers, judge_outputs, dataset_type)

    print(f"Ratio: {ratio}, Accuracy Rate: {accuracy_rate}")
    # print(round(accuracy_rate["accuracy"], 4))



def main():
    random.seed(42)
    np.random.seed(42)

    parser = build_params()
    parser.add_argument("--output-file1", type=str, default=None)
    parser.add_argument("--logit-file1", type=str, default=None)
    parser.add_argument("--output-file2", type=str, default=None)
    parser.add_argument("--logit-file2", type=str, default=None)
    parser.add_argument("--logit-file-gpt", type=str, default=None)
    args = parser.parse_args()

    dataset = build_dataset(args.data_type, "./data")
    answers = [example["score"] for example in dataset]

    relia_scores1 = load_results(args.output_file1)["Entropy"]

    with open(args.logit_file1, 'r') as f:
        judge_output1 = [json.loads(line.strip()) for line in f.readlines()]

    if args.output_file2 is not None:
        relia_scores2 = load_results(args.output_file2)["Entropy"]

        with open(args.logit_file2, 'r') as f:
            judge_output2 = [json.loads(line.strip()) for line in f.readlines()]

        # 计算指标的准确率
        accuracy_rate = compute_accuracy_rate_weak_weak(relia_scores1, relia_scores2, judge_output1, judge_output2, answers, args.data_type)
    
    else:
        with open(args.logit_file_gpt, 'r') as f:
            judge_output_gpt = [json.loads(line.strip())["score"] for line in f.readlines()]
        
        compute_accuracy_rate_weak_strong(relia_scores1, judge_output1, judge_output_gpt, answers, args.data_type, ratio=0.5)

        # for ratio in np.arange(1.0, -0.1, -0.1):
        #     ratio = round(ratio, 1)
        #     accuracy_rate = compute_accuracy_rate_weak_strong(relia_scores1, judge_output1, judge_output_gpt, answers, args.data_type, ratio=ratio)


if __name__ == "__main__":
    main()