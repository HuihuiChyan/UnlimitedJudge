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


def normalize_scores(scores):
    """归一化分数"""
    min_score = min(scores)
    max_score = max(scores)
    normalized_scores = [(score - min_score) /
                         (max_score - min_score) for score in scores]
    return normalized_scores


def compute_combined_score(entropy_scores, variance_scores):
    """计算熵和方差的组合分数"""
    normalized_entropy = normalize_scores(entropy_scores)
    normalized_variance = normalize_scores(variance_scores)
    combined_scores = [(normalized_entropy[i] + normalized_variance[i]
                        ) / 2 for i in range(len(entropy_scores))]
    return combined_scores


def compute_calibrated_score(scores, cali_scores):
    """计算校准后的组合分数"""
    calibrated_score = [(scores[i]) / 2 for i in range(len(scores))]
    return calibrated_score


def select_top_half_indices(average_scores, total_length):
    """获取得分最高的前 50% 数据的索引，包括正向和逆向"""
    sorted_indices = np.argsort(-np.array(average_scores))
    top_half_indices = sorted_indices[:len(sorted_indices) // 2]
    # 添加对应的逆向数据索引
    indices_with_reverse = np.concatenate(
        [top_half_indices, top_half_indices + total_length // 2])
    return indices_with_reverse


def compute_accuracy_rate(metric_results, answers, judge_output, total_length, dataset_type):
    """根据分数结果计算准确率"""
    if dataset_type == "auto-j":
        average_scores = get_average_scores(metric_results)
        top_half_indices = select_top_half_indices(
            average_scores, total_length)
        accuracy_rate = calculate_metrics(
            [answers[i] for i in top_half_indices], [judge_output[i] for i in top_half_indices], dataset_type)
    else:
        average_scores = metric_results
        sorted_indices = np.argsort(-np.array(average_scores))
        bucket_size = len(sorted_indices)//5
        for i in range(5):
            top_half_indices = sorted_indices[bucket_size*i:bucket_size*(i+1)]
            accuracy_rate = calculate_metrics(
                [answers[i] for i in top_half_indices], [judge_output[i] for i in top_half_indices], dataset_type)
            print(accuracy_rate)
    return accuracy_rate

def compute_accuracy_rate(metric_results, answers, judge_output, total_length, dataset_type):
    """根据分数结果计算准确率"""
    if dataset_type == "auto-j":
        average_scores = get_average_scores(metric_results)
        top_half_indices = select_top_half_indices(
            average_scores, total_length)
    else:
        average_scores = metric_results
        sorted_indices = np.argsort(-np.array(average_scores))
        top_half_indices = sorted_indices[:len(sorted_indices) // 2]

    accuracy_rate = calculate_metrics(
        [answers[i] for i in top_half_indices], [judge_output[i] for i in top_half_indices], dataset_type)

    return accuracy_rate

def compute_bucketing_rate(metric_results, answers, judge_output, total_length, dataset_type):
    """根据分数结果计算准确率"""
    if dataset_type == "auto-j":
        pass
    else:
        average_scores = metric_results
        sorted_indices = np.argsort(-np.array(average_scores))
        bucket_size = len(sorted_indices)//5
        for i in range(5):
            top_half_indices = sorted_indices[bucket_size*i:bucket_size*(i+1)]
            accuracy_rate = calculate_metrics(
                [answers[i] for i in top_half_indices], [judge_output[i] for i in top_half_indices], dataset_type)
            print(accuracy_rate)
    return accuracy_rate


def main():
    random.seed(42)
    np.random.seed(42)

    parser = build_params()
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    dataset = build_dataset(args.data_type, "./data")
    answers = [example["score"] for example in dataset]

    entropy_results = load_results(args.output_file)["Entropy"]
    entropy_cali_results = load_results(args.output_file)["entropy_cali"]
    relia_scores = compute_calibrated_score(entropy_results, entropy_cali_results)

    with open(args.logit_file, 'r') as f:
        judge_output = [json.loads(line.strip()) for line in f.readlines()]

    answers = answers[:len(relia_scores)]
    judge_output = judge_output[:len(relia_scores)]
    # 计算指标的准确率
    accuracy_rate = compute_accuracy_rate(
        relia_scores, answers, judge_output, len(relia_scores), args.data_type)

    # 随机选择基线准确率
    if args.data_type == "auto-j":
        # 选取与 top_half_md_indices 相同数量的正向索引
        random_forward_indices = np.random.choice(
            len(answers)//2, len(relia_scores)//4, replace=False)
        # 添加对应的逆向数据索引
        random_indices = np.concatenate(
            [random_forward_indices, random_forward_indices + len(answers)//2])
    else:
        # 随机选取等量的索引作为一个随机基线比较
        random_indices = np.random.choice(
            len(answers), len(relia_scores)//2, replace=False)

    random_accuracy_rate = calculate_metrics(
        [answers[i] for i in random_indices], [judge_output[i] for i in random_indices], args.data_type)

    print(f"Accuracy Rate: {accuracy_rate}")
    print(f"Random Selection Accuracy Rate: {random_accuracy_rate}")


if __name__ == "__main__":
    main()