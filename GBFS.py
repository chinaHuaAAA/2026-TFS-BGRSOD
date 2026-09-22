import numpy as np
from AttributeGB import getAttributeGranularBall
from scipy.stats import entropy
import math


def get_feature_ranking(data):
    """
    返回所有特征的按重要性排序的索引（降序）
    参数:
        data: ndarray, shape (n_samples, n_features)
    返回:
        sorted_indices: 按重要性降序排列的特征索引
        feature_importance: 每个特征的重要性值（熵）
    """
    Gbs = getAttributeGranularBall(data.T)
    total_features = data.shape[1]

    feature_importance = np.zeros(total_features)

    for i in range(len(Gbs[2])):
        features_in_gb = Gbs[1][i][:, -1].astype(int)
        gb_data = Gbs[1][i]

        for j, feature_id in enumerate(features_in_gb):
            feature_values = gb_data[j, :-1]

            value_counts = np.bincount(feature_values.astype(int))
            value_counts = value_counts[value_counts > 0]
            probabilities = value_counts / len(feature_values)
            entropy_value = entropy(probabilities, base=2)

            feature_importance[feature_id] = entropy_value

    sorted_indices = np.argsort(feature_importance)[::-1]

    return sorted_indices, feature_importance


def Feature_selection(data, select_ratio):
    """
    特征选择：返回选中的特征子集（保持兼容）
    """
    sorted_indices, _ = get_feature_ranking(data)
    total_features = data.shape[1]
    target_num = math.ceil(total_features * select_ratio)
    selected_indices = sorted_indices[:target_num]

    if hasattr(data, 'iloc'):
        return data.iloc[:, selected_indices]
    else:
        return data[:, selected_indices]