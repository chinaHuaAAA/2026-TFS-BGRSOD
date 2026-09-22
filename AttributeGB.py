from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.cluster import k_means
import time
import warnings
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

def get_radius(gb):
    n, m = gb.shape
    gb = gb[:, :m - 1]
    num = len(gb)
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    radius = max(distances)
    return radius

def get_radius_mean(gb):
    n, m = gb.shape
    gb = gb[:, :m - 1]
    num = len(gb)
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    radius = np.mean(distances)
    return radius
def spilt_ball(data):
    n, m = data.shape
    cluster = k_means(X=data[:, :m - 1], init='k-means++', n_clusters=2, n_init='auto')[1]
    ball1 = data[cluster == 0, :]
    ball2 = data[cluster == 1, :]
    return [ball1, ball2]


def spilt_ball_2(data):
    n, m = data.shape
    cluster = np.zeros(n)
    X = data[:, :m - 1]
    D = cdist(X, X, 'cosine')
    max_idx = np.argmax(D)  # 找到最大值的索引
    r1 = max_idx // n  # 行索引
    c1 = max_idx % n  # 列索引

    for j in range(0, len(data)):
        if D[j, r1] < D[j, c1]:
            cluster[j] = 0
        else:
            cluster[j] = 1
    ball1 = data[cluster == 0, :]
    ball2 = data[cluster == 1, :]
    return [ball1, ball2]


def get_density_volume(gb):
    gb = gb[:, :-1]
    num = len(gb)
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sum_radius = np.sum(distances)

    result = sum_radius if sum_radius != 0 else num
    return result


def division_ball(gb_list, n):
    gb_list_new = []
    for gb in gb_list:
        if len(gb) >= n:
            ball_1, ball_2 = spilt_ball_2(gb)
            density_parent = get_density_volume(gb)
            density_child_1 = get_density_volume(ball_1)
            density_child_2 = get_density_volume(ball_2)
            w = len(ball_1) + len(ball_2)
            w1 = len(ball_1) / w
            w2 = len(ball_2) / w
            w_child = (w1 * density_child_1 + w2 * density_child_2)
            t2 = (w_child < density_parent)
            if t2:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_new.append(gb)
        else:
            gb_list_new.append(gb)

    return gb_list_new


def normalized_ball(gb_list, radius_detect):
    gb_list_temp = []
    for gb in gb_list:
        if len(gb) < 2:
            gb_list_temp.append(gb)
        else:
            ball_1, ball_2 = spilt_ball_2(gb)
            if get_radius(gb) <= 2 * radius_detect:
                gb_list_temp.append(gb)
            else:
                gb_list_temp.extend([ball_1, ball_2])

    return gb_list_temp


def getAttributeGranularBall(data):
    n, m = data.shape
    index = np.array(range(n)).reshape(n, 1)  # column of index，创建一个索引列, 从0开始的
    data_index = np.hstack((data, index))
    gb_list_temp = [data_index]
    if n ** 0.5 > 64:
        min_split_num = 64
    elif n ** 0.5 > 8:
        min_split_num = 8
    else:
        min_split_num = int(n ** 0.5)

    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = division_ball(gb_list_temp, min_split_num)
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    radius_list = []  # 存储每个粒球的半径
    for gb in gb_list_temp:
        if len(gb) >= 2:
            radius_list.append(get_radius(gb))

    radius_median = np.median(radius_list)
    radius_mean = np.mean(radius_list)
    radius_detect = max(radius_median, radius_mean)
    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = normalized_ball(gb_list_temp, radius_detect)
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    gb_list_final = gb_list_temp

    sample_num = np.ravel([len(gb) for gb in gb_list_final])
    gb_centers = [gb.mean(0) for gb in gb_list_final]
    centers = np.vstack(gb_centers)[:, :-1]

    # 计算最终每个粒球的半径
    final_radius = []
    for gb in gb_list_final:
        if len(gb) >= 2:
            final_radius.append(get_radius_mean(gb)) ##算粒球的平均半径
        else:
            final_radius.append(0)  # 对于单个点的粒球，半径为0

    return centers, gb_list_final, sample_num, np.array(final_radius)  # 添加半径作为第四个返回值


