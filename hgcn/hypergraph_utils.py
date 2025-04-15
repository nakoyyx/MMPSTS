import numpy
import numpy as np
import torch
import torch.nn.functional as F

def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


# 相关距离
def CD(x):
    dist_mat = np.zeros((x.shape[0], x.shape[0]))
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[0]):
            d = 1 - abs(np.corrcoef(x[i], x[j]))
            dist_mat[i][j] = d[0][1]
    return dist_mat
    # 余弦距离


def Cosine(x):
    dist_mat = np.zeros((x.shape[0], x.shape[0]))
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[0]):
            d = np.dot(x[i], x[j]) / (np.linalg.norm(x[i]) * (np.linalg.norm(x[j])))
            dist_mat[i][j] = d
    return dist_mat


# 传过来的是一个特征列表
def feature_concat(*F_list, normal_col=False):
    """
    串联多模态特性。如果特征矩阵的维数大于2， 函数将其化简为二维(将最后一个维度作为特征维度， 另一个维度将被融合为对象维度)
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature    每一列是否规范化规范化
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and len(f)!=0:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])  # 我感觉这句有问题，按照列对齐
            # normal each column
            if normal_col:  # 如果没有正规化，则进行正规化
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix

            if features is None:
                features = f
                # 其实这里是没用的，因为这个代码并没有使用这里进行特征融合
            else:
                features = np.hstack((features, f))  # 将参数元组的元素数组按水平方向进行叠加
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max  # 好像没用
    return features


def hyperedge_concat(*H_list):  # 传输过来的应该是一个H矩阵的列表
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        # print("h shape:",h)
        # if h is not None and h != []:
        if h is not None and len(h)!=0:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, W, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    W_vector = torch.diag(W).flatten()
    # W_vector=W_vector.unsqueeze(0)
    W_vector=W_vector.repeat(H.shape[0],1)
    DV = torch.sum(H * W_vector, dim=1)  # 得到一列顶点权重
    # 检查 DV 中是否存在 0
    zero_mask = (DV == 0)
    # 将 DV 中的 0 替换为一个很小的非零值，例如 1e-6
    DV = torch.where(zero_mask, torch.ones_like(DV) * 1e-6, DV)
    DE = torch.sum(H, dim=0)  # 超边的度的行向量
    invDE = torch.diag(torch.pow(DE,-1))
    DV2 = torch.diag(torch.pow(DV, -0.5))

    H = H  # 不需要转换为torch.mat，PyTorch张量即可
    HT = H.t()

    if variable_weight:
        DV2_H = torch.matmul(DV2, H)
        invDE_HT_DV2 = torch.matmul(torch.matmul(invDE, HT), DV2)
        return DV2_H, W, invDE_HT_DV2
    else:
        G = torch.matmul(torch.matmul(torch.matmul(torch.matmul(torch.matmul(DV2, H), W), invDE), HT), DV2)
        I = torch.eye(G.size(0), dtype=torch.float32).to(G.device)
        G = I + G
        # has_nan = torch.isnan(G).any().item()
        return G,DE


def _generate_G_from_H(H, W, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    W_vector = torch.diag(W).flatten()
    # W_vector=W_vector.unsqueeze(0)
    W_vector=W_vector.repeat(H.shape[0],1)
    DV = torch.sum(H * W_vector, dim=1)  # 得到一列顶点权重
    # 检查 DV 中是否存在 0
    zero_mask = (DV == 0)
    # 将 DV 中的 0 替换为一个很小的非零值，例如 1e-6
    DV = torch.where(zero_mask, torch.ones_like(DV) * 1e-6, DV)
    DE = torch.sum(H, dim=0)  # 超边的度的行向量
    invDE = torch.diag(torch.pow(DE,-1))
    DV2 = torch.diag(torch.pow(DV, -0.5))

    H = H  # 不需要转换为torch.mat，PyTorch张量即可
    HT = H.t()

    if variable_weight:
        DV2_H = torch.matmul(DV2, H)
        invDE_HT_DV2 = torch.matmul(torch.matmul(invDE, HT), DV2)
        return DV2_H, W, invDE_HT_DV2
    else:
        G = torch.matmul(torch.matmul(torch.matmul(torch.matmul(torch.matmul(DV2, H), W), invDE), HT), DV2)
        I = torch.eye(G.size(0), dtype=torch.float32).to(G.device)
        G = I + G
        # has_nan = torch.isnan(G).any().item()
        return G,DE


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):  # 根据距离矩阵构建超图H
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = torch.zeros(n_obj, n_edge)
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        # nearest_idx = np.array(np.argsort(dis_vec)).squeeze()  # 将dis_vec中的元素从小到大排列，提取其对应的index(索引)，
        # # squeeze()，假如某一维只有一项数据，则删除这一维度。
        # avg_dis = np.average(dis_vec)
        # if not np.any(nearest_idx[:k_neig] == center_idx):
        #     nearest_idx[k_neig - 1] = center_idx

        # for node_idx in nearest_idx[:k_neig]:
        #     if is_probH:
        #         H[node_idx, center_idx] = np.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)
        #         # 这里有问题
        #     else:
        #         H[node_idx, center_idx] = 1.0
        # 使用 torch.argsort 方法对 dis_vec 进行排序并提取索引
        nearest_idx = torch.argsort(dis_vec).squeeze()

        # 计算平均值
        avg_dis = torch.mean(dis_vec)
        if not torch.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        # 对 nearest_idx[:k_neig] 中的元素进行操作
        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = torch.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs, split_diff_scale, is_probH, m_prob, subjectnum):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    是否在不同的邻居规模上超边

    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """

    # if len(X.shape) != 2:
    #     X = X.reshape(-1, X.shape[-1])            #按照列对齐
    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    # 欧式距离
    # dis_mat = torch.cdist(X,X, p=2)
    # dis_mat=Eu_dis(X)
    # 相关距离
    # dis_mat =CD(X)
    # 余弦距离
    # 计算余弦距离
    # 计算余弦距离矩阵
    cosine_similarity_matrix = F.cosine_similarity(X.unsqueeze(1), X.unsqueeze(0), dim=2)
    dis_mat = 1 - cosine_similarity_matrix
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        # print(H_tmp.shape)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)  # append命令是将整个对象加在列表末尾，就是直接拼接上

    # shape[0]为行数，shape[1]为列数
    for i in range(X.shape[0]):
        if torch.isnan(X[i][0]):
            for j in range(X.shape[0]):
                H[i][j] = 0
    # print(H)
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[0]):
    #         if (numpy.isnan(H[i][j])):
    #             print("H为nan")
    return H


def statistic_indicators(predicted_labels, true_labels):
    # 计算准确率
    accuracy = (true_labels == predicted_labels).float().mean().item()
    print("Accuracy:", accuracy)

    # 计算精确率、召回率、F1 分数
    # 先计算每个类别的精确率、召回率
    num_classes = len(torch.unique(true_labels))
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    for i in range(num_classes):
        true_positive = ((true_labels == i) & (predicted_labels == i)).float().sum().item()
        false_positive = ((true_labels != i) & (predicted_labels == i)).float().sum().item()
        false_negative = ((true_labels == i) & (predicted_labels != i)).float().sum().item()
        
        precision[i] = true_positive / (true_positive + false_positive + 1e-10)  # 加上一个很小的数以避免除零错误
        recall[i] = true_positive / (true_positive + false_negative + 1e-10)

    # 计算宏平均
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall + 1e-10)

    print("Macro Precision:", macro_precision)
    print("Macro Recall:", macro_recall)
    print("Macro F1 Score:", macro_f1)

    # 计算微平均
    micro_precision = precision.sum() / (precision.sum() + false_positive + 1e-10)
    micro_recall = recall.sum() / (recall.sum() + false_negative + 1e-10)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-10)

    print("Micro Precision:", micro_precision.item())
    print("Micro Recall:", micro_recall.item())
    print("Micro F1 Score:", micro_f1.item())


    return accuracy,macro_precision, macro_recall, macro_f1, micro_precision.item(), micro_recall.item(), micro_f1.item()

# # 示例真实标签和预测结果
# true_labels = torch.tensor([0, 1, 2, 0, 1, 2])  # 真实标签
# predicted_labels = torch.tensor([0, 2, 1, 0, 2, 1])  # 预测结果
# print(statistic_indicators(predicted_labels,true_labels))