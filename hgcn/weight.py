import scipy.io as scio
import numpy as np
import torch
def corrcoef(v_m,v_n):
    # 计算均值
    mean_m = torch.mean(v_m)
    mean_n = torch.mean(v_n)

    # 计算协方差矩阵
    covariance = torch.mean((v_m - mean_m) * (v_n - mean_n))

    # 计算标准差
    std_m = torch.std(v_m)
    std_n = torch.std(v_n)

    # 计算相关系数矩阵
    corr = covariance / (std_m * std_n)
    return corr


# #W是超边权重矩阵
def set_weight(H, a,device):
    H=H.T
    # 每行代表一条超边
    # 计算主体之间的皮尔逊相关系数
    correlations = torch.corrcoef(a)  # 计算相关系数矩阵
    
    # 将相关系数矩阵按照超边张量进行筛选，并求取每行的平均值
    edge_weights = torch.zeros(H.shape[0]).to(device=device)
    for j in range(H.shape[0]):
        # 获取第j条超边连接的主体索引
        connected_subjects = torch.nonzero(H[j]).squeeze()
        
        # 如果没有连接的主体，则权重为0
        if connected_subjects.numel() == 0:
            edge_weights[j] = 0
        else:
           # 选择与当前超边连接的主体之间的相关系数
            relevant_correlations = correlations[connected_subjects][:, connected_subjects]
            
            # 将相关系数矩阵的对角线清零
            relevant_correlations.fill_diagonal_(0)
            # 只取下三角矩阵
            relevant_correlations = torch.tril(relevant_correlations)
            a,b=torch.max(relevant_correlations,1)
            # 计算相关系数矩阵中非零元素的数量
            num=relevant_correlations.shape[0]*(relevant_correlations.shape[0]-1)/2
            relevant_sum=relevant_correlations.sum()
            edge_weights[j] = relevant_sum / num
            if edge_weights[j]<0:
                edge_weights[j]=0
    return torch.diag(edge_weights)

# def set_weight_optimized(H, a, device):
#     H = H.T
#     correlations = torch.corrcoef(a)  # 计算相关系数矩阵一次
    
#     # 将H转换为掩码，以便使用广播
#     mask = H.unsqueeze(0).expand(len(a), len(a), -1)
    
#     # 选择与H连接的主体之间的相关系数
#     relevant_correlations = correlations * mask
    
#     # 清零对角线并只保留下三角
#     relevant_correlations = relevant_correlations.tril() * (1 - torch.eye(len(a), len(a), device=device))
    
#     # 计算每行的权重
#     num_elements = (H.sum(1) * (H.sum(1) - 1)) / 2
#     edge_weights = (relevant_correlations.sum(1) / num_elements).clamp(min=0)
    
#     return torch.diag(edge_weights)