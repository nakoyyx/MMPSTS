import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
from AMTFS.AMTFS1 import AMTFS
from hgcn.HGNN import HGNN
from hgcn.visual_data import load_feature_construct_H
from hgcn import weight,hypergraph_utils
from scipy.optimize import minimize
from hgcn.train_s_w import CustomModel
# from torch_cluster import kmeans
# from HGNN import HGNN
# import weight,hypergraph_utils
# from visual_data import load_feature_construct_H
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import warnings
warnings.filterwarnings("ignore")
def xavier_init(size):  # [114,57]
    in_dim = size[0]  # 114
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size=size, scale=xavier_stddev)  # np.random.normal是一个正态分布， scale:正态分布的标准差，对应分布的宽度 size：输出的shape

class GCN(torch.nn.Module):
    def __init__(self, input_size,hidden_size,device):
        super(GCN, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.Wgcn1 = nn.Parameter(torch.tensor(xavier_init([self.input_size, self.hidden_size])))
        self.bgcn1 = nn.Parameter(torch.zeros(self.hidden_size))
        self.Wgcn12 = nn.Parameter(torch.tensor(xavier_init([self.hidden_size, self.hidden_size])))
        self.bgcn12 = nn.Parameter(torch.zeros(self.hidden_size))
        self.dropout=0.0
        
    def forward(self, input_features, adjacency_matrix):
        # 归一化邻接矩阵
        D = adjacency_matrix.sum(1, keepdim=True)  # 度矩阵
        D=D.squeeze(1)
        D_inv_sqrt = D.pow(-0.5)  # 度矩阵的-0.5次方
        D_inv_sqrt=torch.diag(D_inv_sqrt)
        # normalized_adjacency_matrix = D_inv_sqrt*adjacency_matrix*D_inv_sqrt
        normalized_adjacency_matrix = torch.matmul(D_inv_sqrt,torch.matmul(adjacency_matrix,D_inv_sqrt))
        # 应用图卷积
        support = torch.mm(input_features, self.Wgcn1)
        output = torch.mm(normalized_adjacency_matrix, support) + self.bgcn1
        output=F.relu(output)
        output=F.dropout(output,self.dropout)
        # 应用图卷积
        support = torch.mm(output, self.Wgcn12)
        output = torch.mm(normalized_adjacency_matrix, support) + self.bgcn12

        # 应用非线性激活函数
        return output
class train_hgcn(nn.Module):
    def __init__(self,n_class,in_ch,n_hid,k,device):
        super(train_hgcn, self).__init__()
        self.n_class=n_class
        self.n_hid=n_hid
        self.in_ch=in_ch
        lr=0
        weight_decay=0
        gamma=0
        self.dropout=0.1
        self.lr=lr
        self.weight_decay=weight_decay
        self.gamma=gamma
        self.device=device
        self.k=k
        self.milestones=[100]
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = HGNN(in_ch=self.in_ch,n_class=self.n_class,n_hid=self.n_hid,dropout=self.dropout)
        self.model= self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer ,gamma=self.gamma)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer ,milestones=self.milestones,gamma=self.gamma)
        self.L=0.2
        self.GCN=GCN(self.in_ch,self.n_hid,device)
        
        self.layer_norm = nn.LayerNorm(self.n_hid, eps=1e-6)

    def _get_S_for_H(self,feat1):
        with torch.no_grad():
            feat = feat1.detach() 
        rows=feat.shape[0]
        #2.计算欧氏距离，将对角欧氏距离变为inf
        distX = torch.tensor(np.zeros((rows,rows)),device=self.device)
        # 计算feat中所有向量之间的距离
        distX = torch.cdist(feat, feat)  # cdist 计算每一对之间的距离
        # 创建一个与 dist 大小相同的矩阵，对角线元素为无穷大，其余元素为0
        distX_max=torch.max(distX)
        inf_matrix = torch.diag(torch.full((distX.shape[0],), distX_max.item(), dtype=distX.dtype))
        inf_matrix=inf_matrix.to(device=self.device)
        tmp=torch.eye(rows, dtype=torch.bool)
        tmp=tmp.to(device=self.device)
        # 将 dist 的对角线元素替换为无穷大
        dist = torch.where(tmp, inf_matrix, distX)
        
        dist=dist.to(torch.float32)
        #1.首先初始化S
        S = torch.rand(dist.shape[0],dist.shape[0],requires_grad=True,device=self.device)
        # 目标函数，包括 L2,1 范数的正则化项
        # 惩罚系数
        beta = 1.0
        alpha=0.5
        # 优化器
        optimizer_S = torch.optim.Adam([S], lr=0.1)

        # 优化循环
        for epoch in range(1000):
            tmp=torch.mul(dist, S**2)
            # tmp[tmp<0]=0
            dist_S = torch.sum(tmp)
            D=torch.sqrt(torch.sum(S ** 2, dim=0))
            D = torch.diag(1 / D)
            trace_term = torch.trace(torch.matmul(S.T, torch.matmul(D, S)))
            
            original_objective = dist_S + alpha * trace_term
            # 计算惩罚项
            penalty_term = torch.sum(torch.max(S.sum(dim=1) - 1, torch.tensor(0.0)))
            # 计算 S 中小于 0 的元素的和
            penalty_negative = torch.sum(torch.where(S < 0, S, torch.tensor(0.0).type_as(S)))

            # 计算 S 中大于 1 的元素的和
            penalty_positive = torch.sum(torch.where(S > 1, S - 1, torch.tensor(0.0).type_as(S)))

            # 完整的惩罚项是小于 0 的元素和与大于 1 的元素和的和
            penalty = beta * (-penalty_negative + penalty_positive)
            # 完整的目标函数（原始目标函数 + 惩罚项）
            loss = original_objective + beta * penalty_term+beta*penalty
            if loss.item()<1:
                # print(f"Epoch {epoch}, Loss: {loss.item()}")
                # print('break earlier')
                break
            # 梯度清零
            optimizer_S.zero_grad()      
            # 反向传播
            loss.backward()      
            # 更新参数
            optimizer_S.step()
            # 打印信息
            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch}, Loss: {loss.item()}")
        with torch.no_grad():
            S = S.detach()
        S[S<0]=0
        # 设定每行要保留的前k大元素的数量
        k = 5

        # 获取每一行的前k大元素的索引
        topk_indices = torch.topk(S, k=k, dim=1).indices

        # 创建一个与 S 形状相同的布尔掩码，初始化为 False
        mask = torch.zeros_like(S, dtype=torch.bool)

        # 将 topk_indices 指定的元素位置设置为 True
        for i in range(S.size(0)):
            mask[i][topk_indices[i]] = True

        # 使用掩码来更新 S，将非前k大的元素置为0
        S *= mask

        maxs,max_idx=torch.max(S,dim=0)
        nonzero_count = torch.count_nonzero(S,dim=1)

        # 计算每一行的绝对值和
        row_sums = torch.abs(S).sum(dim=1, keepdim=True)

        # L1 归一化
        S = torch.div(S, row_sums)
        # nonzero_count = torch.count_nonzero(S,dim=1)
        # rowsum=torch.sum(S,dim=1)
        torch.diagonal(S, 0).fill_(1)
        return S.T

    def H_Kmeans(self,feat, H):
        from sklearn.cluster import KMeans
        
        # 将 feat 转换为 CPU 上的 NumPy 数组进行 KMeans 聚类
        feat_np = feat.data.cpu().numpy()
        num_clusters=10
        kmeans = KMeans(n_clusters=num_clusters).fit(feat_np)
        centroids = torch.from_numpy(kmeans.cluster_centers_).to(self.device)
        
        # 计算 feat 中每个样本到每个质心的距离
        distances = torch.norm(feat.unsqueeze(1) - centroids.unsqueeze(0), dim=2)
        
        # 创建 H1 矩阵
        H1 = torch.zeros((feat.shape[0], num_clusters), device=self.device).to(torch.float64)
        
        # 遍历质心
        for j in range(num_clusters):
            # 找到离第 j 个质心最近的 K1 个样本的索引
            _, nearest_indices = torch.topk(distances[:, j], self.k, largest=False)
            
            # 将 H1 中对应位置的值设置为与质心的距离
            H1[nearest_indices, j] = distances[nearest_indices, j]
        
        # 归一化 H1
        H1 /= H1.sum(dim=0, keepdim=True)
        H1 *= H1.sum(dim=1, keepdim=True)
        
        # 将 H1 转换为 float32
        H1 = H1.to(torch.float32)
        H=torch.cat([H,H1],dim=1)
        return H

    def H_KNN(self,feat):
        # 获取特征数量
        N = feat.shape[0] 
        # 初始化H矩阵为零
        H = torch.zeros((N, N), dtype=torch.float, device=self.device) 
        # 计算feat中所有向量之间的距离
        dist_matrix = torch.cdist(feat, feat)
        # 为每个特征向量找到k个最近的距离索引
        _, topk_indices = torch.topk(dist_matrix, k=self.k, dim=1, largest=False)
        # # 确保topk_indices中没有自身的索引
        # topk_indices = topk_indices[topk_indices != torch.arange(N, device=device)].unsqueeze(1)
        
        # 扩展topk_indices以匹配H矩阵的形状
        topk_indices_expanded = topk_indices.expand(-1, self.k)
        
        # 使用scatter_填充H矩阵
        H.scatter_(1, topk_indices_expanded, 1.0)
        return H.T
        
    def construct_hypergraph(self,feat):
        
        # 增加Kmeans，每个时期都做kmeans，得到k个center，每个时期每个subject都得到一条超边，N*5条超边
        # H=self._get_S_for_H(feat)
        H=self.H_KNN(feat)
        H_tmp=H
        # # 检查x是否全为零
        # is_all_zeros = torch.all(torch.eq(similarity, 0))
        # if is_all_zeros!=True:
        #     H=similarity*H_pre+(1-similarity)*H
        # H=self.H_Kmeans(feat,H)
        # W = weight.set_weight(H, feat,self.device).to(torch.float32)
        W=torch.eye(feat.shape[0], device=self.device).to(torch.float32)
        
        H=H.to(torch.float32)
        G,DE = hypergraph_utils.generate_G_from_H(H, W, variable_weight=False)
        G=torch.tensor(G, device=self.device)
        DE=torch.tensor(DE,device=self.device)
        
        G=G.to(torch.float64)
        
        
        
        return feat,G,H_tmp

    def forward(self,feat,demo,a_p):
        #先构造超图
        #得到高阶的feat
        if feat.dim()==3:
            feat=feat.squeeze(0)
        feat=feat.to(torch.float64)
        
        # gcn部分，首先计算demo的值，乘以皮尔逊相关系数，得到A，进行切比雪夫卷积
        demo=demo.squeeze(0)
        A_pre=torch.zeros((demo.shape[0],demo.shape[0]),device=self.device)
        #先求皮尔逊相关系数
        correlations = torch.corrcoef(feat)

        # 计算条件掩码
        gender_mask = torch.eq(demo[:, 0], demo[:, 0].t())
        apoe1_mask = torch.eq(demo[:, 1], demo[:, 1].t())
        apoe2_mask = torch.eq(demo[:, 2], demo[:, 2].t())
        age_mask = torch.abs(demo[:, 3] - demo[:, 3].t()) < 5
        weight_mask = torch.abs(demo[:, 4] - demo[:, 4].t()) < 5

        combined_mask = gender_mask & apoe1_mask & apoe2_mask & age_mask & weight_mask
        # 调整相关系数矩阵
        correlations = correlations * combined_mask
        correlations[correlations<0]=0
        # outputs_pair=self.GCN(feat,correlations+a_p)
        outputs_pair=self.GCN(feat,correlations)
        # feat_pair_hyper=torch.cat((outputs,outputs_pair),dim=1
        feat,G,H=self.construct_hypergraph(feat)
        outputs = self.model(feat,G)
        feat_pair_hyper=outputs+outputs_pair
        outputs=self.layer_norm(feat_pair_hyper)
        return outputs.unsqueeze(0)

    # def forward(self,feat,H_pre,similarity,it,G):
    #     #先构造超图
    #     #得到高阶的feat
    #     feat=feat.to(torch.float64)
    #     if it==0:
    #         feat,G,H=self.construct_hypergraph(feat,H_pre,similarity)
    #     # feat=feat.to(torch.float32)
    #     else:
    #         H=H_pre
    #     outputs = self.model(feat,G)
    #     return outputs.unsqueeze(0),G,H

        

if __name__ == '__main__':
    phase='train'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # num_epochs=40
    # feat = torch.tensor(np.random.rand(5, 23, 50), device=device)
    # s_idx= torch.tensor(np.random.rand(23,23), device=device)
    # lbls=torch.tensor(np.random.randint(1,3,23),device=device).long()
    # tg=train_hgcn(3,
    #                 32,
    #                 250,
    #                 0.5,
    #                 0.01,
    #                 0.1,
    #                 0.99,
    #                 device)
    
    # precision, accuracy, f1, recall,hgcn_loss=tg.new_train_model(100,feat,lbls,s_idx)
    
    # re=tg.train_model(num_epochs,feat,lbls,'train')
        