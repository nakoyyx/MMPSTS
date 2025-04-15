import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# # 假设 S 和 W 是需要优化的张量
# S = torch.randn(10, 10, requires_grad=True)
# W = torch.randn(10, 10, requires_grad=True)

# 创建一个简单的模型，将 S 和 W 作为模型的参数
class CustomModel(nn.Module):
    def __init__(self, s_dim,w_dim,device):
        super(CustomModel, self).__init__()
        self.device=device
        self.beta = 1.0
        self.alpha=0.5
        #1.首先初始化S,w
        S1 = torch.rand(s_dim,s_dim)
        W1=torch.rand(w_dim[0],w_dim[1])
        # 对 W 的每一列执行 softmax 操作
        W1 = F.softmax(W1, dim=1)
        # 将feat的每个时间每个个体对应的feature都乘以w，N表示个体
        # feat：time*N*dim，torch
        # W：time* dim，torch
        # 确保 W 的形状适合广播：(time, 1, dim)
        ########################################
        W1 = W1.unsqueeze(1)        
        # 将 S 和 W 转换为模型的参数
        self.S = nn.Parameter(S1)
        self.W = nn.Parameter(W1)
        # self.register_parameter('S', nn.Parameter(self.S))
        # self.register_parameter('W', nn.Parameter(self.W))

    def L2_distance_1(self,a, b):
        """
        Compute squared Euclidean distance between two matrices a and b.

        Parameters:
            a: numpy array, matrix with each column representing a data point
            b: numpy array, matrix with each column representing a data point

        Returns:
            d: numpy array, distance matrix between a and b
        """
        if len(a.shape)==1:
        # if a.shape[0] == 1:
            # v=a.unsqueeze(0)
            # g=torch.tensor(np.zeros(1, a.shape[1]),device=self.device)
            fa=np.zeros((1,a.shape[0]))
            fb=np.zeros((1,b.shape[0]))
            a = torch.cat((a.unsqueeze(0), torch.tensor(fa,device=self.device)), dim=0)
            b = torch.cat((b.unsqueeze(0), torch.tensor(fb,device=self.device)), dim=0)

        aa = torch.sum(a * a,dim=0)
        bb = torch.sum(b * b,dim=0)

        ab = torch.matmul(a.T, b)
        # 将 aa 在行方向上重复 size(bb, 1) 次，并将其转置以便与 bb 相加
        aa_repeated = aa.unsqueeze(1).repeat(1, bb.shape[0])

        # 将 bb 在列方向上重复 size(aa, 0) 次，并将其转置以便与 aa 相加
        bb_repeated = bb.unsqueeze(0).repeat(aa.shape[0], 1)

        # 计算欧氏距离平方矩阵
        d = bb_repeated + aa_repeated - 2 * ab
        d = d.real
        d = torch.maximum(d, torch.zeros_like(d))
        return d


    def cal_l21(self,matrix):
        D=torch.sqrt(torch.sum(matrix ** 2, dim=0))
        D = torch.diag(1 / D)
        loss = torch.trace(torch.matmul(matrix, torch.matmul(D, matrix.T)))
        return loss
    def cal_sum1_penalty(self,matrix):
        # 惩罚项，元素之和为1
        loss=torch.sum(matrix,dim=0)

        return 100*torch.abs(torch.sum(loss-1))   

    def forward(self,feat):
        # 计算欧氏距离
        #2.计算欧氏距离，将对角欧氏距离变为inf
        # 使用广播特性对 feat 的每个时间步和每个个体的特征乘以 W
        rows=feat.shape[1]
        feat = feat * self.W
        distX = torch.tensor(np.zeros((rows,rows)),device=self.device)
        for i in range(feat.shape[0]):
            # 待改torch
            distX += self.L2_distance_1(feat[i].T, feat[i].T)
        # 创建一个与 dist 大小相同的矩阵，对角线元素为无穷大，其余元素为0
        distX_max=torch.max(distX)
        inf_matrix = torch.diag(torch.full((distX.shape[0],), distX_max.item(), dtype=distX.dtype))
        inf_matrix=inf_matrix.to(device=self.device)
        tmp=torch.eye(rows, dtype=torch.bool)
        tmp=tmp.to(device=self.device)
        # 将 dist 的对角线元素替换为无穷大
        dist = torch.where(tmp, inf_matrix, distX)
        
        dist=dist.to(torch.float32)
        tmp=torch.mul(dist, self.S**2)
        # tmp[tmp<0]=0
        dist_S = torch.sum(tmp)
        W_tmp=self.W
        W_tmp=W_tmp.squeeze(1)
        l21_S=self.cal_l21(self.S)
        l21_W=self.cal_l21(W_tmp)
        
        original_objective = dist_S# + self.alpha * l21_S+self.alpha*l21_W
        # 计算惩罚项
        # ||W||=1&&||S||=1
        penalty_sum1_W=self.cal_sum1_penalty(W_tmp)
        penalty_sum1_S=self.cal_sum1_penalty(self.S)
        # 计算 S 中小于 0 的元素的和
        penalty_negative = torch.sum(torch.where(self.S < 0, self.S, torch.tensor(0.0).type_as(self.S)))

        # 计算 S 中大于 1 的元素的和
        penalty_positive = torch.sum(torch.where(self.S > 1, self.S - 1, torch.tensor(0.0).type_as(self.S)))

        # 完整的惩罚项是小于 0 的元素和与大于 1 的元素和的和
        penalty = self.beta * (-penalty_negative + penalty_positive)
        # 完整的目标函数（原始目标函数 + 惩罚项）
        loss = original_objective + +self.beta*(penalty+penalty_sum1_W+penalty_sum1_S)

        return loss,self.S,self.W

