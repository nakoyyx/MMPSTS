import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
from hgcn.train_hgcn import train_hgcn
from hgcn.HGNN import HGNN
from utils import Get_A,cal_similarity,fat_attention
import math
from xlstm import xLSTM



def xavier_init(size):  # [114,57]
    in_dim = size[0]  # 114
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size=size, scale=xavier_stddev)  # np.random.normal是一个正态分布， scale:正态分布的标准差，对应分布的宽度 size：输出的shape

class HardCalculateLayer(nn.Module):
    def __init__(self, device):
        super(HardCalculateLayer, self).__init__()
        self.device = device

    def forward(self, x):
        input_predict, input_true, true_input_gate = x  #(1,128,133)  (1,128,133) (128,1)
        #首先标记之前模态完全缺失的时刻
        # true_input_gate_1 = true_input_gate.repeat((1, input_predict.shape[-1] - 5))  #(128,1) => (128,128) 感觉这个和M是相等的
        # true_input_gate_1 = torch.cat((true_input_gate_1, torch.ones(true_input_gate.shape[0], 5).to(self.device)), dim=-1)  #这里把数据添加到gpu上  (128，133)
        # true_input_gate_1 = torch.unsqueeze(true_input_gate_1, dim = 0)  #(1, 128,133)添加一个维度

        # pred_input_gate_1 = 1- true_input_gate_1  #本来是1为非缺失，0为缺失，减之后相反 1缺失，0 存在 (1, 128,133)
        # pred_input_gate_1 = pred_input_gate_1.type(input_true.dtype)
        # input_true = -5 * pred_input_gate_1 + input_true * true_input_gate_1  #input_true表示当前时刻的输入，输入特征128缺失的地方填充为-5，不缺失就保留当前值，得分5维缺失为0，不缺失为当前值

        #接着标记缺失的评分
        pred_input_gate = torch.eq(input_true, -5)  #缺失(-5)的地方为true，非缺失为false
        pred_input_gate = pred_input_gate.type(input_true.dtype)#true为1，false为0
        true_input_gate = 1 - pred_input_gate  #这里相反，缺失为0，非缺失为1

        temp1 = pred_input_gate * input_predict  #input_predict是上个时刻的输出值用于插补这个时刻的缺失值，所以乘以pred_input_gate保留input_true的缺失值
        temp2 = true_input_gate * input_true  #保留input_true的非缺失值
        input_temp = temp1 + temp2
        return input_temp

class Temporal_Attention_layer(nn.Module):
    """
    num_of_vertices:Nodes num
    in_channels: input_dim
    num_of_timesteps:

    """
    def __init__(self, dim,T_dim):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.tensor(xavier_init([dim]))) # N
        self.U2 = nn.Parameter(torch.tensor(xavier_init([T_dim,dim]))) # F_in,N
        self.U3 = nn.Parameter(torch.tensor(xavier_init([T_dim]))) # F_in
        # self.be = nn.Parameter(torch.FloatTensor()) # 1, T, T
        # self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps)) # T, T

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        a=torch.matmul(x.permute(1,0,2), self.U1)
        lhs = torch.matmul(torch.matmul(x.permute(1,0,2), self.U1), self.U2)

        rhs = torch.matmul(x.permute(1,2,0),self.U3)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs.T)  # (B,T,N)(B,N,T)->(B,T,T)
        product=torch.sigmoid(product)
        # E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
        E_normalized = F.softmax(product, dim=1)
        zero = torch.zeros_like(E_normalized)
        Sparse_S_normalized = torch.where(E_normalized< torch.div(torch.mean(E_normalized),0.6),zero,E_normalized)
        

        return Sparse_S_normalized   
class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, dim,T_dim):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.tensor(xavier_init([T_dim]))) # (T)
        self.W2 = nn.Parameter(torch.tensor(xavier_init([dim,T_dim])))
        self.W3 = nn.Parameter(torch.tensor(xavier_init([dim])))
        # self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE)) # (1,N,N)
        # self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE)) # (N,N)

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x.permute(1,2,0), self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.sigmoid(product + self.bs)  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized
    
class Generator(nn.Module):
    def __init__(self, Dim, features_out,seq_num, pred_length, run_unit, view_num, latDim,hidden,xlstm_layers,device):
        super(Generator, self).__init__()
        self.G_W1 = nn.Parameter(torch.tensor(xavier_init([Dim * 2, Dim])))  # Data + Hint as inputs  (340，170)
        self.G_b1 = nn.Parameter(torch.tensor(np.zeros(shape=[Dim])))  # 170

        self.G_W2 = nn.Parameter(torch.tensor(xavier_init([Dim, Dim])))  # (170，170)
        self.G_b2 = nn.Parameter(torch.tensor(np.zeros(shape=[Dim])))  # 170

        self.G_W3 = nn.Parameter(torch.tensor(xavier_init([Dim, Dim])))  # (170，170)
        self.G_b3 = nn.Parameter(torch.tensor(np.zeros(shape=[Dim])))
        self.hgcn_k=5
        self.lables=3
        self.latDim = latDim #512
        self.device=device
        self.features_out=features_out
        self.features_out = features_out
        self.pred_length = pred_length
        self.run_unit = run_unit
        self.view_num = view_num
        self.seq_num = seq_num
        self.mid_layer_num = 1
        self.Dim = Dim  #170
        self.H_Dim1, self.H_Dim2 = Dim, Dim
        self.lables = 3
        self.latDim = latDim
        self.hidden = hidden
        self.xlstm_layers=xlstm_layers
        self.hgcn=train_hgcn(self.lables,self.Dim,self.latDim,self.hgcn_k,self.device)
        self.model_list = {"LSTM": LSTM(input_size=self.latDim + self.features_out, hidden_size=self.latDim, batch_first=False),
                           "xLSTM":xLSTM(input_size=self.latDim + self.features_out, head_size=25, num_heads=7, layers=self.xlstm_layers)}
        self.encoder_RNN = self.model_list[self.run_unit]
        self.classfier = nn.ModuleList()
        # self.classfier.append(nn.Linear(self.latDim+5, self.hidden))
        # self.classfier.append(nn.Linear(self.hidden, 64))
        self.classfier.append(nn.Linear(self.latDim+5, self.lables))
        self.temporal_att=Temporal_Attention_layer(self.Dim,5)
        # self.spatial_att=Spatial_Attention_layer(self.Dim,5)
        #Fusion：给MRI和Graphics单独接一个全连接层，然后拼接，再输入到三个全连接层中
        self.MRI = nn.Linear(self.Dim, self.Dim)
        self.Demo = nn.Linear(self.features_out, self.features_out)

        self.Fusion = nn.ModuleList()
        self.Fusion.append(nn.Linear(self.features_out+self.Dim, self.latDim))
        self.Fusion.append(nn.Linear(self.latDim, self.latDim))
        self.Fusion.append(nn.Linear(self.latDim, self.latDim))


    def forward(self, new_x, m,X_demo,X_score,X_missing_mask):
        # 首先插补，base的插补用噪声
        inputs = torch.cat(dim=-1, tensors=[new_x, m])  # Mask + Data Concatenate  [5，128,340]
        G_h1 = F.relu(torch.matmul(inputs, self.G_W1) + self.G_b1)  # (5,128,170)
        G_h2 = F.relu(torch.matmul(G_h1, self.G_W2) + self.G_b2)  # (5,128,170)
        G_prob = torch.sigmoid(torch.matmul(G_h2, self.G_W3) + self.G_b3)  # [0,1] normalized Output (5,128,170)
        input_new=new_x * m + G_prob * (1 - m)
        # MAI = F.relu(self.MRI(input_new)) + input_new #(5,128,170)???????
        # Demo = F.relu(self.Demo(X_demo)) + X_demo #(5,128,5) ?????
        # concat = torch.cat((MAI, Demo), dim=2) #(5,128,175)

        # concat1 = self.Fusion[0](concat) #(5,128,175)-->(5,128,128)[5,64,64]
        # concat2 = self.Fusion[1](concat1) #(5,128,128) [5,64,64]
        # encoder_inputs = self.Fusion[2](concat2) + concat1 #(5,128,128) ？？？？？[5,64,64]
        A_p=self.temporal_att(input_new)
        input_t1=input_new[0,:,:].unsqueeze(0)
        latent_temp = torch.cat([input_new,X_score],dim=2)
        all_encoder_outputs = []  #记录每个时刻的预测输出
        # 混合阶
        X_demo1=X_demo[0,:,:].unsqueeze(dim=0)
        output_hgcn=self.hgcn(input_t1,X_demo1,A_p)#+input_t1
        # output_hgcn=input_t1
        # cat_score
        input_xlstm=torch.cat((output_hgcn,X_score[0,:,:].unsqueeze(0)),dim=2)
        # lstm/xlstm
        encoder_state_h, encoder_state_c = self.encoder_RNN(input_xlstm)
        encoder_outputs=encoder_state_h
        states=encoder_state_c
        encoder_single_output=encoder_outputs
        all_encoder_outputs.append(encoder_single_output)  #保存第一个时间点的输出值
        for i in range(1, self.seq_num):  #从第二个时间点开始就需要进行数据插补了
            missing_mask_temp = X_missing_mask[:,i].unsqueeze(dim=1)  #(128,1)
            m_temp=m[i,:,:].unsqueeze(dim=0)
            input_temp = input_new[i,:,:].unsqueeze(dim=0) #(1,128,133)
            
            # input_temp=torch.cat([input_temp,X_score[i,:,:].unsqueeze(0)],dim=2)
            # 先插补MRI数据
            # input_temp=HardCalculateLayer(self.device)([encoder_single_output[:,:,:self.latDim],input_temp, m]) 
            HardCal1=encoder_single_output[:,:,self.latDim:]
            # HardCal2=input_temp[:,:,self.Dim:]
            # 再补全score数据
            score_t=X_score[i,:,:].unsqueeze(0)
            score_t= HardCalculateLayer(self.device)([HardCal1,score_t, missing_mask_temp])
            input_temp=torch.cat([input_temp,score_t],dim=2)
            #(1,128,133)特征部分缺失的都插补了，评分缺失的还是缺失？？？？？？？？
            # input_temp_score= HardCalculateLayer(self.device)([HardCal1,HardCal2, missing_mask_temp])  #(1,128,133)特征部分缺失的都插补了，评分缺失的还是缺失？？？？？？？？
            # input_temp=torch.cat((input_temp[:,:,:self.Dim],input_temp_score),dim=2)
            input_temp1=input_temp[:,:,:self.latDim]
            # similarity=cal_similarity(input_temp1,encoder_outputs_pre)
            X_demo1=X_demo[i,:,:].unsqueeze(0)
            #将特征经过超图进行降维
            output_hgcn=self.hgcn(input_temp1,X_demo1,A_p)#+input_temp1
            # output_hgcn=input_temp1
            
            input_xlstm=torch.cat((output_hgcn,input_temp[:,:,self.latDim:]),dim=2)
            encoder_state_h, encoder_state_c = self.encoder_RNN(input_xlstm, states)  # (1,128,128)  (1,128,128) (1,128,128)
            encoder_outputs=encoder_state_h#+input_temp
            states = encoder_state_c

            encoder_single_output=encoder_outputs
            if i != self.seq_num - 1:   # 如果不是最后一个时间点的话，就把每个时间点的预测都保存下来
                all_encoder_outputs.append(encoder_single_output)  #{list:4}  每一个是(1,128,133)
            if i== self.seq_num-2:
                encoder_t5=encoder_single_output
            encoder_outputs_pre=encoder_single_output[:,:,:self.latDim] #取非评分数据部分与下一个时间的真实数据计算similarity

        encoder_outputs = torch.cat(all_encoder_outputs ,dim=0) #(4,128,133)

        #Decoder
        decoder_inputs = encoder_single_output #(1,128,133)
        all_outputs=[]
        # all_outputs = all_encoder_outputs
        # all_outputs.append(encoder_single_output)# 将最后一个时间点的数据保存
        inputs = decoder_inputs #(1,128,133)
        decoder_state_c=encoder_state_c
        outputs = 0
        for i in range(self.pred_length):
            input_xlstm=inputs
            decoder_state_h, decoder_state_c = self.encoder_RNN(input_xlstm, states) # (1,128,128)  (1,128,128) (1,128,128)
            decoder_outputs=decoder_state_h#+input_xlstm
            states = decoder_state_c

            # for layer_num in range(self.mid_layer_num):
            #     decoder_outputs = F.relu(self.Fforward[layer_num](decoder_outputs))
            # outputs = self.Fforward[-1](decoder_outputs) + inputs #(1,128,133)？？？？？？
            outputs=decoder_outputs
            encoder_outputs_pre=outputs[:,:,:self.latDim] #取非评分数据部分与下一个时间的真实数据计算similarity

            all_outputs.append(outputs)
            inputs = outputs

        decoder_outputs_temp = torch.cat(all_outputs, dim=0)  #(8,128,133)
        if self.view_num == 2:  #{list:5} 每个元素是[2,16,1]
            # decoder_outputs = [decoder_outputs_temp[:, :, self.Dim + self.features_in[1] + i].unsqueeze(dim=2) for i in range(self.features_out)] #多模态拼接
            # decoder_outputs = [decoder_outputs_temp[:, :, self.Dim + i].unsqueeze(dim=2) for i in range(self.features_out)]  #但模态
            decoder_outputs_out = [decoder_outputs_temp[:, :, self.latDim + i].unsqueeze(dim=2) for i in range(self.features_out)]  #5个(8,128,1)
        else:
            decoder_outputs_out = [decoder_outputs_temp[:, :, self.Dim + i].unsqueeze(dim=2) for i in range(self.features_out)]

        # 分类，使用decoder的最后一个输出，不要评分
        # outputs = outputs.squeeze()[:, :self.latDim] #(128,128)
        # outputs=encoder_t5
        outputs = outputs.squeeze()
        # class_out = torch.sigmoid(self.classfier[0](outputs))#(128,128)-->(128,256)
        # class_out = torch.sigmoid(self.classfier[1](class_out))#(128,256)
        class_out = self.classfier[0](outputs)#(128,3)

        return G_prob, latent_temp, encoder_outputs,decoder_outputs_out, class_out,encoder_t5
        

class Discriminator(nn.Module):
    def __init__(self, input_size,hidden_size):
        super(Discriminator, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.Wgcn1 = nn.Parameter(torch.tensor(xavier_init([self.input_size, self.hidden_size])))
        self.bgcn1 = nn.Parameter(torch.zeros(self.hidden_size))
        self.Wgcn12 = nn.Parameter(torch.tensor(xavier_init([self.hidden_size, self.hidden_size])))
        self.bgcn12 = nn.Parameter(torch.zeros(self.hidden_size))
        self.dropout=0.1
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size, 128,dtype=torch.double),  # w_1 * x + b_1
            nn.BatchNorm1d(128,dtype=torch.double),
            nn.ReLU(),
            nn.Linear(128, 64,dtype=torch.double),
            nn.ReLU(),
            nn.Linear(64, 2,dtype=torch.double),
            nn.ReLU(),

            nn.Softmax()
        )
        self.model_list = {"LSTM": LSTM(input_size=175, hidden_size=self.hidden_size, batch_first=False)}
        self.encoder_RNN = self.model_list['LSTM']

    def forward(self, input,demo):
        # # 两层GCN+全连接
        # #先求皮尔逊相关系数
        correlations = torch.corrcoef(input)
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
        # 归一化邻接矩阵
        adjacency_matrix=correlations
        input_features=input
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
        
        encoder_outputs, (encoder_state_h, encoder_state_c)=self.encoder_RNN(input.unsqueeze(0).to(torch.float32))
        output=self.linear(encoder_outputs.squeeze(0).to(torch.float64))

        return output 
    
class Discriminator1(nn.Module):
    def __init__(self, input_size,hidden_size):
        super(Discriminator1,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.Wgcn1 = nn.Parameter(torch.tensor(xavier_init([self.input_size, self.hidden_size])))
        self.bgcn1 = nn.Parameter(torch.zeros(self.hidden_size))
        self.Wgcn12 = nn.Parameter(torch.tensor(xavier_init([self.hidden_size, self.hidden_size])))
        self.bgcn12 = nn.Parameter(torch.zeros(self.hidden_size))
        self.dropout=0.1
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size, 128,dtype=torch.double),  # w_1 * x + b_1
            nn.BatchNorm1d(128,dtype=torch.double),
            nn.ReLU(),
            nn.Linear(128, 64,dtype=torch.double),
            nn.ReLU(),
            nn.Linear(64, 2,dtype=torch.double),
            nn.ReLU(),

            nn.Softmax()
        )
        self.model_list = {"LSTM": LSTM(input_size=175, hidden_size=self.hidden_size, batch_first=False)}
        self.encoder_RNN = self.model_list['LSTM']
        Dim=170
        H_Dim1=170
        H_Dim2=170
        self.D_W1 = nn.Parameter(torch.tensor(xavier_init([Dim * 2, H_Dim1])))  # Data + Hint as inputs  (340,170)
        self.D_b1 = nn.Parameter(torch.tensor(np.zeros(shape=[H_Dim1])))  # 170

        self.D_W2 = nn.Parameter(torch.tensor(xavier_init([H_Dim1, H_Dim2])))  # (170,170)
        self.D_b2 = nn.Parameter(torch.tensor(np.zeros(shape=[H_Dim2])))  # 170

        self.D_W3 = nn.Parameter(torch.tensor(xavier_init([H_Dim2, Dim])))  # (170,170)
        self.D_b3 = nn.Parameter(torch.tensor(np.zeros(shape=[Dim])))
        
    def forward(self, new_x,h,input,fake_input,demo):
        # # 两层GCN+全连接
        # #先求皮尔逊相关系数
        correlations = torch.corrcoef(input)
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
        # 归一化邻接矩阵
        adjacency_matrix=correlations
        input_features=input
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
        encoder_outputs, (encoder_state_h, encoder_state_c)=self.encoder_RNN(input.unsqueeze(0).to(torch.float32))
        output=self.linear(encoder_outputs.squeeze(0).to(torch.float64))
        real_y=output

        #fake
        correlations = torch.corrcoef(fake_input)

        # 调整相关系数矩阵
        correlations = correlations * combined_mask
        correlations[correlations<0]=0
        # 归一化邻接矩阵
        adjacency_matrix=correlations
        input_features=input
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
        encoder_outputs, (encoder_state_h, encoder_state_c)=self.encoder_RNN(input.unsqueeze(0).to(torch.float32))
        output=self.linear(encoder_outputs.squeeze(0).to(torch.float64))
        fake_y=output
        
        
        inputs = torch.cat(dim=-1, tensors=[new_x, h])  # Hint + Data Concatenate
        D_h1 = F.relu(torch.matmul(inputs, self.D_W1) + self.D_b1)
        D_h2 = F.relu(torch.matmul(D_h1, self.D_W2) + self.D_b2)
        D_logit = torch.matmul(D_h2, self.D_W3) + self.D_b3
        D_prob = torch.sigmoid(D_logit)  # [0,1] Probability Output (128,57)

        return real_y,fake_y,D_prob


