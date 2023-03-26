# -*- coding:utf-8 -*-
# author: Xinge

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
import multiprocessing
import torch_scatter


class cylinder_fea(nn.Module):

    def __init__(self, grid_size, fea_dim=3,  # init第一参数grid_size在yaml中设置 480 360 32
                 out_pt_fea_dim=64, max_pt_per_encode=64, fea_compre=None):  # 第四参数用默认，第五参数 fea_compre在yaml中num_input_features 设的16
        super(cylinder_fea, self).__init__()

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),  # init 第二参数fea_dim在yaml fea_dim设置 9

            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_pt_fea_dim)  #init 第三参数out_pt_fea_dim在yaml out_fea_dim设置 256
        )

        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_pt_fea_dim

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea, xy_ind):  
        # 总之，第一个类输入两个list[tensor([N1,9]),tensor([N2,9])]； list[tensor([N1,3]),tensor([N2,3])]
        # 输出两个tensor unq[m,4]代表了独特m个点的位置和batch； processed_pooled_data[m,16]对应压缩后的16维特征
        cur_dev = pt_fea[0].get_device()
        # print("should be [n,9]",pt_fea[0].shape)  # yes [N,9]
        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):  # len(xy_ind)就是list的长度，即batchsize 2
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))
        cat_pt_fea = torch.cat(pt_fea, dim=0)  # torch.cat让一个list中两个tensor按一定维度合成了一个 所以是tensor，[2N,9]
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)  # tensor,[2N,4]
        pt_num = cat_pt_ind.shape[0]
        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)  # 打乱点 不过特征和位置还是匹配的
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)   # [x,4]不重复的 其中第一维只有 0 1 区分batch
        # print(unq.shape)  # 不重复的最大点数[M,4]
        # print("unq_inv",unq_inv.shape)  # 关键是torch.unique的机制，inv维度是输入维度的（总点数）每个值的范围是0-输出维度-1（不重复的最大数量）
        # process feature
        processed_cat_pt_fea = self.PPmodel(cat_pt_fea)
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]
        # print("guess [m,1]",pooled_data.shape)  # 猜错了，是[m.256] 在同一分组的k个256维向量对每一维找k个里的最大，最终得到[m,256]个向量
        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)  # 再压缩特征到[m,16] 不过我想这么多全连接应该不快
        else:
            processed_pooled_data = pooled_data
        return unq, processed_pooled_data  # 返回m个独特点的[m,4]位置 和 [m,16]特征
