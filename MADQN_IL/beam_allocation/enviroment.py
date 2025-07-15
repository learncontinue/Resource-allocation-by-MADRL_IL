import numpy as np
import random
import math


class LEO_env:
    def __init__(self, n_user, n_beam):  # 初始化有车辆、资源块、距离、路径损耗和快速损耗因子都是0
        self.n_user = n_user
        self.n_beam = n_beam
        self.slot = 40
        self.request = None
        self.request_ = None
        self.PathLoss = None
        self.FastFading = None

    def reset(self):
        n_user = self.n_user
        request = np.random.poisson(lam=self.slot, size=(n_user, self.slot - 10))
        zero = np.zeros((n_user, 10))
        request = np.c_[zero, request]
        self.request = request / self.slot
        return self.request

    def step(self, action):
        request = self.request.copy()
        for i in range(self.n_beam):
            choice = action[i]
            r = 1  # 每个波束分配1的容量
            for j in range(self.slot):  # 前10个时隙没有业务
                if request[choice][j] <= r:  # 如果需求小于r，那么可以满足其需求，r就是准备分配的资源，全分配出去。
                    r = r - request[choice][j]
                    request[choice][j] = 0
                else:
                    request[choice][j] = request[choice][j] - r
                    break
        # step 2：去除数据矩阵最左侧一列 模拟时间前进一个时隙？
        request = np.delete(request, 0, 1)
        # step 3：随机生成泊松分布随机数，加在数据矩阵最右侧一列 新的业务需求的产生
        newData = np.random.poisson(lam=self.slot, size=self.n_user) / self.slot
        request_ = np.c_[request, newData]
        # 新场景的数据包总数
        a = 0
        for i in range(self.n_user):
            a = a + request_[i][self.slot - 1]
        # 吞吐的数据包总数
        throughout= np.sum(self.request) - (np.sum(request_) - a)
        # 换算成吞吐容量/Gbps 每个数据包大小100k，也就是1e-4 Gbps
        return request_, throughout


