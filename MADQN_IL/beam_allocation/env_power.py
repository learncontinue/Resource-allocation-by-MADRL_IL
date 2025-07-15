import numpy as np
import random
import math


class LEO_env:
    def __init__(self, n_user, n_bs):  # 初始化有车辆、资源块、距离、路径损耗和快速损耗因子都是0
        self.user = n_user
        self.bs = n_bs
        self.sig2_dB = -174  # 环境高斯噪声功率（dBm）
        self.sig2 = 10 ** (self.sig2_dB / 10) / 1000  # 环境高斯噪声功率（W）
        self.PO = 5
        self.B = 500
        self.PathLoss = None
        self.FastFading = None
        self.state = np.zeros((self.bs, self.user)) #此时刻状态，信道矩阵
        self.state_ = np.zeros((self.bs, self.user)) #下一时刻状态，信道矩阵
    def reset(self):
        H = self.update_fast_fading()
        self.state = H.copy()
        return H
    def step(self, action):
        H = self.update_fast_fading()  #状态更新
        self.state_ = H.copy()
        reward = self.compute_reward(action)  # 还是用的state，计算
        self.state = H.copy()  #计算，将下一状态转移到此状态
        return H,reward

    def update_fast_fading(self):#快速损耗更新函数
        h = 1/np.sqrt(2) * (np.random.normal(size=(self.bs, self.user)) +
                            1j * np.random.normal(size=(self.bs, self.user)))#h为随机复变量，除于根号2
        FastFading = 20 * np.log10(np.abs(h))
        return np.abs(h)

    def compute_reward(self, action):
        """
        Used for Training
        add the power dimension to the action selection
        :param action:
        :return:
        """
        power_selection = action.copy()
        power_selection_db = np.zeros((self.bs))
        for k in range(self.bs):
            if power_selection[k] != 0:
                power_selection_db[k] = 10 * np.log10(power_selection[k])  # 功率选择
        interference = np.zeros((self.bs))
        v2i_rate_list = np.zeros((self.bs))
        # 计算干扰和噪声
        for n in range(self.user):
            for m in range(self.bs):
                if power_selection[m] != 0:
                    if m != n:
                        interference[n] += power_selection[m] *(self.state[m, n] ** 2)
            interference[n] += self.sig2
        # 计算频谱效率
        for k in range(self.bs):
            if power_selection[k] != 0:
                v2i_rate_list[k] = self.B*np.log2(1 + np.divide(power_selection[k] *(self.state[k, k] ** 2),interference[k]))
        #ee = np.divide(v2i_rate_list, (power_selection + self.PO))
        return np.sum(v2i_rate_list)