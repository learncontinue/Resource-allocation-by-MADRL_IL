import numpy as np
import random
import math


class satellite_env:
    def __init__(self, n_user, n_sat):  # 初始化有车辆、资源块、距离、路径损耗和快速损耗因子都是0
        self.user = n_user
        self.sat = n_sat
        self.sig2_dB = -171.6  # 环境高斯噪声功率谱密度（dBm）
        self.thd_db = -123
        self.thd = 10 ** (self.thd_db / 10)  # 环境高斯噪声功率（mW）
        self.sig2_dB = -144# 环境高斯噪声功率谱密度（dBm）
        self.sig2 = 10 ** (self.sig2_dB / 10)  # 环境高斯噪声功率（mW）
        self.Pb = 24
        self.B = 500
        self.beamb = 125
        #self.Gt = 40.3  # 地球同步卫星
        #self.Gt = 42  # 低轨卫星最大增益
        #self.Gt = 42  # 低轨卫星最大增益
        self.Gt = 36  # 低轨卫星增益少点
        #self.Gr = 31.6  # 地球同步卫星
        self.Gr = 0
        #self.Loss = -209.6  # 地球同步卫星
        self.Loss = -173.4  # 600km的低轨卫星
        self.slot = 2  # 2ms 时隙
        self.TTL = 40

        self.weight = 1  # 权重越大代表吞吐量要求越高
        self.lamda_min = 50
        self.lamda_max = 150
        self.lamda = np.zeros(self.user)
        self.flue = np.zeros((self.user, self.TTL)) #需求矩阵

    def reset(self):
        self.flue = np.zeros((self.user, self.TTL))
        self.lamda = self.random_lamda()  # 得到泊松值
        ft = self.generate_ft(self.lamda)  # 产生当前时刻n个用户业务大小
        self.flue[:,0] = ft
        flue = self.flue
        return flue  # 返回系统的流量

    def step(self, choice, band_s, band_l):
        ## 计算下一状态
        # 先计算信道容量，每个时隙的吞吐量,大小N,其中只有K个不为0
        flue = self.flue.copy()  # 得到当前状态
        ct = self.capacity_t(choice, band_s, band_l)
        ft = np.sum(flue,axis=1)  # 对flue每一行，axis = 1
        tmax = self.TTL
        # 修改用户矩阵，并返回修改后的状态
        thout = np.zeros(self.sat)  # thout函数即为各波束的吞吐量
        for k in range(self.sat):
            index = choice[k]-1  # 取出用户的编号
            # ft 为当前存储池每个用户流量大小
            # 比较每个用户流量总需求和Ct，选出较小的
            thout[k] = min(ct[index], ft[index])  # 计算每个波束为用户提供的容量大小，不大于用户的总需求
            rest = thout[k]  # 拿到k个波束预分配数据大小
            # 对流量矩阵进行消除，从右向左，越往右代表时间延时越大，倒序满足要求，先满足流量时延大的
            for t in range(tmax):  # 最大时延矩阵
                if flue[index,tmax - t -1] <= rest:
                    rest = rest - flue[index,tmax-t-1]
                    flue[index,tmax - t-1] = 0
                else:
                    flue[index, tmax - t-1] = flue[index,tmax-t-1] - rest
                    break
        # 这是的flue是减去提供容量后的
        # 下一时隙
        # 产生新的需求，取原来状态的前40-1位，抛弃最后一位，将新产生的加到最左边变成flue_next
        ft_next = self.generate_ft(self.lamda)  # 产生当前时刻n个用户业务大小
        flue = np.delete(flue, tmax-1, 1)
        flue = np.c_[ft_next,flue]
        next_s = flue
        ## 计算吞吐量奖励
        r_throu = np.sum(thout)
        ## 计算时间延时奖励
        r_time = self.r_timedelay(next_s)
        ## 两部分奖励相加,吞吐量为奖励，时延为惩罚
        #reward = (r_throu/4.576) * self.weight - (1 - self.weight) * (r_time/20.199)
        reward = (r_throu/4.5) * self.weight - (1 - self.weight) * (r_time/20.199)
        return next_s, reward,r_throu,r_time,ct

    def capacity_t(self, choice, band_s, band_l):  # 计算t时刻的吞吐量大小，维度为波束大小
        N = self.user
        K = self.sat
        cover_rate = np.zeros((N,N)) # 用户间信道重叠49个，只有其中的12个有值
        inter = np.zeros(N)  # 对用户干扰，大小为N
        capacity_t = np.zeros(N)
        choice_one = np.zeros((N,K))
        h = self.Gt + self.Gr + self.Loss  # 信道模型为db
        p = self.Pb  # 功率为db模式
        # 生成关联矩阵 choice_one 大小N*K，里头为0/1
        for k in range(K):
            choice_one[choice[k]-1, k] = 1  # 生成信道匹配矩阵，仅有四位为1

        # 计算带宽复用比例,用band_s和band_l (大小都为K)

        for m1 in range(K):
            for m2 in range(K):
                if m1 != m2:
                    cover = sum(band_s[m1]*band_s[m2])  # 计算交叠了几个子信道带宽
                    cover_rate[choice[m1]-1, choice[m2]-1] = cover*self.beamb / band_l[m2]

        # 计算每个用户受到的干扰(仅计算关联用户的)
        for n in range(N):
            # 先计算每个用户干扰
            for k in range(K):  # 对该用户来说，
                if n != choice[k]-1:  #
                    inter[n] = inter[n] + cover_rate[choice[k]-1,n]*(10 ** ((h + p) / 10))  # 当前用户干扰，加上当前用户和与第k个波束关联用户之间的干扰

        # 计算信道容量,单位Mbps（slot）
        for n in range(N):  # 每个用户
            for k in range(K):
                if n == choice[k]-1:  # 如果该用户为关联用户，计算出信道容量
                    capacity = band_l[k] * (10 ** ((h + p) / 10)) / (inter[n] + self.thd + band_l[k] * self.sig2)
                    capacity_t[n] = capacity * 0.002

        return capacity_t


    def random_lamda(self):  # 得每个用户业务分布均值
        min = self.lamda_min
        max = self.lamda_max
        lamda = np.zeros(self.user)
        for i in range(self.user):
            lamda[i] = random.randint(min, max)
        return lamda

    def generate_ft(self, lamda):
        # self.Input_D = np.random.poisson(lam=self.slot, size=(self.user, self.slot - 10))
        ft = np.zeros(self.user)
        for i in range(self.user):
            ft[i] = (np.random.poisson(lam=lamda[i])) * 0.002  # 每个时隙0.02ms
        return ft

    def r_timedelay(self,next_s):
        #输入延时矩阵，N*slot，16*40
        aver_q_delay  = np.zeros(self.user)
        for n in range(self.user):
            sum_t = 0
            sum_d = 0
            for l in range(self.TTL):
                sum_t = sum_t + ( l + 1 ) * next_s[n,l]
                sum_d = sum_d + next_s[n,l]
            aver_q_delay[n] = sum_t/sum_d
        r_delay = np.max(aver_q_delay) - np.min(aver_q_delay)
        #r_delay = r_delay  # 归一化处理
        return r_delay
