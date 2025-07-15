import numpy as np
import random
import math


class terrestrial_env:
    def __init__(self, n_user, n_BS):  # 初始化有车辆、资源块、距离、路径损耗和快速损耗因子都是0
        self.user = n_user
        self.bs = n_BS
        self.Gr = 1  # 卫星
        self.Loss = 1  # 卫星
        self.Gt = 1  # 卫星
        self.Pb = 1  # 卫星
        self.beam = 4
        self.antenna = 3
        self.sig2_dB = -117  # 环境高斯噪声功率（在500MHz情况下）
        self.threshhold = -123
        self.sig2 = 10 ** (self.sig2_dB / 10)  # 环境高斯噪声功率（mW）
        #self.Pb_max_dB = 28  # 地面基站最大发射功率
        self.Pb_max_dB = 20 # 地面基站最大发射功率
        self.Pb_max = 10 ** (self.Pb_max_dB / 10)
        self.Pb_min = 0.1 * self.Pb_max  # 地面基站最大发射功率
        self.B = 500  # 地面基站带宽不变。500MHz
        self.f0 = 20000  # 中心频率20GHz

        self.slot = 2  # 2ms 时隙
        self.TTL = 40

        self.weight = 1  # 权重越大代表吞吐量要求越高
        self.lamda_min = 50
        self.lamda_max = 300
        self.lamda = np.zeros(self.user)
        self.flue = np.zeros((self.user, self.TTL))  # 需求矩阵

        self.bs_pos = np.zeros((n_BS, 2))
        self.user_pos = np.zeros((n_user, 2))
        self.distance = self.get_distance()
        self.angle_bs_user = self.get_angle()

    def reset(self):
        self.flue = np.zeros((self.user, self.TTL))
        self.lamda = self.random_lamda()  # 得到泊松值
        ft = self.generate_ft(self.lamda)  # 产生当前时刻n个用户业务大小
        self.flue[:, 0] = ft
        flue = self.flue
        return flue  # 返回系统的流量

    def step(self, choice_bs, power_bs,user_by_beam):
        ## 计算下一个状态

        # 先计算信道容量，每个时隙的吞吐量,大小N,其中只有K个不为0
        flue = self.flue.copy()  # 得到当前状态
        # 奖励就是吞吐量
        # ct为每个基站的吞吐量，大小为3*3
        tmax = self.TTL
        ct,Imax,choice_ant,h = self.capacity_t_bs(choice_bs, power_bs,user_by_beam)
        c_perslot = ct*0.002

        # 计算对卫星用户的最大干扰
        # 修改用户矩阵，并返回修改后的状态
        # 首先使用生成的吞吐量给用户提供服务
        thout = np.zeros((self.bs,self.antenna))
        ft = np.sum(flue, axis=1)
        for m in range(self.bs):
            for w in range(self.antenna):
                capacity_user = c_perslot[m, w]
                if capacity_user != 0:
                    index_user = int(np.array(np.where(choice_ant[:,m,w]==1)))# 取出用户编号
                    thout[m, w] = min(capacity_user,ft[index_user])
                    rest = thout[m, w].copy()
                    for t in range(tmax):
                        if flue[index_user,tmax-t-1] <= rest:
                            rest = rest - flue[index_user,tmax-t-1]
                            flue[index_user,tmax-1-t] = 0
                        else:
                            flue[index_user,tmax-1-t] = flue[index_user,tmax-t-1]-rest
                            break
        # 其次加上产生新的用户需求
        self.flue = flue
        ft_next = self.generate_ft(self.lamda)  # 产生当前时刻n个用户业务大小
        flue = np.delete(flue, tmax - 1, 1)
        flue = np.c_[ft_next, flue]
        next_s = flue
        reward = np.sum(np.sum(thout)*500)  # 总吞吐量作为奖励。
        if Imax == 0:
            Imax_db = -200
        else:
            Imax_db = 10 * math.log10(Imax)
        if Imax_db > self.threshhold:
            # reward = np.array([1000.0])  # 如果大于门限值，奖励设置为-100，较低
            reward = np.array([0.0])
        #print('capacity', reward, 'Imax', Imax_db)
        return next_s, reward

    def capacity_t_bs(self, choice_bs, power_bs, choice_sat):
        power = power_bs*(self.Pb_max/3)
        B = 500
        n_ant = self.antenna
        n_bs = self.bs
        n_user = self.user
        capacity = np.zeros((n_bs, n_ant))
        d = self.distance
        # 先求信道系数h
        hn = 1.5  # 用户接收机高度0.5m
        hm = 50  # 地面天线高度50m
        h_bs = np.zeros((n_bs, n_ant, n_user))  # 9*16 大致是
        gmax = 20  # 地面基站发射增益20dBi
        Am = 20  # 天线前后比典型值
        ang_3db = 30  # 3db角度
        f0 = 20000
        # 用户和天线位置夹角计算
        angle = self.angle_bs_user

        for m in range(n_bs):
            for x in range(n_ant):
                for n in range(n_user):
                    # 先求两个损耗
                    L_los = 32.44+20* math.log10(f0) + 40* math.log10(d[m, n])
                    #L_bul = 120 + 40 * math.log10(d[m, n]) - 20 * math.log10(hn) - 20 * math.log10(hm)
                    # 再求发射天线增益,假设天线方向为0,120,240
                    ang = angle[m, x, n]  # 用户和天线位置夹角
                    # g = 10**(gmax/10 + max(-0.6*(ang/ang_3db)**2,  -Am/10))
                    g_db = gmax + max(-6 * (ang / ang_3db) ** 2, -Am)
                    h = - L_los + g_db
                    h_bs[m, x, n] = 10 ** (h / 10)  # db转增益
        # 再求干扰I,每个用户受到的干扰，和关联状态有关

        # 关联矩阵大小为3*16，每个基站关联小于3个用户
        # 干扰的计算有问题，就是，这个用户首先要位于基站范围内才会受到干扰，其次它最多受到3个天线的干扰，而不是所有基站对他都有干扰。
        # 如何判断用户是否在基站范围内呢，嗯，，看距离，一个用户和三个基站有距离，看看最短距离是否小于半径，小于则位于范围内，有干扰。
        # 先判断用户和基站的哪个天线相连choice_ant矩阵
        choice_ant = np.zeros((n_user,3,3))  # 维度1为用户，其他两位写着和基站的哪个天线连接
        for m in range(n_bs):  # 一个基站内
            user_connected = np.array(np.where (choice_bs[m]==1))[0]  # 找到相连用户
            num_connected = int(np.sum(choice_bs[m]))
            user_ang_ant = np.zeros((n_ant,num_connected))  # 大小为3*x的角度关联矩阵
            for w in range(n_ant):  # 找到相连用户和不同天线夹角
                muti = choice_bs[m]*angle[m,w]
                user_ang_ant[w] = muti[muti!=0]
            # 找到每列的夹角最小值对应的天线序号wmin
            for x in range(num_connected):
                been = user_connected[x]
                wmin = np.where(user_ang_ant[:,x]==min(user_ang_ant[:,x]))
                # 把wmin对应的用户角度设置为190，不可能的值，使其检索不到
                user_ang_ant[wmin,:] = 180
                # 按照序号，m和been 给choice_ant 赋值
                choice_ant[been,m,wmin] = 1


        # 现在开始计算干扰
        I_bs = np.zeros(n_user)  # 16 大致是
        I_beam = np.zeros((n_user,3))
        for n in range(n_user):
            I = 0
            dis_bs = self.distance[:,n]  # 找出该用户对应的所有基站距离
            dis_min = np.min(dis_bs)
            # 先找到在哪个基站范围内
            near_bs = np.where(dis_bs == dis_min)
            if dis_min >= 10:  # 假设基站有效范围为10km
                I_bs[n] = 0
            else:  # 只有在范围内的才会受到干扰！！！！！！！！！！！111
                for x in range(n_ant):
                    if choice_ant[n,near_bs, x] != 1:  # 排除用户和天线关联的那部分
                        I_beam[n,x] = h_bs[near_bs, x, n] * power[near_bs, x]
                        I = I + I_beam[n,x]
                I_bs[n] = I

        # 计算最大干扰
        I_max = np.zeros(self.beam)
        #计算每个卫星用户受到的干扰
        for k in range(len(choice_sat)):# 找到卫星用户的序号
            I_max[k] = max(I_beam[int(choice_sat[k])-1])
        Imax = np.max(I_max)
        # 计算信道容量  大小为3*3,信道容量只考虑被服务的用户。
        for m in range(n_bs):
            for x in range(n_ant):
                # 一个基站，一个天线，最多一个用户
                n_array = np.where(choice_ant[:,m,x] == 1)
                n_sum_user = int(np.sum(choice_ant[:,m,x]))
                if n_sum_user == 0:
                    capacity[m, x] = 0
                else:
                    been = n_array
                    sinr = (h_bs[m, x, been] * power[m, x]) / (self.sig2 + I_bs[been])  # 500MHz
                    capacity[m,x] = B * math.log2(1 + sinr[0][0])

        return capacity, Imax,choice_ant,h_bs

    def get_angle(self):
        # 得到的角度为幅度值,转角度，再取绝对值
        n_bs = self.bs
        n_ant = self.antenna
        n_user = self.user
        pos_bs = self.bs_pos.copy()
        pos_user = self.user_pos.copy()
        angle = np.zeros((n_bs, n_ant, n_user))
        for m in range(n_bs):
            base_pos = pos_bs[m]  # 得到基站位置
            for n in range(n_user):
                dy = pos_user[n, 1] - base_pos[1]
                dx = pos_user[n, 0] - base_pos[0]
                ang = math.atan((dy) / (dx))  # 用户基站连线同x轴夹角
                ang = 180 * ang / math.pi
                for x in range(n_ant):
                    angle[m, x, n] = abs(ang + x * 120)%180
        return angle

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

    def r_throughout(self, thout):
        # thout 大小为n其中k个有值，
        thout_all = np.sum(thout)
        thout_max = 5.6  # 假设系统最大吞吐量为3000Mbps，6M/slot
        r_throu = thout_all / thout_max
        # 吞吐量奖励，计算前后状态矩阵和的差值即可
        return r_throu

    def r_timedelay(self, next_s):
        # 输入延时矩阵，N*slot，7*40
        aver_q_delay = np.zeros(self.user)
        for n in range(self.user):
            sum_t = 0
            sum_d = 0
            for l in range(self.TTL):
                sum_t = sum_t + (l + 1) * next_s[n, l]
                sum_d = sum_d + next_s[n, l]
            aver_q_delay[n] = sum_t / sum_d
        r_delay = np.max(aver_q_delay) - np.min(aver_q_delay)
        r_delay = r_delay / self.TTL  # 归一化处理
        return r_delay

    def get_distance(self):
        # 作用：得到地面用户位置，条件
        # 半径为5km的基站，三个基站的覆盖面积大约有235平分公里。然后假设在半径为10km的区域（314平方公里）内，分布着16个用户
        # 16*16km的正方形
        # 试试20*20km的
        # 泊松点过程生成16个用户位置PPP，也就是各用户间位置服从泊松分布
        # 输出：基站坐标，用户坐标，用户到基站位置矩阵。
        user = self.user
        bs = self.bs
        # 用户16个，基站3个。
        # 基站位置：先生成一个基站位置，再旋转120度得到另外两个基站位置。
        bs_pos = np.zeros((bs, 2))  # 三组xy坐标
        bs_r = 10  # 基站半径为10
        wall = 30  # 最外面围的是100
        #r0 = 0.75 * bs_r
        r0 =  bs_r
        bs_pos[0, 0] = r0 + wall / 2  # 第一个基站横坐标，纵坐标为0
        bs_pos[0, 1] = 0 + wall / 2
        bs_pos[1, 0] = -r0 * math.cos(math.radians(60)) + wall / 2
        bs_pos[1, 1] = r0 * math.sin(math.radians(60)) + wall / 2
        bs_pos[2, 0] = -r0 * math.cos(math.radians(60)) + wall / 2
        bs_pos[2, 1] = -r0 * math.sin(math.radians(60)) + wall / 2
        # 用户位置：16*16 区域分布。随机抽样出来的样本点在范围内服从均匀分布，样本点之间的距离服从指数分布
        # 这里用户个数确定，均匀分布且保障样本点间距离服从指数分布即可。
        user_pos = np.zeros((user, 2))
        # 上面代表16*16的布点区域
        n = user
        while n > 0:
            # 二维的，
            u1 = random.uniform(0.0, 1.0)  # 生成0-1的随机数
            user_pos[user - n, 0] = wall * u1
            u2 = random.uniform(0.0, 1.0)  # 生成0-1的随机数
            user_pos[user - n, 1] = wall * u2
            n = n - 1
        # 现在计算用户到基站位置矩阵，大小为M*N
        d_user_bs = np.zeros((bs, user))
        for m in range(bs):
            for n in range(user):
                d_user_bs[m, n] = math.sqrt((bs_pos[m, 0] - user_pos[n, 0]) ** 2 + (bs_pos[m, 1] - user_pos[n, 1]) ** 2)

        self.bs_pos = bs_pos.copy()
        self.user_pos = user_pos.copy()
        return d_user_bs
