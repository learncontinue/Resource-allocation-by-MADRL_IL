import math

import numpy as np
import itertools
from matplotlib import pyplot as plt
from environment import terrestrial_env as Env
import torch
import random
import argparse
from DQN import DeepQNetwork as DQN_net
import os

MAX_EPISODES = 100  # 回合次数,一个回合中，每个用户的业务量到达率固定，在50-150之间
MAX_EP_STEPS = 250
B = 500
TTL = 40
n_beam = 4

n_user = 16  # 16 用户
n_bs = 3  #三基站
n_antenna = 3  #天线
M = n_bs
N = n_user
W = n_antenna

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def set_ch():
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号-显示为方块的问题

parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to use gpu or not')
parser.add_argument('--gpu_fraction', default=(0.5, 0), help='idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
parser.add_argument('--random_seed', type=int, default=123, help='Value of random seed')
opt = parser.parse_args()

set_ch()
# Set random seed
setup_seed(opt.random_seed)
random.seed(opt.random_seed)

def draw_position(bs_p, user_p, R):
    x_bs = bs_p[:, 0]
    y_bs = bs_p[:, 1]
    x_user = user_p[:, 0]
    y_user = user_p[:, 1]
    # fig, ax = plt.subplots()
    plt.scatter(x_bs, y_bs, marker='*', color='g', label="基站", s=40)
    # 画基站范围
    theta = np.linspace(0, 2 * np.pi, 100)
    for m in range(len(x_bs)):
        x = x_bs[m] + R * np.cos(theta)
        y = y_bs[m] + R * np.sin(theta)
        # plt.scatter(x, y, color='y',s = 1)
    plt.axis([0, 30, 0, 30])
    plt.scatter(x_user, y_user, marker='o', color='b', label="用户", s=10)
    plt.scatter(15, 15, marker='^', color='r', label="中心", s=50)
    plt.legend()
    plt.title('基站及用户分布情况')
    plt.show()


def calcu_F_N(PMW,R,Loss,A_sat):
    F = np.zeros((n_bs, n_user))
    N = np.ones((n_bs, n_user))
    C = np.zeros((n_bs, n_user))
    Band = 500  # 带宽
    gmax = (10 ** ((20) / 10))  #最大增益
    for m in range(n_bs):
        for n in range(n_user):
            C[m,n] = 0.002*Band * math.log(1+(PMW[m,0]*gmax*(10**(-Loss[m,n]/10)))/((10 ** ((-117) / 10))+(10 ** ((-123) / 10))),2)
            # 第m个基站各天线分配功率相同
            F[m,n] = min(R[n],C[m,n])/R[n]
    #得到F后求N
    for i in range(3):
        N[i,:] = np.arange(n_user)
    #以上得到的是所有用户的F值和序号
    F = np.delete(F,A_sat,axis=1)
    N = np.delete(N, A_sat,axis=1)
    #得到扣除卫星用户的F与序号
    return F,N


def calcu_U(Ith,A,PMW,h,m):
    #计算效用函数，是一个数值，也就是目前基站对卫星用户最大干扰与门限的比值
    #计算最大干扰:所有基站所有天线对卫星用户的干扰。共3*3*4个，选最大的。
    #计算所有干扰:功率+(路径损耗Loss_+发射天线增益)h
    I = np.zeros((n_antenna,n_user))
    for w in range(n_antenna):
        for x in range(n_user):
            #L = 32.44 + 20 * math.log10(20000) + 40 * math.log10(d[m, x])
            I[w,x] = PMW[m,w]*h[m,w,x]
    Im = np.max(I,axis=0)  #基站对每个用户最大干扰,从该基站三个天线中选
    for x in range(n_user):
            Im[x] = Im[x]*A[m,x]
    u = Im/Ith
    U = np.max(u)
    #U = np.sum(u)
    return U


def calcu_CCI(PMW,d,h,A,m):
    Isum = 0
    I = np.zeros((n_antenna, n_user))
    for w in range(n_antenna):
        for nn in range(n_user):
            L = 32.44 + 20 * math.log10(20000) + 20 * math.log10(d[m, nn])
            I[w, nn] = PMW[m, w] * h[m, w, nn]*A[m,nn]
            Isum = Isum + I[w, nn]
    return Isum

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(torch.cuda.is_available())
torch.cuda.set_per_process_memory_fraction(0.5, 0)

env = Env(n_user, n_bs)  # 环境类实例化
dis = env.distance
user_bs = np.zeros((n_bs, n_user))
for n in range(n_user):
    dis_bs = dis[:, n]  # 找出该用户对应的所有基站距离
    dis_min = np.min(dis_bs)
    if dis_min <= 10:
        near_bs = np.where(dis_bs == dis_min)  # 找到在哪个基站范围内
        user_bs[near_bs, n] = 1

Pbmax = env.Pb_max
Pbmin = env.Pb_min
Ith = 10**(env.threshhold/10)
episode_reward = np.zeros(MAX_EPISODES)  # 记录回合奖励
step_reward = np.zeros((MAX_EPISODES,MAX_EP_STEPS))
for i in range(MAX_EPISODES):  # 回合开始前,对环境reset，也就是重置各用户需求到达率和需求矩阵
    s = env.reset()  # 环境初始化
    dis_temp = dis.copy()
    ep_reward = 0  # 记录这一回合的总的奖励reward
    choice_bs_n = np.zeros(3)  # 对应三个基站选择的用户序列序号
    choice_per_shape = np.zeros((3, 3))  # 三个天线 服务的用户序号1-16
    power_bs = np.zeros((n_bs, n_antenna))  # 三行基站，三列波束
    power_n = 0
    for j in range(MAX_EP_STEPS):  # 一个回合200个时隙，一个时隙中进行用户关联和带宽分配
        #user_index = np.arange(1, n_user + 1)  # 记录没有被分配资源的用户
        #user_by_beam = [1, 3, 7, 13]  # 假设得到的卫星用户为2,5,6,9
        #for d in range(n_beam):
            #been = np.where(user_index == user_by_beam[d])
            #user_index = np.delete(user_index, been)
        
        # 这里开始ARA算法
        int_Elist = []
        A = np.zeros((M,n_user))
        #F_sorted = np.zeros((M, n_user-4))
        #Nn = np.zeros((M, n_user - 4))
        #N_sorted = np.zeros((M, n_user-4))
        E = np.zeros((M,W))
        U = np.zeros(M)
        PMW = np.zeros((M, W))
        PtolM = np.zeros(n_bs)
        Imax = Ith
        flue = env.flue
        R = np.sum(flue, axis=1)
        d = env.distance
        Loss_ = np.zeros((M,n_user))
        for m in range(M):
            for n in range(n_user):
                Loss_[m,n] = 32.44 + 20 * math.log10(20000) + 40 * math.log10(d[m, n])
        A_sat = np.array([1,3,7,13])  # 假设选了1,3,5,7个卫星用户
        user_delete = np.array([1,3,7,13])
        for m in range(M):
            Pmaxnow = Pbmax
            while Pmaxnow >= Pbmin:
                PMW[m,:] = Pmaxnow/W
                PtolM[m] = Pmaxnow
                F, N = calcu_F_N(PMW,R,Loss_,user_delete) # 输入需求、功率和损失
                F_sorted = np.sort(F[m,:])[::-1]
                Nn = np.argsort(F[m,:])[::-1]
                lisN = Nn.tolist()
                int_Nn = [int(item) for item in lisN]
                N_sorted = [N[m,i] for i in int_Nn]
                E[m] = N_sorted[:3]     #E记录该基站对应用户服务质量排序，取前3个
                lisE = E[m].tolist()
                int_Elist = [int(item) for item in lisE]
                A[m, :] = 0
                A[m,int_Elist] = 1  #设置匹配矩阵，第m个基站对应的用户关联
                A1,A2,A3,h = env.capacity_t_bs(A, PMW, A_sat)
                unity = calcu_U(Ith,A,PMW,h,m)  # 通过计算基站对卫星用户的干扰来计算效用函数
                U[m] = unity
                if U[m] > 1:
                    Pmaxnow = 0.8*Pmaxnow     #如果不满足干扰条件，减少该基站总功率
                else:
                    break
            if Pmaxnow < Pbmin:
                E[m] = 0
                A[m,:] = np.zeros(n_user)
                PtolM[m] = 0
            # 把listE转成npE
            npE = np.array(int_Elist)
            user_delete = np.append(user_delete, npE)
            #for n in E[m]:
               # A1, A2, A3, h = env.capacity_t_bs(A, PMW, A_sat)
                #CCI = calcu_CCI(PMW,d,h,A,m)
                #Imn = CCI
                #if Imn > Ith:
                #    A = np.zeros((n_bs,n_user))
        # 到此得到了A和PtolM两个矩阵
        choice_bs = A.copy()
        #ARA算法结束，得到用户关联和功率分配
        #choice_bs = np.zeros((n_bs, n_user))  # 用户基站关联矩阵  #  为 1 代表关联，0 代表不关联
        for m in range(n_bs):
            power_bs[m, :] = PtolM[m]/Pbmax
        s_, r = env.step(choice_bs, power_bs, A_sat)  # s_,r为numpy
        s = s_  # 更新flue矩阵
        env.flue = s_  # flue 矩阵更新（环境里的）
        ep_reward += r
        step_reward[i,j] = r/500
    # 每个回合结束时，打印出当前回合数以及总的reward
    episode_reward[i] = ep_reward
    # print('Episode:', i, ' Reward: %.4f' % (ep_reward/200), f"Explore: {dqn_choice.epsilon}")
x = np.arange(0, MAX_EPISODES)  # 这个记录训练回合的标号
np.save(r'E:\data\BS\else\ARA22', episode_reward/MAX_EP_STEPS)
plt.plot(x, episode_reward/MAX_EP_STEPS)
plt.title('基站平均功率分配吞吐量')
plt.ylabel('吞吐量（Mbps）')
plt.xlabel('回合数')
plt.show()
y = np.arange(0, MAX_EPISODES*MAX_EP_STEPS)
step_reward = step_reward.flatten()
plt.plot(y, step_reward)
plt.title('基站平均功率分配每时隙吞吐量')
plt.ylabel('每时隙吞吐量（Mb）')
plt.xlabel('时隙个数')
plt.show()

