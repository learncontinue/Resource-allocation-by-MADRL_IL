import numpy as np
import datetime as dt
import itertools
from matplotlib import pyplot as plt
from environment import satellite_env as Env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
import argparse
from DQN import DeepQNetwork as DQN_net
from choice_DQN import DeepQNetwork as choice_DQN_net
import os

MAX_EPISODES = 1  # 回合次数,一个回合中，每个用户的业务量到达率固定，在50-150之间
MAX_EP_STEPS = 200
B = 500
TTL = 40
n_beam = 4
n_user = 16
# action_space = np.array(range(0, n_user))
level = 4
# 用户和信道关联，使用一个网络
# 带宽选择，用4个网络，一共5个网络
# 动作空间
bandwidth_shape = 10
bandwidth_space = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,0,0],[0,1,1,0],[0,0,1,1],[1,1,1,0],[0,1,1,1],[1,1,1,1]])  # 带宽位置
bandwidth_len = np.array([125,125,125,125,250,250,250,375,375,500])  # 带宽大小
choice_shape = 1820
user = [i for i in range(1,n_user+1)]
choice_space = np.array(list(itertools.combinations(user,n_beam)))  # 选择空间，35行，4列，每一行代表选了哪个用户

# 状态空间
state_shape = n_user * TTL

parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to use gpu or not')
parser.add_argument('--gpu_fraction', default=(0.5, 0), help='idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
parser.add_argument('--random_seed', type=int, default=123, help='Value of random seed')
opt = parser.parse_args()


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





set_ch()
# Set random seed
setup_seed(opt.random_seed)
random.seed(opt.random_seed)
env = Env(n_user,n_beam)
max_throu = 0

# 每个波束都使用子频段
episode_reward_rand  = np.zeros(MAX_EP_STEPS)
episode_throu_rand = np.zeros(MAX_EP_STEPS)
episode_delay_rand = np.zeros(MAX_EP_STEPS)
capacity = np.zeros((MAX_EP_STEPS,n_user))
s = env.reset()
# 得到不同带宽分配下的信道容量。(单个波束)
for j in range(MAX_EP_STEPS):  # 一个回合200个时隙，一个时隙中进行用户关联和带宽分配
    # 关联动作的选择
    n_choice = random.randint(1, choice_shape) - 1  # a为numpy
    choice = choice_space[n_choice, :]  # 确定被波束服务的用户矩阵
    # 带宽分配动作选择
    # n_band 数组提示错误，难道np.zeros(3),这种不对？
    band_s = np.zeros((n_beam, level))
    band_l = np.zeros(n_beam)
    # 找到带宽对应的模式和带宽大小，放入band_s和band_l中

    for k in range(n_beam):
        index = 7
        band_s[k, :] = bandwidth_space[index, :]
        band_l[k] = bandwidth_len[index]
    #  至此，动作选择完毕
    # 在环境中输入动作，choice band_s band_l 三个输入。得到下一个状态和回报
    s_, r,r_throu,r_time,ct = env.step(choice, band_s, band_l)  # s_,r为numpy
    if r > max_throu:
        max_throu = r
    s = s_  # 更新flue矩阵
    env.flue = s_  # fule矩阵更新（环境里的）

# 每个回合结束时，打印出当前回合数以及总的reward
    episode_reward_rand[j] = r  # 计算吞吐量 单位bps
    episode_throu_rand[j] = r_throu
    episode_delay_rand[j] = r_time
    capacity[j:] = ct
print(episode_throu_rand*4.5*500)
print(capacity*500)


# 测试DQN功率分配策略