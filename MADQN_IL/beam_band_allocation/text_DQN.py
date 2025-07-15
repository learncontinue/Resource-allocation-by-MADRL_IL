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

MAX_EPISODES = 100  # 回合次数,一个回合中，每个用户的业务量到达率固定，在50-150之间
MAX_EP_STEPS = 250
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



def div0(a, b):  # 0/0=0
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0  # -inf inf NaN
    return c


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

# 网络参数
CAPACITY = 10000
LR_Q = 0.0001
GAMMA = 0.95
E_GREEDY = 0.00000001
REPLACE_TARGET_ITER = 200
BATCH_SIZE = 256


# 两个DQN网络，一个用来选动作，一个用来选带宽
dqn_choice = choice_DQN_net(CAPACITY, choice_shape, state_shape, LR_Q, GAMMA, E_GREEDY, REPLACE_TARGET_ITER, BATCH_SIZE)
dqn_band1 = DQN_net(CAPACITY, bandwidth_shape, state_shape, LR_Q, GAMMA, E_GREEDY, REPLACE_TARGET_ITER, BATCH_SIZE)
dqn_band2 = DQN_net(CAPACITY, bandwidth_shape, state_shape, LR_Q, GAMMA, E_GREEDY, REPLACE_TARGET_ITER, BATCH_SIZE)
dqn_band3 = DQN_net(CAPACITY, bandwidth_shape, state_shape, LR_Q, GAMMA, E_GREEDY, REPLACE_TARGET_ITER, BATCH_SIZE)
dqn_band4 = DQN_net(CAPACITY, bandwidth_shape, state_shape, LR_Q, GAMMA, E_GREEDY, REPLACE_TARGET_ITER, BATCH_SIZE)
MEMORY_SIZE = 4000  # 样本数达到4000以上开始训练
# 将训练好的参数加载到测试网络上
dqn_choice.load_weight_from_pkl()
dqn_band1.load_weight_from_pkl1()
dqn_band2.load_weight_from_pkl2()
dqn_band3.load_weight_from_pkl3()
dqn_band4.load_weight_from_pkl4()


# DQN训练过程

#先进行马尔科夫过程，产生四元组（状态、动作、奖励、下一状态），存到经验池
#经验池到达一定大小之后，随机选出一个批次的四元组
#使用这一批次计算损失函数
#利用梯度下降法反向更新训练网络
#每更新REPLACE_TARGET_ITER次训练网络，把参数传给目标网络。
# 训练
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 控制GPU资源使用的两种方法
# （1）直接限制gpu的使用率
print(torch.cuda.is_available())
torch.cuda.set_per_process_memory_fraction(1.0, 0)
#
env = Env(n_user, n_beam)  # 环境类实例化
episode_reward = np.zeros(MAX_EPISODES)  # 记录回合奖励
episode_throu_rand = np.zeros(MAX_EPISODES)
episode_delay_rand = np.zeros(MAX_EPISODES)
x = np.zeros(MAX_EPISODES)  # 这个记录训练回合的标号
for i in range(MAX_EPISODES):  # 回合开始前,对环境reset，也就是重置各用户需求到达率和需求矩阵
    x[i] = i
    s = env.reset()  # 环境初始化
    ep_reward = 0  # 记录这一回合的总的奖励reward
    ep_delay = 0
    ep_throu = 0
    for j in range(MAX_EP_STEPS):  # 一个回合200个时隙，一个时隙中进行用户关联和带宽分配
        # 关联动作的选择
        action_choice = dqn_choice.decide_choice(s)  # 得到动作选择
        n_choice = action_choice.numpy()  # a为numpy
        choice = choice_space[n_choice,:] # 确定被波束服务的用户矩阵
        choice = choice[0,0]
        # 带宽分配动作选择
        action_band1 = dqn_band1.decide_band(s)  # 得到动作选择
        action_band2 = dqn_band2.decide_band(s)  # 得到动作选择
        action_band3 = dqn_band3.decide_band(s)  # 得到动作选择
        action_band4 = dqn_band4.decide_band(s)  # 得到动作选择
        #      上面都是tf数组
        # n_band 数组提示错误，难道np.zeros(3),这种不对？
        n_band = np.zeros(4,dtype=int)
        band_s = np.zeros((n_beam,level))
        band_l = np.zeros(n_beam)
        # 这里直接等于action_band2.numpy()，会报错
        n_band[0] = action_band1[0,0].numpy()
        n_band[1] = action_band2[0,0].numpy()
        n_band[2] = action_band3[0,0].numpy()
        n_band[3] = action_band4[0,0].numpy()
        # 找到带宽对应的模式和带宽大小，放入band_s和band_l中
        for k in range(n_beam):
            band_s[k,:] = bandwidth_space[n_band[k],:]
            band_l[k] = bandwidth_len[n_band[k]]
        #  至此，动作选择完毕
        # 在环境中输入动作，choice band_s band_l 三个输入。得到下一个状态和回报
        s_, r,r_throu,r_time,ct = env.step(choice,band_s,band_l)  # s_,r为numpy
        # 得到的s_和r都是numpy
        # 转化为tensor 才能放入记忆池进行存储，记忆池的奖励和状态相同，动作选择不同，五个网络5种动作
        # 奖励转化为tensor
        s = s_  # 更新flue矩阵
        env.flue = s_  # fule矩阵更新（环境里的）
        ep_reward += r
        ep_throu += r_throu
        ep_delay += r_time
    # 每个回合结束时，打印出当前回合数以及总的reward
    episode_reward[i] = ep_reward/MAX_EP_STEPS
    episode_throu_rand[i] = ep_throu*2
    episode_delay_rand[i] = ep_delay / MAX_EP_STEPS
    print('Episode:', i, ' Reward: %.4f' % (ep_reward/MAX_EP_STEPS), f"Explore: {dqn_choice.epsilon}")
#dqn_choice.plot_cost()
#np.save(r'E:\data\LEO\user_16\delay\DQN_test\test_r', episode_reward)
#np.save(r'E:\data\LEO\user_16\delay\DQN_test\test_th', episode_throu_rand)
#np.save(r'E:\data\LEO\user_16\delay\DQN_test\test_de', episode_delay_rand)
print('文件test.npy已保存：', dt.datetime.now())
plt.plot(x, episode_reward,label = "奖励")
plt.plot(x, episode_throu_rand,label = "吞吐量")
plt.plot(x, episode_delay_rand,label = "时延公平性")
plt.legend()
#plt.title("综合考虑时延和吞吐量的目标函数训练的DQN测试")
#plt.title("以吞吐量为目标函数训练的DQN测试(w = 1)")
plt.title("以时延为目标函数训练的DQN测试")
plt.show()

# 测试DQN功率分配策略