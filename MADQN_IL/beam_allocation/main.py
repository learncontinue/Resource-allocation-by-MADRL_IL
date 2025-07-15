import numpy as np
from matplotlib import pyplot as plt
from env_power import LEO_env as LEO
import torch
import random
import argparse
import os


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

MAX_EPISODES = 6000  # 回合次数
MAX_EP_STEPS = 40
state_dim = 37
n_beam = 10
n_user = 37
action_space = np.array(range(0, n_user))
# 测试随机波束分配策略
leo_env = LEO(n_user, n_beam)
sample_rate = int(MAX_EPISODES / 100)
r_episode = np.zeros((MAX_EPISODES, sample_rate))
for i in range(MAX_EPISODES):
    s = leo_env.reset()
    ep_reward = 0  # 记录这一回合的总的奖励reward
    r_record = []  # 记录每一步奖励的数组
    r_100_mean = []  # 记录每100步的平均奖励
    for j in range(MAX_EP_STEPS):  # 一个回合开始
        action = np.random.choice(action_space, size=n_beam, replace=False)
        s_, r = leo_env.step(action)
        leo_env.request = s_
        r_record.append(r)
        if j % 100 == 0:  # 每过100步，计算下均值
            r_100_mean.append(np.mean(r_record[-100:]))
        ep_reward += r
    r_episode[i] = np.array(r_100_mean)  # 回合结束，记录这一回合的奖励，（采样点个数也就是r_100_mean的长度，几个100）

print('Average rewards using random selection:', np.mean(r_episode))
r_episode_mean_random = np.reshape(np.mean(r_episode, axis=0), -1)
plt.plot(100 * np.arange(len(r_episode_mean_random)), r_episode_mean_random)
plt.xlabel('时隙(TS)')
plt.ylabel('上行链路NOMA系统总和能量效率(bit/J)')
plt.title('随机资源分配策略')
plt.show()
