import numpy as np
from matplotlib import pyplot as plt
from env_power import LEO_env as LEO
import torch
import random
import argparse
from DQN import DeepQNetwork as DQN_net
import os

MAX_EPISODES = 40  # 回合次数
MAX_EP_STEPS = 6000
n_beam = 4
n_user = 4
p_max = 100
# action_space = np.array(range(0, n_user))
level = 4
action_space = np.array([25, 50, 75, 100])


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


def choose_action(a,level,n):  # a是一个十进制数字,level为进制,转化后的进制位数最大为n_beam
    temp = a
    n_beam = n
    choice = np.zeros(n_beam)
    # 再把a_n各位拆开赋值给index
    for m in range(n_beam):  # m从0-  n-1
        if temp == 0 :
            break
        x  = temp % level  #取出来余数
        choice[n_beam - m - 1] = action_space[x]
        temp = temp//level  #计算上一位
    return choice

set_ch()
# Set random seed
setup_seed(opt.random_seed)
random.seed(opt.random_seed)

# 网络参数
CAPACITY = 1000
n_actions = level ** 4
state_dim = 16
LR_Q = 0.01
GAMMA = 0.9
E_GREEDY = 0.9
REPLACE_TARGET_ITER = 2
BATCH_SIZE = 100
dqn = DQN_net(CAPACITY, n_actions, state_dim, LR_Q, GAMMA, E_GREEDY, REPLACE_TARGET_ITER, BATCH_SIZE)
MEMORY_SIZE = 1000

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
torch.cuda.set_per_process_memory_fraction(0.5, 0)

leo_env = LEO(n_user, n_beam)  # 环境类实例化
episode_reward = np.zeros(MAX_EPISODES)  # 记录回合奖励
x = np.zeros(MAX_EPISODES)  # 这个记录训练回合的标号
for i in range(MAX_EPISODES):  #训练多少个回合
    x[i] = i
    s = leo_env.reset()  # 环境初始化
    ep_reward = 0  # 记录这一回合的总的奖励reward
    for j in range(MAX_EP_STEPS):
        action = dqn.decide_action(s)  # action为10进制数字
        a = action.numpy()  # a为numpy
        a = choose_action(a, level, n_beam)  # 得到第几个功率分配组合后，使用标号找到a的值，（10机制转level进制,各自位数就是要采取的功率级数
        s_, r = leo_env.step(a)  # s_,r为numpy
        r = torch.from_numpy(r.reshape(1)).type(torch.FloatTensor)
        r = torch.unsqueeze(r, 0)
        # a1_to_bit = torch.unsqueeze(a1_to_bit, 0)  # 将state_dim转换为1×4
        # 这里push的变量应该都是tensor
        tf_s = s.copy()
        tf_s = torch.FloatTensor(tf_s)
        tf_s = torch.reshape(tf_s, [1, 16])
        tf_s_ = s_.copy()
        tf_s_ = torch.FloatTensor(tf_s_)
        tf_s_ = torch.reshape(tf_s_, [1, 16])
        tf_a = action

        dqn.memory.push(tf_s, tf_a, r, tf_s_)
        if (dqn.memory.__len__() >= MEMORY_SIZE) and (dqn.memory.index % 5 == 0):
            dqn.replay()  # 当经验池满了以后，每隔五个数据进行训练
        r = torch.squeeze(r, 0)
        s = s_
        ep_reward += r
    # 每个回合结束时，打印出当前回合数以及总的reward
    ep_reward = ep_reward/MAX_EP_STEPS
    episode_reward[i] = ep_reward
    print('Episode:', i, ' Reward: %i' % ep_reward, f"Explore: {dqn.epsilon}")
plt.plot(x, episode_reward)
plt.show()







# 测试DQN功率分配策略
leo_env = LEO(n_user, n_beam)
sample_rate = int(MAX_EP_STEPS / 100)
r_episode_max = np.zeros((MAX_EPISODES, sample_rate))
for i in range(MAX_EPISODES):
    s = leo_env.reset()  # 状态为H矩阵大小为K*N
    ep_reward = 0  # 记录这一回合的总的奖励reward
    r_record = []  # 记录每一步奖励的数组
    r_100_mean = []  # 记录每100步的平均奖励
    for j in range(MAX_EP_STEPS):  # 一个回合开始
        # action = p_max*np.ones(n_beam)
        action = dqn.decide_action(s)  # action为10进制数字
        a = action.numpy()
        a = choose_action(a,level,n_beam)  # 得到第几个功率分配组合后，使用标号找到a的值，（10机制转level进制,各自位数就是要采取的功率级数
        s_, r = leo_env.step(a)
        r_record.append(r)
        if j % 100 == 0:  # 每过100步，计算下均值
            r_100_mean.append(np.mean(r_record[-100:]))
        ep_reward += r
    r_episode_max[i] = np.array(r_100_mean)  # 回合结束，记录这一回合的奖励，（采样点个数也就是r_100_mean的长度，几个100）
print('Average rewards using max power:', np.mean(r_episode_max))
r_episode_mean_DQN = np.reshape(np.mean(r_episode_max, axis=0), -1)
plt.plot(100 * np.arange(len(r_episode_mean_DQN)), r_episode_mean_DQN)
plt.xlabel('时隙(TS)')
plt.ylabel('下行链路频谱效率(bit/Hz)')
plt.legend(['DQN功率分配策略'])
plt.show()
