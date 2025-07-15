# -*- coding:utf-8 -*-
"""
作者：Admin
日期：2021年09月12日
"""

import numpy as np
import random
import os
import matplotlib.pyplot as plt
from DDPG import DDPG
from environment import terrestrial_env as Environ
from Replay_memory import Memory, Transition
import argparse
import torch

"""
#####################  hyper parameters  ####################
"""

MAX_EPISODES = 100
MAX_EP_STEPS = 6000
LR_A = 0.01    # learning rate for actor
LR_C = 0.01    # learning rate for critic
LR_Q = 0.01    # DQN的学习率
GAMMA = 0.9     # reward discount
E_GREEDY = 0.9
REPLACEMENT = dict(name='soft', tau=0.01)  # 可以选择不同的replacement策略，这里选择了soft replacement
REPLACE_TARGET_ITER = 2  # DON网络的更新频率
MEMORY_SIZE = 1000
BATCH_SIZE = 100
OUTPUT_GRAPH = False
CAPACITY = 200
n_user = 16
n_bs = 3
n_antenna = 3
n_beam = 4
env = Environ(n_user, n_bs)  # 环境类实例化
dis = env.distance
user_bs = np.zeros((n_bs, n_user))
for n in range(n_user):
    dis_bs = dis[:, n]  # 找出该用户对应的所有基站距离
    dis_min = np.min(dis_bs)
    if dis_min <= 10:
        near_bs = np.where(dis_bs == dis_min)  # 找到在哪个基站范围内
        user_bs[near_bs, n] = 1


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


# Set random seed
setup_seed(opt.random_seed)
random.seed(opt.random_seed)

if opt.gpu_fraction == '':
    raise ValueError("--gpu_fraction should be defined")


def div0(a, b):  # 0/0=0
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0  # -inf inf NaN
    return c


def bit_to_array(t, n):  # 将十进制数t转化为n位的二进制数数组
    t = int(t)
    s1 = [0 for _ in range(n)]
    index = -1
    while t != 0:
        s1[index] = t % 2
        t = t >> 1
        index -= 1
    return np.array(s1).astype(np.float32)


def make_mini_batch(memory, batch_size):
    """
    2.创建小批量数据
    :return:
    """
    # 2.1 从经验池中获取小批量数据
    transitions = memory.sample(batch_size)
    # 2.2 将每个变量转换为与小批量数据对应的形式
    batch = Transition(*zip(*transitions))
    # 2.3 将每个变量的元素转换为与小批量数据对应的形式
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_next_batch = torch.cat(batch.state_next)
    return state_batch, action_batch, reward_batch, state_next_batch


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 控制GPU资源使用的两种方法
    # （1）直接限制gpu的使用率
    print(torch.cuda.is_available())
    torch.cuda.set_per_process_memory_fraction(0.5, 0)
    n_user = 16
    n_bs = 3
    n_ant = 9
    TTL = 40
    state_dim = n_user*TTL
    action_dim = n_bs*n_ant #需要分配功率的共9个天线
    action_bound = 1.0
    ddpg = DDPG(state_dim, action_dim, action_bound, LR_A, LR_C, GAMMA, REPLACEMENT)

    M = Memory(capacity=MEMORY_SIZE)

    # 训练
    for i in range(MAX_EPISODES):
        s = env.reset()
        # 获得用户关联
        dis_temp = dis.copy()
        ep_reward = 0  # 记录这一回合的总的奖励reward
        choice_bs_n = np.zeros(3)  # 对应三个基站选择的用户序列序号
        choice_per_shape = np.zeros((3, 3))  # 三个天线 服务的用户序号1-16
        power_bs = np.zeros((n_bs, n_antenna))  # 三行基站，三列波束
        power_n = 0
        for j in range(MAX_EP_STEPS):  # 一个回合200个时隙，一个时隙中进行用户关联和带宽分配
            # 关联动作的选择 三个
            # 这里怎么选好，用户向距离最近的基站发出请求，基站选择<=3个用户提供服务
            user_index = np.arange(1, n_user + 1)  # 记录没有被分配资源的用户
            # 卫星用户服务完了以后，剩下12个## 随机删除了7,9,12,16四个用户
            user_by_beam = [1, 3, 7, 13]  # 假设得到的卫星用户为2,5,6,9
            for d in range(n_beam):
                been = np.where(user_index == user_by_beam[d])
                user_index = np.delete(user_index, been)
            choice_bs = np.zeros((n_bs, n_user))  # 用户基站关联矩阵  #  为 1 代表关联，0 代表不关联
            for m in range(n_bs):
                l = int(sum(user_bs[m]))
                tied_n = np.array(np.where(user_bs[m] == 1))  # 找到基站对应用户编号，应该是多个1/2/3
                if l <= 3:  # 如果范围内用户小于等于3个，那么基站给这几个服务。
                    for l_ in range(l):  # l代表这个基站关联了几个用户
                        choice_bs[m] = user_bs[m]  # 直接把范围内矩阵赋值给关联矩阵
                        been = np.where(user_index == (tied_n[0, l_] + 1))
                        user_index = np.delete(user_index, been)
                else:
                    tied_n = tied_n[0]
                    s_user = np.sum(s, 1)  # 求出每个用户的总需求
                    # 按照基站对应用户找出需求序列s_user_been
                    s_user_been = np.zeros(l)
                    for l_ in range(l):
                        s_user_been[l_] = s_user[tied_n[l_]]  # 用户标号
                    # 找到所有基站对应需求序列，求最大值
                    for x in range(3):
                        f_max = np.max(s_user_been)  # 找到最大需求
                        # 又错了。。。不能以需求大小定义用户，先找到需求中的标号
                        f_max_f = np.array(np.where(s_user_been == f_max))
                        f_max_f = f_max_f[0]
                        if len(f_max_f) > 1:
                            f_max_f = f_max_f[0]
                        f_max_n = tied_n[f_max_f]  # 用户标号
                        # f_max_n = np.array(np.where(s_user == f_max))  # 找到最大需求编号
                        choice_bs[m, f_max_n - 1] = 1
                        # 删除这个用户user_index以及需求s_user_been
                        been = np.where(user_index == f_max_n)
                        user_index = np.delete(user_index, been)
                        s_user_been = np.delete(s_user_been, f_max_f)
                        tied_n = np.delete(tied_n, f_max_f)
            # 获得功率分配
            a = ddpg.decide_action(s)
            s_, r = env.step(a.detach().numpy())
            s_ = torch.from_numpy(s_).type(torch.FloatTensor)
            s_ = torch.unsqueeze(s_, 0)  # 将state_dim转换为1×4
            if 1 in torch.isnan(s_):
                print(f"改状态没有后续状态")
                continue
            r = torch.from_numpy(r.reshape(1)).type(torch.FloatTensor)
            r = torch.unsqueeze(r, 0)
            # a1_to_bit = torch.unsqueeze(a1_to_bit, 0)  # 将state_dim转换为1×4
            M.push(s, a, r, s_)
            if (M.__len__() >= MEMORY_SIZE) and (M.index % 5 == 0):
                ddpg.var *= 0.9998
                state_batch, action_batch, reward_batch, non_final_next_states = make_mini_batch(M, BATCH_SIZE)
                ddpg.learn(state_batch, action_batch, reward_batch, non_final_next_states)
            s = s_
            ep_reward += r
        # 每个回合结束时，打印出当前的回合数以及总的reward
        print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % ddpg.var, )
    # 测试基于DDPG的资源分配策略
    r_episode = np.zeros((MAX_EPISODES, 60))
    for i in range(MAX_EPISODES):
        s = env.reset()
        s = torch.from_numpy(s).type(torch.FloatTensor)
        s = torch.unsqueeze(s, 0)  # 将state_dim转换为1×4
        r_record = []
        r_100_mean = []
        for j in range(MAX_EP_STEPS):
            a = ddpg.decide_action(s)
            s_, r = env.step(a.detach().numpy())
            s_ = torch.from_numpy(s_).type(torch.FloatTensor)
            s_ = torch.unsqueeze(s_, 0)  # 将state_dim转换为1×4
            r = torch.from_numpy(r.reshape(1)).type(torch.FloatTensor)
            r = torch.unsqueeze(r, 0)
            r_record.append(r)
            if j % 100 == 0:
                r_100_mean.append(np.mean(r_record[-100:]))
            s = s_
        r_episode[i] = np.array(r_100_mean)
    print('Average rewards using DDPG:', np.mean(r_episode))
    r_episode_mean_DDPG = np.reshape(np.mean(r_episode, axis=0), -1)
    plt.plot(100 * np.arange(len(r_episode_mean_DDPG)), r_episode_mean_DDPG)
    plt.xlabel('时隙(TS)')
    plt.ylabel('上行链路NOMA系统总和能量效率(bit/J)')
    plt.title('基于DRL的资源分配策略')
    plt.show()
   # deepQNetwork.save_weight_to_pkl()
    ddpg.save_weight_to_pkl()
