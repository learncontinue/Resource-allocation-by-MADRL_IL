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
n_user = 16  # 16个用户
n_bs = 3  # 共三个基站
n_antenna = 3  # 每个基站三个天线
# action_space = np.array(range(0, n_user))
level = 4
# 用户和信道关联，使用一个网络
# 带宽选择，用4个网络，一共5个网络
# --------------卫星动作状态空间-------------------------
# 动作空间
bandwidth_shape = 10
bandwidth_space = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 0],
     [0, 1, 1, 1], [1, 1, 1, 1]])  # 带宽位置
bandwidth_len = np.array([125, 125, 125, 125, 250, 250, 250, 375, 375, 500])  # 带宽大小
choice_shape = 35
user = [i for i in range(1, n_user + 1)]
choice_space = np.array(list(itertools.combinations(user, n_beam)))  # 选择空间，35行，4列，每一行代表选了哪个用户
# 状态空间
state_shape = n_user * TTL

# --------------地面基站动作状态空间和网络-------------------------
# 动作空间
# 一共16个用户，被卫星服务4个，三个基站服务剩下12个用户。
# 要考虑每个基站范围内有几个剩余用户。可能是0-12个。
# 用户基站匹配矩阵A_BS = np.zeros((B,N)),然后一行最多有3个。(约束一)
# 每个用户只能被一个基站服务，所以是12个用户竞争9个子信道，时分复用方式
# 第一个基站选信道状态最好的三个用户 第二个基站再选3个，最后基站再选三个
choice_bs_shape = np.zeros(3)
choice_bs_shape[0] = 220  # C12-3=220
choice_bs_shape[1] = 84  # C1-9-3 = 84
choice_bs_shape[2] = 20  # 最后6个里选3
user_bs1 = [i for i in range(1, 12 + 1)]
choice_space1 = np.array(list(itertools.combinations(user_bs1, n_beam)))
user_bs2 = [i for i in range(1, 9 + 1)]
choice_space2 = np.array(list(itertools.combinations(user_bs2, n_beam)))
user_bs3 = [i for i in range(1, 6 + 1)]
choice_space3 = np.array(list(itertools.combinations(user_bs3, n_beam)))
# 功率分配矩阵,每个基站分配波束功率，但是每个波束为用户分配功率不得少于0.1*Pmax
# 这是由于要保障基站用户的服务质量。所以各波束分配方式有下面五种，每个波束最大不超过Pmax/3
power_bs_level = np.array([0, 0.3, 0.5, 0.75, 1])  # 要么分配的功率在0.1Pmax之上，要么为0也就是不分配功率
level = 5  # 分5个档
# 一个基站的选择方式有5*5*5 = 125种组合。因此功率选择上，每个基站为一个智能体
power_antenna = [i for i in range(1, n_antenna + 1)]
power_bs_shape = 125
# 需要一个函数，将125转化为5进制数字，好选出四个波束的功率分配矩阵

# 所以如果使用DQN需要用6个网络，加上卫星网络共有11个DQN网络。

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


def choose_power(a, level, bit_max):  # a是一个十进制数字,level为进制,转化后的进制位数最大为n_beam
    temp = a
    choice = np.zeros(bit_max)
    # 再把a_n各位拆开赋值给index
    for m in range(bit_max):  # m从0-  n-1
        if temp == 0:
            break
        x = temp % level  # 取出来余数
        choice[bit_max - m - 1] = power_bs_level[x]
        temp = temp // level  # 计算上一位
    return choice


power_bs_space = np.zeros((power_bs_shape, n_antenna))
for n_p in range(power_bs_shape):
    power_bs_space[n_p, :] = choose_power(n_p, level, n_antenna)

set_ch()
# Set random seed
setup_seed(opt.random_seed)
random.seed(opt.random_seed)
# DQN训练过程
CAPACITY = 10000
LR_Q = 0.0001
GAMMA = 0.95
E_GREEDY = 0.00000001
REPLACE_TARGET_ITER = 200
BATCH_SIZE = 256
dqn_power1 = DQN_net(CAPACITY, power_bs_shape, state_shape, LR_Q, GAMMA, E_GREEDY, REPLACE_TARGET_ITER, BATCH_SIZE)
dqn_power2 = DQN_net(CAPACITY, power_bs_shape, state_shape, LR_Q, GAMMA, E_GREEDY, REPLACE_TARGET_ITER, BATCH_SIZE)
dqn_power3 = DQN_net(CAPACITY, power_bs_shape, state_shape, LR_Q, GAMMA, E_GREEDY, REPLACE_TARGET_ITER, BATCH_SIZE)

dqn_power1.load_weight_from_pkl1()
dqn_power2.load_weight_from_pkl2()
dqn_power3.load_weight_from_pkl3()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 控制GPU资源使用的两种方法
# （1）直接限制gpu的使用率
print(torch.cuda.is_available())
torch.cuda.set_per_process_memory_fraction(0.5, 0)
# 随机波束图案，随机带宽分配
env = Env(n_user, n_bs)  # 环境类实例化
dis = env.distance
# 需要考虑用户是否在基站范围内，生成一个基站用户位置内矩阵
user_bs = np.zeros((n_bs, n_user))
for n in range(n_user):
    dis_bs = dis[:, n]  # 找出该用户对应的所有基站距离
    dis_min = np.min(dis_bs)
    if dis_min <= 10:
        near_bs = np.where(dis_bs == dis_min)  # 找到在哪个基站范围内
        user_bs[near_bs, n] = 1

#R_bs = 7.5
# draw_position(env.bs_pos,env.user_pos,R_bs)

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
        # 关联动作的选择 三个？
        # 这里怎么选好，用户向距离最近的基站发出请求，基站选择<=3个用户提供服务
        user_index = np.arange(1, n_user + 1)  # 记录没有被分配资源的用户
        # 卫星用户服务完了以后，剩下12个## 随机删除了7,9,12,16四个用户
        user_by_beam = [1, 3, 7, 13]  # 假设得到的卫星用户为2,5,6,9
        # user_by_beam = np.random.choice(user_index,size = n_beam,replace = False)
        # 从用户序列中删除这几个用户，并将用户和基站的距离设置为100，也就是假设卫星服务的都是较远用户。
        # 回头还要考虑用户位置对卫星服务的影响，即让距离基站远的用户更容易被卫星服务。How？在奖励函数中多加一项。再议
        for d in range(n_beam):
            been = np.where(user_index == user_by_beam[d])
            user_index = np.delete(user_index, been)
            #dis_temp[:, been] = 100  # 把该用户和所有基站的距离都设置为100,好像没啥用。。

        choice_bs = np.zeros((n_bs, n_user))  # 用户基站关联矩阵  #  为 1 代表关联，0 代表不关联
        # 这里直接随机找了3个用户，如果是使用网络，那么
        # choice_bs_n[0] = random.randint(0,int(choice_bs_shape[0])-1)
        # choice_per_shape[0:] = choice_space1[int(choice_bs_n[0]):]
        for m in range(n_bs):
            l = int(sum(user_bs[m]))
            #tied_n = np.zeros((1,l)).astype(int)
            tied_n = np.array(np.where(user_bs[m] == 1))  # 找到基站对应用户编号，应该是多个1/2/3
            if l <= 3:  # 如果范围内用户小于等于3个，那么基站给这几个服务。
                for l_ in range(l):  # l代表这个基站关联了几个用户
                    choice_bs[m] = user_bs[m]  # 直接把范围内矩阵赋值给关联矩阵
                    # 下面两步删去已经关联的用户
                    been = np.where(user_index == (tied_n[0,l_]+ 1))
                    user_index = np.delete(user_index, been)
            # 如果用户数大于三个，需要选出三个来,按照用户需求，
            else:
                tied_n = tied_n[0]
                s_user = np.sum(s,1)  # 求出每个用户的总需求
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
                    if len(f_max_f)>1:
                        f_max_f = f_max_f[0]
                    f_max_n = tied_n[f_max_f]  # 用户标号
                    #f_max_n = np.array(np.where(s_user == f_max))  # 找到最大需求编号
                    choice_bs[m, f_max_n-1] = 1
                    # 删除这个用户user_index以及需求s_user_been
                    been = np.where(user_index == f_max_n)
                    user_index = np.delete(user_index, been)
                    s_user_been = np.delete(s_user_been, f_max_f)
                    tied_n = np.delete(tied_n, f_max_f)
        # 用户关联这样获得，功率分配还是要DQN算法进行
        power_n1 = dqn_power1.decide_action(s)
        power_n2 = dqn_power2.decide_action(s)
        power_n3 = dqn_power3.decide_action(s)
        # 将tf数组转化为np数组
        n_power = np.zeros(3, dtype=int)
        # 这里直接等于action_band2.numpy()，会报错
        n_power[0] = power_n1[0, 0].numpy()
        n_power[1] = power_n2[0, 0].numpy()
        n_power[2] = power_n3[0, 0].numpy()
        # 找到功率对应的大小组合，放入power_bs中
        for m in range(n_bs):
            power_bs[m, :] = power_bs_space[n_power[m]]
        #  至此，动作选择完毕
        # 在环境中输入动作，choice band_s band_l 三个输入。得到下一个状态和回报
        s_, r = env.step(choice_bs, power_bs, user_by_beam)  # s_,r为numpy
        s = s_  # 更新flue矩阵
        env.flue = s_  # fule矩阵更新（环境里的）
        ep_reward += r
        step_reward[i,j] = r/500
    # 每个回合结束时，打印出当前回合数以及总的reward
    episode_reward[i] = ep_reward
    print('Episode:', i, ' Reward: %.4f' % (ep_reward/MAX_EP_STEPS), f"Explore: {dqn_power1.epsilon}")
x = np.arange(0, MAX_EPISODES)  # 这个记录训练回合的s标号
np.save(r'E:\data\BS\DQN\test24_rr20', episode_reward/MAX_EP_STEPS)
print('DQN(L=4)的平均吞吐量',np.mean(episode_reward/MAX_EP_STEPS))
plt.plot(x, episode_reward/MAX_EP_STEPS)
plt.title('基站DQN功率分配吞吐量')
plt.ylabel('每时隙吞吐量（Mb）')
plt.xlabel('时隙个数')
plt.show()
# y = np.arange(0, MAX_EPISODES*MAX_EP_STEPS)
# step_reward = step_reward.flatten()
# plt.plot(y, step_reward)
# plt.show()

# y = np.arange(0, MAX_EPISODES*MAX_EP_STEPS)
# step_reward = step_reward.flatten()
# plt.plot(y, step_reward)
# plt.title('基站DQN功率分配每时隙吞吐量')
# plt.ylabel('每时隙吞吐量（Mb）')
# plt.xlabel('时隙个数')
# plt.show()
# 随机波束图案，平均带宽分配
