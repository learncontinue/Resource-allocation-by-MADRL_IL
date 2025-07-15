import numpy as np
import torch
from matplotlib import pyplot as plt
import tensorflow as tf
def set_ch():
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号-显示为方块的问题

set_ch()


def ema(data, decay):
    new_data = np.zeros(data.shape[0])
    new_data[0] = np.mean(data[:2])
    for idx in range(len(data) - 1):
        new_data[idx + 1] = decay * new_data[idx] + (1 - decay) * data[idx + 1]
    return new_data


#throughout_factor = 4.576*500
#delay_factor = 20.199
throughout_factor = 4.576*500
delay_factor = 20.199
# 导入数据集  奖励
episode_reward_rand = np.load(r'E:\data\LEO\user_16\combine\else\random_r.npy')
episode_reward_aver = np.load(r'E:\data\LEO\user_16\combine\else\average_r.npy')
episode_reward_max = np.load(r'E:\data\LEO\user_16\combine\else\max_r.npy')
episode_reward_DQN_test = np.load(r'E:\data\LEO\user_16\combine\DQN_test\test_r.npy')
# 导入数据集  吞吐量
episode_throu_rand = np.load(r'E:\data\LEO\user_16\combine\else\random_th.npy')
episode_throu_aver = np.load(r'E:\data\LEO\user_16\combine\else\average_th.npy')
episode_throu_max = np.load(r'E:\data\LEO\user_16\combine\else\max_th.npy')
episode_throu_DQN_test = np.load(r'E:\data\LEO\user_16\combine\DQN_test\test_th.npy')
# 导入数据集  时延
episode_delay_rand = np.load(r'E:\data\LEO\user_16\combine\else\random_de.npy')
episode_delay_aver = np.load(r'E:\data\LEO\user_16\combine\else\average_de.npy')
episode_delay_max = np.load(r'E:\data\LEO\user_16\combine\else\max_de.npy')
episode_delay_DQN_test = np.load(r'E:\data\LEO\user_16\combine\DQN_test\test_de.npy')*20.199/40
# 训练集仅考虑奖励。
episode_reward_DQN_train = np.load(r'E:\data\LEO\user_16\combine\DQN_train\data.npy')

MAX_EPISODES = len(episode_reward_rand)
x = np.arange(MAX_EPISODES)  # 这个记录训练回合的标号
#print('随机带宽分配的平均吞吐量',np.mean(episode_reward_rand)*throughout_factor)
#print('均匀带宽分配的平均吞吐量',np.mean(episode_reward_aver)*throughout_factor)
#print('最大带宽分配的平均吞吐量',np.mean(episode_reward_max)*throughout_factor)
#print('DQN的平均吞吐量',np.mean(episode_reward_DQN_test)*throughout_factor)

# 数据输出___________________________________________________________________
print('随机带宽分配的平均奖励',np.mean(episode_reward_rand))
print('均匀带宽分配的平均奖励',np.mean(episode_reward_aver))
print('最大带宽分配的平均奖励',np.mean(episode_reward_max))
print('DQN的平均奖励',np.mean(episode_reward_DQN_test))
# 吞吐量
print('随机带宽分配的平均吞吐量',np.mean(episode_throu_rand))
print('均匀带宽分配的平均吞吐量',np.mean(episode_throu_aver))
print('最大带宽分配的平均吞吐量',np.mean(episode_throu_max))
print('DQN的平均吞吐量',np.mean(episode_throu_DQN_test))
# 时延
print('随机带宽分配的平均时延公平性',np.mean(episode_delay_rand))
print('均匀带宽分配的平均时延公平性',np.mean(episode_delay_aver))
print('最大带宽分配的平均时延公平性',np.mean(episode_delay_max))
print('DQN的平均时延公平性',np.mean(episode_delay_DQN_test))

# 图像输出___________________________________________________________________
#decay=0.85
decay=0.95  # p平滑因子
episode_reward_DQN_test = ema(episode_reward_DQN_test,decay)
episode_reward_DQN_train = ema(episode_reward_DQN_train,decay)
episode_reward_aver = ema(episode_reward_aver,decay)
episode_reward_rand = ema(episode_reward_rand,decay)
episode_reward_max = ema(episode_reward_max,decay)
plt.plot(x, episode_reward_rand,label = "Random bandwidth")
plt.plot(x, episode_reward_aver,label = "Average bandwidth")
plt.plot(x, episode_reward_max,label = "Max bandwidth")
plt.plot(x, episode_reward_DQN_test,label = "MADQN")
plt.title('Satellite resource allocation reward')
plt.legend()
plt.xlabel('episode')
plt.ylabel('Reward')
plt.show()
# 吞吐量
episode_throu_DQN_test = ema(episode_throu_DQN_test,decay)
episode_throu_aver = ema(episode_throu_aver,decay)
episode_throu_rand = ema(episode_throu_rand,decay)
episode_throu_max = ema(episode_throu_max,decay)
#plt.plot(x, episode_throu_rand*throughout_factor,label = "Random bandwidth")
#plt.plot(x, episode_throu_aver*throughout_factor,label = "Average bandwidth")
#plt.plot(x, episode_throu_max*throughout_factor,label = "Max bandwidth")
#plt.plot(x, episode_throu_DQN_test*throughout_factor,label = "MADQN")

plt.plot(x, episode_throu_aver*throughout_factor,label = "MADDPG")
plt.plot(x, episode_throu_rand*throughout_factor,label = "随机波束")
plt.plot(x, episode_throu_DQN_test*throughout_factor,label = "轮询波束")
plt.title('Performance of satellite resource allocation throughput')
plt.legend()
plt.xlabel('episode')
plt.ylabel('System average throughput (Mbps)')
plt.show()

# 时延
episode_delay_DQN_test = ema(episode_delay_DQN_test,decay)
episode_delay_aver = ema(episode_delay_aver,decay)
episode_delay_rand = ema(episode_delay_rand,decay)
episode_delay_max = ema(episode_delay_max,decay)
#plt.plot(x, episode_delay_rand*delay_factor,label = "Random bandwidth")
#plt.plot(x, episode_delay_aver*delay_factor,label = "Average bandwidth")
#plt.plot(x, episode_delay_max*delay_factor,label = "Max bandwidth")
#plt.plot(x, episode_delay_DQN_test*delay_factor,label = "MADQN")
plt.plot(x, episode_delay_rand*delay_factor,label = "随机波束")
plt.plot(x, episode_delay_aver*delay_factor,label = "轮询波束")
plt.plot(x, episode_delay_DQN_test*delay_factor,label = "MADDPG")
plt.title('Delay performance of satellite resource allocation')
plt.legend()
plt.xlabel('episode')
plt.ylabel('System latency difference(ms)')
plt.show()


plt.plot(np.arange(len(episode_reward_DQN_train)), episode_reward_DQN_train,label = "DQN_train")
plt.title('训练收敛过程')
plt.show()

#处理DQN训练数据，采样100个点输出
#smooth_array = int(len(episode_reward_DQN_train)/MAX_EPISODES)
#DQN_train  = np.zeros(MAX_EPISODES)
#DQN_train_reshape = np.reshape(episode_reward_DQN_train,[MAX_EPISODES,smooth_array])

#for i in range(MAX_EPISODES):
    #DQN_train[i] = np.mean(DQN_train_reshape[i:])



