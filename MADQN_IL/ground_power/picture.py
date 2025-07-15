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

# 导入数据集  吞吐量
# episode_throu_rand = np.load(r'E:\data\BS\else\rand220.npy')
# episode_throu_aver = np.load(r'E:\data\BS\else\aver22.npy')
# episode_throu_ARA = np.load(r'E:\data\BS\else\ARA122.npy')
# episode_throu_DQN_4 = np.load(r'E:\data\BS\DQN\test_r22.npy')
# episode_throu_DQN_10 = np.load(r'E:\data\BS\DQN\test1_r22.npy')

episode_throu_rand = np.load(r'E:\data\BS\else\rand240.npy')
episode_throu_aver = np.load(r'E:\data\BS\else\aver24.npy')
episode_throu_ARA = np.load(r'E:\data\BS\else\ARA124.npy')
episode_throu_DQN_4 = np.load(r'E:\data\BS\DQN\test26_rr24.npy')
episode_throu_DQN_10 = np.load(r'E:\data\BS\DQN\test126_r24.npy')


episode_throu_DQN_train = np.load(r'E:\data\BS\DQN\train_24r.npy')
MAX_EPISODES = len(episode_throu_rand)
episode_throu_DQN_train = episode_throu_DQN_train[-100:]/250
x = np.arange(MAX_EPISODES)  # 这个记录训练回合的标号
# 吞吐量
print('随机带宽分配的平均吞吐量',np.mean(episode_throu_rand))
print('均匀带宽分配的平均吞吐量',np.mean(episode_throu_aver))
print('ARA分配的平均吞吐量',np.mean(episode_throu_ARA))
print('DQN的平均吞吐量(L=4)',np.mean(episode_throu_DQN_4))
print('DQN的平均吞吐量(L=10)',np.mean(episode_throu_DQN_10))

# 图像输出___________________________________________________________________
#decay=0.85
decay=0.95  # p平滑因子
# 吞吐量

#plt.plot(x, episode_throu_rand,linewidth = '0.5',color = 'red')
#plt.plot(x, episode_throu_aver,linewidth = '0.5',color = 'blue')
#plt.plot(x, episode_throu_ARA,linewidth = '0.5',color = 'black')
#plt.plot(x, episode_throu_DQN_test,linewidth = '0.5',color = 'green')
#sSSplt.plot(x, episode_throu_DQN_train,linewidth = '0.5',color = 'y')
episode_throu_DQN_4 = ema(episode_throu_DQN_4,decay)
episode_throu_DQN_10 = ema(episode_throu_DQN_10,decay)
episode_throu_aver = ema(episode_throu_aver,decay)
episode_throu_rand = ema(episode_throu_rand,decay)
episode_throu_ARA = ema(episode_throu_ARA,decay)
plt.plot(x, episode_throu_DQN_10,label = "MADQN0-IL(L=10)",linewidth = '3',color = 'blue')
plt.plot(x, episode_throu_DQN_4,label = "MADQN-IL(L=4)",linewidth = '3',color = 'red')
#plt.plot(x, episode_throu_DQN_4,label = "MADQN-IL",linewidth = '1.6',color = 'y')
plt.plot(x, episode_throu_ARA,label = "ARA",linewidth = '3',color = 'black')
plt.plot(x, episode_throu_rand,label = "Random",linewidth = '3',color = 'green')
plt.plot(x, episode_throu_aver,label = "Average",linewidth = '3',color = 'y')
plt.title('Throughput of BS power allocation')
plt.legend()
plt.xlabel('episode')
plt.ylabel('Average system throughput (Mbps)')
plt.show()
y = np.arange(4500)
episode_throu_DQN_train = np.load(r'E:\data\BS\DQN\train_24r.npy')
episode_throu_DQN2_train = np.load(r'E:\data\BS\DQN\train2_24r.npy')
decay_train = 0.99
episode_throu_DQN_train_ping = ema(episode_throu_DQN_train,decay_train)
episode_throu_DQN2_train_ping = ema(episode_throu_DQN2_train,decay_train)
#平滑前
plt.plot(y, episode_throu_DQN_train/250,linewidth = '0.2',color = 'green')
plt.plot(y, episode_throu_DQN2_train/250,linewidth = '0.2',color = 'orange')
#平滑后
plt.plot(y, episode_throu_DQN_train_ping/250,label = "MADQN-IL training process(L=4)",linewidth = '1.2',color = 'Red')
plt.plot(y, episode_throu_DQN2_train_ping/250,label = "MADQN-IL training process(L=10)",linewidth = '1.2',color = 'blue')
plt.legend()
plt.xlabel('episode')
plt.ylabel('system rewards')
plt.show()