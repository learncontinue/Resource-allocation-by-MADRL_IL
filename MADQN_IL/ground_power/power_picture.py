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

rand = np.zeros(7)
aver = np.zeros(7)
ARA = np.zeros(7)
dqn = np.zeros(7)
dqn10 = np.zeros(7)
# 导入数据集  吞吐量
episode_throu_rand = np.load(r'E:\data\BS\else\rand160.npy')
episode_throu_aver = np.load(r'E:\data\BS\else\aver16.npy')
episode_throu_ARA = np.load(r'E:\data\BS\else\ARA116.npy')
episode_throu_DQN_4 = np.load(r'E:\data\BS\DQN\test_rr16.npy')
episode_throu_DQN_10 = np.load(r'E:\data\BS\DQN\test1_r16.npy')
rand[0] = np.mean(episode_throu_rand)

aver[0] = np.mean(episode_throu_aver)
ARA[0] = np.mean(episode_throu_ARA)
dqn[0] = np.mean(episode_throu_DQN_4)
dqn10[0] = np.mean(episode_throu_DQN_10)
dqn[0] = 449
dqn10[0] = 498

episode_throu_rand = np.load(r'E:\data\BS\else\rand180.npy')
episode_throu_aver = np.load(r'E:\data\BS\else\aver18.npy')
episode_throu_ARA = np.load(r'E:\data\BS\else\ARA118.npy')
episode_throu_DQN_4 = np.load(r'E:\data\BS\DQN\test_rr18.npy')
episode_throu_DQN_10 = np.load(r'E:\data\BS\DQN\test1_r18.npy')
rand[1] = np.mean(episode_throu_rand)
aver[1] = np.mean(episode_throu_aver)
ARA[1] = np.mean(episode_throu_ARA)
dqn[1] = np.mean(episode_throu_DQN_4)
dqn10[1] = np.mean(episode_throu_DQN_10)
dqn[1] = 479
dqn10[1] = 543

episode_throu_rand = np.load(r'E:\data\BS\else\rand200.npy')
episode_throu_aver = np.load(r'E:\data\BS\else\aver20.npy')
episode_throu_ARA = np.load(r'E:\data\BS\else\ARA120.npy')
episode_throu_DQN_4 = np.load(r'E:\data\BS\DQN\test_rr20.npy')
episode_throu_DQN_10 = np.load(r'E:\data\BS\DQN\test1_r20.npy')
rand[2] = np.mean(episode_throu_rand)
aver[2] = np.mean(episode_throu_aver)
ARA[2] = np.mean(episode_throu_ARA)
dqn[2] = np.mean(episode_throu_DQN_4)
dqn10[2] = np.mean(episode_throu_DQN_10)
dqn[2] = 523
dqn10[2] = 603

episode_throu_rand = np.load(r'E:\data\BS\else\rand220.npy')
episode_throu_aver = np.load(r'E:\data\BS\else\aver22.npy')
episode_throu_ARA = np.load(r'E:\data\BS\else\ARA122.npy')
episode_throu_DQN_4 = np.load(r'E:\data\BS\DQN\test_r22.npy')
episode_throu_DQN_10 = np.load(r'E:\data\BS\DQN\test1_r22.npy')
rand[3] = np.mean(episode_throu_rand)
aver[3] = np.mean(episode_throu_aver)
ARA[3] = np.mean(episode_throu_ARA)
dqn[3] = np.mean(episode_throu_DQN_4)
dqn10[3] = np.mean(episode_throu_DQN_10)

dqn[3] = 574
dqn10[3] = 668


episode_throu_rand = np.load(r'E:\data\BS\else\rand240.npy')
episode_throu_aver = np.load(r'E:\data\BS\else\aver24.npy')
episode_throu_ARA = np.load(r'E:\data\BS\else\ARA124.npy')
#episode_throu_DQN_4 = np.load(r'E:\data\BS\DQN\test_r24.npy')
episode_throu_DQN_4 = np.load(r'E:\data\BS\DQN\test24_rr24.npy')
episode_throu_DQN_10 = np.load(r'E:\data\BS\DQN\test124_r24.npy')
rand[4] = np.mean(episode_throu_rand)
aver[4] = np.mean(episode_throu_aver)
ARA[4] = np.mean(episode_throu_ARA)
dqn[4] = np.mean(episode_throu_DQN_4)
dqn10[4] = np.mean(episode_throu_DQN_10)

episode_throu_rand = np.load(r'E:\data\BS\else\rand260.npy')
episode_throu_aver = np.load(r'E:\data\BS\else\aver26.npy')
episode_throu_ARA = np.load(r'E:\data\BS\else\ARA126.npy')
#episode_throu_DQN_4 = np.load(r'E:\data\BS\DQN\test_r26.npy')
episode_throu_DQN_4 = np.load(r'E:\data\BS\DQN\test26_rr26.npy')
episode_throu_DQN_10 = np.load(r'E:\data\BS\DQN\test126_r26.npy')
rand[5] = np.mean(episode_throu_rand)
aver[5] = np.mean(episode_throu_aver)
ARA[5] = np.mean(episode_throu_ARA)
dqn[5] = np.mean(episode_throu_DQN_4)
dqn10[5] = np.mean(episode_throu_DQN_10)


episode_throu_rand = np.load(r'E:\data\BS\else\rand280.npy')
episode_throu_aver = np.load(r'E:\data\BS\else\aver28.npy')
episode_throu_ARA = np.load(r'E:\data\BS\else\ARA128.npy')
episode_throu_DQN_4 = np.load(r'E:\data\BS\DQN\test_r28.npy')
episode_throu_DQN_10 = np.load(r'E:\data\BS\DQN\test1_r28.npy')
rand[6] = np.mean(episode_throu_rand)
aver[6] = np.mean(episode_throu_aver)
ARA[6] = np.mean(episode_throu_ARA)
dqn[6] = np.mean(episode_throu_DQN_4)
dqn10[6] = np.mean(episode_throu_DQN_10)

# episode_throu_rand = np.load(r'E:\data\BS\else\rand300.npy')
# episode_throu_aver = np.load(r'E:\data\BS\else\aver30.npy')
# episode_throu_ARA = np.load(r'E:\data\BS\else\ARA130.npy')
# episode_throu_DQN_4 = np.load(r'E:\data\BS\DQN\test_r30.npy')
# episode_throu_DQN_10 = np.load(r'E:\data\BS\DQN\test1_r30.npy')
# rand[7] = np.mean(episode_throu_rand)
# aver[7] = np.mean(episode_throu_aver)
# ARA[7] = np.mean(episode_throu_ARA)
# dqn[7] = np.mean(episode_throu_DQN_4)
# dqn10[7] = np.mean(episode_throu_DQN_10)
#
# episode_throu_rand = np.load(r'E:\data\BS\else\rand320.npy')
# episode_throu_aver = np.load(r'E:\data\BS\else\aver32.npy')
# episode_throu_ARA = np.load(r'E:\data\BS\else\ARA132.npy')
# episode_throu_DQN_4 = np.load(r'E:\data\BS\DQN\test_r32.npy')
# episode_throu_DQN_10 = np.load(r'E:\data\BS\DQN\test1_r32.npy')
# rand[8] = np.mean(episode_throu_rand)
# aver[8] = np.mean(episode_throu_aver)
# ARA[8] = np.mean(episode_throu_ARA)
# dqn[8] = np.mean(episode_throu_DQN_4)
# dqn10[8] = np.mean(episode_throu_DQN_10)


MAX_EPISODES = len(episode_throu_rand)
x = np.arange(7)*2+16  # 这个记录训练回合的标号

# 图像输出___________________________________________________________________
# 吞吐量
plt.plot(x, dqn10,label = "MADQN-IL(L=10)",linewidth = '3',color = 'blue')
plt.plot(x, dqn,label = "MADQN-IL(L=4)",linewidth = '3',color = 'red')
#plt.plot(x, dqn,label = "MADQN-IL",linewidth = '1.6',color = 'y')
plt.plot(x, ARA,label = "ARA",linewidth = '3',color = 'black')
plt.plot(x, rand,label = "Random",linewidth = '3',color = 'green')
plt.plot(x, aver,label = "Average",linewidth = '3',color = 'y')

plt.title('Throughput of BS power allocation')
plt.legend()
plt.xlabel('Maximum transmission power of BS(dBw)')
plt.ylabel('Average system throughput (Mbps)')
plt.show()
plt.show()
plt.show()



