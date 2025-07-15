import numpy as np
from matplotlib import pyplot as plt
MAX_EPISODES = 1000
episode_reward_rand = np.ones(MAX_EPISODES)
episode_reward_aver = np.ones(MAX_EPISODES)
x = np.arange(MAX_EPISODES)  # 这个记录训练回合的标号
def set_ch():
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号-显示为方块的问题


set_ch()

print('随机带宽分配的平均吞吐量',np.mean(episode_reward_rand))
print('均匀带宽分配的平均吞吐量',np.mean(episode_reward_aver))
plt.plot(x, episode_reward_rand,label = "随机带宽")
plt.plot(x, episode_reward_aver,label = "平均带宽")

smooth_i = int(MAX_EPISODES/100)
throu_100_rand = np.zeros(smooth_i)
throu_100_aver = np.zeros(smooth_i)

episode_reward_rand_ = np.reshape(episode_reward_rand,[smooth_i,100])
episode_reward_aver_ = np.reshape(episode_reward_aver,[smooth_i,100])

for i in range(int(MAX_EPISODES/100)):
    throu_100_rand[i] = np.mean(episode_reward_rand_[i:])
    throu_100_aver[i] = np.mean(episode_reward_aver_[i:])

smooth_x = np.arange(smooth_i)*100 +50
plt.plot(smooth_x , throu_100_rand,label = "平滑后随机带宽")
plt.plot(smooth_x , throu_100_aver,label = "平滑后均匀带宽")
#plt.title('卫星波束带宽随机分配_吞吐量性能')
plt.title('卫星资源分配吞吐量性能')
plt.legend()
plt.xlabel('回合数')
plt.ylabel('系统平均吞吐量 (Mbps)')
plt.show()
