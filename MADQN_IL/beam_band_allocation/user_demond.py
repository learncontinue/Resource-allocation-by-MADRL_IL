import numpy as np
import random
from matplotlib import pyplot as plt
def set_ch():
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号-显示为方块的问题

set_ch()


def random_lamda():
    min = 50
    max = 150
    lamda = np.zeros(16)
    for i in range(16):
        lamda[i] = random.randint(min, max)
    return lamda


def generate_ft():
    lamda = random_lamda()
    ft = np.zeros(16)
    for i in range(16):
        ft[i] = (np.random.poisson(lam=lamda[i]))   # 每个时隙0.02ms
    return ft,lamda
ft,lamda = generate_ft()
x = np.arange(16)+1

#plt.bar(x,lamda,alpha = 0.8)
print(ft)
fig, ax = plt.subplots()
bars = ax.bar(x, lamda)

for bar in bars:
    width = bar.get_width()
    center = bar.get_x() + width / 2
    height = bar.get_height()
    ax.text(center, height, f'{height}', ha='center', va='bottom')
#plt.stem(x,lamda)
ax.set_xticks(x)
plt.tick_params(axis='y', labelsize=12)
plt.xlabel('User ID')
plt.ylabel('Average user demand(Mbps)')
plt.show()