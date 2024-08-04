import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 输入因变量
plt.rcParams["font.family"] = 'Times New Roman'

y1 = [0.7623,0.9333,0.7772,0.7494]
y1_1 = [0.7541,0.9445,0.7637,0.7567]
y2 = [0.7488,0.8926,0.7864,0.7790]
y2_1=[0.7368,0.9134,0.7872,0.7871]
y3=[0.7765,0.8951,0.7963,0.7765]
y3_1=[0.7797,0.8969,0.7862,0.7702]
y4=[0.8463,0.8827,0.8593,0.8574]
y4_1=[0.8541,0.8749,0.8569,0.8520]

# assert y1.shape[0]==y2.shape[0], '两个因变量个数不相等！'
x = ['2', '3', '4', '5']
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12), dpi=300)


# 第一个子图
ax1.set_title("WC",fontsize=16)
ax1.plot(x, y1, label='AUC', linestyle='-', marker='*', markersize=10, linewidth=3)
ax1.plot(x, y1_1, label='AUPR', linestyle='-', marker='p', markersize=10, linewidth=3)
ax1.set_ylabel('Value', fontsize=13)
ax1.set_xlabel('Number of layers', fontsize=13)
ax1.tick_params(axis='both', labelsize=11)
ax1.yaxis.grid(True, linestyle='-.')
legend1 = ax1.legend(loc='upper center', bbox_to_anchor=(0.75, 0.9), shadow=True, ncol=2)
# legend1 = ax1.legend([line1], ['AUC'], loc='upper center', bbox_to_anchor=(0.8, 0.95), shadow=True)


# 第二个子图
ax2.set_title("C",fontsize=16)
ax2.plot(x, y2, label='AUC', linestyle='-', marker='*', markersize=10, linewidth=3)
ax2.plot(x, y2_1, label='AUPR', linestyle='-', marker='p', markersize=10, linewidth=3)
ax2.set_ylabel('Value', fontsize=13)
ax2.set_xlabel('Number of layers', fontsize=13)
ax2.tick_params(axis='both', labelsize=11)
ax2.yaxis.grid(True, linestyle='-.')
legend2 = ax2.legend(loc='upper center', bbox_to_anchor=(0.75, 0.9), shadow=True, ncol=2)

#第三个子图
ax3.set_title("A",fontsize=16)
ax3.plot(x, y3, label='AUC', linestyle='-', marker='*', markersize=10, linewidth=3)
ax3.plot(x, y3_1, label='AUPR', linestyle='-', marker='p', markersize=10, linewidth=3)
ax3.set_ylabel('Value', fontsize=13)
ax3.set_xlabel('Number of layers', fontsize=13)
ax3.tick_params(axis='both', labelsize=11)
ax3.yaxis.grid(True, linestyle='-.')
legend3 = ax3.legend(loc='upper center', bbox_to_anchor=(0.75, 0.9), shadow=True, ncol=2)

ax4.set_title("M",fontsize=16)
ax4.plot(x, y4, label='AUC', linestyle='-', marker='*', markersize=10, linewidth=3)
ax4.plot(x, y4_1, label='AUPR', linestyle='-', marker='p', markersize=10, linewidth=3)
ax4.set_ylabel('Value', fontsize=13)
ax4.set_xlabel('Number of layers', fontsize=13)
ax4.tick_params(axis='both', labelsize=11)
ax4.yaxis.grid(True, linestyle='-.')
legend4 = ax4.legend(loc='upper center', bbox_to_anchor=(0.75, 0.9), shadow=True, ncol=2)
plt.suptitle('Impact of the number of graph convolutional layers', fontsize=25, x=0.5,y=0.96)
plt.show()
fig.savefig('Layernum.jpg')