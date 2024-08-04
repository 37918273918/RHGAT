import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams["font.family"] = 'Times New Roman'

# 输入因变量

y1 = [0.9333,0.8667,0.8037,0.7821]
y2 = [0.9445,0.8698,0.8023,0.7844]
# assert y1.shape[0]==y2.shape[0], '两个因变量个数不相等！'
x = ["Inner Product", "DistMult", "Dedicom", "Bilinear"]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12), dpi=300)

# 第一个子图
ax1.plot(x, y1, label='AUC', linestyle='-', marker='*', markersize=10, linewidth=3)
ax1.set_ylabel('AUC', fontsize=13)
ax1.set_xlabel('Decoder', fontsize=13)
ax1.tick_params(axis='both', labelsize=11)
ax1.yaxis.grid(True, linestyle='-.')
# legend1 = ax1.legend(loc='upper center', bbox_to_anchor=(0.9, 0.9), shadow=True, ncol=2)

# 第二个子图
ax2.plot(x, y2, label='AUPR', linestyle='-', marker='p', markersize=10,linewidth=3,color='green')
ax2.set_ylabel('AUPR', fontsize=13)
ax2.set_xlabel('Decoder', fontsize=13)
ax2.tick_params(axis='both', labelsize=11)
ax2.yaxis.grid(True, linestyle='-.')
# legend2 = ax2.legend(loc='upper center', bbox_to_anchor=(0.9, 0.9), shadow=True, ncol=2)
plt.suptitle('Selection of decoders', fontsize=25, x=0.5,y=0.96)
plt.show()
fig.savefig('Decoder.jpg')