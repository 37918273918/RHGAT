import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

# 设置字体为Arial
plt.rcParams["font.family"] = 'Times New Roman'


# 数据
k = np.arange(4)
x = ['GPCR', 'Enzyme', 'IC', 'NR']
A1 = [0.7658, 0.7594, 0.7947, 0.7049]#GCN
B1 = [0.7753, 0.8485, 0.9084, 0.8204]#GAT
C1 = [0.7622, 0.9627, 0.7867, 0.9120]#DTIGAT
D1 = [0.8634, 0.9132, 0.8879, 0.8603]# DTIMGNN
E1 = [0.8043, 0.8634, 0.8311, 0.8737]# MK-TCMF
F1 = [0.8814, 0.9440, 0.9173, 0.9099]#MHADTI
G1 = [0.9544, 0.9808, 0.9854, 0.9333]#RHGAT
# GPCR Enzyme IC NR
     #AUC，  AUPR   AUC，  AUPR    AUC，  AUPR   AUC，  AUPR
#GCN 0.7658 0.7676 0.7594 0.7970 0.7947 0.8156 0.7049 0.7553
# GAT 0.7753 0.7680 0.8485 0.8335 0.9084 0.8816 0.8204 0.8082
# DTIGAT 0.7622 0.7649 0.9627 0.9613 0.7867 0.7778 0.9120 0.9032
# DTIMGNN 0.8634 0.8552 0.9132 0.8488 0.8879 0.8320 0.8603 0.8429
# MK-TCMF 0.8043 0.8339 0.8634 0.8799 0.8311 0.8589 0.8737 0.8846
# MHADTI 0.8814 0.8596 0.9440 0.9373 0.9173 0.8948 0.9099 0.9150
# RHGAT 0.9544 0.9521 0.9808 0.9849 0.9854 0.9846 0.9259 0.9384

A2 = [0.7676, 0.7970, 0.8156, 0.7553]
B2 = [0.7680, 0.8335, 0.8816, 0.8082]
C2 = [0.7649, 0.9613, 0.7778, 0.9032]
D2 = [0.8552, 0.8488, 0.8320, 0.8429]
E2 = [0.8339, 0.8799, 0.8589, 0.8846]
F2 = [0.8596, 0.9373, 0.8948, 0.9150]
G2 = [0.9521, 0.9849, 0.9846, 0.9445]
# 创建第一个图
fontsize = 15  # 调整图例字体大小
w =9
h = w/1.5
plt.figure(figsize=(w, h), dpi=300)

# 第一个子图
plt.subplot(211)  # 第一个子图，2行1列，位置1
total_width, n = 0.7, 7  # 设置宽度和实例个数
width = total_width / n
plt.bar(k, A1, width=width, label='GCN',color=(0.5,0.6,0.9))
plt.bar(k + 1 * width, B1, width=width, label='GAT',color=(0.8,0.4,0.5), tick_label=x)
plt.bar(k + 2 * width, C1, width=width, label='DTIGAT',color=(0.8,0.7,0.5), tick_label=x)
plt.bar(k + 3 * width, D1, width=width, label='DTIMGNN',color=(0.5,0.5,0.8), tick_label=x)
plt.bar(k + 4 * width, E1, width=width, label='MK-TCMF',color=(0.5,0.7,0.6), tick_label=x)
plt.bar(k + 5 * width, F1, width=width, label='MHADTI',color=(0.8,0.9,0.3), tick_label=x)
plt.bar(k + 6 * width, G1, width=width, label='RHGAT',color=(0.8,0.2,0.7), tick_label=x)
plt.xticks(k + 5 * width / 2, x)
plt.ylim(0.65,1)
plt.title('Model performance with baseline models', fontsize=fontsize)
plt.ylabel('AUC',fontsize=10)
plt.xlabel('Subdatasets',fontsize=10)
# plt.legend(ncol=1, bbox_to_anchor=(1.15, 0), loc='lower center', edgecolor='w', fontsize=10)  # 缩小图例字体大小

# 创建第二个图
plt.subplot(212)  # 第二个子图，2行1列，位置2
total_width, n = 0.7, 7  # 设置宽度和实例个数
width = total_width / n
plt.bar(k, A2, width=width, label='GCN',color=(0.5,0.6,0.9))
plt.bar(k + 1 * width, B2, width=width, label='GAT',color=(0.8,0.4,0.5), tick_label=x)
plt.bar(k + 2 * width, C2, width=width, label='DTIGAT',color=(0.8,0.7,0.5), tick_label=x)
plt.bar(k + 3 * width, D2, width=width, label='DTIMGNN',color=(0.5,0.5,0.8), tick_label=x)
plt.bar(k + 4 * width, E2, width=width, label='MK-TCMF',color=(0.5,0.7,0.6), tick_label=x)
plt.bar(k + 5 * width, F2, width=width, label='MHADTI',color=(0.8,0.9,0.3), tick_label=x)
plt.bar(k + 6 * width, G2, width=width, label='RHGAT',color=(0.8,0.2,0.7), tick_label=x)
plt.xticks(k + 5 * width / 2, x)
plt.ylim(0.65,1)
plt.ylabel('AUPR',fontsize=10)
plt.xlabel('Subdatasets',fontsize=10)
plt.legend(ncol=1, bbox_to_anchor=(1.15, 0), loc='lower center', edgecolor='w', fontsize=10)  # 缩小图例字体大小
# plt.legend(ncol=1, bbox_to_anchor=(1.15,0), loc='upper center', edgecolor='w', fontsize=10)
plt.tight_layout()  # 自动调整子图间距
plt.savefig('Two_Figs.jpg', dpi=300, bbox_inches='tight')
plt.show()
