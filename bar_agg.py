import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

# 设置字体为Arial
plt.rcParams["font.family"] = 'Times New Roman'


# 数据
k = np.arange(1)
x = ['NR']
A1 = [0.9333]#RHGAT
B1 = [0.8926]#RHGAT-S1
C1 = [0.8951]#RHGAT-S2
D1 = [0.8827]# RHGAT-S3



# std1=[0.0580]
# std2=[0.0522]
# std3=[0.0603]
# std4=[0.0870]
# std5=[0.0634]
# GPCR Enzyme IC NR
#AUC，  AUPR   AUC，  AUPR    AUC，  AUPR   AUC，  AUPR
#GCN 0.7658 0.7676 0.7594 0.7970 0.7947 0.8156 0.7049 0.7553
# GAT 0.7753 0.7680 0.8485 0.8335 0.9084 0.8816 0.8204 0.8082
# DTIGAT 0.7622 0.7649 0.9627 0.9613 0.7867 0.7778 0.9120 0.9032
# DTIMGNN 0.8634 0.8552 0.9132 0.8488 0.8879 0.8320 0.8603 0.8429
# MK-TCMF 0.8043 0.8339 0.8634 0.8799 0.8311 0.8589 0.8737 0.8846
# MHADTI 0.8814 0.8596 0.9440 0.9373 0.9173 0.8948 0.9099 0.9150
# RHGAT 0.9544 0.9521 0.9808 0.9849 0.9854 0.9846 0.9259 0.9384

A2 = [0.9445]
B2 = [0.9134]
C2 = [0.8969]
D2 = [0.8749]

# 创建第一个图
fontsize = 15  # 调整图例字体大小
w =9
h = w/1.5
plt.figure(figsize=(w, h), dpi=300)
# plt.title('Model performances', fontsize=fontsize)
# 第一个子图
plt.subplot(121)  # 第一个子图，2行1列，位置1
total_width, n = 0.7, 4  # 设置宽度和实例个数
width = total_width / n
plt.bar(k, A1, width=width, label='CA',color='dodgerblue')
plt.bar(k + 1 * width, B1, width=width, label='C',color='fuchsia', tick_label=x)
plt.bar(k + 2 * width, C1, width=width, label='A',color='gold', tick_label=x)
plt.bar(k + 3 * width, D1, width=width, label='M',color='burlywood', tick_label=x)
plt.xticks(k + 2 * width / 2, x)
plt.ylim(0.8,1)
# plt.title('Model performances', fontsize=fontsize)
plt.ylabel('AUC',fontsize=10)
plt.xlabel('Subdataset',x=0.42,fontsize=10)
# plt.legend(ncol=1, bbox_to_anchor=(1.15, 0), loc='lower center', edgecolor='w', fontsize=10)  # 缩小图例字体大小

# 创建第二个图
plt.subplot(122)  # 第一个子图，2行1列，位置1
total_width, n = 0.7, 4  # 设置宽度和实例个数
width = total_width / n
plt.bar(k, A2, width=width, label='WA',color='dodgerblue')
plt.bar(k + 1 * width, B2, width=width, label='C',color='fuchsia', tick_label=x)
plt.bar(k + 2 * width, C2, width=width, label='A',color='gold', tick_label=x)
plt.bar(k + 3 * width, D2, width=width, label='M',color='burlywood', tick_label=x)

plt.xticks(k + 2 * width / 2, x)
plt.ylim(0.8,1)
# plt.title('Model performances', fontsize=fontsize)
plt.ylabel('AUPR',fontsize=10)
plt.xlabel('Subdataset',x=0.43,fontsize=10)
plt.legend(ncol=1, bbox_to_anchor=(1.3,0.6), loc='upper center', edgecolor='w', fontsize=14)
plt.suptitle('Selection of feature embedding aggregation methods', fontsize=20, x=0.45,y=0.99)
plt.tight_layout()  # 自动调整子图间距
plt.savefig('Aggregate.jpg', dpi=300, bbox_inches='tight')
plt.show()
