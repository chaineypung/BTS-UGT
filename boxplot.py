import pandas as pd
import matplotlib.pyplot as plt

# 读取excel文件
file_01 = pd.read_excel('./revise/hd_et.xlsx')
file_02 = pd.read_excel('./revise/hd_tc.xlsx')
file_03 = pd.read_excel('./revise/hd_wt.xlsx')

fig = plt.figure(figsize=(8, 6))
d1 = (file_01['3DUnet'] + file_02['3DUnet'] + file_03['3DUnet']) / 3
print(d1)
d2 = (file_01['SEVnet'] + file_02['SEVnet'] + file_03['SEVnet']) / 3.
d3 = (file_01['DMFnet'] + file_02['DMFnet'] + file_03['DMFnet']) / 3.
d4 = (file_01['nnUnet'] + file_02['nnUnet'] + file_03['nnUnet']) / 3.
d5 = (file_01['TransBTS'] + file_02['TransBTS'] + file_03['TransBTS']) / 3.
d6 = (file_01['VTUnet'] + file_02['VTUnet'] + file_03['VTUnet']) / 3.
d7 = (file_01['Ours(No Mask)'] + file_02['Ours(No Mask)'] + file_03['Ours(No Mask)']) / 3.
d8 = (file_01['Ours'] + file_02['Ours'] + file_03['Ours']) / 3.

label = '3DUnet', 'SEVnet', 'DMFnet', 'nnUnet', 'TransBTS', 'VTUnet', 'Ours(No Mask)', 'Ours'
plt.boxplot([d1, d2, d3, d4, d5, d6, d7, d8], labels=label, showmeans=True)  # label设置横轴每个箱图对应的横坐标
plt.xticks(fontproperties='Times New Roman', size=20, rotation=45)
plt.yticks(fontproperties='Times New Roman', size=20)
# plt.xlabel('Methods', fontproperties='Times New Roman', size=30)
plt.ylabel('HD(mm)', fontproperties='Times New Roman', size=20)
plt.gca().set(ylim=(-0.3, 30))
# plt.gca().set(ylim=(0.7, 1.01))
plt.savefig('./revise/hd_avg.pdf', # ⽂件名：png、jpg、pdf)
dpi = 300, # 保存图⽚像素密度
bbox_inches = 'tight')# 保存图⽚完整
plt.show()