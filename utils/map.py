import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = np.load('map.npz')
dis = data['dis'][4]
inten = data['ref'][4]
knn = data['knn'][4]
result = knn
power = 16

range_data = np.copy(result)
# print(range_data.max(), range_data.min())
range_data = range_data ** (1 / power)
# print(range_data.max(), range_data.min())
viridis_range = ((range_data - range_data.min()) /
                 (range_data.max() - range_data.min()) *
                 255).astype(np.uint8)
p1 = sns.heatmap(viridis_range, square=False, cmap="viridis")
# sns.heatmap(knn, linewidth=0.1, cmap="viridis")
plt.show()
s1 = p1.get_figure()
s1.savefig('knn.png', dpi=600, bbox_inches='tight')

#
# def get_data(size):
#     # R = np.linspace(0, 2 * np.pi, size)
#     #
#     # x = np.linspace(start=0, stop=size, num=size, dtype=np.int)
#     # y = np.linspace(start=0, stop=size, num=size, dtype=np.int)
#     #
#     # z_value = []
#     #
#     # for i in range(len(R)):
#     #     v = -np.cos(R)
#     #     z_value.append(v)
#     #
#     # np_z_value = np.asarray(z_value)
#     # result = np_z_value + np.expand_dims(-np.cos(R), axis=1)
#
#     x = np.arange(0, 64)
#     y = np.arange(0, 400)
#     data = np.load('map.npz')
#     dis = data['dis'][4]
#     inten = data['ref'][4]
#     knn = data['knn'][2]
#     result = knn
#     df = pd.DataFrame(data=[v for v in zip(x, y, result)], columns=['x', 'y', 'z'])
#     return df
#
#
# if __name__ == '__main__':
#     SIZE = 100
#     df = get_data(SIZE)
#
#     layout = go.Layout(
#         # plot_bgcolor='red',  # 图背景颜色
#         paper_bgcolor='white',  # 图像背景颜色
#         autosize=True,
#         # width=2000,
#         # height=1200,
#         # title='数据密度热力图',
#         # titlefont=dict(size=30, color='gray'),
#
#         # 图例相对于左下角的位置
#         legend=dict(
#             x=0.02,
#             y=0.02
#         ),
#
#         # x轴的刻度和标签
#         xaxis=dict(title='x坐标轴数据',  # 设置坐标轴的标签
#                    titlefont=dict(color='red', size=20),
#                    tickfont=dict(color='blue', size=18, ),
#                    tickangle=45,  # 刻度旋转的角度
#                    showticklabels=False,  # 是否显示坐标轴
#                    # 刻度的范围及刻度
#                    # autorange=False,
#                    # range=[0, 100],
#                    # type='linear',
#                    ),
#
#         # y轴的刻度和标签
#         yaxis=dict(title='y坐标轴数据',  # 坐标轴的标签
#                    titlefont=dict(color='blue', size=18),  # 坐标轴标签的字体及颜色
#                    tickfont=dict(color='green', size=20, ),  # 刻度的字体大小及颜色
#                    showticklabels=False,  # 设置是否显示刻度
#                    tickangle=-45,
#                    # 设置刻度的范围及刻度
#                    autorange=False,
#                    # range=[0, 100],
#                    # type='linear',
#                    ),
#     )
#
#     fig = go.Figure(data=go.Heatmap(
#         showlegend=False,
#         name='数据',
#         x=df['x'],
#         y=df['y'],
#         z=df['z'],
#     ),
#         layout=layout
#     )
#
#     fig.update_layout(margin=dict(t=100, r=150, b=100, l=100), autosize=True)
#
#     fig.show()
