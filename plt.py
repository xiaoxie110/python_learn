import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('csv/room13.csv')
df.info();df.head()
#热力图
def heatmap(data, method='pearson', camp='RdYlGn', figsize=(10 ,8)):
    """
    data: 整份数据
    method：默认为 pearson 系数
    camp：默认为：RdYlGn-红黄蓝；YlGnBu-黄绿蓝；Blues/Greens 也是不错的选择
    figsize: 默认为 10，8
    """
    ## 消除斜对角颜色重复的色块
    #     mask = np.zeros_like(df2.corr())
    #     mask[np.tril_indices_from(mask)] = True
    plt.figure(figsize=figsize, dpi= 80)
    sns.heatmap(data.corr(method=method), \
                xticklabels=data.corr(method=method).columns, \
                yticklabels=data.corr(method=method).columns, cmap=camp, \
                center=0, annot=True)
    plt.show()
    # 要想实现只是留下对角线一半的效果，括号内的参数可以加上 mask=mask
heatmap(data=df[['age', 'building_area', 'attach_area', 'trans_month', 'floor', 'price']], figsize=(16,15))

# 刚才的探索我们发现，style 与 neighborhood 的类别都是三类，
## 如果只是两类的话我们可以进行卡方检验，所以这里我们使用方差分析

# ## 利用回归模型中的方差分析
# ## 只有 statsmodels 有方差分析库
# ## 从线性回归结果中提取方差分析结果
import statsmodels.api as sm
from statsmodels.formula.api import ols  # ols 为建立线性回归模型的统计学库
from statsmodels.stats.anova import anova_lm

# C 表示告诉 Python 这是分类变量，否则 Python 会当成连续变量使用
## 这里直接使用方差分析对所有分类变量进行检验
## 下面几行代码便是使用统计学库进行方差分析的标准姿势
lm = ols('price ~ C(street) + C(subway)', data=df).fit()
anova_lm(lm)
print(anova_lm(lm))
#
# # Residual 行表示模型不能解释的组内的，其他的是能解释的组间的
# # df: 自由度（n-1）- 分类变量中的类别个数减1
# # sum_sq: 总平方和（SSM），residual行的 sum_eq: SSE
# # mean_sq: msm, residual行的 mean_sq: mse
# # F：F 统计量，查看卡方分布表即可
# # PR(>F): P 值
#
# # 反复刷新几次，发现都很显著，所以这两个变量也挺值得放入模型中
# from statsmodels.formula.api import ols
#
# lm = ols('price ~ area + bedrooms + bathrooms', data=df).fit()
# lm.summary()
# # 设置虚拟变量
# # 以名义变量 neighborhood 街区为例
# nominal_data = df['neighborhood']
#
# # 设置虚拟变量
# dummies = pd.get_dummies(nominal_data)
# dummies.sample()  # pandas 会自动帮你命名
#
# # 每个名义变量生成的虚拟变量中，需要各丢弃一个，这里以丢弃C为例
# dummies.drop(columns=['C'], inplace=True)
# dummies.sample()
#
# # 将结果与原数据集拼接
# results = pd.concat(objs=[df, dummies], axis='columns')  # 按照列来合并
# results.sample(3)
# # 对名义变量 style 的处理可自行尝试
#
# # 再次建模
# lm = ols('price ~ area + bedrooms + bathrooms + A + B', data=results).fit()
# lm.summary()
#
# # 自定义方差膨胀因子的检测公式
# def vif(df, col_i):
#     """
#     df: 整份数据
#     col_i：被检测的列名
#     """
#     cols = list(df.columns)
#     cols.remove(col_i)
#     cols_noti = cols
#     formula = col_i + '~' + '+'.join(cols_noti)
#     r2 = ols(formula, df).fit().rsquared
#     return 1. / (1. - r2)
#
# test_data = results[['area', 'bedrooms', 'bathrooms', 'A', 'B']]
# for i in test_data.columns:
#     print(i, '\t', vif(df=test_data, col_i=i))
# print(lm.params)
# lm = ols(formula='price ~ area + bathrooms + A + B', data=results).fit()
# lm.summary()
#
# test_data = df[['area', 'bathrooms']]
# for i in test_data.columns:
#     print(i, '\t', vif(df=test_data, col_i=i))
#
# print(lm.params)