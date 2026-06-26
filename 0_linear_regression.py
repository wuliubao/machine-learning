import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#
# lib: StatModels 偏解释性，有统计意义，类似R语言，偏研究类、论文
# model: Linear Regression / OLS 最小二乘法ordinary least squares
#

# STEP1 特征工程
football_df = pd.read_table('dataset/premier_league_data.csv')
# F.C.真实积分 y
true_point = football_df.iloc[:, 5] / 38
# F.C.整体评分 x
all_rating = football_df.iloc[:, 4]
# F.C.不同位置球员评分 x1 x2 x3 x4
position_rating = football_df.iloc[:, [0, 1, 2, 3]]

# STEP2 模型训练
#   sm.OLS(y, X)     → 创建模型对象（未训练）
#          .fit()     → 训练模型，计算参数
#          ↓
#       返回结果对象
#       ├── .params      回归系数
#       ├── .rsquared    R²决定系数
#       ├── .pvalues     p值
#       ├── .fittedvalues 拟合值
#       └── .summary()   完整统计报告

# 单个参数线性回归
all_rating_add = sm.add_constant(all_rating)
# y = ax + b
all_result = sm.OLS(true_point, all_rating_add).fit()

# 多元参数线性回归
position_rating_add = sm.add_constant(position_rating)
# y = a1x1 + a2x2 + a3x3 + a4x4
position_result = sm.OLS(true_point, position_rating_add).fit()

# STEP4
# 1.summary
print(all_result.summary())
print(position_result.summary())

# 2.plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方框的问题
fig = plt.figure()

# 添加第1个子图来
ax1 = fig.add_subplot(121)
# 线图
plt.plot(all_rating, all_result.fittedvalues, c="r", linewidth=4)
# 散点图
plt.scatter(all_rating, true_point, c="b", s=5)
plt.xlabel("F.C.整体评分")
plt.ylabel("F.C.真实积分")
plt.title("一元线性回归")

# 添加第2个子图来
ax2 = fig.add_subplot(122)
plt.scatter(range(len(true_point)), position_result.model.endog, c="b", s=5)
plt.scatter(range(len(true_point)), position_result.fittedvalues, c="r", s=5)
plt.title("多元线性回归")

# 展示
plt.show()
