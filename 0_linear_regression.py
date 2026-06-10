import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 使用StatsModels库OLS训练

# 获取数据
football_df = pd.read_table('dataset/premier_league_data.csv')
# 真实积分
true_point = football_df.iloc[:, 5] / 38
# F.C.整体评分
all_rating = football_df.iloc[:, 4]
# F.C.不同位置球员评分
position_rating = football_df.iloc[:, [0, 1, 2, 3]]
# 创建画布
fig = plt.figure()

# 使用最小二乘法来定义模型、训练模型
# 单个参数线性回归
all_rating_add = sm.add_constant(all_rating)
all_result = sm.OLS(true_point, all_rating_add).fit()
print(all_result.summary())
# 添加第1个子图来
ax1 = fig.add_subplot(121)
# 线图
plt.plot(all_rating, all_result.fittedvalues, c="r", linewidth=4)
# 散点图
plt.scatter(all_rating, true_point, c="b", s=5)
plt.xlabel("rating")
plt.ylabel("point")
plt.title("Simple Linear Regression")

# 多元参数线性回归
position_rating_add = sm.add_constant(position_rating)
position_result = sm.OLS(true_point, position_rating_add).fit()
print(position_result.summary())
# 添加第2个子图来
ax2 = fig.add_subplot(122)
plt.scatter(range(len(true_point)), position_result.model.endog, c="b", s=5)
plt.scatter(range(len(true_point)), position_result.fittedvalues, c="r", s=5)
plt.title("Multivariate Linear Regression")

# 展示
plt.show()
