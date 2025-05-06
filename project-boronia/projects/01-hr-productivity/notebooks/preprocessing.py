#%%
!pip install --upgrade pandas-profiling
!pip install --upgrade scikit-optimize
# %%
# 데이터 로드
import kagglehub
import os
import pandas as pd
path = kagglehub.dataset_download("ishadss/productivity-prediction-of-garment-employees")
csv_file = os.path.join(path, 'garments_worker_productivity.csv')
df = pd.read_csv(csv_file)
# %%
# 기본 정보 확인
df.info()
list_cat_cols = list(df.select_dtypes(include = ['object']).columns)
list_num_cols = list(df.select_dtypes(include = ['int64', 'float64']).columns)
target_col = 'actual_productivity'
list_num_cols.remove(target_col)
df.isna().sum().sort_values(ascending = False)
# %%
# target 변수 확인
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import scipy.stats as stats
df[target_col].describe()
plt.figure(figsize=(10, 5))
sns.violinplot(x = target_col, data = df)
plt.show()
stats.probplot(df[target_col], dist = "norm", plot = plt)
plt.title("shaprio test pvalue : {}".format(stats.shapiro(df[target_col])[1]))
plt.show()
print("왜도", skew(df[target_col]))
print("첨도",kurtosis(df[target_col]))