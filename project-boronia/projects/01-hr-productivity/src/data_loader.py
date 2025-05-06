# %%
import kagglehub
path = kagglehub.dataset_download("ishadss/productivity-prediction-of-garment-employees")
print("Path to dataset files:", path)
# %%
import os
files = os.listdir(path)
print("Downloaded files:", files)
# %%
import pandas as pd
csv_file = os.path.join(path, 'garments_worker_productivity.csv')
df = pd.read_csv(csv_file)
df.head()