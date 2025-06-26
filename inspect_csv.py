import pandas as pd

file_path = "data/RELIANCE_intraday.csv"

# Load first 5 rows without parsing dates or skipping
df = pd.read_csv(file_path, nrows=5)
print("🧾 Raw Columns:", df.columns.tolist())
print(df)
