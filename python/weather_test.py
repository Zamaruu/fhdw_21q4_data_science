import numpy as np
import pandas as pd
import os, glob, csv

df = pd.read_csv("../backend/import.csv", header=None)
print(df.head())
df = pd.to_numeric(pd.to_datetime(df[0]))

with open("../backend/output.csv", 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(df)