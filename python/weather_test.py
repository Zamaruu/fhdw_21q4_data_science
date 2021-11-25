import numpy as np
import pandas as pd
import os, glob, csv

from weather_api import saveDFtoJSON, getApiArguments

args = getApiArguments()
df = pd.date_range(args['start'], args['end'], freq="d").to_frame()
print(df)