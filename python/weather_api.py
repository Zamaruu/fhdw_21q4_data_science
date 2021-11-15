import os
from pandas.core.frame import DataFrame

def getSavePath():
    save_path = os.path.dirname(__file__)
    save_path = save_path[:len(save_path) - 6]
    save_path = save_path + "backend/"
    return save_path

def saveDFtoCSV(df: DataFrame):
    removeOutputFile()
    df.to_csv(getSavePath() + 'output.csv', index=False,)

def removeOutputFile():
    if os.path.exists(getSavePath() + "output.csv"):
        os.remove(getSavePath() + "output.csv")
    else:
        print("The file does not exist")