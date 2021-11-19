import os
from pandas.core.frame import DataFrame
import json

def getApiArguments():
    path = getBackendPath()
    import_file = path + "import.json"
    json_file = open(import_file) 
    args = json.load(json_file)
    return args

def getBackendPath():
    save_path = os.path.dirname(__file__)
    save_path = save_path[:len(save_path) - 6]
    save_path = save_path + "backend/"
    return save_path

def saveDFtoCSV(df: DataFrame):
    removeOutputFile()
    df.to_json(getBackendPath() + 'output.json')

def removeOutputFile():
    if os.path.exists(getBackendPath() + "output.json"):
        os.remove(getBackendPath() + "output.json")
    else:
        print("The file does not exist")