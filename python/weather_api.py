import os
from types import new_class
from pandas.core.frame import DataFrame
import json
import numpy as np
import datetime as dt

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

def saveDictToJSON(name, dict):
    removeOutputFile()
    with open(getBackendPath() + name, 'w') as fp:
        json.dump(dict, fp)

def convertDateTimeListToString(py_dt_list):
    #To numpy_datetime64
    new_dates = np.array(py_dt_list, dtype='datetime64')
    new_dates = np.datetime_as_string(new_dates, timezone='local', unit='D', casting="unsafe")
    # new_dates = [date.astype(dt.datetime) for date in new_dates]
    # if(type(new_dates[0]) == int):
    #     new_dates = [dt.datetime.fromtimestamp(int(ts)) for ts in new_dates]
    # print(new_dates)
    # new_dates = [date.strftime('%d.%m.%Y') for date in new_dates]

    print(new_dates)
    return new_dates
    #for py_date in py_dt_list:


def saveDFtoJSON(df: DataFrame):
    removeOutputFile()
    df.to_json(getBackendPath() + 'output.json')

def removeOutputFile():
    if os.path.exists(getBackendPath() + "output.json"):
        os.remove(getBackendPath() + "output.json")
    else:
        print("The file does not exist")