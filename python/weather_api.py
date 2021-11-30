"""
@author: Maximilian Ditz
"""

import os
from types import new_class
from pandas.core.frame import DataFrame
import json
import numpy as np
import datetime as dt

def getApiArguments():
    """Liest die JSON-Datei aus und gibt sie als Dictionary zurück."""
    path = getBackendPath()
    import_file = path + "import.json"
    json_file = open(import_file) 
    args = json.load(json_file)
    return args

def getBackendPath():
    """Gibt den reltiven Pfad des Backends OS-Unabhängig zurück."""

    save_path = os.path.dirname(__file__)
    save_path = save_path[:len(save_path) - 6]
    save_path = save_path + "backend/"
    return save_path

def saveDictToJSON(name, dict):
    """Speichert ein bereits formatiertes Dict als JSON-Datei im Backendordner ab."""

    removeOutputFile()
    with open(getBackendPath() + name, 'w') as fp:
        json.dump(dict, fp)


def convertDateTimeListToString(py_dt_list):
    """Konvertiert eine Liste mit dynamischem Datumstyp in eine Liste aus Stringdaten."""

    new_dates = np.array(py_dt_list, dtype='datetime64')
    new_dates = np.datetime_as_string(new_dates, timezone='local', unit='D', casting="unsafe")
    return new_dates

def saveDFtoJSON(df: DataFrame):
    """Speichert ein Pandas DataFrame asl JSON-Datei im Backendordner ab."""
    
    removeOutputFile()
    df.to_json(getBackendPath() + 'output.json')

def removeOutputFile():
    """Wenn aufgerufen löscht die Funktion eine bereits vorhandene output.json Datei im backenordner."""

    if os.path.exists(getBackendPath() + "output.json"):
        os.remove(getBackendPath() + "output.json")
    else:
        print("The file does not exist")