import numpy as np
import pandas as pd
import torch

def sort_data(path):
    data = pd.read_csv(path)
    data.sort_values("OS.time",ascending = False, inplace = True)
    x = data.drop(["sub_id", "race", "clinical_stage","histological_grade", "OS", "OS.time","DSS","DSS.time","DFI","DFI.time","PFI","PFI.time", "age_at_initial_pathologic_diagnosis", "race_black", "race_white"], axis = 1).values
    ytime = data.loc[:, ["OS.time"]].values
    yevent = data.loc[:, ["OS"]].values
    age = data.loc[:, ["age_at_initial_pathologic_diagnosis"]].values
    cstage = data.loc[:, ["clinical_stage"]].values
    hgrade = data.loc[:, ["histological_grade"]].values
    race_black = data.loc[:, ["race_black"]].values
    race_white = data.loc[:, ["race_white"]].values
    return(x, ytime, yevent, age, cstage, hgrade, race_black, race_white)

def load_data(path, dtype):
    x, ytime, yevent, age, cstage, hgrade, race_black, race_white = sort_data(path)
    X = torch.from_numpy(x).type(dtype)
    YTIME = torch.from_numpy(ytime).type(dtype)
    YEVENT = torch.from_numpy(yevent).type(dtype)
    AGE = torch.from_numpy(age).type(dtype)
    CSTAGE = torch.from_numpy(cstage).type(dtype)
    HGRADE = torch.from_numpy(hgrade).type(dtype)
    RACE_BLACK = torch.from_numpy(race_black).type(dtype)
    RACE_WHITE = torch.from_numpy(race_white).type(dtype)
    if torch.cuda.is_available():
        X = X.cuda()
        YTIME = YTIME.cuda()
        YEVENT = YEVENT.cuda()
        AGE = AGE.cuda()
        CSTAGE = CSTAGE.cuda()
        HGRADE = HGRADE.cuda()
        RACE_BLACK = RACE_BLACK.cuda()
        RACE_WHITE = RACE_WHITE.cuda()
    return(X, YTIME, YEVENT, AGE, CSTAGE, HGRADE, RACE_BLACK, RACE_WHITE)
