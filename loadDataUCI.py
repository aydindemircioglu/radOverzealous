import pandas as pd
from glob import glob
from sklearn.utils import resample
import numpy as np


def listDatasetsUCI():
    datasetsUCI = glob("./dataUCI/*/alldata.tab")
    datasetsUCI = [z.split("/")[2] for z in datasetsUCI]
    return datasetsUCI


def loadDatasetUCI (dataset):
    df = pd.read_csv(f"dataUCI/{dataset}/alldata.tab", sep = " ")
    if len(df) > 5000:
        df = resample(df, n_samples=5000, stratify=df["clase"],random_state=42)
    y = df["clase"].values
    X = df.drop(["clase"], axis = 1).values
    return X,y


if __name__ == '__main__':
    for dataset in listDatasetsUCI():
        X, y = loadDatasetUCI(dataset)

        print (dataset, X.shape, np.round(100*np.sum(y)/y.shape[0],2))

    print (len(listDatasetsUCI()))
#
