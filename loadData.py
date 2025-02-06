import pandas as pd
from glob import glob
from sklearn.utils import resample
import radMLBench


def listDatasets():
    datasets = radMLBench.listDatasets()
    datasetsUCI = glob("./dataUCI/*/alldata.tab")
    datasetsUCI = [z.split("/")[2] for z in datasetsUCI]
    datasets.extend(datasetsUCI)
    return datasets


def loadDataset (dataset):
    if dataset in radMLBench.listDatasets():
        X, y = radMLBench.loadData(dataset, return_X_y=True, local_cache_dir="./datasets")
    else:
        df = pd.read_csv(f"dataUCI/{dataset}/alldata.tab", sep = " ")
        if len(df) > 5000:
            df = resample(df, n_samples=5000, stratify=df["clase"],random_state=42)
        y = df["clase"].values
        X = df.drop(["clase"], axis = 1).values
    return X,y


if __name__ == '__main__':
    for dataset in listDatasets():
        X, y = loadDataset(dataset)

    print (len(listDatasets()))
#
