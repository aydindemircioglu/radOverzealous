from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_curve
from scipy.optimize import minimize_scalar

import os

from featureselection import *

import hashlib
from functools import partial
from joblib import dump, Parallel, delayed
import numpy as np
import random
import time

import radMLBench

import optuna
import warnings
optuna.logging.set_verbosity(optuna.logging.FATAL)


# overall repeats of the whole procedures
nRepeats = 25


# around 1029 experiments 3*7*(15+15+9+9+1)
search_space = {
    'fs_method': ["LASSO", "ET", "MRMRe"],
    'N': [2**k for k in range(0,6)],
    'clf_method': ["RBFSVM", "RandomForest", "LogisticRegression", "NaiveBayes"],
    'RF_n_estimators': [50,100,250],
    'RF_max_depth': [3, 5, 7],
    'C_LR': [2**k for k in range(-7,8,2)]+[1],
    'C_SVM': [2**k for k in range(-7,8,2)]+[1]
}


selection_cache = {}


def get_md5_checksum(X, y):
    md5 = hashlib.md5()
    md5.update(X.tobytes())
    md5.update(y.tobytes())
    return md5.hexdigest()



def cached_select_features(X_train, y_train, fs_method, N, cfg, exp_number):
    ''' if its in cache, apply it to X_train, if not, train and then apply.'''
    global selection_cache
    checksum = get_md5_checksum(X_train, y_train)
    cache_key = (checksum, fs_method, N)

    if cache_key in selection_cache[exp_number]:
        return selection_cache[exp_number][cache_key]

    X_train_selected, fsel = select_features(X_train, y_train, fs_method, N)
    selection_cache[exp_number][cache_key] = X_train_selected, fsel
    return X_train_selected, fsel



def select_features(X, y, fs_method = None, N = None, best_params = None):
    if best_params is not None:
        fs_method = best_params["fs_method"]
        N = best_params["N"]

    if fs_method == "LASSO":
        clf_fs = LogisticRegression(penalty='l1', max_iter=100, solver='liblinear', C=1, random_state=42)
        fsel = SelectFromModel(clf_fs, prefit=False, max_features=N, threshold=-np.inf)
    elif fs_method == "MRMRe":
        mrmre_score_fct = partial(mrmre_score, nFeatures = N)
        fsel = SelectKBest(mrmre_score_fct, k = N)
    elif fs_method == "ET":
        clf_fs = ExtraTreesClassifier(random_state=42)
        fsel = SelectFromModel(clf_fs, prefit=False, max_features=N, threshold=-np.inf)

    X_selected = fsel.fit_transform(X, y)
    return X_selected, fsel



def getClassifier(params):
    if params['clf_method'] == "LogisticRegression":
        clf = LogisticRegression(max_iter=500, solver='liblinear', C=params['C_LR'], random_state=42)
    elif params['clf_method'] == "NaiveBayes":
        clf = GaussianNB()
    elif params['clf_method'] == "RandomForest":
        clf = RandomForestClassifier(n_estimators=params['RF_n_estimators'], max_depth=params["RF_max_depth"], random_state=42)
    elif params['clf_method'] == "RBFSVM":
        clf = SVC(kernel="rbf", C=params['C_SVM'], gamma='auto', probability=True, random_state=42)
    return clf



def objective_flat_cv (trial, X, y, k, n_repeats, cfg, exp_number):
    np.random.seed(42)
    random.seed(42)

    fs_method = trial.suggest_categorical("fs_method", search_space['fs_method'])
    N = trial.suggest_categorical("N", search_space['N'])

    clf_method = trial.suggest_categorical("clf_method", search_space['clf_method'])
    if clf_method == "LogisticRegression":
        C_LR = trial.suggest_categorical("C_LR", search_space['C_LR'])
        clf = LogisticRegression(max_iter=500, solver='liblinear', C=C_LR, random_state=42)
    if clf_method == "NaiveBayes":
        clf = GaussianNB()
    if clf_method == "RandomForest":
        RF_n_estimators = trial.suggest_categorical("RF_n_estimators", search_space['RF_n_estimators'])
        RF_max_depth = trial.suggest_categorical("RF_max_depth", search_space['RF_max_depth'])
        clf = RandomForestClassifier(n_estimators=RF_n_estimators, max_depth=RF_max_depth, random_state=42)
    if clf_method == "RBFSVM":
        C_SVM = trial.suggest_categorical("C_SVM", search_space['C_SVM'])
        clf = SVC(kernel="rbf", C=C_SVM, gamma='auto', probability=True, random_state=42)

    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats, random_state=42)
    y_preds = []
    y_gts = []
    models = []
    for fold_idx, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_selected, fsel = cached_select_features(X_train, y_train, fs_method = fs_method, N = N, cfg = cfg, exp_number = exp_number)
        X_test_selected = fsel.transform(X_test)

        clf.fit(X_train_selected, y_train)
        y_pred = clf.predict_proba(X_test_selected)[:, 1]
        y_preds.append(y_pred)
        y_gts.append(y_test)
        models.append([fsel, clf])

    y_preds_flat = [p for y in y_preds for p in y]
    y_gts_flat = [gt for y in y_gts for gt in y]
    cv_auc = roc_auc_score(y_gts_flat, y_preds_flat)

    # corresponds to performance on all validation folds across repeats
    trial.set_user_attr("ensemble_models", models)
    trial.set_user_attr("y_preds_int", y_preds_flat)
    trial.set_user_attr("y_gts_int", y_gts_flat)
    trial.set_user_attr("auc_int", cv_auc)
    return cv_auc # used to find the best model




def retrain_model (X_train, y_train, best_params):
    X_train_selected, fsel = select_features(X_train, y_train, best_params=best_params)
    clf = getClassifier(best_params)
    clf.fit(X_train_selected, y_train)
    return fsel, clf



def get_metrics (y_test, ensemble_y_probs):
    auc_true = roc_auc_score(y_test, ensemble_y_probs)

    # mertics
    fpr, tpr, thresholds = roc_curve(y_test, ensemble_y_probs)
    thresholds = thresholds[np.isfinite(thresholds)]

    def youden_index(thresh):
        idx = np.argmin(np.abs(thresholds - thresh))
        return -(tpr[idx] - fpr[idx])

    result = minimize_scalar(youden_index, bounds=(thresholds.min(), thresholds.max()), method='bounded')
    optimal_threshold_raw = result.x
    idx_optimal = np.argmin(np.abs(thresholds - optimal_threshold_raw))
    optimal_threshold = thresholds[idx_optimal]

    sensitivity = tpr[idx_optimal]
    specificity = 1 - fpr[idx_optimal]
    y_pred_bin = (ensemble_y_probs >= optimal_threshold).astype(int)

    metrics = {
        "auc": auc_true,
        "accuracy": accuracy_score(y_test, y_pred_bin),
        "precision": precision_score(y_test, y_pred_bin),
        "recall": recall_score(y_test, y_pred_bin),
        "f1": f1_score(y_test, y_pred_bin),
        "mcc": matthews_corrcoef(y_test, y_pred_bin),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "optimal_threshold": optimal_threshold,
        "y_test_prob": ensemble_y_probs,
        "y_test_true": y_test
    }
    return metrics



def eval_model (models, X_test, y_test):
    # just ensure
    if len(np.unique(y_test)) == 1:
        raise Exception ("y_test must have more than one unique class")

    # we always treat it as an ensemble
    y_probs = []
    for (fsel, clf) in models:
        X_test_selected = fsel.transform(X_test)
        y_prob = clf.predict_proba(X_test_selected)[:, 1]
        y_probs.append(y_prob)

    ensemble_y_probs = np.mean(y_probs, axis=0)
    metrics = get_metrics(y_test, ensemble_y_probs)
    return metrics



def rep_flat_cv (X, y, k, valrepeats, cfg):
    exp_number = str(random.randint(1, 10**12))
    selection_cache[exp_number] = {}

    print (f'\t-Flat CV: {cfg["dataset"]} for repeat {cfg["repeat"]}')
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler(), direction="maximize")
        study.optimize(lambda trial: objective_flat_cv (trial, X, y, k, valrepeats, cfg, exp_number))
        best_params = study.best_params
        best_trial = study.best_trial
        best_auc = study.best_value

    # compute metrics of best model
    metrics = get_metrics (best_trial.user_attrs.get("y_gts_int"), best_trial.user_attrs.get("y_preds_int"))

    # refit model
    fsel, clf = retrain_model (X, y, best_params)

    end_time = time.time()
    total_time = end_time - start_time
    del selection_cache[exp_number]

    results = {
        "type": "flat_cv_k",
        "k": k,
        "valrepeats": valrepeats,
        "dataset": cfg["dataset"],
        "repeat": cfg["repeat"],
        "metrics_refit": metrics,
        "metrics_ensemble": metrics,
        "predicted_AUC_refit": best_auc,
        "predicted_AUC_ensemble": best_auc, # same since no ensembling internally
        "best_params": best_params, # single best param
        "ensemble_models": best_trial.user_attrs.get("ensemble_models"),
        "refit_models": [(fsel, clf)],
        "time": total_time
    }
    return results



def rep_holdout_cv (X, y, k, valrepeats, cfg):
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state=cfg["repeat"], stratify=y)
    print ("Holdout: y_holdout has in nested-cv ", np.sum(y_holdout), "/", len(y_holdout))
    print ("Holdout: y_train has in nested-cv ", np.sum(y_train), "/", len(y_train))
    # return None

    start_time = time.time()
    results = rep_flat_cv (X_train, y_train, k, valrepeats, cfg)

    # evaluate model on holdout, getting metrics
    metrics_refit = eval_model (results["refit_models"], X_holdout, y_holdout)
    metrics_ensemble = eval_model (results["ensemble_models"], X_holdout, y_holdout)

    # retrain refit model on all data for final model, makes no sense to
    # use not all data when refitting anyway
    fsel, clf = retrain_model (X, y, results["best_params"])
    end_time = time.time()
    total_time = end_time - start_time

    results = {
        "type": "holdout_cv_k",
        "k": k,
        "valrepeats": valrepeats,
        "dataset": cfg["dataset"],
        "repeat": cfg["repeat"],
        "best_params": results["best_params"], # best param of refitted model
        "metrics_refit": metrics_refit,
        "metrics_ensemble": metrics_ensemble,
        "refit_models": [(fsel, clf)],
        "ensemble_models": results["ensemble_models"],
        "time": total_time
    }
    return results



def rep_nested_cv(X, y, k, l, valrepeats, cfg):
    ''' k is outer CV, l inner
    '''
    print(f'\t-Nested CV: {cfg["dataset"]} for repeat {cfg["repeat"]}')
    outer_cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=valrepeats, random_state=cfg["repeat"])

    refit_models = []
    ensemble_models = []
    refit_metrics = []
    ensemble_metrics = []

    start_time = time.time()
    for fold_idx, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        results = rep_flat_cv (X_train, y_train, l, 1, cfg)

        # evaluate model on test, getting metrics
        metrics_refit = eval_model (results["refit_models"], X_test, y_test)
        metrics_ensemble = eval_model (results["ensemble_models"], X_test, y_test)
        # outer_auc_refit = metrics_refit["auc"]
        # outer_auc_ensemble = metrics_ensemble["auc"]

        refit_metrics.append(metrics_refit)
        ensemble_metrics.append(metrics_ensemble)
        refit_models.extend(results["refit_models"])
        ensemble_models.extend(results["ensemble_models"])

    # compute refitted model via flat cv
    end_time = time.time()
    total_time = end_time - start_time

    start_time = time.time()
    flat_model = rep_flat_cv (X_train, y_train, k, valrepeats, cfg)
    end_time = time.time()
    total_refit_time = end_time - start_time

    results = {
        "type": "nested_cv",
        "k": k,
        "l": l,
        "valrepeats": valrepeats,
        "dataset": cfg["dataset"],
        "repeat": cfg["repeat"],
        "metrics_refit": refit_metrics,
        "metrics_ensemble": ensemble_metrics,
        "refit_models": refit_models,
        "ensemble_models": ensemble_models,
        "flat_models": flat_model,
        "time": total_time,
        "refit_time": total_refit_time
    }

    return results



def apply_validation (cfg):
    try:
        os.makedirs("./results", exist_ok=True)

        dataset, valscheme, valrepeats, repeat = (
            cfg["dataset"], cfg["valscheme"], cfg["valrepeats"], cfg["repeat"]
        )

        X, y = radMLBench.loadData(dataset, return_X_y=True, local_cache_dir="./datasets")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=repeat, stratify=y)

        resFile = f"./results/{dataset}_{valrepeats}_{valscheme}_{repeat}.dump"
        if os.path.exists(resFile):
            return None

        valmethod, valparam = valscheme.split("-")
        print (f"Optimizing {dataset} for repeat {repeat} with scheme {valscheme}")
        study = None

        if valmethod == "CV":
            k = int(valparam)
            flat_cv_results = rep_flat_cv (X_train, y_train, k, valrepeats, cfg)
            final_metrics_refit = eval_model (flat_cv_results["refit_models"], X_test, y_test)
            final_metrics_ensemble = eval_model (flat_cv_results["ensemble_models"], X_test, y_test)
            flat_cv_results["final_metrics_refit"] = final_metrics_refit
            flat_cv_results["final_metrics_ensemble"] = final_metrics_ensemble
            dump(flat_cv_results, resFile)
        elif valmethod == "CVHoldout":
            k = int(valparam)
            holdout_cv_results = rep_holdout_cv (X_train, y_train, k, valrepeats, cfg)
            final_metrics_refit = eval_model (holdout_cv_results["refit_models"], X_test, y_test)
            final_metrics_ensemble = eval_model (holdout_cv_results["ensemble_models"], X_test, y_test)
            holdout_cv_results["final_metrics_refit"] = final_metrics_refit
            holdout_cv_results["final_metrics_ensemble"] = final_metrics_ensemble
            dump(holdout_cv_results, resFile)
        elif valmethod == "NestedCV":
            k, l = valparam.split("+")
            k = int(k)
            l = int(l)
            nested_cv_results = rep_nested_cv (X_train, y_train, k, l, valrepeats, cfg)
            final_metrics_refit = eval_model (nested_cv_results["refit_models"], X_test, y_test)
            final_metrics_ensemble = eval_model (nested_cv_results["ensemble_models"], X_test, y_test)
            final_metrics_flat = eval_model (nested_cv_results["flat_models"]["refit_models"], X_test, y_test)
            final_metrics_flat_ensemble = eval_model (nested_cv_results["flat_models"]["ensemble_models"], X_test, y_test)
            nested_cv_results["final_metrics_refit"] = final_metrics_refit
            nested_cv_results["final_metrics_flat"] = final_metrics_flat
            nested_cv_results["final_metrics_flat_ensemble"] = final_metrics_flat_ensemble
            nested_cv_results["final_metrics_ensemble"] = final_metrics_ensemble
            dump(nested_cv_results, resFile)
        else:
            raise Exception ("Unknown CV method")
    except Exception as e:
        print (f"Error occured while optimizing {dataset} for repeat {repeat} with scheme {valscheme}. {e}")
        with open("errors.txt", "a") as file:
            file.write(f"Error occurred while optimizing {dataset} for repeat {repeat} with scheme {valscheme}. {e}\n")
        raise (e)


if __name__ == '__main__':
    large_datasets = []
    for dataset in radMLBench.listDatasets():
        meta = radMLBench.getMetaData(dataset)
        if meta["nInstances"] > 100:
            large_datasets.append(dataset)

    print(f"Have {len(large_datasets)} datasets with more than 100 samples.")
    # large_datasets.remove("UCSF-PDGM")

    # large_datasets = ["Ahn2021"]
    valSchemes = ["CV-5", "CV-10", "CVHoldout-5", "CVHoldout-10", \
                        "NestedCV-5+10", "NestedCV-10+5"] # "NestedCV-5+5",

    experiments = []
    for r in range(nRepeats):
        subexp = []
        for dataset in large_datasets:
            for valscheme in valSchemes:
                for valrepeats in [1, 5]:
                    # every experiment needs to be called R times
                    exp_params = {
                        "dataset": dataset,
                        "valscheme": valscheme,
                        "valrepeats": valrepeats,
                        "repeat": r
                    }
                    subexp.append(exp_params)
        random.shuffle(subexp)
        experiments.extend(subexp)

    print (f"Computing {len(experiments)} experiments.")

    results = Parallel(n_jobs=30)(
        delayed(apply_validation)(e) for e in experiments
    )

    # for e in experiments:
    #     apply_validation(e)


#
