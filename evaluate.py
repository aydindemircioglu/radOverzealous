from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import wilcoxon, linregress

import pickle
from glob import glob

from joblib import Parallel, delayed
from joblib import dump, load
import numpy as np
import pandas as pd
import seaborn as sns
import os
import cv2

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from PIL import Image
from PIL import ImageDraw, ImageFont

import radMLBench
from loadDataUCI import *
from utils import *



name_mapping = {
    'flat_cv_k_1_5_Refit': 'Flat 5-fold CV, refit',
    'flat_cv_k_1_10_Refit': 'Flat 10-fold CV, refit',
    'flat_cv_k_5_5_Refit': 'Flat 5-fold CV, refit, 5x',
    'flat_cv_k_5_10_Refit': 'Flat 10-fold CV, refit, 5x',

    'flat_cv_k_1_5_Ensemble': 'Flat 5-fold CV, ensemble',
    'flat_cv_k_1_10_Ensemble': 'Flat 10-fold CV, ensemble',
    'flat_cv_k_5_5_Ensemble': 'Flat 5-fold CV, ensemble, 5x',
    'flat_cv_k_5_10_Ensemble': 'Flat 10-fold CV, ensemble, 5x',

    'holdout_cv_k_1_5_Refit': 'Holdout 5-fold CV, refit',
    'holdout_cv_k_1_10_Refit': 'Holdout 10-fold CV, refit',
    'holdout_cv_k_5_5_Refit': 'Holdout 5-fold CV, refit, 5x',
    'holdout_cv_k_5_10_Refit': 'Holdout 10-fold CV, refit, 5x',

    'holdout_cv_k_1_5_Ensemble': 'Holdout 5-fold CV, ensemble',
    'holdout_cv_k_1_10_Ensemble': 'Holdout 10-fold CV, ensemble',
    'holdout_cv_k_5_5_Ensemble': 'Holdout 5-fold CV, ensemble, 5x',
    'holdout_cv_k_5_10_Ensemble': 'Holdout 10-fold CV, ensemble, 5x',

    'nested_cv_1_5_10_Flat': 'Nested (5+10)-fold CV, refit',
    'nested_cv_1_10_5_Flat': 'Nested (10+5)-fold CV, refit',
    'nested_cv_5_5_10_Flat': 'Nested (5+10)-fold CV, refit, 5x',
    'nested_cv_5_10_5_Flat': 'Nested (10+5)-fold CV, refit, 5x',

    'nested_cv_1_5_10_Flat-Ensemble': 'Nested (5+10)-fold CV, refit ensemble',
    'nested_cv_1_10_5_Flat-Ensemble': 'Nested (10+5)-fold CV, refit ensemble',
    'nested_cv_5_5_10_Flat-Ensemble': 'Nested (5+10)-fold CV, refit ensemble, 5x',
    'nested_cv_5_10_5_Flat-Ensemble': 'Nested (10+5)-fold CV, refit ensemble, 5x',

    'nested_cv_1_5_10_Ensemble-Refit': 'Nested (5+10)-fold CV, simple ensemble',
    'nested_cv_1_10_5_Ensemble-Refit': 'Nested (10+5)-fold CV, simple ensemble',
    'nested_cv_5_5_10_Ensemble-Refit': 'Nested (5+10)-fold CV, simple ensemble, 5x',
    'nested_cv_5_10_5_Ensemble-Refit': 'Nested (10+5)-fold CV, simple ensemble, 5x',

    'nested_cv_1_5_10_Ensemble-Ensemble': 'Nested (5+10)-fold CV, full ensemble ',
    'nested_cv_1_10_5_Ensemble-Ensemble': 'Nested (10+5)-fold CV, full ensemble',
    'nested_cv_5_5_10_Ensemble-Ensemble': 'Nested (5+10)-fold CV, full ensemble, 5x',
    'nested_cv_5_10_5_Ensemble-Ensemble': 'Nested (10+5)-fold CV, full ensemble, 5x',



    'flat_cv_k_Refit': 'Flat CV, refit',
    'flat_cv_k_Ensemble': 'Flat CV, ensemble',
    'holdout_cv_k_Refit': 'Holdout CV, refit',
    'holdout_cv_k_Ensemble': 'Holdout CV, ensemble',
    'nested_cv_Flat': 'Nested CV, refit',
    'nested_cv_Flat-Ensemble': 'Nested CV, refit ensemble',
    'nested_cv_Ensemble-Ensemble': 'Nested CV, full ensemble',
    'nested_cv_Ensemble-Refit': 'Nested CV, simple ensemble',
}

metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1', 'mcc', 'sensitivity', 'specificity']




def process_file(z, propTbl, metrics):
    try:
         results = []
         if "results_" in z: # just in case
             return []

         df = load(z)

         base = {"Dataset": df["dataset"], "CV": df["type"]}
         base.update(propTbl[df["dataset"]])
         base["CV-Repeats"] = df["valrepeats"]
         base["Repeat"] = df["repeat"]
         if base["CV"] == "flat_cv_k" or base["CV"] == "holdout_cv_k":
             base["Folds"] = df["k"]
             base["Time"] = df["time"]

             row = base.copy()
             row["Evaluation"] = "Refit"
             for metric in metrics:
                 metric_key = metric.upper() if metric in ['auc', 'mcc'] else metric.capitalize()
                 row[f"{metric_key}-Int"] = df["metrics_refit"][metric]
                 row[f"{metric_key}-Ext"] = df["final_metrics_refit"][metric]
                 row[f"{metric_key}-Overfitting"] = row[f"{metric_key}-Int"] - row[f"{metric_key}-Ext"]
             results.append(row)

             row = base.copy()
             row["Evaluation"] = "Ensemble"
             for metric in metrics:
                 metric_key = metric.upper() if metric in ['auc', 'mcc'] else metric.capitalize()
                 row[f"{metric_key}-Int"] = df["metrics_ensemble"][metric]
                 row[f"{metric_key}-Ext"] = df["final_metrics_ensemble"][metric]
                 row[f"{metric_key}-Overfitting"] = row[f"{metric_key}-Int"] - row[f"{metric_key}-Ext"]
             results.append(row)

         elif base["CV"] == "nested_cv":
             base["Folds"] = f'{df["k"]}_{df["l"]}'
             base["Time"] = df["time"]
             # nested cv has no 'identify best model config',
             # so in each inner loop one model is there and its auc,
             # and there is no selection process. thus we obtain #loops estimates

             # this is the internal CV refits used as ensemble
             row = base.copy()
             row["Evaluation"] = "Ensemble-Refit"

             for metric in metrics:
                 metric_key = metric.upper() if metric in ['auc', 'mcc'] else metric.capitalize()
                 values = [r[metric] for r in df["metrics_refit"]]
                 row[f"{metric_key}-Int"] = np.mean(values)
                 row[f"{metric_key}-Ext"] = df["final_metrics_refit"][metric]
                 row[f"{metric_key}-Overfitting"] = row[f"{metric_key}-Int"] - row[f"{metric_key}-Ext"]
             results.append(row)

             # here we use all internal CV models as ensemble
             row = base.copy()
             row["Evaluation"] = "Ensemble-Ensemble"
             for metric in metrics:
                 metric_key = metric.upper() if metric in ['auc', 'mcc'] else metric.capitalize()
                 values = [r[metric] for r in df["metrics_ensemble"]]
                 row[f"{metric_key}-Int"] = np.mean(values)
                 row[f"{metric_key}-Ext"] = df["final_metrics_ensemble"][metric]
                 row[f"{metric_key}-Overfitting"] = row[f"{metric_key}-Int"] - row[f"{metric_key}-Ext"]
             results.append(row)

             # for flat we use the internal estimation but another model,
             # sowe can decide whether to use refit or ensemble
             # since refit is 'normal', we use that
             row = base.copy()
             row["Evaluation"] = "Flat"
             for metric in metrics:
                 metric_key = metric.upper() if metric in ['auc', 'mcc'] else metric.capitalize()
                 values = [r[metric] for r in df["metrics_refit"]]
                 row[f"{metric_key}-Int"] = np.mean(values)
                 row[f"{metric_key}-Ext"] = df["final_metrics_flat"][metric]
                 row[f"{metric_key}-Overfitting"] = row[f"{metric_key}-Int"] - row[f"{metric_key}-Ext"]
             results.append(row)

             # here we use the internal refit CV as model, but ensemble it
             row = base.copy()
             row["Evaluation"] = "Flat-Ensemble"
             for metric in metrics:
                 metric_key = metric.upper() if metric in ['auc', 'mcc'] else metric.capitalize()
                 values = [r[metric] for r in df["metrics_refit"]]
                 row[f"{metric_key}-Int"] = np.mean(values)
                 row[f"{metric_key}-Ext"] = df["final_metrics_flat_ensemble"][metric]
                 row[f"{metric_key}-Overfitting"] = row[f"{metric_key}-Int"] - row[f"{metric_key}-Ext"]
             results.append(row)
         else:
             raise Exception (f'Unknown CV: {base["CV"]}')
         return results
    except Exception as e:
        print(f"Error in file {z}: {str(e)}")
        raise  # Re-raise the exception to stop the process


def readResults(cohort):
    try:
        results = load(f"./paper/results_{cohort}.dump")
        return pd.DataFrame(results).reset_index(drop=True)
    except:
        pass

    propTbl = {}
    datasets = getDatasetList (cohort)
    files = [f for f in glob("./results/*.dump") if any(os.path.basename(f).startswith(ds) for ds in datasets)]

    with Parallel(n_jobs=30, verbose = 10) as parallel:
        results = parallel(delayed(process_file)(z, propTbl, metrics) for z in files)
    results = [item for sublist in results for item in sublist]

    dump(results, "./paper/results_{cohort}.dump")
    return pd.DataFrame(results).reset_index(drop=True)



def groupData (df_org, metric = "AUC"):
    df = df_org.groupby(['Dataset', 'CV', 'CV-Repeats', 'Folds', 'Evaluation']).mean().reset_index()
    df[f"{metric}-Ext-Best"] = df.groupby('Dataset')[f'{metric}-Ext'].transform('max')
    df[f"{metric}-Performance"] = df[f"{metric}-Ext"] - df[f"{metric}-Ext-Best"]
    df[f"{metric}-Overfitting"] = df[f"{metric}-Int"] - df[f"{metric}-Ext"]
    return df


def getDatasetList (cohort):
    large_datasets = []
    if cohort == "radMLBench":
        for dataset in radMLBench.listDatasets():
            m = radMLBench.getMetaData(dataset)
            if m["nInstances"] > 100:
                large_datasets.append(dataset)
    else:
        for dataset in listDatasetsUCI():
            X, y = loadDatasetUCI(dataset)
            if y.shape[0] >= 100 and np.sum(y) > 20:
                large_datasets.append(dataset)
    return large_datasets



def createDatasetTable(cohort):
    tbl = []
    large_datasets = getDatasetList(cohort)
    if cohort == "radMLBench":
        for dataset in large_datasets:
            m = radMLBench.getMetaData(dataset)
            tbl.append({"Dataset": dataset, "Modality": m["modality"], "Outcome": m["outcome"],
                "Instances": m['nInstances'],
                "Features": m["nFeatures"], "Dimensionality": m["Dimensionality"], "Balance": m["ClassBalance"]})
    else:
        for dataset in large_datasets:
            X, y = loadDatasetUCI(dataset)
            tbl.append({"Dataset": dataset,
                "Instances": X.shape[0],
                "Features": X.shape[1], "Dimensionality": np.round(X.shape[1] / X.shape[0], 3),
                "Balance": np.round( np.sum(y)/X.shape[0]*100)})
    tbl = pd.DataFrame(tbl)
    tbl = tbl.sort_values(["Dataset"])
    tbl.to_excel(f"./paper/Table_1_{cohort}.xlsx", index=False)



def bootstrap_regression(x, y, n_bootstrap=1000):
    bootstrap_slopes = []
    bootstrap_intercepts = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(range(len(x)), size=len(x), replace=True)
        x_sample = x[indices]
        y_sample = y[indices]
        slope, intercept, *_ = linregress(x_sample, y_sample)
        bootstrap_slopes.append(slope)
        bootstrap_intercepts.append(intercept)
    return np.array(bootstrap_slopes), np.array(bootstrap_intercepts)



def testRelations(diffs, ID, cohort = None, metric = "AUC", DPI = 300):
    diffs = diffs.sort_values([f"{metric}-Overfitting"])
    nFeatures_values = diffs["Features"].values
    nInstances_values = diffs["Instances"].values
    dimensionality_values = diffs["Dimensionality"].values
    diff_values = diffs[f"{metric}-Overfitting"].values

    sns.set(style="white")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi = DPI)
    sns.set(style="white")
    axis_fontsize = 16
    title_fontsize = 15

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "text.usetex": False,
    })

    plt.subplots_adjust(wspace=4.7)

    slope, intercept, r_value, p_value, std_err = linregress(nFeatures_values, diff_values)
    if p_value < 0.10:
        print (f"Feature is significant for {ID}: {p_value}")
    slopes, intercepts = bootstrap_regression(nFeatures_values, diff_values)
    y_pred_mean = slope * nFeatures_values + intercept
    y_pred_bootstrap = [s * nFeatures_values + b for s, b in zip(slopes, intercepts)]
    y_pred_lower = np.percentile(y_pred_bootstrap, 2.5, axis=0)
    y_pred_upper = np.percentile(y_pred_bootstrap, 97.5, axis=0)

    axes[0].scatter(nFeatures_values, diff_values, s=15, color='black')
    axes[0].plot(nFeatures_values, y_pred_mean, color='black')
    #axes[0].fill_between(nFeatures_values, y_pred_lower, y_pred_upper, color='grey', alpha=0.3)
    sorted_indices = np.argsort(nFeatures_values)
    axes[0].fill_between(nFeatures_values[sorted_indices], y_pred_lower[sorted_indices], y_pred_upper[sorted_indices], color='grey', alpha=0.3)
    axes[0].text(0.42, 0.97, f"$R^2 = {r_value**2:.2f}$\n(p = {p_value:.2g})", transform=axes[0].transAxes, fontsize=15, verticalalignment='top')
    axes[0].set_xlabel("Number of Features", fontsize=axis_fontsize)
    axes[0].set_ylabel(f"Overfitting (in {metric})", fontsize=axis_fontsize)
#    axes[0].text(-0.17, -0.08, "(a)", transform=axes[0].transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left')
    axes[0].axhline(y=0.0, color='red', linewidth=2, zorder=2,  linestyle='--')

    slope, intercept, r_value, p_value, std_err = linregress(nInstances_values, diff_values)
    if p_value < 0.10:
        print (f"Instances is significant for {ID}: {p_value}")
    slopes, intercepts = bootstrap_regression(nInstances_values, diff_values)
    y_pred_mean = slope * nInstances_values + intercept
    y_pred_bootstrap = [s * nInstances_values + b for s, b in zip(slopes, intercepts)]
    y_pred_lower = np.percentile(y_pred_bootstrap, 2.5, axis=0)
    y_pred_upper = np.percentile(y_pred_bootstrap, 97.5, axis=0)

    axes[1].scatter(nInstances_values, diff_values, s=15, color='black')
    axes[1].plot(nInstances_values, y_pred_mean, color='black')
    sorted_indices = np.argsort(nInstances_values)
    axes[1].fill_between(nInstances_values[sorted_indices], y_pred_lower[sorted_indices], y_pred_upper[sorted_indices], color='grey', alpha=0.3)
    axes[1].text(0.42, 0.97, f"$R^2 = {r_value**2:.2f}$\n(p = {p_value:.2g})", transform=axes[1].transAxes, fontsize=15, verticalalignment='top')
    axes[1].set_xlabel("Number of Instances", fontsize=axis_fontsize)
    axes[1].set_ylabel(f"Overfitting (in {metric})", fontsize=axis_fontsize)
#    axes[1].text(-0.17, -0.08, "(b)", transform=axes[1].transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left')
    axes[1].axhline(y=0.0, color='red', linewidth=2, zorder=2,  linestyle='--')

    slope, intercept, r_value, p_value, std_err = linregress(dimensionality_values, diff_values)
    if p_value < 0.10:
        print (f"Dimensionality is significant for {ID}: {p_value}")
    slopes, intercepts = bootstrap_regression(dimensionality_values, diff_values)
    y_pred_mean = slope * dimensionality_values + intercept
    y_pred_bootstrap = [s * dimensionality_values + b for s, b in zip(slopes, intercepts)]
    y_pred_lower = np.percentile(y_pred_bootstrap, 2.5, axis=0)
    y_pred_upper = np.percentile(y_pred_bootstrap, 97.5, axis=0)

    mask = (dimensionality_values < 40)
    axes[2].scatter(dimensionality_values[mask], diff_values[mask], s=15, color='black')
    axes[2].plot(dimensionality_values[mask], y_pred_mean[mask], color='black')
    sorted_indices = np.argsort(dimensionality_values[mask])
    axes[2].fill_between(dimensionality_values[mask][sorted_indices],
                          y_pred_lower[mask][sorted_indices],
                          y_pred_upper[mask][sorted_indices],
                          color='grey', alpha=0.3)
    #axes[2].fill_between(dimensionality_values[sorted_indices], y_pred_lower[sorted_indices], y_pred_upper[sorted_indices], color='grey', alpha=0.3)
    axes[2].text(0.42, 0.97, f"$R^2 = {r_value**2:.2f}$\n(p = {p_value:.2g})", transform=axes[2].transAxes, fontsize=15, verticalalignment='top')
    axes[2].set_xlabel("Dimensionality", fontsize=axis_fontsize)
    axes[2].set_ylabel(f"Overfitting (in {metric})", fontsize=axis_fontsize)
#    axes[2].text(-0.17, -0.08, "(c)", transform=axes[2].transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left')
    axes[2].axhline(y=0.0, color='red', linewidth=2, zorder=2,  linestyle='--')

    plt.tight_layout()
    os.makedirs(f"./paper/relations_{cohort}", exist_ok = True)
    plt.savefig(f"./paper/relations_{cohort}/FigRelation_{ID}.png")
    plt.close()




def generateRelationPlots (df_org, cohort = None, metric = "AUC"):
    df = groupData (df_org)

    diffs = df.copy()
    #diffs = diffs.query("DiffAUC < 0.4").copy()
    id_columns = ['CV', 'CV-Repeats', 'Folds', 'Evaluation']
    diffs['ID'] = diffs[id_columns].astype(str).agg('_'.join, axis=1)

    for ID in diffs["ID"].unique():
        subdf = diffs.query("ID == @ID")
        testRelations(subdf, ID, cohort = cohort, metric = metric, DPI = 300)



def plotVariance (df_org, cohort = None ,metric = "AUC"):
    # want to see the variance per dataset
    df = df_org.copy()
    id_columns = ['CV', 'CV-Repeats', 'Folds', 'Evaluation']
    df['ID'] = df[id_columns].astype(str).agg('_'.join, axis=1)
    df["ID_Group"] = pd.Categorical(df["ID"].str.split("_").str[0], categories=["flat", "holdout", "nested"], ordered=True)

    df = df.sort_values(by=["Dataset", "ID_Group", "ID"])
    mapping_order = list(name_mapping.keys())[:32]

    # compute std deviations first, its now 25600/25=1024 rows
    df_std = df.groupby(['ID', 'Dataset'])[f"{metric}-Overfitting"].std().reset_index()
    df_std["ID_Group"] = pd.Categorical(df_std["ID"].str.split("_").str[0], categories=["flat", "holdout", "nested"], ordered=True)

    fig, ax = plt.subplots(figsize=(15, 10))

    # # Process each subplot
    # for ax, dataset in zip(g.axes.flat, df_std['Dataset'].unique()):
    #     # Filter data for the current dataset
    #     subset = df_std[df_std['Dataset'] == dataset]
    #     ax.clear()
    #     unique_ids = mapping_order

    positions = []
    current_pos = 0
    group_to_positions = {}
    id_to_position = {}

    for id_val in mapping_order:
        group = df_std[df_std['ID'] == id_val]['ID_Group'].iloc[0]
        if group not in group_to_positions:
            if positions:
                current_pos += 1
            group_to_positions[group] = []
        group_to_positions[group].append(current_pos)
        positions.append(current_pos)
        id_to_position[id_val] = current_pos
        current_pos += 1


    # Create the boxplot
    bp = sns.boxplot(
        data=df_std,
        x="ID", y=f"{metric}-Overfitting",
        ax=ax,
        positions=positions,
        width=0.7,
        order=mapping_order,
        color='blue',
        linewidth=1.5,
        fliersize=0
    )

    # Adjust the transparency of the boxplot patches
    for patch in bp.patches:
        patch.set_alpha(0.2)

    # Add scatter plot for each ID
    for id_val in mapping_order:
        id_data = df_std[df_std['ID'] == id_val]
        x_pos = id_to_position[id_val]
        x_jitter = 0 * np.random.normal(0, 0.05, size=len(id_data))
        ax.scatter(x_pos + x_jitter, id_data[f"{metric}-Overfitting"],
                   color='black', s=10, alpha=0.6, zorder=3)

    # Set x-ticks and labels
    ax.set_xticks(positions)
    ax.set_xticklabels([name_mapping[u] for u in mapping_order], rotation=45, ha='right', fontsize=14)

    # Adjust tick parameters
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    # Set y-axis limits
    ax.set_ylim(0.0, 0.23)

    # Set title and labels
    ax.set_title("", fontsize=23, pad=20)
    ax.set_ylabel(f"Standard deviation of {metric}-Overfitting", fontsize=14)
    ax.set_xlabel(f"", fontsize=14)


    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"paper/Figure_7_{cohort}.png", dpi=300, bbox_inches='tight')
    plt.close()



def checkVariancevsSamplesize(df_org, metric = "AUC", cohort = cohort, DPI=300):
    df = df_org.copy()
    id_columns = ['CV', 'CV-Repeats', 'Folds', 'Evaluation']
    df['ID'] = df[id_columns].astype(str).agg('_'.join, axis=1)

    # compute std deviations first, its now 25600/25=1024 rows
    std_dev = df.groupby(['ID', 'Dataset'])[f"{metric}-Overfitting"].std().reset_index()

    # now we take the mean std over all datasets
    mean_std_dev = std_dev.groupby('Dataset')[f"{metric}-Overfitting"].mean().reset_index()

    # add back metadata
    dataset_info = df[['Dataset', 'Instances', 'Dimensionality', 'Features']].drop_duplicates()
    mean_std_dev = mean_std_dev.merge(dataset_info, on='Dataset', how='left')

    nInstances_values = mean_std_dev["Instances"].values
    nFeatures_values = mean_std_dev["Features"].values
    nDimensionality_values = mean_std_dev["Dimensionality"].values
    overfitting_values = mean_std_dev[f"{metric}-Overfitting"].values

    # Create the plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=DPI)
    sns.set(style="white")
    axis_fontsize = 16
    title_fontsize = 15

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "text.usetex": False,
    })

    plt.subplots_adjust(wspace=4.0)

    # Plot 1: Overfitting vs Instances
    slope, intercept, r_value, p_value, std_err = linregress(nInstances_values, overfitting_values)
    if p_value < 0.10:
        print (f"Instances is significant: {p_value}")
    slopes, intercepts = bootstrap_regression(nInstances_values, overfitting_values)
    y_pred_mean = slope * nInstances_values + intercept
    y_pred_bootstrap = [s * nInstances_values + b for s, b in zip(slopes, intercepts)]
    y_pred_lower = np.percentile(y_pred_bootstrap, 2.5, axis=0)
    y_pred_upper = np.percentile(y_pred_bootstrap, 97.5, axis=0)

    axes[0].scatter(nInstances_values, overfitting_values, s=15, color='black')
    axes[0].plot(nInstances_values, y_pred_mean, color='black')
    sorted_indices = np.argsort(nInstances_values)
    axes[0].fill_between(nInstances_values[sorted_indices], y_pred_lower[sorted_indices], y_pred_upper[sorted_indices], color='grey', alpha=0.3)
    axes[0].text(0.42, 0.97, f"$R^2 = {r_value**2:.2f}$\n(p = {p_value:.2g})", transform=axes[0].transAxes, fontsize=15, verticalalignment='top')
    axes[0].set_xlabel("Number of Instances", fontsize=axis_fontsize)
    axes[0].set_ylabel(f"Observed variance in {metric}", fontsize=axis_fontsize)
    axes[0].text(-0.17, -0.08, "(a)", transform=axes[0].transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left')
    axes[0].axhline(y=0.0, color='red', linewidth=2, zorder=2, linestyle='--')


    # Plot 2: Overfitting vs Features
    slope, intercept, r_value, p_value, std_err = linregress(nFeatures_values, overfitting_values)
    if p_value < 0.10:
        print (f"Features is significant: {p_value}")
    slopes, intercepts = bootstrap_regression(nFeatures_values, overfitting_values)
    y_pred_mean = slope * nFeatures_values + intercept
    y_pred_bootstrap = [s * nFeatures_values + b for s, b in zip(slopes, intercepts)]
    y_pred_lower = np.percentile(y_pred_bootstrap, 2.5, axis=0)
    y_pred_upper = np.percentile(y_pred_bootstrap, 97.5, axis=0)

    axes[1].scatter(nFeatures_values, overfitting_values, s=15, color='black')
    axes[1].plot(nFeatures_values, y_pred_mean, color='black')
    sorted_indices = np.argsort(nFeatures_values)
    axes[1].fill_between(nFeatures_values[sorted_indices], y_pred_lower[sorted_indices], y_pred_upper[sorted_indices], color='grey', alpha=0.3)
    axes[1].text(0.42, 0.97, f"$R^2 = {r_value**2:.2f}$\n(p = {p_value:.2g})", transform=axes[1].transAxes, fontsize=15, verticalalignment='top')
    axes[1].set_xlabel("Number of Features", fontsize=axis_fontsize)
    axes[1].set_ylabel(f"Observed variance in {metric}", fontsize=axis_fontsize)
    axes[1].text(-0.17, -0.08, "(b)", transform=axes[1].transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left')
    axes[1].axhline(y=0.0, color='red', linewidth=2, zorder=2,  linestyle='--')

    # Plot 2: Overfitting vs nDimensionality_values
    slope, intercept, r_value, p_value, std_err = linregress(nDimensionality_values, overfitting_values)
    if p_value < 0.10:
        print (f"nDimensionality_values is significant: {p_value}")
    slopes, intercepts = bootstrap_regression(nDimensionality_values, overfitting_values)
    y_pred_mean = slope * nDimensionality_values + intercept
    y_pred_bootstrap = [s * nDimensionality_values + b for s, b in zip(slopes, intercepts)]
    y_pred_lower = np.percentile(y_pred_bootstrap, 2.5, axis=0)
    y_pred_upper = np.percentile(y_pred_bootstrap, 97.5, axis=0)

    mask = (nDimensionality_values < 40)
    axes[2].scatter(nDimensionality_values[mask], overfitting_values[mask], s=15, color='black')
    axes[2].plot(nDimensionality_values[mask], y_pred_mean[mask], color='black')
    sorted_indices = np.argsort(nDimensionality_values[mask])
    axes[2].fill_between(nDimensionality_values[mask][sorted_indices],
                          y_pred_lower[mask][sorted_indices],
                          y_pred_upper[mask][sorted_indices],
                          color='grey', alpha=0.3)
    #axes[2].fill_between(dimensionality_values[sorted_indices], y_pred_lower[sorted_indices], y_pred_upper[sorted_indices], color='grey', alpha=0.3)
    axes[2].text(0.42, 0.97, f"$R^2 = {r_value**2:.2f}$\n(p = {p_value:.2g})", transform=axes[2].transAxes, fontsize=15, verticalalignment='top')
    axes[2].set_xlabel("Dimensionality", fontsize=axis_fontsize)
    axes[2].set_ylabel(f"Observed variance in {metric}", fontsize=axis_fontsize)
    axes[2].text(-0.17, -0.08, "(c)", transform=axes[2].transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left')
    axes[2].axhline(y=0.0, color='red', linewidth=2, zorder=2, linestyle='--')

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"./paper/Figure_8_{cohort}.png")



def generatePlotsPerDataset (df_org, cohort = None):
    # want to see the variance per dataset
    df = df_org.copy()
    id_columns = ['CV', 'CV-Repeats', 'Folds', 'Evaluation']
    df['ID'] = df[id_columns].astype(str).agg('_'.join, axis=1)
    df["ID_Group"] = pd.Categorical(df["ID"].str.split("_").str[0], categories=["flat", "holdout", "nestedcv"], ordered=True)
    df = df.sort_values(by=["Dataset", "ID_Group", "ID"])
    mapping_order = list(name_mapping.keys())[:32]

    df_data = df.copy()

    for dataset in df_data.Dataset.unique():
        df = df_data.query("Dataset == @dataset")
        g = sns.catplot(
            data=df,
            x="ID", y="AUC-Overfitting",
            kind="box",
            height=10, aspect=1.5,  # Increased height
            col="Dataset",
            sharey=True,
            col_wrap=1,
            order=mapping_order
        )

        # Process each subplot
        for ax in g.axes.flat:
            # Get current dataset name from subplot title
            dataset = ax.get_title().split(" = ")[-1]
            subset = df[df["Dataset"] == dataset]
            ax.clear()
            unique_ids = mapping_order

            positions = []
            current_pos = 0
            group_to_positions = {}
            id_to_position = {}

            for id_val in mapping_order:
                group = subset[subset['ID'] == id_val]['ID_Group'].iloc[0]
                if group not in group_to_positions:
                    if positions:
                        current_pos += 1
                    group_to_positions[group] = []
                group_to_positions[group].append(current_pos)
                positions.append(current_pos)
                id_to_position[id_val] = current_pos
                current_pos += 1

            bp = sns.boxplot(
                data=subset,
                x="ID", y="AUC-Overfitting",
                ax=ax,
                positions=positions,
                width=0.7,
                order=mapping_order,
                color='blue',
                linewidth=1.5,
                fliersize=0
            )

            for patch in bp.patches:
                patch.set_alpha(0.2)

            for id_val in mapping_order:
                id_data = subset[subset['ID'] == id_val]
                x_pos = id_to_position[id_val]
                x_jitter = 0*np.random.normal(0, 0.05, size=len(id_data))
                ax.scatter(x_pos + x_jitter, id_data['AUC-Overfitting'],
                          color='black', s=10, alpha=0.6, zorder=3)

            ax.set_xticks(positions)
            ax.set_xticklabels([name_mapping[u] for u in unique_ids], rotation=45, ha='right', fontsize=14)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.set_ylim(-0.4, 0.4)
            ax.set_title(dataset, fontsize=23, pad=20)
            ax.set_xlabel("", fontsize=14)
            ax.set_ylabel("AUC-Overfitting", fontsize=14)
            ax.axhline(y=0.0, color='red', linewidth=2, zorder=2)

        plt.tight_layout()
        os.makedirs(f"paper/dataset_{cohort}", exist_ok = True)
        plt.savefig(f"paper/dataset_{cohort}/{dataset}.png", dpi=300, bbox_inches='tight')
        plt.close()



def getTimings(df_org, cohort = cohort):
    color_palette = {
        'flat_cv_k_1_5_Refit': '#4c8bf5',
        'flat_cv_k_1_10_Refit': '#4c8bf5',
        'flat_cv_k_5_5_Refit': '#4c8bf5',
        'flat_cv_k_5_10_Refit': '#4c8bf5',

        'holdout_cv_k_1_5_Refit': '#ff9d3b',
        'holdout_cv_k_1_10_Refit': '#ff9d3b',
        'holdout_cv_k_5_5_Refit': '#ff9d3b',
        'holdout_cv_k_5_10_Refit': '#ff9d3b',

        'nested_cv_1_5_10_Flat': '#bb5dbf',
        'nested_cv_1_10_5_Flat': '#bb5dbf',
        'nested_cv_5_5_10_Flat': '#bb5dbf',
        'nested_cv_5_10_5_Flat': '#bb5dbf'
    }

    order_within_group = {
        'flat': ['flat_cv_k_1_5_Refit', 'flat_cv_k_1_10_Refit',
                'flat_cv_k_5_5_Refit', 'flat_cv_k_5_10_Refit'],
        'holdout': ['holdout_cv_k_1_5_Refit', 'holdout_cv_k_1_10_Refit',
                   'holdout_cv_k_5_5_Refit', 'holdout_cv_k_5_10_Refit'],
        'nested': ['nested_cv_1_5_10_Flat', 'nested_cv_1_10_5_Flat',
                  'nested_cv_5_5_10_Flat', 'nested_cv_5_10_5_Flat']
    }

    df = df_org.query("Evaluation == 'Flat' or Evaluation == 'Refit' ").copy()
    id_columns = ['CV', 'CV-Repeats', 'Folds', 'Evaluation']
    df['ID'] = df[id_columns].astype(str).agg('_'.join, axis=1)
    df["ID_Group"] = pd.Categorical(df["ID"].str.split("_").str[0], categories=["flat", "holdout", "nested"], ordered=True)

    # Convert time to hours
    subset = (
        df.groupby(["ID", "ID_Group", "Dataset"], observed=True)["Time"]
        .mean()
        .reset_index()
    )
    subset["Time"] = subset["Time"] / 3600  # Convert to hours

    positions = []
    id_to_position = {}
    current_pos = 0

    for group in ["flat", "holdout", "nested"]:
        for id_val in order_within_group[group]:
            id_to_position[id_val] = current_pos
            positions.append(current_pos)
            current_pos += 1
        current_pos += 1

    plt.figure(figsize=(9, 6.6))
    plt.grid(False)

    for group in ["flat", "holdout", "nested"]:
        for id_val in order_within_group[group]:
            id_data = subset[subset['ID'] == id_val]
            if len(id_data) == 0:
                continue

            x_pos = id_to_position[id_val]
            mean_time = id_data["Time"].mean()
            stderr = id_data["Time"].std() / np.sqrt(len(id_data))

            color = color_palette[id_val]
            plt.bar(x_pos, mean_time, width=0.6, color=color, alpha=0.3)
            plt.errorbar(x_pos, mean_time, yerr=stderr, color=color, capsize=5, capthick=1, elinewidth=1)

            jitter = 0.2 * (2 * np.random.rand(len(id_data)) - 1)
            plt.scatter(
                x_pos + jitter,
                id_data["Time"],
                color=color,
                s=20,
                alpha=0.7
            )

    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')

    ordered_ids = []
    for group in ["flat", "holdout", "nested"]:
        ordered_ids.extend(order_within_group[group])

    # Use the name mapping for x-axis labels
    clean_labels = [name_mapping[id_val] for id_val in ordered_ids]
    plt.xticks(positions[:len(ordered_ids)], clean_labels, rotation=45, ha='right', fontsize=10)
    plt.xlabel("")
    plt.ylabel("Time (hours)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"paper/Figure_S2_{cohort}.png", dpi=500, bbox_inches="tight")
    #plt.show()
    plt.close()



def getOverfittingStats(df_org, fname = None, metric = "AUC"):
    print ("\n\nOverfitting stats")
    df = groupData (df_org, metric)

    id_columns = ['CV', 'CV-Repeats', 'Folds', 'Evaluation']
    mean_table = df.groupby(id_columns)[[f'{metric}-Int', f'{metric}-Ext', f'{metric}-Overfitting', f'{metric}-Performance']].mean().round(3)
    D5 = mean_table.query("`CV-Repeats` == 5").reset_index()
    D1 = mean_table.query("`CV-Repeats` == 1 ").reset_index()
    print ("Reduction of overfitting for repeats 1 -> 5", np.mean(D5["AUC-Overfitting"]-D1["AUC-Overfitting"]))

    # to see what really happens
    D5['ID'] = D5[id_columns].astype(str).agg('_'.join, axis=1)
    z= D5["AUC-Overfitting"]-D1["AUC-Overfitting"]
    z.index = D5.ID
    print (z)

    D5 = mean_table.query("Folds == 5 or Folds == '5_10' ").reset_index()
    D10 = mean_table.query("Folds == 10 or Folds == '10_5' ").reset_index()
    print ("Reduction of overfitting for ext. folds 5 -> 10", np.mean(D10["AUC-Overfitting"]-D5["AUC-Overfitting"]))

    D10['ID'] = D10[id_columns].astype(str).agg('_'.join, axis=1)
    z= D10["AUC-Overfitting"]-D5["AUC-Overfitting"]
    z.index = D10.ID
    print (z)

    D5 = mean_table.query("Folds == 5 ").reset_index()
    D10 = mean_table.query("Folds == 10").reset_index()
    print ("Reduction of overfitting for ext. folds 5 -> 10, no nested-CV", np.mean(D10["AUC-Overfitting"]-D5["AUC-Overfitting"]))




def getSummaryTable(df_org, cohort = None, fname = None, metric = "AUC"):
    df = groupData (df_org, metric)

    mean_table = df.groupby(['CV', 'Folds', 'CV-Repeats', 'Evaluation'])[[f'{metric}-Int', f'{metric}-Ext', f'{metric}-Overfitting', f'{metric}-Performance']].mean().round(3)
    std_table = df.groupby(['CV', 'Folds', 'CV-Repeats', 'Evaluation'])[[f'{metric}-Int', f'{metric}-Ext', f'{metric}-Overfitting', f'{metric}-Performance']].std().round(3)

    CV_mapping = {'flat_cv_k': 'Flat CV', 'holdout_cv_k': 'Holdout CV', "nested_cv": 'Nested CV'}
    Folds_mapping = {5: '5', 10: '10', '5_10': '5+10', '10_5': '10+5'}
    Model_mapping = {"Flat": "Refit", "Flat-Ensemble": "Refit ensemble",
        "Ensemble-Ensemble": "Full ensemble", "Ensemble-Refit": "Simple ensemble"}

    summary_table = mean_table.astype(str) + " ± " + std_table.astype(str)
    summary_table = summary_table.reset_index()
    summary_table["CV"] = summary_table["CV"].map(CV_mapping)
    summary_table["Folds"] = summary_table["Folds"].map(Folds_mapping)
    summary_table.loc[summary_table["CV"] == "Nested CV", "Evaluation"] = summary_table["Evaluation"].map(Model_mapping)

    eval_order = {
        "Flat CV": ["Refit", "Ensemble"],
        "Holdout CV": ["Refit", "Ensemble"],
        "Nested CV": ["Refit", "Refit ensemble", "Simple ensemble", "Full ensemble"]
    }

    summary_table["temp"] = summary_table["CV"].map({"Flat CV": 0, "Holdout CV": 10, "Nested CV": 20})
    summary_table["temp"] += summary_table["Folds"].map({'5': 0, '10': 1, '5+10': 2, '10+5': 3})
    summary_table["temp"] += summary_table["CV-Repeats"].map({1: 0, 5: 1}) * 0.1
    summary_table["temp"] += summary_table.apply(lambda row: eval_order.get(row["CV"], []).index(row["Evaluation"]) * 0.01, axis=1)

    summary_table = summary_table.sort_values(by=["temp"]).drop(columns=["temp"]).reset_index(drop=True)
    summary_table.to_excel(f"./paper/{fname}.xlsx")



def addText (finalImage, text = '', org = (0,0), fontFace = '', fontSize = 12, color = (255,255,255)):
     # Convert the image to RGB (OpenCV uses BGR)
     #tmpImg = cv2.cvtColor(finalImage, cv2.COLOR_BGR2RGB)
     tmpImg = finalImage
     pil_im = Image.fromarray(tmpImg)
     draw = ImageDraw.Draw(pil_im)
     font = ImageFont.truetype(fontFace + ".ttf", fontSize)
     draw.text(org, text, font=font, fill = color)
     #tmpImg = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
     tmpImg = np.array(pil_im)
     return (tmpImg.copy())



def addBorder (img, pos, thickness):
    if pos == "H":
        img = np.hstack([255*np.ones(( img.shape[0],int(img.shape[1]*thickness), 3), dtype = np.uint8),img])
    if pos == "V":
        img = np.vstack([255*np.ones(( int(img.shape[0]*thickness), img.shape[1], 3), dtype = np.uint8),img])
    return img



def addBlackBorder(img, pixel):
    return cv2.copyMakeBorder(img, pixel, pixel, pixel, pixel, cv2.BORDER_CONSTANT, value=(0, 0, 0))



def joinRepeatsPlot():
    fontFace = "Arial"

    imA = cv2.imread("./paper/dataset_radMLBench/WORC-Lipo.png")
    #imA = addText(imA, "a", (40, 40), fontFace, 112, color=(0, 0, 0))

    imB = cv2.imread("./paper/dataset_radMLBench/PI-CAI.png")
    imB = cv2.resize(imB, (imA.shape[1], imA.shape[0]), interpolation=cv2.INTER_LINEAR)

    imA = addBlackBorder(imA, 10)
    imB = addBlackBorder(imB, 10)
    #imB = addText(imB, "b", (40, 40), fontFace, 112, color=(0, 0, 0))

    imB = addBorder(imB, "V", 0.05)
    imgU = np.vstack([imA, imB])

    cv2.imwrite("./paper/Figure_S1.png", imgU)



def joinRelationPlot():
    fontFace = "Arial"

    imA = cv2.imread("./paper/relations_radMLBench/FigRelation_flat_cv_k_1_5_Refit.png")
    #imA = addText(imA, "a", (40, 40), fontFace, 112, color=(0, 0, 0))

    imB = cv2.imread("./paper/relations_radMLBench/FigRelation_holdout_cv_k_1_5_Refit.png")
    imB = cv2.resize(imB, (imA.shape[1], imA.shape[0]), interpolation=cv2.INTER_LINEAR)

    # imA = addBlackBorder(imA, 10)
    # imB = addBlackBorder(imB, 10)
    #imB = addText(imB, "b", (40, 40), fontFace, 112, color=(0, 0, 0))

    imB = addBorder(imB, "V", 0.05)
    imgU = np.vstack([imA, imB])

    cv2.imwrite("./paper/Figure_6.png", imgU)



def getAUCIntvsExt(df_org, fname = None, cohort = None, metric = "AUC", leg_anchor = 1.05):
    df = df_org.groupby(['Dataset', 'CV', 'CV-Repeats', 'Folds', 'Evaluation']).mean().reset_index()

    legend_order = ['flat_cv_k_Refit',\
            'flat_cv_k_Ensemble',\
            'holdout_cv_k_Refit',\
            'holdout_cv_k_Ensemble',\
            'nested_cv_Flat',\
            'nested_cv_Flat-Ensemble',\
            'nested_cv_Ensemble-Refit',\
            'nested_cv_Ensemble-Ensemble']

    summary_table = (
        df.groupby(['CV', 'Folds', 'CV-Repeats', 'Evaluation'])[[f'{metric}-Ext', f'{metric}-Int']]
        .mean()
        .reset_index()
    )
    summary_table['ID'] = summary_table[["CV", "Evaluation"]].astype(str).agg('_'.join, axis=1)
    summary_table['Outer_Folds'] = [int(str(z).split("_")[-1]) for z in summary_table["Folds"].values]
    summary_table['ID'] = summary_table['ID'].map(name_mapping)
    mapped_legend_order = [name_mapping.get(id, id) for id in legend_order]

    sns.set(style="whitegrid")
    if cohort == "radMLBench":
        plt.figure(figsize=(10, 8))
    else:
        plt.figure(figsize=(8, 5))
    ax = plt.gca()
    scatter = sns.scatterplot(
        data=summary_table,
        x=f'{metric}-Int',
        y=f'{metric}-Ext',
        hue='ID',
        hue_order=mapped_legend_order,
        palette=color_palette,
        style='Outer_Folds',
        markers={5: 'o', 10: '^'},
        size='CV-Repeats',
        sizes=(70, 244),
        alpha = 0.7,
    )

    if cohort == "radMLBench":
        plt.plot([0.615, 0.69], [0.615, 0.69], color='red', linestyle='--', linewidth=1.5, alpha = 0.7)
        plt.gca().margins(0)
        plt.gca().set_aspect('equal', adjustable='box')  # Ensure 1:1 aspect ratio
        plt.xlim(0.615, 0.77)
        plt.ylim(0.615, 0.69)
    else:
        plt.plot([0.872, 0.897], [0.872, 0.897], color='red', linestyle='--', linewidth=1.5, alpha = 0.7)
        plt.gca().margins(0)
        plt.gca().set_aspect('equal', adjustable='box')  # Ensure 1:1 aspect ratio
        plt.xlim(0.872, 0.897)
        plt.ylim(0.872, 0.897)

    plt.title(f' ', fontsize=14)
    plt.xlabel(f'{metric} [Validation estimate]', fontsize=12)
    plt.ylabel(f'{metric} [Test performance]', fontsize=12)

    plt.legend(title='', frameon=False, fontsize=10, title_fontsize=12, loc='upper left', bbox_to_anchor=(1, leg_anchor), markerscale=1.5)
    legend_labels = {
        'ID': '\nCV scheme',
        'CV-Repeats': '\nNumber of repeats',
        'Outer_Folds': '\nNumber of outer folds',
    }

    legend_lines = ax.get_legend().get_lines()
    # Print out the colors of each marker (line object)
    for line in legend_lines:
        color = line.get_color()  # Get the color of the line (marker)
        size = line.get_markersize()  # Get the size of the marker
        if size > 10 and size < 16:
            line.set_markersize(9)  # Set the new size for the marker
        if size > 21:
            line.set_markersize(16)  # Set the new size for the marker
        #print(f"Marker color: {color}, Marker size: {size}")

    for text in ax.get_legend().get_texts():
        label = text.get_text()
        if label in legend_labels:
            text.set_text(legend_labels[label])
            text.set_fontweight('bold')  # Make the matched key bold
    #
    #
    # min_val, max_val = summary_table[f'{metric}-Ext'].min(), summary_table[f'{metric}-Int'].max()
    # start = np.ceil(min_val / 0.025) * 0.025
    # end = np.floor(max_val / 0.025) * 0.025
    # ticks = np.arange(start, end + 0.025, 0.025)
    # if len(ticks) > 8:
    #     ticks = np.arange(start, end + 0.05, 0.05)
    # plt.xticks(ticks, [f"{round(tick, 3):.3f}".rstrip('0').rstrip('.') for tick in ticks])
    # plt.yticks(ticks, [f"{round(tick, 3):.3f}".rstrip('0').rstrip('.') for tick in ticks])

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./paper/{fname}", dpi=500, bbox_inches='tight')


col_lb = '#88aaff' # light blue
col_b = '#2266ff' #  blue
col_gold = '#d06000'  # Orange
col_lgold = '#ffbb55'    # Lighter Orange
col_lv = '#dd66dd'    # Lighter violet
col_v = '#992299'       # Violet
col_lg = '#44dd44'
col_g = '#118811'  # Green (nested_cv, Ensemble-Ensemble)

color_palette = {
    'flat_cv_k_Ensemble': col_b,
    'flat_cv_k_Refit': col_lb,

    'holdout_cv_k_Ensemble': col_gold,
    'holdout_cv_k_Refit': col_lgold,

    'nested_cv_Ensemble-Ensemble': col_g,
    'nested_cv_Ensemble-Refit': col_lg,
    'nested_cv_Flat': col_lv,
    'nested_cv_Flat-Ensemble': col_v,

    'Flat CV, ensemble': col_b,
    'Flat CV, refit': col_lb,

    'Holdout CV, ensemble': col_gold,
    'Holdout CV, refit': col_lgold,

    'Nested CV, full ensemble': col_g,
    'Nested CV, refit ensemble': col_v,
    'Nested CV, refit': col_lv,
    'Nested CV, simple ensemble': col_lg,
}



if __name__ == '__main__':
    os.makedirs("paper", exist_ok = True)
    for cohort in ["radMLBench", "UCI"]:
        print (f"\n\n\nProcessing cohort {cohort}")
        createDatasetTable(cohort)
        df_org = readResults(cohort)
        getSummaryTable(df_org, cohort = cohort, fname = f'Table_2_{cohort}', metric = "AUC")
        getOverfittingStats(df_org, fname = None, metric = "AUC")
        checkVariancevsSamplesize(df_org, cohort = cohort, metric = "AUC", DPI=300)
        getAUCIntvsExt(df_org, metric = "AUC", cohort = cohort, fname=f"Figure_5_{cohort}", leg_anchor = 1.085)
        getTimings(df_org, cohort = cohort)
        generateRelationPlots(df_org, cohort = cohort) # AUC only
        generatePlotsPerDataset(df_org, cohort = cohort) # AUC only
        plotVariance(df_org, cohort = cohort)

    joinRelationPlot()
    joinRepeatsPlot()



#
