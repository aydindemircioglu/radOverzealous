from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import wilcoxon, linregress
from statsmodels.stats.multitest import multipletests

import pickle
from glob import glob

from joblib import Parallel, delayed
from joblib import dump, load
import numpy as np
import pandas as pd
import seaborn as sns
import os
import cv2
import time

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from PIL import Image
from PIL import ImageDraw, ImageFont

import radMLBench
from loadDataUCI import *
from utils import *



name_mapping = {
    'flat_cv_k_1_5_Refit': 'Simple 5-fold CV, refit',
    'flat_cv_k_1_10_Refit': 'Simple 10-fold CV, refit',
    'flat_cv_k_5_5_Refit': 'Simple 5-fold CV, refit, 5x',
    'flat_cv_k_5_10_Refit': 'Simple 10-fold CV, refit, 5x',

    'flat_cv_k_1_5_Ensemble': 'Simple 5-fold CV, ensemble',
    'flat_cv_k_1_10_Ensemble': 'Simple 10-fold CV, ensemble',
    'flat_cv_k_5_5_Ensemble': 'Simple 5-fold CV, ensemble, 5x',
    'flat_cv_k_5_10_Ensemble': 'Simple 10-fold CV, ensemble, 5x',

    'holdout_cv_k_1_5_Refit': 'Holdout 5-fold CV, refit',
    'holdout_cv_k_1_10_Refit': 'Holdout 10-fold CV, refit',
    'holdout_cv_k_5_5_Refit': 'Holdout 5-fold CV, refit, 5x',
    'holdout_cv_k_5_10_Refit': 'Holdout 10-fold CV, refit, 5x',

    'holdout_cv_k_1_5_Ensemble': 'Holdout 5-fold CV, ensemble',
    'holdout_cv_k_1_10_Ensemble': 'Holdout 10-fold CV, ensemble',
    'holdout_cv_k_5_5_Ensemble': 'Holdout 5-fold CV, ensemble, 5x',
    'holdout_cv_k_5_10_Ensemble': 'Holdout 10-fold CV, ensemble, 5x',

    'nested_cv_1_5_10_Flat': 'Nested (5x10)-fold CV, refit',
    'nested_cv_1_10_5_Flat': 'Nested (10x5)-fold CV, refit',
    'nested_cv_5_5_10_Flat': 'Nested (5x10)-fold CV, refit, 5x',
    'nested_cv_5_10_5_Flat': 'Nested (10x5)-fold CV, refit, 5x',

    'nested_cv_1_5_10_Flat-Ensemble': 'Nested (5x10)-fold CV, refit ensemble',
    'nested_cv_1_10_5_Flat-Ensemble': 'Nested (10x5)-fold CV, refit ensemble',
    'nested_cv_5_5_10_Flat-Ensemble': 'Nested (5x10)-fold CV, refit ensemble, 5x',
    'nested_cv_5_10_5_Flat-Ensemble': 'Nested (10x5)-fold CV, refit ensemble, 5x',

    'nested_cv_1_5_10_Ensemble-Refit': 'Nested (5x10)-fold CV, simple ensemble',
    'nested_cv_1_10_5_Ensemble-Refit': 'Nested (10x5)-fold CV, simple ensemble',
    'nested_cv_5_5_10_Ensemble-Refit': 'Nested (5x10)-fold CV, simple ensemble, 5x',
    'nested_cv_5_10_5_Ensemble-Refit': 'Nested (10x5)-fold CV, simple ensemble, 5x',

    'nested_cv_1_5_10_Ensemble-Ensemble': 'Nested (5x10)-fold CV, full ensemble ',
    'nested_cv_1_10_5_Ensemble-Ensemble': 'Nested (10x5)-fold CV, full ensemble',
    'nested_cv_5_5_10_Ensemble-Ensemble': 'Nested (5x10)-fold CV, full ensemble, 5x',
    'nested_cv_5_10_5_Ensemble-Ensemble': 'Nested (10x5)-fold CV, full ensemble, 5x',



    'flat_cv_k_Refit': 'Simple CV, refit',
    'flat_cv_k_Ensemble': 'Simple CV, ensemble',
    'holdout_cv_k_Refit': 'Holdout CV, refit',
    'holdout_cv_k_Ensemble': 'Holdout CV, ensemble',
    'nested_cv_Flat': 'Nested CV, refit',
    'nested_cv_Flat-Ensemble': 'Nested CV, refit ensemble',
    'nested_cv_Ensemble-Ensemble': 'Nested CV, full ensemble',
    'nested_cv_Ensemble-Refit': 'Nested CV, simple ensemble',
}





def process_file(z, propTbl, metric):
    try:
         results = []
         if "results_" in z: # just in case
             return []

         df = load(z)

         base = {"Dataset": df["dataset"], "CV": df["type"]}
         dataset = df["dataset"]
         dataProp = propTbl.query("Dataset == @dataset")
         assert len(dataProp) == 1
         dataProp = dict(dataProp.iloc[0])
         base.update(dataProp)
         base["CV-Repeats"] = df["valrepeats"]
         base["Repeat"] = df["repeat"]
         if base["CV"] == "flat_cv_k" or base["CV"] == "holdout_cv_k":
             base["Folds"] = df["k"]
             base["Time"] = df["time"]

             row = base.copy()
             row["Evaluation"] = "Refit"
             metric_key = metric.upper()
             row[f"{metric_key}-Int"] = df["metrics_refit"][metric]
             row[f"{metric_key}-Ext"] = df["final_metrics_refit"][metric]
             row[f"{metric_key}-Overfitting"] = row[f"{metric_key}-Int"] - row[f"{metric_key}-Ext"]
             results.append(row)

             row = base.copy()
             row["Evaluation"] = "Ensemble"
             metric_key = metric.upper()
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

             metric_key = metric.upper()
             values = [r[metric] for r in df["metrics_refit"]]
             row[f"{metric_key}-Int"] = np.mean(values)
             row[f"{metric_key}-Ext"] = df["final_metrics_refit"][metric]
             row[f"{metric_key}-Overfitting"] = row[f"{metric_key}-Int"] - row[f"{metric_key}-Ext"]
             results.append(row)

             # here we use all internal CV models as ensemble
             row = base.copy()
             row["Evaluation"] = "Ensemble-Ensemble"
             metric_key = metric.upper()
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
             metric_key = metric.upper()
             values = [r[metric] for r in df["metrics_refit"]]
             row[f"{metric_key}-Int"] = np.mean(values)
             row[f"{metric_key}-Ext"] = df["final_metrics_flat"][metric]
             row[f"{metric_key}-Overfitting"] = row[f"{metric_key}-Int"] - row[f"{metric_key}-Ext"]
             results.append(row)

             # here we use the internal refit CV as model, but ensemble it
             row = base.copy()
             row["Evaluation"] = "Flat-Ensemble"
             metric_key = metric.upper()
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



def readResults(cohort, metric, propTbl):
    # propTbl is properties of each dataset, so we can directly have them along the results
    try:
        print ("Loading ", f"./paper/results_{cohort}_{metric}.dump")
        results = load(f"./paper/results_{cohort}_{metric}.dump")
        return pd.DataFrame(results).reset_index(drop=True)
    except:
        print ("FAILED. Recomputing..")
        pass

    datasets = getDatasetList (cohort)
    files = [f for f in glob(f"./results/*{metric}.dump") if any(os.path.basename(f).startswith(ds) for ds in datasets)]

    print ("###", metric)
    print ("Processing", len(files), "files")

    time.sleep(3)
    with Parallel(n_jobs=30, verbose = 10) as parallel:
        results = parallel(delayed(process_file)(z, propTbl, metric.lower()) for z in files)
    # results = [process_file(z, propTbl, metric) for z in files]
    results = [item for sublist in results for item in sublist]

    dump(results, f"./paper/results_{cohort}_{metric}.dump")
    return pd.DataFrame(results).reset_index(drop=True)



def groupData (df_org, metric = "AUC"):
    df = df_org.groupby(['Dataset', 'CV', 'CV-Repeats', 'Folds', 'Evaluation']).mean(numeric_only=True).reset_index()
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
            m["publication_doi"]
            tbl.append({"Dataset": dataset, "Modality": m["modality"], "Outcome": m["outcome"],
                "Instances": m['nInstances'],
                "Features": m["nFeatures"], "Dimensionality": m["Dimensionality"], "Balance": m["ClassBalance"],
                "Source": m["publication_doi"]})
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
    return tbl


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
    tmp = {} # gather for heatmap

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
    print (f"### {ID}")
    printSig ("Features", p_value)
    tmp["Features"] = p_value
    slopes, intercepts = bootstrap_regression(nFeatures_values, diff_values)
    y_pred_mean = slope * nFeatures_values + intercept
    y_pred_bootstrap = [s * nFeatures_values + b for s, b in zip(slopes, intercepts)]
    y_pred_lower = np.percentile(y_pred_bootstrap, 2.5, axis=0)
    y_pred_upper = np.percentile(y_pred_bootstrap, 97.5, axis=0)

    axes[0].scatter(nFeatures_values, diff_values, s=15, color='black')
    axes[0].plot(nFeatures_values, y_pred_mean, color='black')
    sorted_indices = np.argsort(nFeatures_values)
    axes[0].fill_between(nFeatures_values[sorted_indices], y_pred_lower[sorted_indices], y_pred_upper[sorted_indices], color='grey', alpha=0.3)
    axes[0].text(0.42, 0.97, f"$R^2 = {r_value**2:.2f}$\n(p = {p_value:.2g})", transform=axes[0].transAxes, fontsize=15, verticalalignment='top')
    axes[0].set_xlabel("Number of Features", fontsize=axis_fontsize)
    axes[0].set_ylabel(f"Overfitting (in {metric})", fontsize=axis_fontsize)
    axes[0].axhline(y=0.0, color='red', linewidth=2, zorder=2,  linestyle='--')

    slope, intercept, r_value, p_value, std_err = linregress(nInstances_values, diff_values)
    printSig ("Instances", p_value)
    tmp["Instances"] = p_value

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
    printSig ("Dimensionality", p_value)
    tmp["Dimensionality"] = p_value

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
    os.makedirs(f"./paper/relations_{metric}_{cohort}", exist_ok = True)
    plt.savefig(f"./paper/relations_{metric}_{cohort}/FigRelation_{ID}.png")
    plt.close()
    return tmp


def get_p_value_color(p_value):
    """Convert p-value to significance level (0-4 scale)"""
    if p_value < 0.001:
        return 4  # ***
    elif p_value < 0.01:
        return 3  # **
    elif p_value < 0.05:
        return 2  # *
    elif p_value < 0.10:
        return 1  # .
    else:
        return 0  # not significant



def generateRelationPlots (df_org, cohort = None, metric = "AUC", fname = None):
    df = groupData (df_org, metric = metric)

    diffs = df.copy()
    id_columns = ['CV', 'CV-Repeats', 'Folds', 'Evaluation']
    diffs['ID'] = diffs[id_columns].astype(str).agg('_'.join, axis=1)

    sTable = []
    for ID in diffs["ID"].unique():
        subdf = diffs.query("ID == @ID")
        tmp = testRelations(subdf, ID, cohort = cohort, metric = metric, DPI = 300)
        tmp["ID"] = ID
        tmp["metric"] = metric
        sTable.append(tmp)
    sTable = pd.DataFrame(sTable)
    print (sTable)
    sTable['Instances'] = sTable['Instances'].apply(get_p_value_color)
    sTable['Features'] = sTable['Features'].apply(get_p_value_color)
    sTable['Dimensionality'] = sTable['Dimensionality'].apply(get_p_value_color)

    sig_matrix = sTable
    sig_matrix["ID_Group"] = pd.Categorical(
        sig_matrix["ID"].str.split("_").str[0],
        categories=["flat", "holdout", "nested"],
        ordered=True
    )

    flat_data = sig_matrix[sig_matrix["ID_Group"] == "flat"]
    holdout_data = sig_matrix[sig_matrix["ID_Group"] == "holdout"]
    nested_data = sig_matrix[sig_matrix["ID_Group"] == "nested"]

    # fix order
    mapping_order = list(name_mapping.keys())[:32]

    flat_data = sig_matrix[sig_matrix["ID_Group"] == "flat"]
    flat_data = flat_data.set_index('ID').loc[[x for x in mapping_order if x in flat_data['ID'].values]].reset_index()

    holdout_data = sig_matrix[sig_matrix["ID_Group"] == "holdout"]
    holdout_data = holdout_data.set_index('ID').loc[[x for x in mapping_order if x in holdout_data['ID'].values]].reset_index()

    nested_data = sig_matrix[sig_matrix["ID_Group"] == "nested"]
    nested_data = nested_data.set_index('ID').loc[[x for x in mapping_order if x in nested_data['ID'].values]].reset_index()


    DPI = 300
    # fig, axes = plt.subplots(1, 3, figsize=(18, 8), dpi=DPI)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi = DPI)
    cmap = sns.color_palette("Reds", 5)
    metric_labels = ['Features', 'Instances', 'Dimensionality']
    for idx, (data, ax, title) in enumerate(zip(
        [flat_data, holdout_data, nested_data],
        axes,
        ['Simple CV', 'Holdout', 'Nested CV']
    )):
        print (data)
        # Prepare data matrix
        heatmap_data = data[metric_labels].values
        scheme_labels = [name_mapping[id_val] for id_val in data['ID'].values]

        # Create heatmap
        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=4,
            cbar=(idx == 2),  # Only show colorbar on rightmost plot
            # cbar_kws={'label': 'Significance',
            #           'ticks': [0, 1, 2, 3, 4],
            #           'format': plt.FuncFormatter(lambda x, p: ['n.s.', 'p<0.1', 'p<0.05', 'p<0.01', 'p<0.001'][int(x)])},
            cbar_kws={'label': 'Significance',
                      'ticks': [0.5, 1.5, 2.5, 3.5, 4.5],
                      'boundaries': [0, 1, 2, 3, 4, 5],
                      'format': plt.FuncFormatter(lambda x, p: ['n.s.', 'p<0.1', 'p<0.05', 'p<0.01', 'p<0.001'][int(x) if int(x) < 5 else 4])},
            linewidths=0.5,
            linecolor='white',
            square=False,
            xticklabels=metric_labels,
            yticklabels=scheme_labels
        )

        ax.set_title(title, fontsize=16, pad=10)
        ax.set_xlabel('')
        if idx == 0:
            ax.set_ylabel('Validation Scheme', fontsize=14)
        else:
            ax.set_ylabel('')

        # Rotate x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

        # Add grid
        ax.set_xticks(np.arange(len(metric_labels)) + 0.5, minor=False)
        ax.set_yticks(np.arange(len(scheme_labels)) + 0.5, minor=False)

    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
    plt.close()



def plotVariance (df_org, cohort = None, fname = None, metric = "AUC"):
    # want to see the variance per dataset
    df = df_org.copy()
    id_columns = ['CV', 'CV-Repeats', 'Folds', 'Evaluation']
    df['ID'] = df[id_columns].astype(str).agg('_'.join, axis=1)
    df["ID_Group"] = pd.Categorical(df["ID"].str.split("_").str[0], categories=["flat", "holdout", "nested"], ordered=True)

    df = df.sort_values(by=["Dataset", "ID_Group", "ID"])
    mapping_order = list(name_mapping.keys())[:32]

    # compute std deviations first
    df_std = df.groupby(['ID', 'Dataset'])[f"{metric}-Overfitting"].std().reset_index()
    df_std["ID_Group"] = pd.Categorical(df_std["ID"].str.split("_").str[0], categories=["flat", "holdout", "nested"], ordered=True)

    fig, ax = plt.subplots(figsize=(15, 10))


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
    if metric == "AUC":
        ax.set_ylim(0.0, 0.23)
    else:
        ax.set_ylim(0.0, 0.42)


    # Set title and labels
    ax.set_title("", fontsize=23, pad=20)
    ax.set_ylabel(f"Standard deviation of {metric}-Overfitting", fontsize=14)
    ax.set_xlabel(f"", fontsize=14)


    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def printSig (tgtstr, p_value):
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    elif p_value < 0.10:
        stars = "."
    else:
        stars = ""

    if stars:
        sigstr = "significant"
    else:
        sigstr = "NOT significant"
    print(f"{stars:<4} {tgtstr:<15} is {sigstr:<17}: {p_value:.3f}")



def checkVariancevsSamplesize(df_org, metric = "AUC", cohort = None, fname = None, DPI=300):
    print ("Variance vs SampleSize:")
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
    printSig ("Instances", p_value)

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
    axes[0].text(-0.17, -0.08, "", transform=axes[0].transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left')
    axes[0].axhline(y=0.0, color='red', linewidth=2, zorder=2, linestyle='--')


    # Plot 2: Overfitting vs Features
    slope, intercept, r_value, p_value, std_err = linregress(nFeatures_values, overfitting_values)
    printSig ("Features", p_value)

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
    axes[1].text(-0.17, -0.08, "", transform=axes[1].transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left')
    axes[1].axhline(y=0.0, color='red', linewidth=2, zorder=2,  linestyle='--')

    # Plot 2: Overfitting vs nDimensionality_values
    slope, intercept, r_value, p_value, std_err = linregress(nDimensionality_values, overfitting_values)
    printSig ("Dimensionality_values", p_value)
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
    axes[2].text(-0.17, -0.08, "", transform=axes[2].transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left')
    axes[2].axhline(y=0.0, color='red', linewidth=2, zorder=2, linestyle='--')

    # Save the plot
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()



def generatePlotsPerDataset (df_org, cohort = None, metric = "AUC"):
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
            x="ID", y=f"{metric}-Overfitting",
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
                x="ID", y=f"{metric}-Overfitting",
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
                ax.scatter(x_pos + x_jitter, id_data[f'{metric}-Overfitting'],
                          color='black', s=10, alpha=0.6, zorder=3)

            ax.set_xticks(positions)
            ax.set_xticklabels([name_mapping[u] for u in unique_ids], rotation=45, ha='right', fontsize=14)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.set_ylim(-0.4, 0.4)
            ax.set_title(dataset, fontsize=23, pad=20)
            ax.set_xlabel("", fontsize=14)
            ax.set_ylabel(f"{metric}-Overfitting", fontsize=14)
            ax.axhline(y=0.0, color='red', linewidth=2, zorder=2)

        plt.tight_layout()
        os.makedirs(f"paper/dataset_{metric}_{cohort}", exist_ok = True)
        plt.savefig(f"paper/dataset_{metric}_{cohort}/{dataset}.png", dpi=300, bbox_inches='tight')
        plt.close()



def getTimings(df_org, cohort = None):
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
    print (f"\n\nOverfitting stats for {metric}")
    df = groupData (df_org, metric)

    id_columns = ['CV', 'CV-Repeats', 'Folds', 'Evaluation']
    mean_table = df.groupby(id_columns)[[f'{metric}-Int', f'{metric}-Ext', f'{metric}-Overfitting', f'{metric}-Performance']].mean().round(3)
    D5 = mean_table.query("`CV-Repeats` == 5").reset_index()
    D1 = mean_table.query("`CV-Repeats` == 1 ").reset_index()
    #print ("Reduction of overfitting for repeats 1 -> 5", np.mean(D5[f"{metric}-Overfitting"]-D1[f"{metric}-Overfitting"]))

    # to see what really happens
    D5['ID'] = D5[id_columns].astype(str).agg('_'.join, axis=1)
    z = D5[f"{metric}-Overfitting"]-D1[f"{metric}-Overfitting"]
    z.index = D5.ID

    flat_subset = z[z.index.str.contains("flat_cv_k_5")].abs()
    min_v, max_v = flat_subset.min(), flat_subset.max()
    #print (z)
    print(f"Reduction in overfitting for {metric} repeat 1 -> 5: {min_v:.3f} - {max_v:.3f}")

    other_subset = z[~z.index.str.contains("flat_cv_k")].abs()
    min_o, max_o = other_subset.min(), other_subset.max()
    print(f"Reduction in overfitting for {metric} repeat 1 -> 5 (other schemes): {min_o:.3f} - {max_o:.3f}")

    D5 = mean_table.query("Folds == 5 or Folds == '5_10' ").reset_index()
    D10 = mean_table.query("Folds == 10 or Folds == '10_5' ").reset_index()
    #print ("Reduction of overfitting for ext. folds 5 -> 10", np.mean(D10[f"{metric}-Overfitting"]-D5[f"{metric}-Overfitting"]))

    D10['ID'] = D10[id_columns].astype(str).agg('_'.join, axis=1)
    z = D10[f"{metric}-Overfitting"]-D5[f"{metric}-Overfitting"]
    z.index = D10.ID
    # print (z)

    flat_subset = z[z.index.str.contains("flat_cv_k_5")].abs()
    min_v, max_v = flat_subset.min(), flat_subset.max()
    print(f"Reduction in overfitting for {metric} folds 5 -> 10: {min_v:.3f} - {max_v:.3f}")


    other_subset = z[~z.index.str.contains("flat_cv_k")].abs()
    min_o, max_o = other_subset.min(), other_subset.max()
    print(f"Reduction in overfitting for {metric} folds 5 -> 10 (other schemes): {min_o:.3f} - {max_o:.3f}")

    D5 = mean_table.query("Folds == 5 ").reset_index()
    D10 = mean_table.query("Folds == 10").reset_index()
#    print ("Reduction of overfitting for ext. folds 5 -> 10, no nested-CV", np.mean(D10[f"{metric}-Overfitting"]-D5[f"{metric}-Overfitting"]))




def getSummaryTable(df_org, cohort = None, fname = None, metric = "AUC"):
    df = groupData (df_org, metric)

    mean_table = df.groupby(['CV', 'Folds', 'CV-Repeats', 'Evaluation'])[[f'{metric}-Int', f'{metric}-Ext', f'{metric}-Overfitting', f'{metric}-Performance']].mean().round(3)
    std_table = df.groupby(['CV', 'Folds', 'CV-Repeats', 'Evaluation'])[[f'{metric}-Int', f'{metric}-Ext', f'{metric}-Overfitting', f'{metric}-Performance']].std().round(3)

    CV_mapping = {'flat_cv_k': 'Simple CV', 'holdout_cv_k': 'Holdout CV', "nested_cv": 'Nested CV'}
    Folds_mapping = {5: '5', 10: '10', '5_10': '5+10', '10_5': '10+5'}
    Model_mapping = {"Flat": "Refit", "Flat-Ensemble": "Refit ensemble",
        "Ensemble-Ensemble": "Full ensemble", "Ensemble-Refit": "Simple ensemble"}

    summary_table = mean_table.astype(str) + " ± " + std_table.astype(str)
    summary_table = summary_table.reset_index()
    summary_table["CV"] = summary_table["CV"].map(CV_mapping)
    summary_table["Folds"] = summary_table["Folds"].map(Folds_mapping)
    summary_table.loc[summary_table["CV"] == "Nested CV", "Evaluation"] = summary_table["Evaluation"].map(Model_mapping)

    eval_order = {
        "Simple CV": ["Refit", "Ensemble"],
        "Holdout CV": ["Refit", "Ensemble"],
        "Nested CV": ["Refit", "Refit ensemble", "Simple ensemble", "Full ensemble"]
    }

    summary_table["temp"] = summary_table["CV"].map({"Simple CV": 0, "Holdout CV": 10, "Nested CV": 20})
    summary_table["temp"] += summary_table["Folds"].map({'5': 0, '10': 1, '5+10': 2, '10+5': 3})
    summary_table["temp"] += summary_table["CV-Repeats"].map({1: 0, 5: 1}) * 0.1
    summary_table["temp"] += summary_table.apply(lambda row: eval_order.get(row["CV"], []).index(row["Evaluation"]) * 0.01, axis=1)

    summary_table = summary_table.sort_values(by=["temp"]).drop(columns=["temp"]).reset_index(drop=True)

    # post-add pvalues
    pvals = []
    df_map = df_org.copy() # stupid
    df_map['CV'] = df_map["CV"].map(CV_mapping)
    df_map['Folds'] = df_map["Folds"].map(Folds_mapping)
    df_map.loc[df_map["CV"] == "Nested CV", "Evaluation"] = df_map["Evaluation"].map(Model_mapping)

    for idx, row in summary_table.iterrows():
        group_data = df_map[
            (df_map['CV'] == row['CV']) &
            (df_map['Folds'] == row['Folds']) &
            (df_map['CV-Repeats'] == row['CV-Repeats']) &
            (df_map['Evaluation'] == row['Evaluation'])
        ][f'{metric}-Overfitting'].values
        if len(group_data) == 0:
            print ("### p-value computation failed.")
            exit(-1)
        stat, p_val = wilcoxon(group_data, alternative='two-sided')
        pvals.append(p_val)

    #summary_table[f'{metric}-Pvalue'] = pvals
    _, corrected_pvals, _, _ = multipletests(pvals, method='holm')
    summary_table[f'{metric}-Pvalue'] = corrected_pvals

    summary_table[f'{metric}-Pvalue'] = summary_table[f'{metric}-Pvalue'].apply(
        lambda p: '<0.001' if p < 0.001 else f'{p:.3f}'
    )

    summary_table.to_excel(f"./paper/{fname}.xlsx")
    return summary_table



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



def createFigure6():
    fontFace = "Arial"

    for cohort in ["radMLBench", "UCI"]:
        for metric in ["AUC", "MCC", "F1"]:
            imA = cv2.imread(f"./paper/relations_{metric}_{cohort}/FigRelation_flat_cv_k_1_5_Refit.png")

            imB = cv2.imread(f"./paper/Figure_S6_lower_{metric}_{cohort}.png")
            imB = cv2.resize(imB, (imA.shape[1], imA.shape[0]), interpolation=cv2.INTER_LINEAR)
            imB = addBorder(imB, "V", 0.05)

            imgU = np.vstack([imA, imB])
            if cohort == "radMLBench" and metric == "AUC":
                cv2.imwrite("./paper/Figure_6.png", imgU)
            cv2.imwrite(f"./paper/Figure_S6_{cohort}_{metric}.png", imgU)
    pass


def getAUCIntvsExt(df_org, fname = None, cohort = None, metric = "AUC", leg_anchor = 1.05):
    # 1. Calculate the mean for the points (as you did before)
    df_grouped = df_org.groupby(['Dataset', 'CV', 'CV-Repeats', 'Folds', 'Evaluation']).mean(numeric_only=True).reset_index()

    # 2. Calculate the Mean AND the Standard Error for the error bars
    # We group by the experimental setup to get the spread across datasets
    summary_stats = (
        df_grouped.groupby(['CV', 'Folds', 'CV-Repeats', 'Evaluation'])[[f'{metric}-Ext', f'{metric}-Int']]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )

    # Flatten multi-index columns
    summary_stats.columns = ['_'.join(col).strip('_') for col in summary_stats.columns.values]

    # Calculate Standard Error for the CIs
    summary_stats['x_err'] = summary_stats[f'{metric}-Int_std'] / np.sqrt(summary_stats[f'{metric}-Int_count'])
    summary_stats['y_err'] = summary_stats[f'{metric}-Ext_std'] / np.sqrt(summary_stats[f'{metric}-Ext_count'])

    # Prepare IDs for mapping and legend
    summary_stats['ID_raw'] = summary_stats[["CV", "Evaluation"]].astype(str).agg('_'.join, axis=1)
    summary_stats['ID'] = summary_stats['ID_raw'].map(name_mapping)
    summary_stats['Outer_Folds'] = [int(str(z).split("_")[-1]) for z in summary_stats["Folds"].values]

    legend_order = ['flat_cv_k_Refit', 'flat_cv_k_Ensemble', 'holdout_cv_k_Refit',
                    'holdout_cv_k_Ensemble', 'nested_cv_Flat', 'nested_cv_Flat-Ensemble',
                    'nested_cv_Ensemble-Refit', 'nested_cv_Ensemble-Ensemble']
    mapped_legend_order = [name_mapping.get(id, id) for id in legend_order]

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    scatter = sns.scatterplot(
        data=summary_stats,
        x=f'{metric}-Int_mean',
        y=f'{metric}-Ext_mean',
        hue='ID',
        hue_order=mapped_legend_order,
        palette=color_palette,
        style='Outer_Folds',
        markers={5: 'o', 10: '^'},
        size='CV-Repeats',
        sizes=(70, 244),
        alpha=0.9,
        ax=ax,
        zorder=2
    )

    if metric == "AUC":
        if cohort == "radMLBench":
            plt.plot([0.605, 0.695], [0.605, 0.695], color='red', linestyle='--', linewidth=1.5, alpha = 0.7, zorder=0)
            #plt.xlim(0.61, 0.77)
            plt.ylim(0.605, 0.695)
        else:
            plt.plot([0.872, 0.897], [0.872, 0.897], color='red', linestyle='--', linewidth=1.5, alpha = 0.7, zorder=0)
            plt.xlim(0.872, 0.897); plt.ylim(0.872, 0.897)
    elif metric == "F1":
        if cohort == "radMLBench":
            plt.plot([0.585, 0.675], [0.585, 0.675], color='red', linestyle='--', linewidth=1.5, alpha = 0.7, zorder=0)
            #plt.xlim(0.58, 0.68)
            plt.ylim(0.585, 0.675)
        else:
            plt.plot([0.765, 0.81], [0.765, 0.81], color='red', linestyle='--', linewidth=1.5, alpha = 0.7, zorder=0)
            plt.xlim(0.765, 0.81); plt.ylim(0.77, 0.8)
            # plt.plot([0.0, 1.0], [0.0, 1.0], color='red', linestyle='--', linewidth=1.5, alpha = 0.7, zorder=0)
            # plt.xlim(0.872, 0.897); plt.ylim(0.872, 0.897)
    elif metric == "MCC":
        pass
        if cohort == "radMLBench":
            plt.plot([0.13, 0.32], [0.13, 0.32], color='red', linestyle='--', linewidth=1.5, alpha = 0.7, zorder=0)
            # plt.xlim(0.615, 0.77);
            plt.ylim(0.13, 0.32)
        else:
            plt.plot([0.6, 0.7], [0.6, 0.7], color='red', linestyle='--', linewidth=1.5, alpha = 0.7, zorder=0)
            # plt.plot([0.872, 0.897], [0.872, 0.897], color='red', linestyle='--', linewidth=1.5, alpha = 0.7, zorder=0)
            plt.xlim(0.61, 0.69); plt.ylim(0.615, 0.665)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(f'{metric} [Validation estimate]')
    plt.ylabel(f'{metric} [Test performance]')
    plt.legend(title='', frameon=False, loc='upper left', bbox_to_anchor=(1, leg_anchor))

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./paper/{fname}", dpi=500, bbox_inches='tight')
    plt.close()


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

    'Simple CV, ensemble': col_b,
    'Simple CV, refit': col_lb,

    'Holdout CV, ensemble': col_gold,
    'Holdout CV, refit': col_lgold,

    'Nested CV, full ensemble': col_g,
    'Nested CV, refit ensemble': col_v,
    'Nested CV, refit': col_lv,
    'Nested CV, simple ensemble': col_lg,
}


def convertToTiff():
    for p in glob('./paper/Figure_?.png'):
        img = cv2.imread(p)
        cv2.imwrite(p.replace('.png', '.tiff'), img, [cv2.IMWRITE_TIFF_COMPRESSION, 5])


def createFigure5():
    fontFace = "Arial"

    imA = cv2.imread("./paper/Figure_S5_AUC_radMLBench.png")
    #imA = addText(imA, "a", (40, 40), fontFace, 112, color=(0, 0, 0))

    imB = cv2.imread("./paper/Figure_S5_F1_radMLBench.png")
    width_scale = imA.shape[1] / imB.shape[1]
    new_height = int(imB.shape[0] * width_scale)
    imB = cv2.resize(imB, (imA.shape[1], new_height), interpolation=cv2.INTER_LINEAR)

    imC = cv2.imread("./paper/Figure_S5_MCC_radMLBench.png")
    width_scale = imA.shape[1] / imC.shape[1]
    new_height = int(imC.shape[0] * width_scale)
    imC = cv2.resize(imC, (imA.shape[1], new_height), interpolation=cv2.INTER_LINEAR)

    # imA = addBlackBorder(imA, 10)
    # imB = addBlackBorder(imB, 10)
    #imB = addText(imB, "b", (40, 40), fontFace, 112, color=(0, 0, 0))

    imB = addBorder(imB, "V", 0.05)
    imC = addBorder(imC, "V", 0.05)
    imgU = np.vstack([imA, imB, imC])

    cv2.imwrite("./paper/Figure_5.png", imgU)


def createFigure7():
    fontFace = "Arial"

    imA = cv2.imread("./paper/Figure_S7_AUC_radMLBench.png")
    cv2.imwrite("./paper/Figure_7.png", imA)


def createFigure8():
    imA = cv2.imread("./paper/Figure_S8_AUC_radMLBench.png")
    #imA = addText(imA, "a", (40, 40), fontFace, 112, color=(0, 0, 0))

    imB = cv2.imread("./paper/Figure_S8_F1_radMLBench.png")
    width_scale = imA.shape[1] / imB.shape[1]
    new_height = int(imB.shape[0] * width_scale)
    imB = cv2.resize(imB, (imA.shape[1], new_height), interpolation=cv2.INTER_LINEAR)

    imC = cv2.imread("./paper/Figure_S8_MCC_radMLBench.png")
    width_scale = imA.shape[1] / imC.shape[1]
    new_height = int(imC.shape[0] * width_scale)
    imC = cv2.resize(imC, (imA.shape[1], new_height), interpolation=cv2.INTER_LINEAR)

    imB = addBorder(imB, "V", 0.05)
    imC = addBorder(imC, "V", 0.05)
    imgU = np.vstack([imA, imB, imC])

    cv2.imwrite("./paper/Figure_8.png", imgU)



if __name__ == '__main__':
    os.makedirs("paper", exist_ok = True)
    for cohort in ["radMLBench", "UCI"]:
        print (f"\n\n\nProcessing cohort {cohort}")
        propTbl = createDatasetTable(cohort)
        for metric in ["AUC", "MCC", "F1"]:
            df_org = readResults(cohort, metric, propTbl)
            print ("\n\n\n\n######", metric)
            fname = f'Table_S2_{metric}_{cohort}'
            # if metric == "AUC":
            #     fname = f'Table_2_{cohort}'
            summary_table = getSummaryTable(df_org, cohort = cohort, fname = fname, metric = metric)
            print (summary_table)
            largest_row = summary_table.sort_values(by=f'{metric}-Overfitting', key=lambda x: x.str.split(" ").str[0].astype(float), ascending=False).iloc[0:1]
            print (f"Overfitting for {metric} on {cohort}")
            print (largest_row)

            getOverfittingStats(df_org, fname = None, metric = metric)

            # Figure 6
            fname = f"./paper/Figure_S6_lower_{metric}_{cohort}.png"
            generateRelationPlots(df_org, cohort = cohort, metric = metric, fname = fname)

            fname = f"./paper/Figure_S8_{metric}_{cohort}.png"
            checkVariancevsSamplesize(df_org, cohort = cohort, fname = fname, metric = metric, DPI=300)

            fname = f"Figure_S5_{metric}_{cohort}.png"
            getAUCIntvsExt(df_org, metric = metric, cohort = cohort, fname = fname, leg_anchor = 1.085)

            generatePlotsPerDataset(df_org, cohort = cohort, metric = metric)

            fname = f"./paper/Figure_S7_{metric}_{cohort}.png"
            plotVariance(df_org, fname = fname, cohort = cohort, metric = metric)
        getTimings(df_org, cohort = cohort)

    createFigure5()
    createFigure6()
    createFigure7()
    createFigure8()
    convertToTiff()



#
