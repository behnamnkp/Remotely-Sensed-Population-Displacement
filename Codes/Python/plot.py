import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from ast import literal_eval
import geopandas as gp
import seaborn as sns
import statsmodels.formula.api as smf
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from matplotlib.pyplot import figure
from splot.esda import lisa_cluster
import json
import pickle


with open('config.json') as config_file:
    config = json.load(config_file)

MAIN = config['file_paths']['MAIN']
INPUT = config['file_paths']['INPUT']
OUTPUT = config['file_paths']['OUTPUT']
DOCS = config['file_paths']['DOCS']
FIGS = config['file_paths']['FIGS']
CODES = config['file_paths']['CODES']
EMPTY_SUFFIXES = config['flags']['EMPTY_SUFFIXES']
START_YEAR = config['flags']['START_YEAR']
END_YEAR = config['flags']['END_YEAR']
LANDUSE = config['flags']['LANDUSE']
NIGHTLIGHT = config['flags']['NIGHTLIGHT']
INTERVAL = config['flags']['INTERVAL']
STATISTIC = config['flags']['STATISTIC']
NIGHTLIGHT_ANGLE_CORRECTION = config['flags']['NIGHTLIGHT_ANGLE_CORRECTION']
TARGET_LAYER = config['flags']['TARGET_LAYER']

def plot_clusters(moran_loc, residuals, year, name):

    lisa_cluster(moran_loc, residuals, p=0.05, figsize=(9, 9))
    plt.title(f'Cluster Map of {name} Residuals {year}', size=20)
    plt.show()
    plt.savefig(FIGS + f'{name} Residuals{year}.png', dpi=500, bbox_inches='tight')

def compare_models(cluster, ntl_level, year, name):

    # Load the saved regression models
    with open(OUTPUT + f'ols_{name}{year}.pickle', 'rb') as file:
        model = pickle.load(file)
    with open(OUTPUT + f'spatial_ols_{name}{year}.pickle', 'rb') as file:
        spatial_model = pickle.load(file)

    # Generate predictions using the models
    model_pred = (
        cluster[f'area_bg{year}'] * model.params[3] +
        cluster[f'area_hr{year}'] * model.params[1] +
        cluster[f'area_nr{year}'] * model.params[2] +
        cluster[f'area_lr{year}'] * model.params[4]
    )
    model_pred = pd.DataFrame(model_pred, columns=['pred' + year])
    predictions = pd.concat((ntl_level, model_pred), axis=1)

    spatial_model_pred = (
        cluster[f'area_bg{year}'] * spatial_model.params[3] +
        cluster[f'area_hr{year}'] * spatial_model.params[1] +
        cluster[f'area_nr{year}'] * spatial_model.params[2] +
        cluster[f'clusters{year}_HH'] * spatial_model.params[4] +
        cluster[f'clusters{year}_HL'] * spatial_model.params[5] +
        cluster[f'clusters{year}_NS'] * spatial_model.params[6] +
        cluster[f'area_lr{year}'] * spatial_model.params[7]
    )
    spatial_model_pred = pd.DataFrame(spatial_model_pred, columns=['ntlpred' + year])
    predictions_spatial = pd.concat((ntl_level, spatial_model_pred), axis=1)

    # Compare mode predictions
    sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
    f, axes = plt.subplots(2, 1)
    f.subplots_adjust(hspace=.5)
    sns.scatterplot(x=predictions['ntlpred' + year], y=predictions['CNTL' + year], ax=axes[0], color='black')
    axes[0].set(xlabel=f'{name} surface reflectance {year} (model)')
    axes[0].set(ylabel=f'{name} surface reflectance {year}')
    sns.scatterplot(x=predictions_spatial['ntlpred' + year], y=predictions['CNTL' + year], ax=axes[1], color='black')
    axes[1].set(xlabel=f'{name} surface reflectance {year} (spatial model)')
    axes[1].set(ylabel=f'{name} surface reflectance {year}')