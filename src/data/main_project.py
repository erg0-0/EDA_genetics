import os
import sys
import pandas as pd


while os.path.basename(os.getcwd()) != "group_8":
    os.chdir("..")
project_dir = os.getcwd()
sys.path.append(os.path.join(project_dir, "src"))

from utils.data_preparation import (
    preprocess_data,
    data_connection,
    find_patho_genes_df,
    binarizee,
    selected_genes,
)
from utils.statistical_tests import create_model_matrix
from utils.visualisation import generate_heatmap
from models.model_teeth import (
    prepare_data_for_model_teeth,
    train_test_random_forest_with_undersampling,
)
from models.model_cleft import XGBoost_model

input_path = os.path.abspath(os.path.join(project_dir, "..", "local_files"))
output_path = input_path
data_connection(input_path, output_path)
save_dir = os.path.join(project_dir, "src", "visualisation")

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

df = pd.read_csv(
    os.path.abspath(os.path.join(project_dir, "..", "local_files", "genetics.csv"))
)
df = preprocess_data(df)
all_filters_df = find_patho_genes_df(
    df, MHD_min3=True, freq_threshold=0.05, silent_mutation=False, malicious=True
)
selected_genes_teeth = selected_genes(all_filters_df, df, illness="teeth")
selected_genes_cleft = selected_genes(all_filters_df, df, illness="cleft")
generate_heatmap(
    all_filters_df,
    df,
    selected_genes_teeth,
    selected_genes_cleft,
    save_dir=save_dir,
    illness="teeth",
)
generate_heatmap(
    all_filters_df,
    df,
    selected_genes_teeth,
    selected_genes_cleft,
    save_dir=save_dir,
    illness="cleft",
)
model_matrix_teeth = create_model_matrix(
    all_filters_df, df, selected_genes_teeth, selected_genes_cleft, illness="teeth"
)
model_matrix_cleft = create_model_matrix(
    all_filters_df, df, selected_genes_teeth, selected_genes_cleft, illness="cleft"
)
binmatrix_teeth = binarizee(all_filters_df, df, illness="teeth")
binmatrix_cleft = binarizee(all_filters_df, df, illness="cleft")
teeth_base = prepare_data_for_model_teeth(df)
print('Results for teeth subgroup:')
print()
train_test_random_forest_with_undersampling(teeth_base, save_dir, random_seed=42)
print()
print()
print('Results for cleft subgroup:')
print()
XGBoost_model(model_matrix_cleft, save_dir)