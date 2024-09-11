import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_heatmap(all_filters_df, df, selected_genes_teeth, selected_genes_cleft, save_dir, illness=False):
    """
    Generates a heatmap based on the given illness.

    Args:
    - all_filters_df (pd.DataFrame): DataFrame containing all filters.
    - df (pd.DataFrame): DataFrame containing some data.
    - selected_genes_teeth (pd.DataFrame): return from selected_genes with 'teeth' argument.
    - selected_genes_cleft (pd.DataFrame): return from selected_genes with 'cleft' argument
    - illness (str): Specifies the illness to consider. Type 'teeth', 'cleft' or teeth&cleft'.
    - save_dir (str) : Directory for saving output.

    Returns:
    - heatmap saved in save_dir
    """

    combined_selected_genes = pd.concat([selected_genes_teeth, selected_genes_cleft])
    combined_selected_genes.drop_duplicates(inplace=True)

    if illness == 'teeth':
        selected_genes = selected_genes_teeth
    elif illness == 'cleft':
        selected_genes = selected_genes_cleft
    elif illness == 'teeth&cleft':
        selected_genes = combined_selected_genes
    else:
        raise ValueError("Invalid illness specified. Please choose either 'teeth', 'cleft', or 'teeth&cleft'.")
  
    all_filters_df['is_pathogenic'] = 1
    selected_columns = ['sample_ID', 'Gene_Name', 'Variant_Id', 'target', 'is_pathogenic']
    all_filters_df_for_bin = all_filters_df[selected_columns]
    df2 = df[selected_columns[:-1]]
    merged_data = pd.concat([all_filters_df_for_bin, df2])
    bin_matrix = merged_data.drop_duplicates(subset=['Gene_Name', 'Variant_Id', 'sample_ID', 'target', 'is_pathogenic'], keep='first')
    bin_matrix['is_pathogenic'].fillna(0, inplace=True)
    bin_matrix['is_pathogenic2'] = bin_matrix['is_pathogenic'].replace({1: '_with_pathogenic', 0: '_without_pathogenic'})
    bin_matrix['target_eng'] = bin_matrix['target'].replace({'braki zębowe': 'teeth', 'rozszczepy': 'cleft', 'kontrole': 'control'})
    bin_matrix['target_long'] = bin_matrix['target_eng'] + bin_matrix['is_pathogenic2']
    bin_matrix = bin_matrix.groupby(['sample_ID', 'Gene_Name', 'target_long'], as_index=False)['is_pathogenic'].max()
    bin_matrix_sorted = bin_matrix.merge(selected_genes, on='Gene_Name').sort_values(by='Gene_Name')
    bin_matrix_sorted['Combined_Column'] = bin_matrix_sorted['target_long'] + '_' + bin_matrix_sorted['sample_ID'].astype(str)

  
    if illness == 'teeth':
        bin_matrix_sorted_filtered = bin_matrix_sorted[~bin_matrix_sorted['target_long'].isin(['cleft_with_pathogenic', 'cleft_without_pathogenic'])]
    elif illness == 'cleft':
        bin_matrix_sorted_filtered = bin_matrix_sorted[~bin_matrix_sorted['target_long'].isin(['teeth_with_pathogenic', 'teeth_without_pathogenic'])]
    elif illness == 'teeth&cleft':
        bin_matrix_sorted_filtered = bin_matrix_sorted
    else:
        raise ValueError("Invalid illness specified. Please choose either 'teeth', 'cleft', or 'teeth&cleft'.")


    pivot_table = bin_matrix_sorted_filtered.pivot_table(index='Combined_Column', columns='Gene_Name', values='is_pathogenic', fill_value=0)
    plt.figure(figsize=(8, 40))
    sns.heatmap(pivot_table, cmap='coolwarm', cbar=True)
    plt.title(f'Heatmap for {illness} and control')
    plt.xlabel('Gene_Name')
    plt.ylabel('Combined_Column')
    plt.savefig(os.path.join(save_dir,  f'heatmap_{illness}.png'))
    plt.close()