import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency
from statsmodels.stats.multitest import multipletests

def perform_fischer_test(matrix, sick_without_pathogenic, sick_with_pathogenic, control_without_pathogenic, control_with_pathogenic):
    """
    Perform Fisher's exact test for each gene based on the provided patient groups.

    Parameters:
    - matrix (DataFrame): matrix prepared for fischer or chi2 tests.
    - sick_without_pathogenic (str): Column name representing sick patients without pathogenic mutations.
    - sick_with_pathogenic (str): Column name representing sick patients with pathogenic mutations.
    - control_without_pathogenic (str): Column name representing control patients without pathogenic mutations.
    - control_with_pathogenic (str): Column name representing control patients with pathogenic mutations.

    Returns:
    - results (dict): A dictionary containing the results of Fisher's exact test for each gene.
        Each key is a gene name (index from chi2_fisher_matrix), and each value is a dictionary with the following keys:
        - 'Contingency Table': Pandas DataFrame representing the contingency table.
        - 'Odds Ratio': The odds ratio calculated from the contingency table.
        - 'P-value': The p-value calculated from Fisher's exact test.
    """
    
    results = {}
    for gene_name in matrix.index:
        gene_data = matrix.loc[gene_name, [sick_without_pathogenic, sick_with_pathogenic, control_without_pathogenic, control_with_pathogenic]]
        sick_without_pathogenic_count = gene_data[sick_without_pathogenic]
        sick_with_pathogenic_count = gene_data[sick_with_pathogenic]
        control_without_pathogenic_count = gene_data[control_without_pathogenic]
        control_with_pathogenic_count = gene_data[control_with_pathogenic]
        contingency_table = pd.DataFrame({
            'Sick_patients': [sick_without_pathogenic_count, sick_with_pathogenic_count],
            'Control_patients': [control_without_pathogenic_count, control_with_pathogenic_count]
        }, index=['Pathogenic_absent', 'Pathogenic_present'])
        
        odds_ratio, p_value = fisher_exact(contingency_table)
    
        results[gene_name] = {
            'Contingency Table': contingency_table,
            'Odds Ratio': odds_ratio,
            'P-value': p_value
        }
    return results

def matrix_fisher (matrix,target_name='braki zębowe', method = "hommel" ):
    """
    Perform Fisher's exact test for cleft and teeth patients and merge the results with the input matrix.

    Parameters:
    - matrix (DataFrame): The input matrix containing gene information in the rows and prepared column names.
    Expected columns are: 'cleft_without_pathogenic', 'cleft_with_pathogenic', 'control_without_pathogenic', 
    'control_with_pathogenic', 'teeth_without_pathogenic', 'teeth_with_pathogenic'. 
    - target_name (str): The target condition for Fisher's exact test. 
        'braki zębowe' for teeth patients and 'rozszczepy' for cleft patients. Default is 'braki zębowe'.
    - method (str): The method used for correcting multiple testing. 
        Options are: 'bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 
        'fdr_tsbh', 'fdr_tsbky'. Default is 'hommel'.

    Returns:
    - matrix (DataFrame): The input matrix with additional columns containing Fisher's exact test results.
        Two additional columns are added:
        - If target_name is 'braki zębowe':
            - 'teeth_fisher_p-value_0.05': The p-values from Fisher's exact test for teeth patients, 
              with values below 0.05 considered significant.
            - 'corrected_p_value': Corrected p-values based on the specified method.
            - 'is_significant_teeth_fisher_corrected': True if the corrected p-value is less than 0.05, False otherwise.
        - If target_name is 'rozszczepy':
            - 'cleft_fisher_p-value_0.05': The p-values from Fisher's exact test for cleft patients, 
              with values below 0.05 considered significant.
            - 'corrected__cleft_p_value': Corrected p-values based on the specified method.
            - 'is_significant_cleft_fisher_corrected': True if the corrected p-value is less than 0.05, False otherwise.
    """
    if target_name=='braki zębowe':  
        teeth_fisher = perform_fischer_test(matrix,'teeth_without_pathogenic', 'teeth_with_pathogenic', 'control_without_pathogenic', 'control_with_pathogenic')
        teeth_fisher_df = pd.DataFrame(teeth_fisher).T.reset_index()[['index', 'P-value']]
        teeth_fisher_df.columns = ['Gene_Name', 'teeth_fisher_p-value_0.05']
        teeth_fisher_df['teeth_fisher_p-value_0.05'] = pd.to_numeric(teeth_fisher_df['teeth_fisher_p-value_0.05'], errors='coerce')
        matrix = pd.merge(matrix, teeth_fisher_df, on='Gene_Name', how='left')
        p_values = matrix["teeth_fisher_p-value_0.05"].to_list()
        rejected, corrected_p_values, _, _ = multipletests(p_values, method=method)
        matrix['corrected_p_value'] = corrected_p_values
        matrix['is_significant_teeth_fisher_corrected'] = rejected
    
    elif target_name=='rozszczepy':
        cleft_fisher = perform_fischer_test(matrix,'cleft_without_pathogenic', 'cleft_with_pathogenic', 'control_without_pathogenic', 'control_with_pathogenic')
        cleft_fisher_df = pd.DataFrame(cleft_fisher).T.reset_index()[['index', 'P-value']]
        cleft_fisher_df.columns = ['Gene_Name', 'cleft_fisher_p-value_0.05']
        cleft_fisher_df['cleft_fisher_p-value_0.05'] = pd.to_numeric(cleft_fisher_df['cleft_fisher_p-value_0.05'], errors='coerce')
        matrix = pd.merge(matrix, cleft_fisher_df, on='Gene_Name', how='left')
        p_values = matrix["cleft_fisher_p-value_0.05"].to_list()
        rejected, corrected_p_values, _, _ = multipletests(p_values, method=method)
        matrix['corrected__cleft_p_value'] = corrected_p_values
        matrix['is_significant_cleft_fisher_corrected'] = rejected
     
    return matrix

def perform_chi2_test(matrix, sick_without_pathogenic, sick_with_pathogenic, control_without_pathogenic, control_with_pathogenic):
    """
    Perform Chi-square test for each gene based on the provided patient groups.

    Parameters:
    - matrix (str): matrix prepared for fischer or chi2 tests.
    - sick_without_pathogenic (str): Column name representing sick patients without pathogenic mutations.
    - sick_with_pathogenic (str): Column name representing sick patients with pathogenic mutations.
    - control_without_pathogenic (str): Column name representing control patients without pathogenic mutations.
    - control_with_pathogenic (str): Column name representing control patients with pathogenic mutations.

    Returns:
    - results (dict): A dictionary containing the results of Chi-square test for each gene.
        Each key is a gene name (index from chi2_fisher_matrix), and each value is a dictionary with the following keys:
        - 'Contingency Table': Pandas DataFrame representing the contingency table.
        - 'Chi-square statistic': The chi-square statistic calculated from the contingency table.
        - 'P-value': The p-value calculated from Chi-square test.
    """
    results = {}
    for gene_name in matrix.index:
        gene_data = matrix.loc[gene_name, [sick_without_pathogenic, sick_with_pathogenic, control_without_pathogenic, control_with_pathogenic]]
        contingency_table = pd.DataFrame({
            'Sick_patients': [gene_data[sick_without_pathogenic], gene_data[sick_with_pathogenic]],
            'Control_patients': [gene_data[control_without_pathogenic], gene_data[control_with_pathogenic]]
        }, index=['Pathogenic_absent', 'Pathogenic_present'])
        
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
    
        results[gene_name] = {
            'Contingency Table': contingency_table,
            'Chi-square statistic': chi2_stat,
            'P-value': p_value
        }
    return results

def matrix_chi2(matrix, target_name='braki zębowe', method="hommel"):
    """
    Perform Chi-square test for cleft and teeth patients and merge the results with the input matrix.

    Parameters:
    - matrix (DataFrame): The input matrix containing gene information in the rows and prepared column names.
    Expected columns are: 'cleft_without_pathogenic', 'cleft_with_pathogenic', 'control_without_pathogenic', 
    'control_with_pathogenic', 'teeth_without_pathogenic', 'teeth_with_pathogenic'. 
    - target_name (str): The target condition for Chi-square test. 
        'braki zębowe' for teeth patients and 'rozszczepy' for cleft patients. Default is 'braki zębowe'.
    - method (str): The method used for correcting multiple testing. 
        Options are: 'bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 
        'fdr_tsbh', 'fdr_tsbky'. Default is 'hommel'.

    Returns:
    - matrix (DataFrame): The input matrix with additional columns containing Chi-square test results.
        Two additional columns are added:
        - If target_name is 'braki zębowe':
            - 'teeth_chi2_p-value_0.05': The p-values from Chi-square test for teeth patients, 
              with values below 0.05 considered significant.
            - 'corrected_p_value': Corrected p-values based on the specified method.
            - 'is_significant_teeth_chi2_corrected': True if the corrected p-value is less than 0.05, False otherwise.
        - If target_name is 'rozszczepy':
            - 'cleft_chi2_p-value_0.05': The p-values from Chi-square test for cleft patients, 
              with values below 0.05 considered significant.
            - 'corrected_cleft_p_value': Corrected p-values based on the specified method.
            - 'is_significant_cleft_chi2_corrected': True if the corrected p-value is less than 0.05, False otherwise.
    """
    if target_name == 'braki zębowe':  
        teeth_chi2 = perform_chi2_test(matrix, 'teeth_without_pathogenic', 'teeth_with_pathogenic', 'control_without_pathogenic', 'control_with_pathogenic')
        teeth_chi2_df = pd.DataFrame(teeth_chi2).T.reset_index()[['index', 'P-value']]
        teeth_chi2_df.columns = ['Gene_Name', 'teeth_chi2_p-value_0.05']
        teeth_chi2_df['teeth_chi2_p-value_0.05'] = pd.to_numeric(teeth_chi2_df['teeth_chi2_p-value_0.05'], errors='coerce')
        matrix = pd.merge(matrix, teeth_chi2_df, on='Gene_Name', how='left')
        p_values = matrix["teeth_chi2_p-value_0.05"].to_list()
        rejected, corrected_p_values, _, _ = multipletests(p_values, method=method)
        matrix['corrected_p_value'] = corrected_p_values
        matrix['is_significant_teeth_chi2_corrected'] = rejected
    
    elif target_name == 'rozszczepy':
        cleft_chi2 = perform_chi2_test(matrix, 'cleft_without_pathogenic', 'cleft_with_pathogenic', 'control_without_pathogenic', 'control_with_pathogenic')
        cleft_chi2_df = pd.DataFrame(cleft_chi2).T.reset_index()[['index', 'P-value']]
        cleft_chi2_df.columns = ['Gene_Name', 'cleft_chi2_p-value_0.05']
        cleft_chi2_df['cleft_chi2_p-value_0.05'] = pd.to_numeric(cleft_chi2_df['cleft_chi2_p-value_0.05'], errors='coerce')
        matrix = pd.merge(matrix, cleft_chi2_df, on='Gene_Name', how='left')
        p_values = matrix["cleft_chi2_p-value_0.05"].to_list()
        rejected, corrected_p_values, _, _ = multipletests(p_values, method=method)
        matrix['corrected_cleft_p_value'] = corrected_p_values
        matrix['is_significant_cleft_chi2_corrected'] = rejected
     
    return matrix

def create_model_matrix(all_filters_df, df, selected_genes_teeth, selected_genes_cleft, illness=False):
    """

    Args:
    - all_filters_df (pd.DataFrame): DataFrame containing all filters.
    - df (pd.DataFrame): DataFrame containing some data.
    - selected_genes_teeth (pd.DataFrame): return from selected_genes with 'teeth' argument.
    - selected_genes_cleft (pd.DataFrame): return from selected_genes with 'cleft' argument
    - illness (str): Specifies the illness to consider. Type 'teeth', 'cleft' or teeth&cleft'.

    Return:
    - Matrix for Models
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
    selected_columns = ['sample_ID', 'Gene_Name', 'target_num', 'is_pathogenic']
    all_filters_df_for_bin = all_filters_df[selected_columns]
    df2 = df[selected_columns[:-1]]
    merged_data = pd.concat([all_filters_df_for_bin, df2])
    bin_matrix = merged_data.drop_duplicates(subset=['Gene_Name', 'sample_ID', 'target_num', 'is_pathogenic'], keep='first')
    bin_matrix['is_pathogenic'].fillna(0, inplace=True)

    #bin_matrix = bin_matrix.groupby(['sample_ID', 'Gene_Name'], as_index=False)['is_pathogenic'].max()
    bin_matrix_sorted = bin_matrix.merge(selected_genes, on='Gene_Name').sort_values(by='Gene_Name')
    
    if illness == 'teeth':
        bin_matrix_sorted_filtered = bin_matrix_sorted[~bin_matrix_sorted['target_num'].isin([2])]
    elif illness == 'cleft':
        bin_matrix_sorted_filtered = bin_matrix_sorted[~bin_matrix_sorted['target_num'].isin([1])]
    elif illness == 'teeth&cleft':
        bin_matrix_sorted_filtered = bin_matrix_sorted
    else:
        raise ValueError("Invalid illness specified. Please choose either 'teeth', 'cleft', or 'teeth&cleft'.")
   
    bin_matrix_sorted_filtered.drop('sample_ID', axis=1, inplace=True)

    return bin_matrix_sorted_filtered