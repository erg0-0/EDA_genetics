import pandas as pd
import os
from utils.statistical_tests import matrix_fisher, matrix_chi2 

pd.options.mode.chained_assignment = None

def data_connection(input_path:str , output_path:str):
    """
    Concatenates data from  directories: BRAKI ZĘBOWE_wyniki, ROZSZCZEPY_wyniki, KONTROLE_wyniki.
    Outputs in a new file.

    Parameters:
    - input_path (str): The path to the first directory containing data. Example: '../../data/'
    - output_path (str): The path to the second directory where new data is stored data. example: '../../data/'

    Returns:
    New .csv file in the output location
    """
    directories = [os.path.join(input_path,'BRAKI ZĘBOWE_wyniki'), os.path.join(input_path, 'ROZSZCZEPY_wyniki'), os.path.join(input_path,'KONTROLE_wyniki')]
    data_frames = []
    for directory in directories:
        files = os.listdir(directory)
        for file in files:
            if file.endswith('.tabular'):
                file_path = os.path.join(directory, file)
                df = pd.read_csv(file_path, sep='\t')
                df['target'] = os.path.basename(directory)
                data_frames.append(df)
    df = pd.concat(data_frames, ignore_index=True)
    df['target'] = df['target'].apply(lambda x: x.replace('_wyniki', '').lower())
    encoding = {'braki zębowe': 1, 'rozszczepy': 2, 'kontrole': 0}
    df['target_num'] = df['target'].replace(encoding)
    df.to_csv(os.path.join(output_path, 'genetics.csv'), index=False)

    
def count_MHD(row, columns_to_consider):
    """ 
    Function for counting occurrences of 'M', 'H', and 'D' in specified columns, 
    ignoring columns with value 'Na'.
    """
    count = 0
    for col in columns_to_consider:
        value = row.get(col)
        if pd.notna(value) and value != 'Na':
            count += int(value in ['M', 'H', 'D'])
    return count

def is_homoheterozygot(row):
    """
    Determine if a row represents a homozygous or heterozygous genotype.

    Parameters:
    - row (pandas.Series): A row from a DataFrame containing genotype information.

    Returns:
    - int: 1 if the genotype is heterozygous ('heterozygota'), 0 otherwise.
    """
    if row['Zygosity'] == 'heterozygota':
        return 1
    else:
        return 0


def preprocess_data(dataframe):
    """
    Process the data in the dataframe by replacing dots with NaNs and replacing spaces with underscores in column names.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to process.

    Returns:
    pandas.DataFrame: The processed dataframe.
    """
    #general cleaning
    dataframe = (dataframe
                 .rename(columns=lambda x: x.replace(' ', '_'))
                 .replace(['.'], pd.NA)
                 )  
    #other predictions
    other_preds = ['SIFT_pred', 'LRT_pred', 'MutationAssessor_pred', 'FATHMM_pred', 'PROVEAN_pred', 'MetaSVM_pred']
    dataframe[other_preds] = dataframe[other_preds].replace('.', '0')
    dataframe['MHD_count'] = dataframe.apply(count_MHD, axis=1, columns_to_consider=other_preds)
    dataframe['MHD_min3'] = dataframe['MHD_count'] >= 3
    #frequency in population
    dataframe['1000gp_EUR_freq'] = pd.to_numeric(dataframe['1000gp_EUR_freq'].replace(['.', 'ND'], 0))
    dataframe['frequency1%'] = dataframe['1000gp_EUR_freq'] < 0.01
    #dataframe['frequency1%'] = dataframe.loc[dataframe['1000gp_EUR_freq'] < 0.01, '1000gp_EUR_freq'] powyżej poprawiony kod
    #flagging silent mutation
    dataframe['reference_amino_acid'], dataframe['mutated_amino_acid'] = zip(*dataframe['HGVS.p'].apply(extract_amino_acids))
    dataframe['is_silent_mutation'] = dataframe['reference_amino_acid'] == dataframe['mutated_amino_acid']
    #flagging pathogenic
    exclude = ['Wariant o nieznanej patogennosci', 'Wariant potencjalnie patogenny', 'Wariant patogenny']
    dataframe['is_malicious'] = dataframe['Patogennosc'].isin(exclude)
    #homozygots
    dataframe['is_homozygot'] = dataframe['Zygosity'].apply(lambda x: 1 if x == 'homozygota' else 0)
    dataframe['is_homoheterozygot'] = dataframe.apply(is_homoheterozygot, axis=1)
    #general cleaning
    dataframe = (dataframe.replace(['.', 'ND;;', 'ND'], pd.NA)  )
    return dataframe

def extract_amino_acids(hgvs_p):
    """
    Extract reference and variant amino acids from the given HGVS protein notation.

    Parameters:
    - hgvs_p (str or NaN): The HGVS protein notation.

    Returns:
    - tuple: A tuple containing the reference and variant amino acids.
             If `hgvs_p` is NaN, returns ('Silent', 'Silent').
             If `hgvs_p` does not contain a variant, returns (None, None).
             Otherwise, returns a tuple of the reference and variant amino acids.
    """
    if pd.isna(hgvs_p):
        return 'Silent', 'Silent' 
    else:
        parts = hgvs_p.split('.')
        if len(parts) > 1:
            ref_variant = parts[1]
            reference, mutation = ref_variant[0:3], ref_variant[-3:]
            return reference, mutation
        else:
            return None, None

def find_patho_genes_df(dataframe, MHD_min3=True, freq_threshold=0.01, silent_mutation=False, malicious=True):
    """
        Filter out noise from a DataFrame based on specified conditions.

        Args:
        dataframe (DataFrame): The DataFrame to filter.
        MHD_min3 (bool, optional): Whether to filter based on the 'MHD_min3' column. Defaults to True.
        freq_threshold (float or None, optional): The maximum value for the '1000gp_EUR_freq' column. Defaults to 0.01.
        silent_mutation (bool or None, optional): Whether to filter based on the 'is_silent_mutation' column. Defaults to False.
        malicious (bool or None, optional): Whether to filter based on the 'is_malicious' column. Defaults to True.

        Returns:
        DataFrame: The filtered DataFrame.
        """
    if MHD_min3:
        dataframe = dataframe[dataframe['MHD_min3'] == True]
    if freq_threshold is not None:
        dataframe = dataframe[dataframe['1000gp_EUR_freq'] <= freq_threshold]
    if silent_mutation is not None:
        dataframe = dataframe[dataframe['is_silent_mutation'] == silent_mutation]
    if malicious is not None:
        dataframe = dataframe[dataframe['is_malicious'] == malicious]
    return dataframe



def binarizee(all_filters_df, preprocessed_dataframe, illness="teeth"):
    """
    Process data based on selected columns, merge them, and create a pivot matrix, allowing to choose the type of illness.

    Args:
    all_filters_df (DataFrame): DataFrame containing all the data after filtering.
    preprocessed_data (DataFrame): DataFrame containing preprocessed data.
    illness (str, optional): Type of illness to consider. Default is "teeth", you can choose "cleft" also.

    Returns:
    DataFrame: Pivot matrix with processed data.
    """
    patients_per_class = preprocessed_dataframe.groupby('target')['sample_ID'].nunique()
    total_control = patients_per_class.get('kontrole',0)
    total_cleft = patients_per_class.get('rozszczepy',0)
    total_teeth = patients_per_class.get('braki zębowe',0)

    all_filters_df['is_pathogenic'] = 1
    selected_columns = ['sample_ID', 'Gene_Name', 'Variant_Id', 'target', 'is_pathogenic']
    bin_matrix = all_filters_df[selected_columns]
    bin_matrix = bin_matrix.groupby(['sample_ID', 'Gene_Name', 'target'], as_index=False)['is_pathogenic'].max()
    bin_matrix['target_eng'] = bin_matrix['target'].replace({'braki zębowe': 'teeth', 'rozszczepy': 'cleft', 'kontrole': 'control'})
    bin_matrix['target_long'] = bin_matrix['target_eng'] + "_with_pathogenic"
    count_df = bin_matrix.groupby(['Gene_Name', 'target_long']).agg({'sample_ID': 'nunique'}).reset_index()
    pivot_df = count_df.pivot_table(index='Gene_Name', columns='target_long', values='sample_ID', aggfunc='first').fillna(0)
    pivot_df["cleft_without_pathogenic"] =  total_cleft - pivot_df['cleft_with_pathogenic']
    pivot_df["teeth_without_pathogenic"] =  total_teeth - pivot_df['teeth_with_pathogenic']
    pivot_df["control_without_pathogenic"] = total_control - pivot_df['control_with_pathogenic']
    if illness == "teeth":
        pivot_df = pivot_df.drop(columns=['cleft_with_pathogenic', 'cleft_without_pathogenic'], errors='ignore')
    elif illness == "cleft":
        pivot_df = pivot_df.drop(columns=['teeth_with_pathogenic', 'teeth_without_pathogenic'], errors='ignore')
    return pivot_df  


def selected_genes(all_filters_df, df, illness=False):
    """
    Selects pathogenic genes for illness based on  chi2 and fisher tests.

    Args:
    - all_filters_df (pd.DataFrame): DataFrame containing all filters.
    - df (pd.DataFrame): DataFrame containing raw data.
    - illness (str): Specifies the illness to consider. Default is 'teeth'.

    Returns:
    - pd.DataFrame: DataFrame containing selected genes without duplicates.
    """

    if illness == 'teeth':
        target_name = 'braki zębowe'
    elif illness == 'cleft':
        target_name = 'rozszczepy'
    else:
        raise ValueError("Invalid illness specified. Please choose either 'teeth' or 'cleft'.")

    binmatrix = binarizee(all_filters_df, df, illness=illness)
    above_0_matrix = binmatrix[(binmatrix['control_with_pathogenic'] > 0) & (binmatrix[f'{illness}_with_pathogenic'] > 0)]
    matrix_chi2_df = matrix_chi2(above_0_matrix, target_name=target_name)
    matrix_fisher_df = matrix_fisher(binmatrix, target_name=target_name, method='hommel')
    filtered_chi2 = matrix_chi2_df.loc[matrix_chi2_df[f'{illness}_chi2_p-value_0.05'] < 0.05, ['Gene_Name']]
    filtered_fisher = matrix_fisher_df.loc[matrix_fisher_df[f'{illness}_fisher_p-value_0.05'] < 0.05, ['Gene_Name']]
    selected_genes = pd.concat([filtered_chi2, filtered_fisher])
    selected_genes.drop_duplicates(inplace=True)

    return selected_genes
