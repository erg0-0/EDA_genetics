import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, learning_curve, cross_validate
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

from utils.data_preparation import find_patho_genes_df, selected_genes, preprocess_data

def prepare_data_for_model_teeth(df):
    """
    Preprocesses the input DataFrame for model training.

    Args:
        df (DataFrame): Input DataFrame containing raw data.

    Returns:
        DataFrame: Preprocessed DataFrame ready for model training.
    """

    df = preprocess_data(df)
    columns_to_drop = [
        "sample_ID", 'transcript_ID', 'Variant_Id', 'HGVS.c', 'HGVS.p', 'Zygosity',
        'Filter', 'Patogennosc', 'ClinVar_Alt_Info', 'target', 'Disease', 'SIFT_pred',
        'LRT_pred', 'MutationAssessor_pred', 'FATHMM_pred', 'PROVEAN_pred',
        'MetaSVM_pred', 'is_homoheterozygot', "frequency1%"
    ]

    all_filters_df = find_patho_genes_df(df, MHD_min3=True, freq_threshold=0.05, silent_mutation=False, malicious=True)
    selected_genes_teeth = selected_genes(all_filters_df, df, illness="teeth")
    all_filters_df['selected_genes_teeth'] = (all_filters_df['target_num'] == 1) & (all_filters_df['Gene_Name'].isin(selected_genes_teeth))

    all_filters_df.drop(columns=columns_to_drop, inplace=True)

    all_filters_df.replace({True: 1, False: 0}, inplace=True)

    all_filters_df['Chrom'].replace('X', 0, inplace=True)
    all_filters_df['Chrom'] = all_filters_df['Chrom'].astype(int)

    object_columns = all_filters_df.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()

    for col in object_columns:
        all_filters_df[f"{col}_encoded"] = label_encoder.fit_transform(all_filters_df[col])

    all_filters_df.drop(columns=object_columns, inplace=True)
    all_filters_df = all_filters_df[all_filters_df['target_num'].isin([0, 1])]

    return all_filters_df


def train_random_forest(X_train, y_train, random_seed=42):
    """
    Trains a Random Forest classifier.

    Args:
        X_train (array-like): Feature matrix for training.
        y_train (array-like): Target labels for training.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        RandomForestClassifier: Trained Random Forest classifier.
    """
    rf_classifier = RandomForestClassifier(n_estimators=1000, max_depth=6, min_samples_split=2, min_samples_leaf=2, max_features='sqrt', random_state=random_seed)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

def evaluate_model(classifier, X_test, y_test):
    """
    Evaluates the performance of the classifier on the test set.

    Args:
        classifier (object): Trained classifier model.
        X_test (array-like): Feature matrix for testing.
        y_test (array-like): Target labels for testing.
    """
    y_pred = classifier.predict(X_test)
    print("Classification Report for Test Set:")
    print(classification_report(y_test, y_pred))

def perform_cross_validation(classifier, X_resampled, y_resampled):
    """
    Performs cross-validation and prints the results.

    Args:
        classifier (object): Classifier model.
        X_resampled (array-like): Feature matrix for training after resampling.
        y_resampled (array-like): Target labels for training after resampling.
    """
    cv_results = cross_validate(classifier, X_resampled, y_resampled, cv=3, scoring=['precision', 'recall', 'f1'])
    print("\nCross-validation Results:")
    print("Precision:", cv_results['test_precision'])
    print("Recall:", cv_results['test_recall'])
    print("F1-score:", cv_results['test_f1'])
    print("Mean Precision:", cv_results['test_precision'].mean())
    print("Mean Recall:", cv_results['test_recall'].mean())
    print("Mean F1-score:", cv_results['test_f1'].mean())

def plot_learning_curve(classifier, X_train, y_train, save_dir):
    """
    Plots the learning curve of the classifier.

    Args:
        classifier (object): Trained classifier model.
        X_train (array-like): Feature matrix for training.
        y_train (array-like): Target labels for training.
        save_dir (str): Directory to save the plot.
    """
    train_sizes, train_scores, test_scores = learning_curve(classifier, X_train, y_train, train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0], scoring='f1', random_state=42)
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Train F1 Score', color='blue')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.3)
    plt.plot(train_sizes, test_mean, label='Test F1 Score', color='red')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='red', alpha=0.3)
    plt.title('Learning Curve - Hypodontia')
    plt.xlabel('Training Size')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'learning_curve_hypodontia.png'))
    plt.close()

def plot_feature_importance(rf_classifier, X, save_dir):
    """
    Plots the feature importances of the Random Forest classifier.

    Args:
        rf_classifier (object): Trained Random Forest classifier.
        X (DataFrame): DataFrame containing features.
        save_dir (str): Directory to save the plot.
    """
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': rf_classifier.feature_importances_})
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(20, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances.head(10))
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig(os.path.join(save_dir, 'feature_importance_hypodontia.png'))
    plt.close()

def plot_roc_curve(rf_classifier, X_test, y_test, save_dir):
    """
    Plots the ROC curve of the Random Forest classifier.

    Args:
        rf_classifier (object): Trained Random Forest classifier.
        X_test (array-like): Feature matrix for testing.
        y_test (array-like): Target labels for testing.
        save_dir (str): Directory to save the plot.
    """ 
    y_prob = rf_classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Hypodontia ')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve_hypodontia.png'))
    plt.close()

def train_test_random_forest_with_undersampling(teeth_base, save_dir, random_seed=42):
    """
    Trains a Random Forest classifier with undersampling and evaluates its performance.

    Args:
        teeth_base (DataFrame): DataFrame containing the dataset.
        save_dir (str): Directory to save the visualizations.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    X = teeth_base.drop(columns=['target_num'])
    y = teeth_base['target_num']
    
    under_sampler = RandomUnderSampler(random_state=random_seed)
    X_resampled, y_resampled = under_sampler.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=random_seed, stratify=y_resampled)
    
    rf_classifier = train_random_forest(X_train, y_train, random_seed)
    
    evaluate_model(rf_classifier, X_test, y_test)
    
    perform_cross_validation(rf_classifier, X_resampled, y_resampled)
    
    plot_learning_curve(rf_classifier, X_train, y_train, save_dir)
    
    plot_feature_importance(rf_classifier, X, save_dir)
    
    plot_roc_curve(rf_classifier, X_test, y_test, save_dir)