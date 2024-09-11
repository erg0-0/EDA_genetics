import xgboost as xgb
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score
from imblearn.under_sampling import NearMiss
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def XGBoost_model(model_df, save_dir):
    
    """
    Perform XGBoost classification using the provided dataframe.

    Parameters:
    - model_df (DataFrame): The input dataframe containing the dataset.
    - save_dir (str): Directory for saving plots.
    Returns:
    None

    This function performs XGBoost classification on the given dataset.
    It preprocesses the data, performs cross-validation, trains the model,
    evaluates its performance, and generates various plots including ROC curve,
    learning curve, and feature importance plot. The plots are saved in the
    current directory.
    """
    ohe = pd.get_dummies(model_df, columns=['Gene_Name'])
    ohe['target'] = ohe['target_num'].map(lambda x: 1 if x == 2 else x)
    ohe.drop('target_num', axis=1, inplace=True)
    X = ohe.drop(['target'], axis=1)
    y = ohe[['target']]
    undersampler = NearMiss()
    X_resampled, y_resampled = undersampler.fit_resample(X, y)
    y_resampled = y_resampled.values.ravel()

    xgb_model = xgb.XGBClassifier(learning_rate=0.01, n_estimators=50)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

    cv_precision = cross_val_score(xgb_model, X_train, y_train, cv=3, scoring='precision')
    cv_recall = cross_val_score(xgb_model, X_train, y_train, cv=3, scoring='recall')
    cv_f1 = cross_val_score(xgb_model, X_train, y_train, cv=3, scoring='f1')

    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Cross-validation Results:")
    print("Precision:", cv_precision)
    print("Recall:", cv_recall)
    print("F1-score:", cv_f1)
    print("Mean Precision:", np.mean(cv_precision))
    print("Mean Recall:", np.mean(cv_recall))
    print("Mean F1-score:", np.mean(cv_f1))

    fpr, tpr, _ = roc_curve(y_test, xgb_model.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve_cleft.png'))
    plt.close()

    train_sizes, train_scores, test_scores = learning_curve(xgb_model, X_train, y_train, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.savefig(os.path.join(save_dir, 'learning_curve_cleft.png'))
    plt.close()

    plt.figure(figsize=(10,6))
    xgb.plot_importance(xgb_model, importance_type='gain', max_num_features=10)
    plt.title('XGBoost Feature Importance')
    plt.savefig(os.path.join(save_dir, 'feature_importance_cleft.png'))
    plt.close()
