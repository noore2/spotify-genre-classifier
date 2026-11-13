#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    ConfusionMatrixDisplay
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12

import random
random.seed(187904)
np.random.seed(187904)

df = pd.read_csv("musicData.csv")
print(f"Raw shape: {df.shape}")

df.dropna(how='all', inplace=True)
df = df[df['music_genre'].notna()].reset_index(drop=True)
print(f"After dropping empty rows: {df.shape}")

df.drop(columns=['instance_id', 'artist_name', 'track_name'], inplace=True)

df['duration_ms'] = df['duration_ms'].replace(-1.0, np.nan)

df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')

le_key = LabelEncoder()
df['key'] = df['key'].astype(str)
df['key'] = le_key.fit_transform(df['key'])

df['mode'] = df['mode'].map({'Major': 1, 'Minor': 0})

month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

def parse_date(d):
    try:
        parts = str(d).split('-')
        return month_map.get(parts[1], np.nan)
    except:
        return np.nan

df['obtained_month'] = df['obtained_date'].apply(parse_date)
df.drop(columns=['obtained_date'], inplace=True)

le_genre = LabelEncoder()
df['music_genre'] = le_genre.fit_transform(df['music_genre'])
genre_names = le_genre.classes_
print(f"Genres: {genre_names}")

feature_cols = [c for c in df.columns if c != 'music_genre']
X_full = df[feature_cols].copy()
y_full = df['music_genre'].values

print(f"\nMissing values before imputation:\n{X_full.isnull().sum()[X_full.isnull().sum() > 0]}")

# stratified train/test split: 500 per genre test, 4500 per genre train
X_train_list, X_test_list = [], []
y_train_list, y_test_list = [], []

for genre_id in range(len(genre_names)):
    idx = np.where(y_full == genre_id)[0]
    np.random.shuffle(idx)
    test_idx = idx[:500]
    train_idx = idx[500:]
    X_train_list.append(X_full.iloc[train_idx])
    X_test_list.append(X_full.iloc[test_idx])
    y_train_list.append(y_full[train_idx])
    y_test_list.append(y_full[test_idx])

X_train = pd.concat(X_train_list).reset_index(drop=True)
X_test = pd.concat(X_test_list).reset_index(drop=True)
y_train = np.concatenate(y_train_list)
y_test = np.concatenate(y_test_list)

print(f"\nTrain shape: {X_train.shape} | Test shape: {X_test.shape}")
print(f"Train genre counts: {pd.Series(y_train).value_counts().sort_index().values}")
print(f"Test  genre counts: {pd.Series(y_test).value_counts().sort_index().values}")

# impute missing with median fitted on train only
medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

print(f"\nMissing after imputation - Train: {X_train.isnull().sum().sum()} | Test: {X_test.isnull().sum().sum()}")

# standardize fitted on train only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA 2D visualization
pca = PCA(n_components=2, random_state=18790439)
X_train_pca = pca.fit_transform(X_train_scaled)

genre_colors = [
    'steelblue', 'tomato', 'forestgreen', 'goldenrod', 'mediumpurple',
    'darkorange', 'deeppink', 'teal', 'saddlebrown', 'gray'
]

fig, ax = plt.subplots(figsize=(10, 7))
for i, genre in enumerate(genre_names):
    mask = y_train == i
    ax.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1],
               c=genre_colors[i], label=genre,
               alpha=0.4, s=10, edgecolors='none')
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("PCA 2D: Music Genres in Reduced Feature Space")
ax.legend(title="Genre", fontsize=9, markerscale=3)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("pca_genre_clusters.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPCA variance explained: PC1={pca.explained_variance_ratio_[0]*100:.2f}%, "
      f"PC2={pca.explained_variance_ratio_[1]*100:.2f}%, "
      f"Total={pca.explained_variance_ratio_.sum()*100:.2f}%")

# validation split from training set for early stopping only
rng = np.random.default_rng(18790439)
shuffle_idx = rng.permutation(len(X_train_scaled))
X_train_shuffled = X_train_scaled[shuffle_idx]
y_train_shuffled = y_train[shuffle_idx]

val_size = int(0.05 * len(X_train_scaled))
X_val = X_train_shuffled[:val_size]
y_val = y_train_shuffled[:val_size]
X_tr = X_train_shuffled[val_size:]
y_tr = y_train_shuffled[val_size:]

print(f"Classes in y_val: {np.unique(y_val)}")
print(f"Train: {X_tr.shape[0]} rows | Val: {X_val.shape[0]} rows")

# XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    early_stopping_rounds=20,
    tree_method='hist',
    verbosity=0,
    random_state=18790439,
    n_jobs=-1
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"\nXGBoost trained {model.best_iteration + 1} trees (early stopping)")

# predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# AUC
auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
print(f"\nFinal AUC (macro, OvR): {auc:.4f}")

# classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=genre_names))

# ROC curves
fig, ax = plt.subplots(figsize=(10, 7))
for i, genre in enumerate(genre_names):
    y_binary = (y_test == i).astype(int)
    fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
    genre_auc = roc_auc_score(y_binary, y_prob[:, i])
    ax.plot(fpr, tpr, linewidth=2,
            label=f'{genre} (AUC={genre_auc:.3f})',
            color=genre_colors[i])

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'ROC Curves — XGBoost Multiclass (One-vs-Rest)\nMacro AUC = {auc:.4f}')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches='tight')
plt.show()

# confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=genre_names,
    cmap='Blues',
    colorbar=False,
    ax=ax
)
ax.set_title(f'Confusion Matrix — XGBoost\nAUC = {auc:.4f}')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

# feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
importances.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
ax.set_xlabel('Feature Importance (XGBoost gain)')
ax.set_title('Feature Importances — XGBoost')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\nTop 5 most important features:")
print(importances.tail(5).iloc[::-1])
print(f"\nFinal AUC: {auc:.4f}")


