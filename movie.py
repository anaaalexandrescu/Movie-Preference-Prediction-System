import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

DATA_PATH = 'ml-100k/' 

ratings = pd.read_csv(DATA_PATH + 'u.data', sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp'], encoding='latin-1')

movies = pd.read_csv(DATA_PATH + 'u.item', sep='|',
    names=['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
           'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
           'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
           'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
    encoding='latin-1')

users = pd.read_csv(DATA_PATH + 'u.user', sep='|',
    names=['user_id', 'age', 'gender', 'occupation', 'zip_code'], encoding='latin-1')

print(f"ratings: {len(ratings):,} | movies: {len(movies):,} | users: {len(users):,}")

print(f"ratings - missing: {ratings.isnull().sum().sum()} values")
print(f"movies - missing: {movies.isnull().sum().sum()} values")
print(f"users - missing: {users.isnull().sum().sum()} values")

print(f"ratings - duplicates: {ratings.duplicated().sum()}")
print(f"movies - duplicates: {movies.duplicated(subset=['item_id']).sum()}")
print(f"users - duplicates: {users.duplicated(subset=['user_id']).sum()}")

invalid_ratings = ratings[(ratings['rating'] < 1) | (ratings['rating'] > 5)]
print(f"invalid ratings: {len(invalid_ratings)}")

print(ratings['rating'].value_counts().sort_index())
print(f"\nrating stats - Mean: {ratings['rating'].mean():.2f}, Std: {ratings['rating'].std():.2f}")
print(f"Min: {ratings['rating'].min()}, Max: {ratings['rating'].max()}")

user_activity = ratings.groupby('user_id').size()
Q1 = user_activity.quantile(0.25)
Q3 = user_activity.quantile(0.75)
IQR = Q3 - Q1
outliers_users = ((user_activity < Q1 - 1.5*IQR) | (user_activity > Q3 + 1.5*IQR)).sum()
print(f"Users with outlier activity: {outliers_users} ({outliers_users/len(user_activity)*100:.1f}%)")
print(f"User activity - Q1: {Q1:.0f}, Median: {user_activity.median():.0f}, Q3: {Q3:.0f}")

movie_popularity = ratings.groupby('item_id').size()
Q1_m = movie_popularity.quantile(0.25)
Q3_m = movie_popularity.quantile(0.75)
IQR_m = Q3_m - Q1_m
outliers_movies = ((movie_popularity < Q1_m - 1.5*IQR_m) | (movie_popularity > Q3_m + 1.5*IQR_m)).sum()
print(f"Movies with outlier popularity: {outliers_movies} ({outliers_movies/len(movie_popularity)*100:.1f}%)")
print(f"Movie popularity - Q1: {Q1_m:.0f}, Median: {movie_popularity.median():.0f}, Q3: {Q3_m:.0f}")

print(f"Age - Mean: {users['age'].mean():.1f}, Std: {users['age'].std():.1f}")
print(f"Age - Min: {users['age'].min()}, Max: {users['age'].max()}")
print(f"Age - Q1: {users['age'].quantile(0.25):.0f}, Median: {users['age'].median():.0f}, Q3: {users['age'].quantile(0.75):.0f}")

n_users = ratings['user_id'].nunique()
n_movies = ratings['item_id'].nunique()
sparsity = 1 - (len(ratings) / (n_users * n_movies))
print(f"\n8. matrix sparsity: {sparsity*100:.2f}%")

fig = plt.figure(figsize=(16, 10))

ax1 = plt.subplot(2, 3, 1)
rating_counts = ratings['rating'].value_counts().sort_index()
ax1.bar(rating_counts.index, rating_counts.values, color='steelblue', edgecolor='black')
ax1.set_xlabel('Rating')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of Ratings', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

ax2 = plt.subplot(2, 3, 2)
ax2.hist(user_activity, bins=50, color='coral', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Nr of Ratings')
ax2.set_ylabel('Nr of Users')
ax2.set_title('User Activity Distribution', fontweight='bold')
ax2.axvline(user_activity.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'mean: {user_activity.mean():.0f}')
ax2.legend()

ax3 = plt.subplot(2, 3, 3)
ax3.hist(movie_popularity, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Nr of Ratings')
ax3.set_ylabel('Nr of Movies')
ax3.set_title('Movie Popularity Distribution', fontweight='bold')
ax3.axvline(movie_popularity.mean(), color='red', linestyle='--', linewidth=2,
            label=f'mean: {movie_popularity.mean():.0f}')
ax3.legend()

ax4 = plt.subplot(2, 3, 4)
ax4.hist(users['age'], bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
ax4.set_xlabel('Age')
ax4.set_ylabel('Nr of Users')
ax4.set_title('User Age Distribution', fontweight='bold')
ax4.axvline(users['age'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'mean: {users['age'].mean():.1f}')
ax4.legend()

ax5 = plt.subplot(2, 3, 5)
genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
genre_counts = movies[genre_cols].sum().sort_values(ascending=False)
ax5.barh(range(len(genre_counts)), genre_counts.values, color='teal', edgecolor='black')
ax5.set_yticks(range(len(genre_counts)))
ax5.set_yticklabels(genre_counts.index)
ax5.set_xlabel('Nr of Movies')
ax5.set_title('Movies per Genre', fontweight='bold')
ax5.invert_yaxis()

ax6 = plt.subplot(2, 3, 6)
gender_counts = users['gender'].value_counts()
ax6.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
        colors=['skyblue', 'pink'], startangle=90)
ax6.set_title('Gender Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('01_eda_plots.png', dpi=300, bbox_inches='tight')
plt.show()

data_full = ratings.merge(movies, on='item_id').merge(users, on='user_id')
print(f"Merged dataset: {data_full.shape}")

data_full['liked'] = (data_full['rating'] >= 4).astype(int)
liked_counts = data_full['liked'].value_counts()
print(f"\nTarget - Liked: {liked_counts[1]:,} ({liked_counts[1]/len(data_full)*100:.1f}%) | Not Liked: {liked_counts[0]:,} ({liked_counts[0]/len(data_full)*100:.1f}%)")

print("\nCalculating user features...")
user_stats = ratings.groupby('user_id').agg({'rating': ['mean', 'std', 'count']}).reset_index()
user_stats.columns = ['user_id', 'user_avg_rating', 'user_rating_std', 'user_num_ratings']
user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)
print(f"User features created: {user_stats.shape}")

print("Calculating movie features...")
movie_stats = ratings.groupby('item_id').agg({'rating': ['mean', 'std', 'count']}).reset_index()
movie_stats.columns = ['item_id', 'movie_avg_rating', 'movie_rating_std', 'movie_num_ratings']
movie_stats['movie_rating_std'] = movie_stats['movie_rating_std'].fillna(0)
print(f"Movie features created: {movie_stats.shape}")

print("Calculating genre preferences...")
ratings_with_movies = ratings.merge(movies, on='item_id')
genre_prefs = []

for user_id in user_stats['user_id']:
    user_ratings = ratings_with_movies[ratings_with_movies['user_id'] == user_id]
    total_ratings = len(user_ratings)
    user_genre_pref = {'user_id': user_id}
    for genre in genre_cols:
        genre_count = user_ratings[user_ratings[genre] == 1].shape[0]
        user_genre_pref[f'pct_{genre}'] = (genre_count / total_ratings * 100) if total_ratings > 0 else 0
    genre_prefs.append(user_genre_pref)

genre_prefs_df = pd.DataFrame(genre_prefs)
print(f"Genre preferences created: {genre_prefs_df.shape}")

users_encoded = users.copy()
users_encoded['gender_encoded'] = users_encoded['gender'].map({'M': 1, 'F': 0})

classification_data = data_full[['user_id', 'item_id', 'liked']].copy()
classification_data = classification_data.merge(user_stats, on='user_id')
classification_data = classification_data.merge(movie_stats, on='item_id')
classification_data = classification_data.merge(users_encoded[['user_id', 'age', 'gender_encoded']], on='user_id')
classification_data = classification_data.merge(movies[['item_id'] + genre_cols], on='item_id')
classification_data = classification_data.merge(genre_prefs_df, on='user_id')

for genre in genre_cols:
    classification_data[f'match_{genre}'] = classification_data[f'pct_{genre}'] * classification_data[genre]

match_cols = [f'match_{genre}' for genre in genre_cols]
classification_data['genre_match_score'] = classification_data[match_cols].sum(axis=1)
classification_data['rating_diff'] = abs(classification_data['user_avg_rating'] - classification_data['movie_avg_rating'])

feature_cols = [
    'user_avg_rating', 'user_rating_std', 'user_num_ratings',
    'movie_avg_rating', 'movie_rating_std', 'movie_num_ratings',
    'age', 'gender_encoded', 'genre_match_score', 'rating_diff'
] + genre_cols

X = classification_data[feature_cols].values
y = classification_data['liked'].values

print("\n1. Checking for NaN/Inf values:")
nan_count = np.isnan(X).sum()
inf_count = np.isinf(X).sum()
print(f"NaN values: {nan_count}")
print(f"Inf values: {inf_count}")

if nan_count > 0 or inf_count > 0:
    print("Fixing NaN/Inf values...")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print("Fixed!")

print(f"\n2. Final feature matrix: {X.shape}")
print(f"Number of features: {len(feature_cols)}")

print("\n4. Class balance check:")
class_0 = (y == 0).sum()
class_1 = (y == 1).sum()
imbalance_ratio = class_1 / class_0
print(f"Class 0 (Not Liked): {class_0:,} ({class_0/len(y)*100:.1f}%)")
print(f"Class 1 (Liked): {class_1:,} ({class_1/len(y)*100:.1f}%)")

print("\n5. Train-test split (stratified):")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")

print("\n6. Feature scaling (StandardScaler):")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaling applied (mean=0, std=1)")

print("\n[Random Forest]")
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=20, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"Accuracy: {acc_rf:.4f} | Precision: {prec_rf:.4f} | Recall: {rec_rf:.4f} | F1: {f1_rf:.4f}")
cv_scores_rf = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='f1', n_jobs=-1)
print(f"CV F1: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

feature_importance_rf = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n[XGBoost]")
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    use_label_encoder=False, eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

acc_xgb = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb)
rec_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

print(f"Accuracy: {acc_xgb:.4f} | Precision: {prec_xgb:.4f} | Recall: {rec_xgb:.4f} | F1: {f1_xgb:.4f}")
cv_scores_xgb = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='f1', n_jobs=-1)
print(f"CV F1: {cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std():.4f})")

feature_importance_xgb = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'Accuracy': [acc_rf, acc_xgb],
    'Precision': [prec_rf, prec_xgb],
    'Recall': [rec_rf, rec_xgb],
    'F1-Score': [f1_rf, f1_xgb]
})

print("\n" + comparison.to_string(index=False))

best_model_idx = comparison['F1-Score'].idxmax()
best_model_name = comparison.loc[best_model_idx, 'Model']
print(f"\nBest Model: {best_model_name} (F1: {comparison['F1-Score'].max():.4f})")

fig, ax = plt.subplots(figsize=(14, 6))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics_to_plot))
width = 0.35

bars1 = ax.bar(x - width/2, comparison.loc[0, metrics_to_plot], width, label='Random Forest', color='steelblue')
bars2 = ax.bar(x + width/2, comparison.loc[1, metrics_to_plot], width, label='XGBoost', color='coral')

ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot)
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('02_model_comparison.png', dpi=300)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Not Liked', 'Liked'])
disp_rf.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('Random Forest', fontweight='bold')

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=['Not Liked', 'Liked'])
disp_xgb.plot(ax=axes[1], cmap='Oranges', values_format='d')
axes[1].set_title('XGBoost', fontweight='bold')

plt.tight_layout()
plt.savefig('03_confusion_matrices.png', dpi=300)
plt.show()

# ROC Curves and Precision-Recall Curves sections REMOVED

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

top_n = 15
top_features_rf = feature_importance_rf.head(top_n)
axes[0].barh(range(top_n), top_features_rf['importance'], color='steelblue', edgecolor='black')
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(top_features_rf['feature'])
axes[0].set_title('Random Forest - Top 15 Features', fontweight='bold')
axes[0].set_xlabel('Importance')
axes[0].invert_yaxis()

top_features_xgb = feature_importance_xgb.head(top_n)
axes[1].barh(range(top_n), top_features_xgb['importance'], color='coral', edgecolor='black')
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(top_features_xgb['feature'])
axes[1].set_title('XGBoost - Top 15 Features', fontweight='bold')
axes[1].set_xlabel('Importance')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('04_feature_importance.png', dpi=300)
plt.show()


print("\nRandom Forest:")
print(classification_report(y_test, y_pred_rf, target_names=['Not Liked', 'Liked'], digits=4))

print("\nXGBoost:")
print(classification_report(y_test, y_pred_xgb, target_names=['Not Liked', 'Liked'], digits=4))

print(f"\nBest Model: {best_model_name} | F1: {comparison['F1-Score'].max():.4f}")