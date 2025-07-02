"""
Advanced Feature Engineering for Personality Prediction
Playground Series S5E7 - Enhanced Feature Creation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.knn_imputer = KNNImputer(n_neighbors=5)
        self.feature_selector = SelectKBest(f_classif)
        
    def create_statistical_features(self, df):
        """統計的特徴量を作成"""
        df_new = df.copy()
        
        # 数値特徴量の統計量
        numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                       'Friends_circle_size', 'Post_frequency']
        
        # 行ごとの統計量
        df_new['numeric_mean'] = df[numeric_cols].mean(axis=1, skipna=True)
        df_new['numeric_std'] = df[numeric_cols].std(axis=1, skipna=True)
        df_new['numeric_median'] = df[numeric_cols].median(axis=1, skipna=True)
        df_new['numeric_min'] = df[numeric_cols].min(axis=1, skipna=True)
        df_new['numeric_max'] = df[numeric_cols].max(axis=1, skipna=True)
        df_new['numeric_range'] = df_new['numeric_max'] - df_new['numeric_min']
        df_new['numeric_skew'] = df[numeric_cols].skew(axis=1, skipna=True)
        
        # 外向性スコア（直感的特徴量）
        df_new['extroversion_score'] = (
            (10 - df['Time_spent_Alone'].fillna(5)) + 
            df['Social_event_attendance'].fillna(5) + 
            df['Going_outside'].fillna(4) + 
            (df['Friends_circle_size'].fillna(8) / 2) + 
            df['Post_frequency'].fillna(5)
        ) / 5
        
        # 社交性指標
        df_new['social_activity'] = (
            df['Social_event_attendance'].fillna(5) + 
            df['Going_outside'].fillna(4) + 
            df['Post_frequency'].fillna(5)
        ) / 3
        
        # 孤独指標
        df_new['isolation_tendency'] = (
            df['Time_spent_Alone'].fillna(3) + 
            (10 - df['Social_event_attendance'].fillna(5)) + 
            (15 - df['Friends_circle_size'].fillna(8))
        ) / 3
        
        return df_new
    
    def create_interaction_features(self, df):
        """相互作用特徴量を作成"""
        df_new = df.copy()
        
        # 重要な相互作用
        df_new['social_vs_alone'] = (
            df['Social_event_attendance'].fillna(5) - 
            df['Time_spent_Alone'].fillna(3)
        )
        
        df_new['friends_vs_posts'] = (
            df['Friends_circle_size'].fillna(8) * 
            df['Post_frequency'].fillna(5)
        )
        
        df_new['fear_vs_social'] = (
            df['Stage_fear'].map({'Yes': 1, 'No': 0}).fillna(0.5) * 
            df['Social_event_attendance'].fillna(5)
        )
        
        df_new['drained_vs_going_out'] = (
            df['Drained_after_socializing'].map({'Yes': 1, 'No': 0}).fillna(0.5) * 
            df['Going_outside'].fillna(4)
        )
        
        # 比率特徴量
        df_new['alone_to_social_ratio'] = (
            df['Time_spent_Alone'].fillna(3) / 
            (df['Social_event_attendance'].fillna(5) + 1)
        )
        
        df_new['posts_per_friend'] = (
            df['Post_frequency'].fillna(5) / 
            (df['Friends_circle_size'].fillna(8) + 1)
        )
        
        return df_new
    
    def create_missing_features(self, df):
        """欠損値パターン特徴量を作成"""
        df_new = df.copy()
        
        # 欠損値フラグ
        missing_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
                       'Post_frequency']
        
        for col in missing_cols:
            df_new[f'{col}_missing'] = df[col].isnull().astype(int)
        
        # 欠損値数
        df_new['total_missing'] = df[missing_cols].isnull().sum(axis=1)
        df_new['missing_ratio'] = df_new['total_missing'] / len(missing_cols)
        
        # 欠損パターン
        df_new['numeric_missing'] = df[['Time_spent_Alone', 'Social_event_attendance', 
                                       'Going_outside', 'Friends_circle_size', 
                                       'Post_frequency']].isnull().sum(axis=1)
        
        df_new['categorical_missing'] = df[['Stage_fear', 'Drained_after_socializing']].isnull().sum(axis=1)
        
        return df_new
    
    def create_frequency_encoding(self, train_df, test_df, cols):
        """頻度エンコーディング"""
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        for col in cols:
            freq_map = train_df[col].value_counts(normalize=True).to_dict()
            train_new[f'{col}_freq'] = train_df[col].map(freq_map)
            test_new[f'{col}_freq'] = test_df[col].map(freq_map)
            
        return train_new, test_new
    
    def create_target_encoding_features(self, train_df, test_df, target_col, cat_cols, cv_folds):
        """Target encodingベースの特徴量"""
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        for col in cat_cols:
            # クロスバリデーションでのターゲットエンコーディング
            train_new[f'{col}_target_mean'] = 0
            
            for fold in cv_folds:
                train_idx, val_idx = fold
                target_mean = train_df.iloc[train_idx].groupby(col)[target_col].mean()
                train_new.loc[val_idx, f'{col}_target_mean'] = train_new.loc[val_idx, col].map(target_mean)
            
            # テストセット用
            target_mean = train_df.groupby(col)[target_col].mean()
            test_new[f'{col}_target_mean'] = test_new[col].map(target_mean)
            
        return train_new, test_new
    
    def advanced_imputation(self, df):
        """KNN補完による高度な欠損値処理"""
        df_imputed = df.copy()
        
        # 数値特徴量のKNN補完
        numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                       'Friends_circle_size', 'Post_frequency']
        
        if len(numeric_cols) > 0:
            df_imputed[numeric_cols] = self.knn_imputer.fit_transform(df[numeric_cols])
        
        # カテゴリカル特徴量の頻度補完
        for col in ['Stage_fear', 'Drained_after_socializing']:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'No'
            df_imputed[col] = df[col].fillna(mode_val)
            
        return df_imputed
    
    def create_personality_specific_features(self, df):
        """性格特異的特徴量"""
        df_new = df.copy()
        
        # Big Five関連の推定特徴量
        df_new['openness_proxy'] = (
            df['Post_frequency'].fillna(5) + 
            (10 - df['Stage_fear'].map({'Yes': 8, 'No': 2}).fillna(5))
        ) / 2
        
        df_new['conscientiousness_proxy'] = (
            (10 - df['Time_spent_Alone'].fillna(3)) + 
            df['Social_event_attendance'].fillna(5)
        ) / 2
        
        df_new['agreeableness_proxy'] = (
            (10 - df['Drained_after_socializing'].map({'Yes': 8, 'No': 2}).fillna(5)) + 
            (df['Friends_circle_size'].fillna(8) / 2)
        ) / 2
        
        df_new['neuroticism_proxy'] = (
            df['Stage_fear'].map({'Yes': 8, 'No': 2}).fillna(5) + 
            df['Drained_after_socializing'].map({'Yes': 6, 'No': 4}).fillna(5)
        ) / 2
        
        # 一貫性メトリクス
        # 社交的回答の一貫性
        social_answers = [
            11 - df['Time_spent_Alone'].fillna(5),
            df['Social_event_attendance'].fillna(5),
            df['Going_outside'].fillna(4),
            df['Friends_circle_size'].fillna(8) / 2,
            df['Post_frequency'].fillna(5),
            10 - df['Stage_fear'].map({'Yes': 8, 'No': 2}).fillna(5),
            10 - df['Drained_after_socializing'].map({'Yes': 8, 'No': 2}).fillna(5)
        ]
        
        social_df = pd.DataFrame(social_answers).T
        df_new['social_consistency'] = social_df.std(axis=1)
        df_new['social_mean'] = social_df.mean(axis=1)
        
        return df_new
    
    def fit_transform(self, train_df, test_df, target_col=None, cv_folds=None):
        """全特徴量エンジニアリングパイプライン"""
        print("高度な特徴量エンジニアリング開始...")
        
        # 1. 統計的特徴量
        print("1. 統計的特徴量作成中...")
        train_processed = self.create_statistical_features(train_df)
        test_processed = self.create_statistical_features(test_df)
        
        # 2. 相互作用特徴量  
        print("2. 相互作用特徴量作成中...")
        train_processed = self.create_interaction_features(train_processed)
        test_processed = self.create_interaction_features(test_processed)
        
        # 3. 欠損値特徴量
        print("3. 欠損値パターン特徴量作成中...")
        train_processed = self.create_missing_features(train_processed)
        test_processed = self.create_missing_features(test_processed)
        
        # 4. 性格特異的特徴量
        print("4. 性格特異的特徴量作成中...")
        train_processed = self.create_personality_specific_features(train_processed)
        test_processed = self.create_personality_specific_features(test_processed)
        
        # 5. 頻度エンコーディング
        print("5. 頻度エンコーディング実行中...")
        cat_cols = ['Stage_fear', 'Drained_after_socializing']
        train_processed, test_processed = self.create_frequency_encoding(
            train_processed, test_processed, cat_cols
        )
        
        # 6. 高度な欠損値処理
        print("6. KNN補完実行中...")
        train_processed = self.advanced_imputation(train_processed)
        test_processed = self.advanced_imputation(test_processed)
        
        print(f"特徴量エンジニアリング完了: {train_processed.shape[1]}特徴量")
        
        return train_processed, test_processed