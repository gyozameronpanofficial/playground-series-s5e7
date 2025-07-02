"""
革新的特徴量エンジニアリング: GMベースライン0.975708突破戦略
分析結果を基にした高度特徴量生成
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from itertools import combinations, permutations
from scipy import stats
import math

class RevolutionaryFeatureEngineer:
    """革新的特徴量エンジニアリングクラス"""
    
    def __init__(self):
        self.config = {
            'DATA_PATH': Path('/Users/osawa/kaggle/playground-series-s5e7/data/raw'),
            'OUTPUT_PATH': Path('/Users/osawa/kaggle/playground-series-s5e7/data/processed'),
            'TARGET_MAPPING': {"Extrovert": 0, "Introvert": 1},
            'RANDOM_STATE': 42
        }
        self.config['OUTPUT_PATH'].mkdir(exist_ok=True)
        
    def load_data(self):
        """データ読み込み"""
        print("=== データ読み込み ===")
        self.train_df = pd.read_csv(self.config['DATA_PATH'] / 'train.csv')
        self.test_df = pd.read_csv(self.config['DATA_PATH'] / 'test.csv')
        
        self.feature_cols = [col for col in self.train_df.columns 
                           if col not in ['id', 'Personality']]
        
        print(f"データ形状: train={self.train_df.shape}, test={self.test_df.shape}")
        
    def create_psychological_features(self, X_train, X_test):
        """心理学的特徴量生成（分析結果ベース）"""
        print("=== 心理学的特徴量生成 ===")
        
        # 数値特徴量のみ処理
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return pd.DataFrame(index=X_train.index), pd.DataFrame(index=X_test.index)
        
        X_train_num = X_train[numeric_cols].fillna(-1)
        X_test_num = X_test[numeric_cols].fillna(-1)
        
        psych_features_train = pd.DataFrame(index=X_train.index)
        psych_features_test = pd.DataFrame(index=X_test.index)
        
        # 1. 回答一貫性特徴量群
        psych_features_train['response_consistency'] = X_train_num.std(axis=1)
        psych_features_test['response_consistency'] = X_test_num.std(axis=1)
        
        psych_features_train['response_variance'] = X_train_num.var(axis=1)
        psych_features_test['response_variance'] = X_test_num.var(axis=1)
        
        psych_features_train['response_range'] = X_train_num.max(axis=1) - X_train_num.min(axis=1)
        psych_features_test['response_range'] = X_test_num.max(axis=1) - X_test_num.min(axis=1)
        
        # 2. 中央傾向特徴量群
        median_values = X_train_num.median()
        psych_features_train['median_deviation'] = (X_train_num - median_values).abs().mean(axis=1)
        psych_features_test['median_deviation'] = (X_test_num - median_values).abs().mean(axis=1)
        
        mean_values = X_train_num.mean()
        psych_features_train['mean_deviation'] = (X_train_num - mean_values).abs().mean(axis=1)
        psych_features_test['mean_deviation'] = (X_test_num - mean_values).abs().mean(axis=1)
        
        # 3. 極端回答特徴量群
        threshold_high = X_train_num.quantile(0.8).mean()
        threshold_low = X_train_num.quantile(0.2).mean()
        
        psych_features_train['high_response_rate'] = (X_train_num > threshold_high).mean(axis=1)
        psych_features_test['high_response_rate'] = (X_test_num > threshold_high).mean(axis=1)
        
        psych_features_train['low_response_rate'] = (X_train_num < threshold_low).mean(axis=1)
        psych_features_test['low_response_rate'] = (X_test_num < threshold_low).mean(axis=1)
        
        psych_features_train['extreme_response_rate'] = psych_features_train['high_response_rate'] + psych_features_train['low_response_rate']
        psych_features_test['extreme_response_rate'] = psych_features_test['high_response_rate'] + psych_features_test['low_response_rate']
        
        # 4. 分布特徴量群
        psych_features_train['response_skewness'] = X_train_num.apply(lambda row: stats.skew(row), axis=1)
        psych_features_test['response_skewness'] = X_test_num.apply(lambda row: stats.skew(row), axis=1)
        
        psych_features_train['response_kurtosis'] = X_train_num.apply(lambda row: stats.kurtosis(row), axis=1)
        psych_features_test['response_kurtosis'] = X_test_num.apply(lambda row: stats.kurtosis(row), axis=1)
        
        # 5. 順序統計量
        psych_features_train['q75_q25_ratio'] = X_train_num.quantile(0.75, axis=1) / (X_train_num.quantile(0.25, axis=1) + 1e-8)
        psych_features_test['q75_q25_ratio'] = X_test_num.quantile(0.75, axis=1) / (X_test_num.quantile(0.25, axis=1) + 1e-8)
        
        # 6. エントロピー特徴量
        for i, row in X_train_num.iterrows():
            values, counts = np.unique(row, return_counts=True)
            if len(counts) > 1:
                entropy = -np.sum((counts/counts.sum()) * np.log2(counts/counts.sum()))
                psych_features_train.loc[i, 'response_entropy'] = entropy
            else:
                psych_features_train.loc[i, 'response_entropy'] = 0
                
        for i, row in X_test_num.iterrows():
            values, counts = np.unique(row, return_counts=True)
            if len(counts) > 1:
                entropy = -np.sum((counts/counts.sum()) * np.log2(counts/counts.sum()))
                psych_features_test.loc[i, 'response_entropy'] = entropy
            else:
                psych_features_test.loc[i, 'response_entropy'] = 0
        
        print(f"心理学的特徴量: {len(psych_features_train.columns)} 個生成")
        return psych_features_train, psych_features_test
    
    def create_interaction_features(self, X_train, X_test):
        """相互作用特徴量生成（分析結果ベース）"""
        print("=== 相互作用特徴量生成 ===")
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return pd.DataFrame(index=X_train.index), pd.DataFrame(index=X_test.index)
        
        # 欠損値処理
        X_train_num = X_train[numeric_cols].fillna(-1)
        X_test_num = X_test[numeric_cols].fillna(-1)
        
        interaction_train = pd.DataFrame(index=X_train.index)
        interaction_test = pd.DataFrame(index=X_test.index)
        
        # 分析で発見された有効な相互作用パターン
        high_value_pairs = [
            ('Time_spent_Alone', 'Social_event_attendance'),
            ('Time_spent_Alone', 'Post_frequency'),
            ('Social_event_attendance', 'Going_outside'),
            ('Time_spent_Alone', 'Going_outside'),
            ('Going_outside', 'Post_frequency'),
            ('Social_event_attendance', 'Post_frequency'),
            ('Time_spent_Alone', 'Friends_circle_size'),
            ('Social_event_attendance', 'Friends_circle_size')
        ]
        
        for col1, col2 in high_value_pairs:
            if col1 in numeric_cols and col2 in numeric_cols:
                base_name = f"{col1}_{col2}"
                
                # 基本演算
                interaction_train[f'{base_name}_multiply'] = X_train_num[col1] * X_train_num[col2]
                interaction_test[f'{base_name}_multiply'] = X_test_num[col1] * X_test_num[col2]
                
                interaction_train[f'{base_name}_add'] = X_train_num[col1] + X_train_num[col2]
                interaction_test[f'{base_name}_add'] = X_test_num[col1] + X_test_num[col2]
                
                interaction_train[f'{base_name}_subtract'] = X_train_num[col1] - X_train_num[col2]
                interaction_test[f'{base_name}_subtract'] = X_test_num[col1] - X_test_num[col2]
                
                # 比率（分析で最も有効だった）
                interaction_train[f'{base_name}_ratio'] = X_train_num[col1] / (X_train_num[col2] + 1e-8)
                interaction_test[f'{base_name}_ratio'] = X_test_num[col1] / (X_test_num[col2] + 1e-8)
                
                # 最大・最小
                interaction_train[f'{base_name}_max'] = np.maximum(X_train_num[col1], X_train_num[col2])
                interaction_test[f'{base_name}_max'] = np.maximum(X_test_num[col1], X_test_num[col2])
                
                interaction_train[f'{base_name}_min'] = np.minimum(X_train_num[col1], X_train_num[col2])
                interaction_test[f'{base_name}_min'] = np.minimum(X_test_num[col1], X_test_num[col2])
                
                # 絶対差
                interaction_train[f'{base_name}_abs_diff'] = np.abs(X_train_num[col1] - X_train_num[col2])
                interaction_test[f'{base_name}_abs_diff'] = np.abs(X_test_num[col1] - X_test_num[col2])
        
        # 3つの特徴量の組み合わせ（トップ5パターンのみ）
        top_triplets = [
            ('Time_spent_Alone', 'Social_event_attendance', 'Going_outside'),
            ('Time_spent_Alone', 'Social_event_attendance', 'Post_frequency'),
            ('Social_event_attendance', 'Going_outside', 'Post_frequency'),
            ('Time_spent_Alone', 'Going_outside', 'Post_frequency'),
            ('Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size')
        ]
        
        for col1, col2, col3 in top_triplets:
            if all(col in numeric_cols for col in [col1, col2, col3]):
                base_name = f"{col1}_{col2}_{col3}"
                
                # 三項演算
                interaction_train[f'{base_name}_sum'] = X_train_num[col1] + X_train_num[col2] + X_train_num[col3]
                interaction_test[f'{base_name}_sum'] = X_test_num[col1] + X_test_num[col2] + X_test_num[col3]
                
                interaction_train[f'{base_name}_mean'] = (X_train_num[col1] + X_train_num[col2] + X_train_num[col3]) / 3
                interaction_test[f'{base_name}_mean'] = (X_test_num[col1] + X_test_num[col2] + X_test_num[col3]) / 3
                
                interaction_train[f'{base_name}_std'] = np.array([X_train_num.loc[i, [col1, col2, col3]].std() for i in X_train_num.index])
                interaction_test[f'{base_name}_std'] = np.array([X_test_num.loc[i, [col1, col2, col3]].std() for i in X_test_num.index])
        
        print(f"相互作用特徴量: {len(interaction_train.columns)} 個生成")
        return interaction_train, interaction_test
    
    def create_missing_pattern_features(self, X_train, X_test):
        """欠損パターン特徴量生成"""
        print("=== 欠損パターン特徴量生成 ===")
        
        missing_train = pd.DataFrame(index=X_train.index)
        missing_test = pd.DataFrame(index=X_test.index)
        
        # 各特徴量の欠損フラグ
        for col in self.feature_cols:
            missing_train[f'{col}_missing'] = X_train[col].isnull().astype(int)
            missing_test[f'{col}_missing'] = X_test[col].isnull().astype(int)
        
        # 欠損数統計
        missing_train['total_missing'] = X_train[self.feature_cols].isnull().sum(axis=1)
        missing_test['total_missing'] = X_test[self.feature_cols].isnull().sum(axis=1)
        
        missing_train['missing_ratio'] = missing_train['total_missing'] / len(self.feature_cols)
        missing_test['missing_ratio'] = missing_test['total_missing'] / len(self.feature_cols)
        
        # 分析で発見された重要な欠損パターン
        important_missing = ['Drained_after_socializing', 'Stage_fear', 'Post_frequency', 
                           'Social_event_attendance', 'Going_outside', 'Friends_circle_size']
        
        for col in important_missing:
            if col in self.feature_cols:
                # 他の特徴量との欠損パターン組み合わせ
                for other_col in important_missing:
                    if col != other_col and other_col in self.feature_cols:
                        pattern_name = f'{col}_{other_col}_both_missing'
                        missing_train[pattern_name] = (X_train[col].isnull() & X_train[other_col].isnull()).astype(int)
                        missing_test[pattern_name] = (X_test[col].isnull() & X_test[other_col].isnull()).astype(int)
        
        print(f"欠損パターン特徴量: {len(missing_train.columns)} 個生成")
        return missing_train, missing_test
    
    def create_clustering_features(self, X_train, X_test, y_train):
        """クラスタリング特徴量生成"""
        print("=== クラスタリング特徴量生成 ===")
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return pd.DataFrame(index=X_train.index), pd.DataFrame(index=X_test.index)
        
        # 数値データの準備
        X_train_num = X_train[numeric_cols].fillna(-1)
        X_test_num = X_test[numeric_cols].fillna(-1)
        
        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_num)
        X_test_scaled = scaler.transform(X_test_num)
        
        cluster_train = pd.DataFrame(index=X_train.index)
        cluster_test = pd.DataFrame(index=X_test.index)
        
        # 複数のクラスタ数でクラスタリング
        for n_clusters in [2, 3, 4, 5]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.config['RANDOM_STATE'], n_init=10)
            
            # 訓練データでクラスタリング
            train_clusters = kmeans.fit_predict(X_train_scaled)
            test_clusters = kmeans.predict(X_test_scaled)
            
            cluster_train[f'cluster_k{n_clusters}'] = train_clusters
            cluster_test[f'cluster_k{n_clusters}'] = test_clusters
            
            # クラスタ中心からの距離
            train_distances = kmeans.transform(X_train_scaled)
            test_distances = kmeans.transform(X_test_scaled)
            
            # 最近クラスタ中心からの距離
            cluster_train[f'cluster_k{n_clusters}_min_distance'] = train_distances.min(axis=1)
            cluster_test[f'cluster_k{n_clusters}_min_distance'] = test_distances.min(axis=1)
            
            # クラスタ内での相対位置
            for cluster_id in range(n_clusters):
                cluster_train[f'cluster_k{n_clusters}_distance_{cluster_id}'] = train_distances[:, cluster_id]
                cluster_test[f'cluster_k{n_clusters}_distance_{cluster_id}'] = test_distances[:, cluster_id]
        
        # PCA特徴量
        pca = PCA(n_components=min(5, len(numeric_cols)), random_state=self.config['RANDOM_STATE'])
        train_pca = pca.fit_transform(X_train_scaled)
        test_pca = pca.transform(X_test_scaled)
        
        for i in range(train_pca.shape[1]):
            cluster_train[f'pca_component_{i}'] = train_pca[:, i]
            cluster_test[f'pca_component_{i}'] = test_pca[:, i]
        
        print(f"クラスタリング特徴量: {len(cluster_train.columns)} 個生成")
        return cluster_train, cluster_test
    
    def create_advanced_ngram_features(self, X_train, X_test):
        """高度n-gram特徴量（GMを超える戦略）"""
        print("=== 高度n-gram特徴量生成 ===")
        
        # GMベースライン前処理を再現
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        # 数値→文字列変換
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train_processed[numeric_cols] = X_train_processed[numeric_cols].fillna(-1).astype(np.int16).astype(str)
        X_test_processed[numeric_cols] = X_test_processed[numeric_cols].fillna(-1).astype(np.int16).astype(str)
        
        # カテゴリカル→文字列
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        X_train_processed[categorical_cols] = X_train_processed[categorical_cols].astype(str).fillna("missing")
        X_test_processed[categorical_cols] = X_test_processed[categorical_cols].astype(str).fillna("missing")
        
        # 2-gram
        for col1, col2 in combinations(self.feature_cols, 2):
            feature_name = f"{col1}-{col2}"
            X_train_processed[feature_name] = X_train_processed[col1] + "-" + X_train_processed[col2]
            X_test_processed[feature_name] = X_test_processed[col1] + "-" + X_test_processed[col2]
        
        # 3-gram
        for col1, col2, col3 in combinations(self.feature_cols, 3):
            feature_name = f"{col1}-{col2}-{col3}"
            X_train_processed[feature_name] = (X_train_processed[col1] + "-" + 
                                            X_train_processed[col2] + "-" + 
                                            X_train_processed[col3])
            X_test_processed[feature_name] = (X_test_processed[col1] + "-" + 
                                           X_test_processed[col2] + "-" + 
                                           X_test_processed[col3])
        
        # 4-gram（限定版）- 計算量削減のため有効な組み合わせのみ
        important_features = ['Time_spent_Alone', 'Stage_fear', 'Drained_after_socializing', 
                            'Social_event_attendance', 'Going_outside', 'Post_frequency']
        
        important_4grams = list(combinations([col for col in important_features if col in self.feature_cols], 4))[:10]
        
        for col1, col2, col3, col4 in important_4grams:
            feature_name = f"{col1}-{col2}-{col3}-{col4}"
            X_train_processed[feature_name] = (X_train_processed[col1] + "-" + 
                                            X_train_processed[col2] + "-" + 
                                            X_train_processed[col3] + "-" + 
                                            X_train_processed[col4])
            X_test_processed[feature_name] = (X_test_processed[col1] + "-" + 
                                           X_test_processed[col2] + "-" + 
                                           X_test_processed[col3] + "-" + 
                                           X_test_processed[col4])
        
        print(f"高度n-gram特徴量: {X_train_processed.shape[1]} 個（元の{len(self.feature_cols)}から拡張）")
        return X_train_processed, X_test_processed
    
    def run_feature_engineering(self):
        """全特徴量エンジニアリングの実行"""
        print("=== 革新的特徴量エンジニアリング開始 ===")
        
        # データ読み込み
        self.load_data()
        
        # 基本データ準備
        X_train = self.train_df[self.feature_cols].copy()
        y_train = self.train_df['Personality'].map(self.config['TARGET_MAPPING'])
        X_test = self.test_df[self.feature_cols].copy()
        
        # 各種特徴量生成
        print("\n--- 特徴量生成フェーズ ---")
        
        # 1. 心理学的特徴量
        psych_train, psych_test = self.create_psychological_features(X_train, X_test)
        
        # 2. 相互作用特徴量
        interact_train, interact_test = self.create_interaction_features(X_train, X_test)
        
        # 3. 欠損パターン特徴量
        missing_train, missing_test = self.create_missing_pattern_features(X_train, X_test)
        
        # 4. クラスタリング特徴量
        cluster_train, cluster_test = self.create_clustering_features(X_train, X_test, y_train)
        
        # 5. 高度n-gram特徴量
        ngram_train, ngram_test = self.create_advanced_ngram_features(X_train, X_test)
        
        # 特徴量統合
        print("\n--- 特徴量統合フェーズ ---")
        
        # n-gram特徴量をベースに追加特徴量を結合
        final_train = ngram_train.copy()
        final_test = ngram_test.copy()
        
        # 数値特徴量を追加
        if not psych_train.empty:
            for col in psych_train.columns:
                final_train[f'psych_{col}'] = psych_train[col]
                final_test[f'psych_{col}'] = psych_test[col]
        
        if not interact_train.empty:
            for col in interact_train.columns:
                final_train[f'interact_{col}'] = interact_train[col].astype(str)
                final_test[f'interact_{col}'] = interact_test[col].astype(str)
        
        if not missing_train.empty:
            for col in missing_train.columns:
                final_train[f'missing_{col}'] = missing_train[col].astype(str)
                final_test[f'missing_{col}'] = missing_test[col].astype(str)
        
        if not cluster_train.empty:
            for col in cluster_train.columns:
                final_train[f'cluster_{col}'] = cluster_train[col].astype(str)
                final_test[f'cluster_{col}'] = cluster_test[col].astype(str)
        
        # 最終結果
        print(f"\n=== 特徴量エンジニアリング完了 ===")
        print(f"最終特徴量数: {final_train.shape[1]} (元の{len(self.feature_cols)}から{final_train.shape[1]/len(self.feature_cols):.1f}倍)")
        print(f"心理学的特徴量: {len(psych_train.columns) if not psych_train.empty else 0}")
        print(f"相互作用特徴量: {len(interact_train.columns) if not interact_train.empty else 0}")
        print(f"欠損パターン特徴量: {len(missing_train.columns) if not missing_train.empty else 0}")
        print(f"クラスタリング特徴量: {len(cluster_train.columns) if not cluster_train.empty else 0}")
        
        # 保存
        final_train_with_target = final_train.copy()
        final_train_with_target['Personality'] = self.train_df['Personality']
        final_train_with_target['id'] = self.train_df['id']
        
        final_test_with_id = final_test.copy()
        final_test_with_id['id'] = self.test_df['id']
        
        # 保存
        train_output_path = self.config['OUTPUT_PATH'] / 'revolutionary_train.csv'
        test_output_path = self.config['OUTPUT_PATH'] / 'revolutionary_test.csv'
        
        final_train_with_target.to_csv(train_output_path, index=False)
        final_test_with_id.to_csv(test_output_path, index=False)
        
        print(f"\n保存完了:")
        print(f"訓練データ: {train_output_path}")
        print(f"テストデータ: {test_output_path}")
        
        return final_train_with_target, final_test_with_id

if __name__ == "__main__":
    engineer = RevolutionaryFeatureEngineer()
    train_data, test_data = engineer.run_feature_engineering()