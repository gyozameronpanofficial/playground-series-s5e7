"""
拡張特徴量エンジニアリング
4-gram/5-gramと統計的特徴量の実装
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from scipy import stats
import xgboost as xgb

class EnhancedFeatureEngineering:
    """拡張特徴量エンジニアリングクラス"""
    
    def __init__(self):
        self.config = {
            'RANDOM_STATE': 42,
            'N_SPLITS': 5,
            'TARGET_COL': 'Personality',
            'TARGET_MAPPING': {"Extrovert": 0, "Introvert": 1},
            'DATA_PATH': Path('/Users/osawa/kaggle/playground-series-s5e7/data/raw')
        }
        
    def load_data(self):
        """データ読み込み"""
        print("=== データ読み込み ===")
        train_df = pd.read_csv(self.config['DATA_PATH'] / 'train.csv')
        test_df = pd.read_csv(self.config['DATA_PATH'] / 'test.csv')
        
        feature_cols = [col for col in train_df.columns 
                       if col not in ['id', self.config['TARGET_COL']]]
        
        X_train = train_df[feature_cols].copy()
        y_train = train_df[self.config['TARGET_COL']].map(self.config['TARGET_MAPPING'])
        X_test = test_df[feature_cols].copy()
        
        print(f"訓練データ形状: {X_train.shape}")
        print(f"テストデータ形状: {X_test.shape}")
        
        return X_train, y_train, X_test
    
    def baseline_preprocessing(self, X_train, X_test):
        """ベースライン前処理"""
        print("\n=== ベースライン前処理 ===")
        
        # 数値特徴量処理
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train[numeric_cols] = X_train[numeric_cols].fillna(-1).astype(np.int16).astype(str)
        X_test[numeric_cols] = X_test[numeric_cols].fillna(-1).astype(np.int16).astype(str)
        
        # カテゴリカル特徴量処理
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        X_train[categorical_cols] = X_train[categorical_cols].astype(str).fillna("missing")
        X_test[categorical_cols] = X_test[categorical_cols].astype(str).fillna("missing")
        
        print(f"前処理後形状: {X_train.shape}")
        return X_train, X_test
    
    def create_ngram_features(self, X_train, X_test, max_n=5):
        """拡張N-gram特徴量生成（4-gram, 5-gramまで）"""
        print(f"\n=== {max_n}-gramまでの特徴量生成 ===")
        
        original_cols = list(X_train.columns)
        original_count = len(original_cols)
        
        print(f"元の特徴量数: {original_count}")
        
        # 2-gram
        if max_n >= 2:
            print("2-gram特徴量生成中...")
            for col1, col2 in combinations(original_cols, 2):
                feature_name = f"{col1}-{col2}"
                X_train[feature_name] = X_train[col1] + "-" + X_train[col2]
                X_test[feature_name] = X_test[col1] + "-" + X_test[col2]
            print(f"2-gram後: {X_train.shape[1]} 特徴量")
        
        # 3-gram
        if max_n >= 3:
            print("3-gram特徴量生成中...")
            for col1, col2, col3 in combinations(original_cols, 3):
                feature_name = f"{col1}-{col2}-{col3}"
                X_train[feature_name] = (X_train[col1] + "-" + 
                                        X_train[col2] + "-" + 
                                        X_train[col3])
                X_test[feature_name] = (X_test[col1] + "-" + 
                                       X_test[col2] + "-" + 
                                       X_test[col3])
            print(f"3-gram後: {X_train.shape[1]} 特徴量")
        
        # 4-gram（新規）
        if max_n >= 4:
            print("4-gram特徴量生成中...")
            count_4gram = 0
            for col1, col2, col3, col4 in combinations(original_cols, 4):
                feature_name = f"{col1}-{col2}-{col3}-{col4}"
                X_train[feature_name] = (X_train[col1] + "-" + 
                                        X_train[col2] + "-" + 
                                        X_train[col3] + "-" + 
                                        X_train[col4])
                X_test[feature_name] = (X_test[col1] + "-" + 
                                       X_test[col2] + "-" + 
                                       X_test[col3] + "-" + 
                                       X_test[col4])
                count_4gram += 1
            print(f"4-gram後: {X_train.shape[1]} 特徴量 (+{count_4gram}個)")
        
        # 5-gram（新規）
        if max_n >= 5:
            print("5-gram特徴量生成中...")
            count_5gram = 0
            for col1, col2, col3, col4, col5 in combinations(original_cols, 5):
                feature_name = f"{col1}-{col2}-{col3}-{col4}-{col5}"
                X_train[feature_name] = (X_train[col1] + "-" + 
                                        X_train[col2] + "-" + 
                                        X_train[col3] + "-" + 
                                        X_train[col4] + "-" + 
                                        X_train[col5])
                X_test[feature_name] = (X_test[col1] + "-" + 
                                       X_test[col2] + "-" + 
                                       X_test[col3] + "-" + 
                                       X_test[col4] + "-" + 
                                       X_test[col5])
                count_5gram += 1
            print(f"5-gram後: {X_train.shape[1]} 特徴量 (+{count_5gram}個)")
        
        print(f"最終特徴量数: {X_train.shape[1]} (元の{X_train.shape[1]/original_count:.1f}倍)")
        
        return X_train, X_test
    
    def create_statistical_features(self, X_train_orig, X_test_orig, X_train, X_test):
        """統計的特徴量の追加"""
        print("\n=== 統計的特徴量生成 ===")
        
        # 元データを数値として扱うバージョンを作成
        X_train_numeric = X_train_orig.copy()
        X_test_numeric = X_test_orig.copy()
        
        # 数値列の処理
        numeric_cols = X_train_numeric.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            # 基本統計量
            X_train[f'numeric_mean'] = X_train_numeric[numeric_cols].mean(axis=1).astype(str)
            X_train[f'numeric_std'] = X_train_numeric[numeric_cols].std(axis=1).fillna(0).astype(str)
            X_train[f'numeric_median'] = X_train_numeric[numeric_cols].median(axis=1).astype(str)
            X_train[f'numeric_sum'] = X_train_numeric[numeric_cols].sum(axis=1).astype(str)
            X_train[f'numeric_min'] = X_train_numeric[numeric_cols].min(axis=1).astype(str)
            X_train[f'numeric_max'] = X_train_numeric[numeric_cols].max(axis=1).astype(str)
            
            X_test[f'numeric_mean'] = X_test_numeric[numeric_cols].mean(axis=1).astype(str)
            X_test[f'numeric_std'] = X_test_numeric[numeric_cols].std(axis=1).fillna(0).astype(str)
            X_test[f'numeric_median'] = X_test_numeric[numeric_cols].median(axis=1).astype(str)
            X_test[f'numeric_sum'] = X_test_numeric[numeric_cols].sum(axis=1).astype(str)
            X_test[f'numeric_min'] = X_test_numeric[numeric_cols].min(axis=1).astype(str)
            X_test[f'numeric_max'] = X_test_numeric[numeric_cols].max(axis=1).astype(str)
            
            print(f"基本統計量6個追加")
        
        # 欠損値パターン特徴量
        missing_pattern_train = X_train_orig.isnull().astype(int)
        missing_pattern_test = X_test_orig.isnull().astype(int)
        
        # 欠損値の総数
        X_train['missing_count'] = missing_pattern_train.sum(axis=1).astype(str)
        X_test['missing_count'] = missing_pattern_test.sum(axis=1).astype(str)
        
        # 欠損値パターンのハッシュ
        missing_pattern_hash_train = missing_pattern_train.apply(
            lambda x: hash(tuple(x)), axis=1
        ).astype(str)
        missing_pattern_hash_test = missing_pattern_test.apply(
            lambda x: hash(tuple(x)), axis=1
        ).astype(str)
        
        X_train['missing_pattern'] = missing_pattern_hash_train
        X_test['missing_pattern'] = missing_pattern_hash_test
        
        print(f"欠損値パターン特徴量2個追加")
        
        # カテゴリカル特徴量の統計
        categorical_cols = X_train_orig.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_cols) > 0:
            # カテゴリカル特徴量のユニーク数
            cat_unique_train = X_train_orig[categorical_cols].nunique(axis=1)
            cat_unique_test = X_test_orig[categorical_cols].nunique(axis=1)
            
            X_train['categorical_unique_count'] = cat_unique_train.astype(str)
            X_test['categorical_unique_count'] = cat_unique_test.astype(str)
            
            print(f"カテゴリカル統計量1個追加")
        
        print(f"統計的特徴量後の形状: {X_train.shape}")
        
        return X_train, X_test
    
    def create_clustering_features(self, X_train_orig, X_test_orig, X_train, X_test, n_clusters=5):
        """クラスタリング特徴量の追加"""
        print(f"\n=== クラスタリング特徴量生成 (k={n_clusters}) ===")
        
        # 数値データのみでクラスタリング
        numeric_cols = X_train_orig.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            # 欠損値を中央値で補完
            X_train_cluster = X_train_orig[numeric_cols].fillna(
                X_train_orig[numeric_cols].median()
            )
            X_test_cluster = X_test_orig[numeric_cols].fillna(
                X_train_orig[numeric_cols].median()  # 訓練データの中央値を使用
            )
            
            # K-Meansクラスタリング
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.config['RANDOM_STATE'])
            
            train_clusters = kmeans.fit_predict(X_train_cluster)
            test_clusters = kmeans.predict(X_test_cluster)
            
            # クラスタ番号を特徴量として追加
            X_train[f'cluster_{n_clusters}'] = train_clusters.astype(str)
            X_test[f'cluster_{n_clusters}'] = test_clusters.astype(str)
            
            # クラスタ中心からの距離
            train_distances = kmeans.transform(X_train_cluster).min(axis=1)
            test_distances = kmeans.transform(X_test_cluster).min(axis=1)
            
            X_train[f'cluster_distance'] = train_distances.astype(str)
            X_test[f'cluster_distance'] = test_distances.astype(str)
            
            print(f"クラスタリング特徴量2個追加")
        
        print(f"クラスタリング後の形状: {X_train.shape}")
        
        return X_train, X_test
    
    def evaluate_features(self, X_train, y_train, feature_description=""):
        """特徴量評価"""
        print(f"\n=== {feature_description}評価 ===")
        
        # XGBoostモデルで評価
        model = Pipeline([
            ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
            ('model', xgb.XGBClassifier(
                objective='binary:logistic',
                learning_rate=0.02,
                n_estimators=300,  # 高速化のため削減
                max_depth=5,
                colsample_bytree=0.45,
                random_state=self.config['RANDOM_STATE'],
                verbosity=0
            ))
        ])
        
        # クロスバリデーション
        cv = StratifiedKFold(
            n_splits=self.config['N_SPLITS'], 
            shuffle=True, 
            random_state=self.config['RANDOM_STATE']
        )
        
        scores = []
        for train_idx, valid_idx in cv.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_valid = X_train.iloc[valid_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_valid = y_train.iloc[valid_idx]
            
            model.fit(X_fold_train, y_fold_train)
            preds = model.predict(X_fold_valid)
            score = accuracy_score(y_fold_valid, preds)
            scores.append(score)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"スコア: {mean_score:.6f} (±{std_score:.6f})")
        print(f"特徴量数: {X_train.shape[1]}")
        
        return mean_score, std_score
    
    def run_enhanced_feature_engineering(self):
        """拡張特徴量エンジニアリングの実行"""
        print("=== 拡張特徴量エンジニアリング実行 ===\n")
        
        # データ読み込み
        X_train_orig, y_train, X_test_orig = self.load_data()
        
        # ベースライン前処理
        X_train, X_test = self.baseline_preprocessing(X_train_orig.copy(), X_test_orig.copy())
        
        # 段階的特徴量追加と評価
        results = {}
        
        # 1. 3-gramまで（ベースライン）
        X_train_3gram, X_test_3gram = self.create_ngram_features(
            X_train.copy(), X_test.copy(), max_n=3
        )
        score_3gram, std_3gram = self.evaluate_features(
            X_train_3gram, y_train, "3-gramベースライン"
        )
        results['3gram'] = {'score': score_3gram, 'std': std_3gram}
        
        # 2. 4-gram追加
        X_train_4gram, X_test_4gram = self.create_ngram_features(
            X_train.copy(), X_test.copy(), max_n=4
        )
        score_4gram, std_4gram = self.evaluate_features(
            X_train_4gram, y_train, "4-gram拡張"
        )
        results['4gram'] = {'score': score_4gram, 'std': std_4gram}
        print(f"4-gramによる改善: {score_4gram - score_3gram:+.6f}")
        
        # 3. 5-gram追加
        X_train_5gram, X_test_5gram = self.create_ngram_features(
            X_train.copy(), X_test.copy(), max_n=5
        )
        score_5gram, std_5gram = self.evaluate_features(
            X_train_5gram, y_train, "5-gram拡張"
        )
        results['5gram'] = {'score': score_5gram, 'std': std_5gram}
        print(f"5-gramによる改善: {score_5gram - score_4gram:+.6f}")
        
        # 4. 統計的特徴量追加
        X_train_stats, X_test_stats = self.create_statistical_features(
            X_train_orig, X_test_orig, X_train_5gram.copy(), X_test_5gram.copy()
        )
        score_stats, std_stats = self.evaluate_features(
            X_train_stats, y_train, "統計的特徴量追加"
        )
        results['statistical'] = {'score': score_stats, 'std': std_stats}
        print(f"統計的特徴量による改善: {score_stats - score_5gram:+.6f}")
        
        # 5. クラスタリング特徴量追加
        X_train_final, X_test_final = self.create_clustering_features(
            X_train_orig, X_test_orig, X_train_stats.copy(), X_test_stats.copy()
        )
        score_final, std_final = self.evaluate_features(
            X_train_final, y_train, "最終（全特徴量）"
        )
        results['final'] = {'score': score_final, 'std': std_final}
        print(f"クラスタリング特徴量による改善: {score_final - score_stats:+.6f}")
        
        # 総合結果
        print(f"\n=== 総合改善結果 ===")
        print(f"ベースライン（3-gram）: {score_3gram:.6f}")
        print(f"最終スコア: {score_final:.6f}")
        print(f"総改善: {score_final - score_3gram:+.6f}")
        print(f"目標0.975708まで: {0.975708 - score_final:+.6f}")
        
        # 最終データ保存
        print(f"\n=== 拡張特徴量データ保存 ===")
        
        # 処理済みデータを保存
        processed_path = Path('/Users/osawa/kaggle/playground-series-s5e7/data/processed')
        processed_path.mkdir(exist_ok=True)
        
        X_train_final['target'] = y_train
        X_train_final.to_csv(processed_path / 'enhanced_train.csv', index=False)
        X_test_final.to_csv(processed_path / 'enhanced_test.csv', index=False)
        
        print(f"保存完了:")
        print(f"  訓練データ: {processed_path / 'enhanced_train.csv'}")
        print(f"  テストデータ: {processed_path / 'enhanced_test.csv'}")
        print(f"  最終特徴量数: {X_train_final.shape[1] - 1}")  # target列を除く
        
        return results, X_train_final, X_test_final

if __name__ == "__main__":
    engineer = EnhancedFeatureEngineering()
    results, X_train_enhanced, X_test_enhanced = engineer.run_enhanced_feature_engineering()