"""
Phase 2b: 高度Target Encoding実装

Phase 2aのTF-IDF失敗を受けて、よりシンプルで効果的なTarget Encodingに注力
- Smoothing Target Encoding（ベイズ平均）
- 複数CV戦略での頑健性向上
- ノイズ注入による汎化性能向上

Author: Osawa
Date: 2025-07-03
Purpose: GM超越のための高品質特徴量エンジニアリング
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedTargetEncoder:
    """高度Target Encodingエンジン"""
    
    def __init__(self, smoothing_alpha=100, n_splits=5, random_state=42):
        self.smoothing_alpha = smoothing_alpha
        self.n_splits = n_splits
        self.random_state = random_state
        self.target_encoders = {}
        self.global_mean = None
        
    def smooth_target_encoding(self, feature_series, target_series, alpha=None):
        """
        Smoothing Target Encodingの実装
        
        Parameters:
        -----------
        feature_series : pd.Series
            エンコード対象の特徴量
        target_series : pd.Series
            ターゲット変数
        alpha : float
            平滑化パラメータ（大きいほど保守的）
        
        Returns:
        --------
        dict : カテゴリ別のエンコード値
        """
        if alpha is None:
            alpha = self.smoothing_alpha
        
        # グローバル平均
        global_mean = target_series.mean()
        
        # カテゴリ別統計計算
        stats_df = pd.DataFrame({
            'feature': feature_series,
            'target': target_series
        })
        
        category_stats = stats_df.groupby('feature').agg({
            'target': ['count', 'mean']
        }).reset_index()
        
        category_stats.columns = ['category', 'count', 'mean']
        
        # Smoothing Target Encoding計算
        # smoothed_mean = (count * mean + alpha * global_mean) / (count + alpha)
        category_stats['smoothed_mean'] = (
            category_stats['count'] * category_stats['mean'] + 
            alpha * global_mean
        ) / (category_stats['count'] + alpha)
        
        # 辞書形式で返す
        encoding_dict = dict(zip(category_stats['category'], category_stats['smoothed_mean']))
        
        return encoding_dict, global_mean
    
    def create_cv_target_encoding(self, X, y, feature_name, cv_strategy='stratified'):
        """
        クロスバリデーション内でのTarget Encoding
        
        Parameters:
        -----------
        X : pd.DataFrame
            特徴量データフレーム
        y : pd.Series
            ターゲット変数
        feature_name : str
            エンコード対象の特徴量名
        cv_strategy : str
            CV戦略（'stratified' or 'kfold'）
        
        Returns:
        --------
        pd.Series : エンコード済み特徴量
        """
        
        # CV設定
        if cv_strategy == 'stratified':
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # エンコード結果を格納する配列
        encoded_feature = np.zeros(len(X))
        
        # CV内でのTarget Encoding
        for train_idx, valid_idx in cv.split(X, y):
            # Train fold内でエンコード辞書を作成
            train_feature = X.iloc[train_idx][feature_name]
            train_target = y.iloc[train_idx]
            
            encoding_dict, global_mean = self.smooth_target_encoding(
                train_feature, train_target
            )
            
            # Valid foldに適用
            valid_feature = X.iloc[valid_idx][feature_name]
            
            # エンコード適用（未知カテゴリはグローバル平均）
            encoded_valid = valid_feature.map(encoding_dict).fillna(global_mean)
            encoded_feature[valid_idx] = encoded_valid
        
        return pd.Series(encoded_feature, index=X.index, name=f'{feature_name}_target_encoded')
    
    def create_noise_augmented_encoding(self, X, y, feature_name, noise_level=0.01):
        """
        ノイズ注入によるTarget Encodingの頑健性向上
        
        Parameters:
        -----------
        X : pd.DataFrame
            特徴量データフレーム
        y : pd.Series
            ターゲット変数
        feature_name : str
            エンコード対象の特徴量名
        noise_level : float
            ノイズレベル（標準偏差）
        
        Returns:
        --------
        pd.Series : ノイズ注入済みエンコード特徴量
        """
        
        # 基本のTarget Encoding
        base_encoded = self.create_cv_target_encoding(X, y, feature_name)
        
        # ガウシアンノイズ注入
        noise = np.random.normal(0, noise_level, len(base_encoded))
        noise_encoded = base_encoded + noise
        
        return pd.Series(noise_encoded, index=X.index, name=f'{feature_name}_noise_encoded')
    
    def create_multiple_cv_encodings(self, X, y, feature_name, n_encodings=3):
        """
        複数の異なるCV設定でのTarget Encodingアンサンブル
        
        Parameters:
        -----------
        X : pd.DataFrame
            特徴量データフレーム
        y : pd.Series
            ターゲット変数
        feature_name : str
            エンコード対象の特徴量名
        n_encodings : int
            作成するエンコード数
        
        Returns:
        --------
        pd.DataFrame : 複数のエンコード特徴量
        """
        
        encodings = []
        
        for i in range(n_encodings):
            # 異なるランダムシードでCV設定
            encoder = AdvancedTargetEncoder(
                smoothing_alpha=self.smoothing_alpha,
                n_splits=self.n_splits,
                random_state=self.random_state + i
            )
            
            # CV戦略をランダムに選択
            cv_strategy = 'stratified' if i % 2 == 0 else 'kfold'
            
            encoded = encoder.create_cv_target_encoding(X, y, feature_name, cv_strategy)
            encoded.name = f'{feature_name}_cv_encoded_{i+1}'
            encodings.append(encoded)
        
        return pd.concat(encodings, axis=1)
    
    def fit_transform(self, X, y, categorical_features=None):
        """
        全特徴量に対するTarget Encodingの適用
        
        Parameters:
        -----------
        X : pd.DataFrame
            特徴量データフレーム
        y : pd.Series
            ターゲット変数
        categorical_features : list
            カテゴリカル特徴量のリスト
        
        Returns:
        --------
        pd.DataFrame : Target Encoding適用済みデータ
        """
        
        print("=== 高度Target Encoding実行 ===")
        
        # カテゴリカル特徴量の自動検出
        if categorical_features is None:
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"対象カテゴリカル特徴量: {categorical_features}")
        
        # 結果を格納するDataFrame
        encoded_X = X.copy()
        
        for feature in categorical_features:
            print(f"\n{feature}のTarget Encoding中...")
            
            # 1. 基本のSmoothing Target Encoding
            basic_encoded = self.create_cv_target_encoding(encoded_X, y, feature)
            encoded_X[f'{feature}_basic_encoded'] = basic_encoded
            
            # 2. ノイズ注入版
            noise_encoded = self.create_noise_augmented_encoding(encoded_X, y, feature)
            encoded_X[f'{feature}_noise_encoded'] = noise_encoded
            
            # 3. 複数CV版（2個）
            multi_cv_encoded = self.create_multiple_cv_encodings(encoded_X, y, feature, n_encodings=2)
            encoded_X = pd.concat([encoded_X, multi_cv_encoded], axis=1)
            
            print(f"  生成特徴量数: {4}個")
        
        print(f"\n元特徴量数: {X.shape[1]} → 拡張後: {encoded_X.shape[1]}")
        
        return encoded_X
    
    def transform(self, X_test):
        """
        テストデータへのTarget Encodingの適用
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            テストデータ
        
        Returns:
        --------
        pd.DataFrame : エンコード済みテストデータ
        """
        
        # 実装の簡略化のため、fit_transformで保存した辞書を使用
        # 実際の実装では、fit時に辞書を保存し、transformで適用
        print("警告: transformメソッドは簡略実装です")
        return X_test

def create_phase2b_features():
    """Phase 2b用特徴量データの作成"""
    
    print("=== Phase 2b 特徴量作成開始 ===")
    
    # 1. 元データ読み込み
    print("1. 元データ読み込み中...")
    train_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/raw/train.csv')
    test_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/raw/test.csv')
    
    print(f"   訓練データ: {train_data.shape}")
    print(f"   テストデータ: {test_data.shape}")
    
    # 2. Target Encodingの適用
    print("\n2. Target Encoding適用中...")
    
    # 特徴量とターゲット分離
    feature_cols = [col for col in train_data.columns if col not in ['id', 'Personality']]
    X_train = train_data[feature_cols]
    y_train = train_data['Personality'].map({'Extrovert': 1, 'Introvert': 0})
    X_test = test_data[feature_cols]
    
    # Target Encoderの初期化
    target_encoder = AdvancedTargetEncoder(
        smoothing_alpha=50,  # より保守的な値
        n_splits=5,
        random_state=42
    )
    
    # 訓練データへのTarget Encoding適用
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    
    # 3. テストデータの簡易処理
    print("\n3. テストデータ処理中...")
    
    # 簡易実装：訓練データから得られたエンコード値の平均を使用
    X_test_encoded = X_test.copy()
    
    # 各カテゴリカル特徴量について
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for feature in categorical_features:
        # 訓練データのエンコード値から統計を計算
        encoded_cols = [col for col in X_train_encoded.columns if col.startswith(f'{feature}_') and 'encoded' in col]
        
        for encoded_col in encoded_cols:
            # 各カテゴリの平均エンコード値を計算
            encoding_dict = X_train_encoded.groupby(X_train[feature])[encoded_col].mean().to_dict()
            global_mean = X_train_encoded[encoded_col].mean()
            
            # テストデータに適用
            X_test_encoded[encoded_col] = X_test[feature].map(encoding_dict).fillna(global_mean)
    
    # 4. 最終データ作成
    print("\n4. 最終データ作成中...")
    
    # id列とPersonality列を追加
    train_final = pd.concat([
        train_data[['id', 'Personality']], 
        X_train_encoded
    ], axis=1)
    
    test_final = pd.concat([
        test_data[['id']],
        X_test_encoded
    ], axis=1)
    
    # 5. 保存
    print("\n5. データ保存中...")
    train_path = '/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2b_train_features.csv'
    test_path = '/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2b_test_features.csv'
    
    train_final.to_csv(train_path, index=False)
    test_final.to_csv(test_path, index=False)
    
    print(f"✅ Phase 2b特徴量作成完了")
    print(f"   訓練データ: {train_final.shape} → {train_path}")
    print(f"   テストデータ: {test_final.shape} → {test_path}")
    
    # 6. 特徴量統計
    print(f"\n📊 特徴量統計:")
    print(f"   元特徴量数: {len(feature_cols)}")
    print(f"   拡張後特徴量数: {train_final.shape[1] - 2}")  # id, Personalityを除く
    print(f"   追加特徴量数: {train_final.shape[1] - 2 - len(feature_cols)}")
    
    return train_final, test_final

def main():
    """メイン実行関数"""
    
    # Phase 2b特徴量作成
    train_data, test_data = create_phase2b_features()
    
    print(f"\n" + "="*50)
    print("Phase 2b Target Encoding実装完了")
    print("="*50)
    print("🎯 次のステップ: CV評価とSubmission作成")
    
    return train_data, test_data

if __name__ == "__main__":
    main()