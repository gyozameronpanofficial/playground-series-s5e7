"""
Phase 2b + フェーズ1+2 統合実装

Phase 2bの成功（PB 0.975708 = GM基準達成）を踏まえ、
フェーズ1+2の心理学特徴量+擬似ラベリング（CV 0.974211）と統合
GM超越を確実にする統合手法の実装

統合要素:
1. 心理学ドメイン特徴量（Big Five理論ベース）
2. Target Encoding（実証済み有効性）
3. 擬似ラベリング（データ拡張効果）

期待効果: CV 0.975000+ & PB 0.976500+

Author: Osawa
Date: 2025-07-03
Purpose: GM超越確実な統合手法の実装
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

class HybridFeatureEngineer:
    """統合特徴量エンジニア"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.target_encoders = {}
        
    def create_psychological_features(self, df):
        """心理学ドメイン特徴量の作成（フェーズ1から）"""
        
        print("   心理学ドメイン特徴量作成中...")
        
        # Big Five理論ベーススコア
        extroversion_features = ['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
        introversion_features = ['Time_spent_Alone', 'Stage_fear', 'Drained_after_socializing']
        
        # 数値化関数
        def convert_to_numeric(series):
            if series.dtype == 'object':
                mapping = {'No': 0, 'Sometimes': 1, 'Yes': 2}
                if series.name in ['Friends_circle_size', 'Post_frequency']:
                    mapping = {'Small/Low': 0, 'Medium': 1, 'Large/High': 2}
                return series.map(mapping).fillna(1)
            return series
        
        df_processed = df.copy()
        
        # 全特徴量を数値化
        for col in df_processed.columns:
            if col not in ['id', 'Personality']:
                df_processed[col] = convert_to_numeric(df_processed[col])
        
        # 外向性スコア
        extroversion_cols = [col for col in extroversion_features if col in df_processed.columns]
        df_processed['extroversion_score'] = df_processed[extroversion_cols].mean(axis=1)
        
        # 内向性スコア
        introversion_cols = [col for col in introversion_features if col in df_processed.columns]
        df_processed['introversion_score'] = df_processed[introversion_cols].mean(axis=1)
        
        # 社交バランス
        df_processed['social_balance'] = df_processed['extroversion_score'] - df_processed['introversion_score']
        
        # 社交疲労度
        if 'Drained_after_socializing' in df_processed.columns and 'Social_event_attendance' in df_processed.columns:
            df_processed['social_fatigue'] = df_processed['Drained_after_socializing'] * df_processed['Social_event_attendance']
        
        # 社交積極度
        if 'Going_outside' in df_processed.columns and 'Friends_circle_size' in df_processed.columns:
            df_processed['social_proactivity'] = df_processed['Going_outside'] * df_processed['Friends_circle_size']
        
        # 孤独嗜好度
        if 'Time_spent_Alone' in df_processed.columns and 'Stage_fear' in df_processed.columns:
            df_processed['solitude_preference'] = df_processed['Time_spent_Alone'] * (2 - df_processed['Stage_fear'])
        
        print(f"     心理学特徴量追加数: 6個")
        return df_processed
    
    def apply_target_encoding(self, train_df, test_df, target_col='Personality'):
        """Target Encodingの適用（Phase 2bから）"""
        
        print("   Target Encoding適用中...")
        
        # カテゴリカル特徴量の特定
        categorical_features = []
        for col in train_df.columns:
            if col not in ['id', 'Personality'] and train_df[col].dtype == 'object':
                categorical_features.append(col)
        
        if not categorical_features:
            print("     カテゴリカル特徴量なし、Target Encodingスキップ")
            return train_df.copy(), test_df.copy()
        
        print(f"     対象カテゴリカル特徴量: {categorical_features}")
        
        # 数値化
        y_train = train_df[target_col].map({'Extrovert': 1, 'Introvert': 0})
        
        train_encoded = train_df.copy()
        test_encoded = test_df.copy()
        
        # CV内でのTarget Encoding
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for feature in categorical_features:
            print(f"       {feature}のTarget Encoding...")
            
            # CV内エンコーディング
            encoded_train = np.zeros(len(train_df))
            
            for train_idx, valid_idx in cv.split(train_df, y_train):
                # Train foldでエンコード辞書作成
                train_fold_feature = train_df.iloc[train_idx][feature]
                train_fold_target = y_train.iloc[train_idx]
                
                # カテゴリ別平均計算
                encoding_dict = train_fold_feature.groupby(train_fold_feature).apply(
                    lambda x: train_fold_target.iloc[x.index].mean()
                ).to_dict()
                
                global_mean = train_fold_target.mean()
                
                # Valid foldにエンコード適用
                valid_feature = train_df.iloc[valid_idx][feature]
                encoded_valid = valid_feature.map(encoding_dict).fillna(global_mean)
                encoded_train[valid_idx] = encoded_valid
            
            # 訓練データに追加
            train_encoded[f'{feature}_target_encoded'] = encoded_train
            
            # テストデータ用の全体エンコード辞書作成
            full_encoding_dict = train_df[feature].groupby(train_df[feature]).apply(
                lambda x: y_train.iloc[x.index].mean()
            ).to_dict()
            
            test_encoded[f'{feature}_target_encoded'] = test_df[feature].map(full_encoding_dict).fillna(y_train.mean())
        
        print(f"     Target Encoding特徴量追加数: {len(categorical_features)}個")
        return train_encoded, test_encoded
    
    def create_statistical_features(self, df):
        """統計的特徴量の作成"""
        
        print("   統計的特徴量作成中...")
        
        df_processed = df.copy()
        
        # 数値特徴量の統計
        numeric_cols = []
        for col in df_processed.columns:
            if col not in ['id', 'Personality'] and pd.api.types.is_numeric_dtype(df_processed[col]):
                numeric_cols.append(col)
        
        if len(numeric_cols) > 1:
            numeric_data = df_processed[numeric_cols]
            
            # 特徴量統計
            df_processed['feature_mean'] = numeric_data.mean(axis=1)
            df_processed['feature_std'] = numeric_data.std(axis=1)
            df_processed['feature_max'] = numeric_data.max(axis=1)
            df_processed['feature_min'] = numeric_data.min(axis=1)
            
            print(f"     統計的特徴量追加数: 4個")
        else:
            print("     数値特徴量不足、統計的特徴量スキップ")
        
        return df_processed
    
    def create_hybrid_features(self, train_df, test_df):
        """統合特徴量作成メイン関数"""
        
        print("=== 統合特徴量エンジニアリング実行 ===")
        print(f"元データ形状 - 訓練: {train_df.shape}, テスト: {test_df.shape}")
        
        # 1. 心理学ドメイン特徴量
        train_psych = self.create_psychological_features(train_df)
        test_psych = self.create_psychological_features(test_df)
        
        # 2. Target Encoding
        train_encoded, test_encoded = self.apply_target_encoding(train_psych, test_psych)
        
        # 3. 統計的特徴量
        train_final = self.create_statistical_features(train_encoded)
        test_final = self.create_statistical_features(test_encoded)
        
        print(f"最終データ形状 - 訓練: {train_final.shape}, テスト: {test_final.shape}")
        print(f"特徴量増加数: {train_final.shape[1] - train_df.shape[1]}個")
        
        return train_final, test_final

def create_pseudo_labeled_data(train_df, confidence_threshold=0.85):
    """擬似ラベリングデータの作成（フェーズ2から）"""
    
    print("=== 擬似ラベリング実行 ===")
    
    # テストデータ読み込み
    test_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/raw/test.csv')
    
    # 特徴量エンジニア適用
    feature_engineer = HybridFeatureEngineer()
    train_features, test_features = feature_engineer.create_hybrid_features(train_df, test_data)
    
    # 前処理
    feature_cols = [col for col in train_features.columns if col not in ['id', 'Personality']]
    
    # カテゴリカル特徴量エンコード
    X_train = train_features[feature_cols].copy()
    X_test = test_features[feature_cols].copy()
    
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()
            combined = pd.concat([X_train[col], X_test[col]]).astype(str)
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    
    X_train = X_train.fillna(0).values
    X_test = X_test.fillna(0).values
    y_train = train_features['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values
    
    # 擬似ラベル生成用モデル
    pseudo_models = [
        lgb.LGBMClassifier(n_estimators=1000, random_state=42, verbosity=-1),
        xgb.XGBClassifier(n_estimators=1000, random_state=42, verbosity=0),
        CatBoostClassifier(iterations=1000, random_seed=42, verbose=False)
    ]
    
    # アンサンブル予測
    test_predictions = []
    
    for i, model in enumerate(pseudo_models):
        print(f"   モデル{i+1}訓練中...")
        model.fit(X_train, y_train)
        pred_proba = model.predict_proba(X_test)[:, 1]
        test_predictions.append(pred_proba)
    
    # アンサンブル平均
    ensemble_proba = np.mean(test_predictions, axis=0)
    
    # 高信頼度サンプル選択
    confident_mask = (ensemble_proba >= confidence_threshold) | (ensemble_proba <= 1 - confidence_threshold)
    confident_indices = np.where(confident_mask)[0]
    
    if len(confident_indices) == 0:
        print("   高信頼度サンプルなし、元データのみ返却")
        return train_features
    
    # 擬似ラベル作成
    pseudo_labels = (ensemble_proba[confident_indices] >= 0.5).astype(int)
    pseudo_labels_str = ['Extrovert' if label == 1 else 'Introvert' for label in pseudo_labels]
    
    # 擬似ラベルデータフレーム作成
    pseudo_df = test_features.iloc[confident_indices].copy()
    pseudo_df['Personality'] = pseudo_labels_str
    pseudo_df['is_pseudo'] = True
    pseudo_df['confidence'] = np.maximum(ensemble_proba[confident_indices], 
                                       1 - ensemble_proba[confident_indices])
    
    # 元データにフラグ追加
    train_features['is_pseudo'] = False
    train_features['confidence'] = 1.0
    
    # 結合
    augmented_data = pd.concat([train_features, pseudo_df], ignore_index=True)
    
    print(f"   元データ: {len(train_features)}サンプル")
    print(f"   擬似ラベル: {len(pseudo_df)}サンプル")
    print(f"   総計: {len(augmented_data)}サンプル")
    print(f"   拡張率: {len(pseudo_df)/len(train_features)*100:.1f}%")
    
    return augmented_data

def main():
    """メイン実行関数"""
    
    print("=== Phase 2b + フェーズ1+2 統合実装 ===")
    
    # 1. 元データ読み込み
    print("1. 元データ読み込み中...")
    train_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/raw/train.csv')
    test_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/raw/test.csv')
    
    print(f"   訓練データ: {train_data.shape}")
    print(f"   テストデータ: {test_data.shape}")
    
    # 2. 統合特徴量エンジニアリング
    print("\\n2. 統合特徴量エンジニアリング実行中...")
    feature_engineer = HybridFeatureEngineer()
    train_features, test_features = feature_engineer.create_hybrid_features(train_data, test_data)
    
    # 3. 擬似ラベリング
    print("\\n3. 擬似ラベリング実行中...")
    augmented_train = create_pseudo_labeled_data(train_features)
    
    # 4. データ保存
    print("\\n4. データ保存中...")
    
    # 訓練データ保存
    train_path = '/Users/osawa/kaggle/playground-series-s5e7/data/processed/hybrid_train_features.csv'
    augmented_train.to_csv(train_path, index=False)
    
    # テストデータ保存
    test_path = '/Users/osawa/kaggle/playground-series-s5e7/data/processed/hybrid_test_features.csv'
    test_features.to_csv(test_path, index=False)
    
    print(f"✅ 統合データ作成完了!")
    print(f"   訓練データ: {augmented_train.shape} → {train_path}")
    print(f"   テストデータ: {test_features.shape} → {test_path}")
    
    # 5. 統計サマリー
    print(f"\\n📊 統合実装サマリー:")
    print(f"   元特徴量数: {train_data.shape[1] - 2}")  # id, Personalityを除く
    print(f"   統合特徴量数: {test_features.shape[1] - 1}")  # idを除く
    print(f"   追加特徴量数: {test_features.shape[1] - train_data.shape[1] + 1}")
    print(f"   擬似ラベル拡張率: {(len(augmented_train) - len(train_features))/len(train_features)*100:.1f}%")
    
    print(f"\\n🎯 期待効果:")
    print(f"   CV予想: 0.975000+ (フェーズ1+2の0.974211 + Phase 2b効果)")
    print(f"   PB予想: 0.976500+ (Phase 2bの0.975708 + 統合効果)")
    
    return augmented_train, test_features

if __name__ == "__main__":
    main()