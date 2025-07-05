"""
Phase 4 拡張統合実装 - CV 0.980+ 目標

Phase 3の統合手法（CV 0.976404）に以下の機能を追加:
1. 外れ値特徴量（統計的閾値ベース）
2. 戦略的N-gram特徴量（心理学的組み合わせ）
3. 両向性特徴量（バランス指標）
4. 高度欠損値処理（personality-aware）

期待効果: CV 0.976404 → 0.978404 (+0.002000)

Author: Osawa
Date: 2025-07-04
Purpose: GM分析を踏まえた拡張実装
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class EnhancedHybridFeatureEngineer:
    """Phase 4 拡張統合特徴量エンジニア"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.target_encoders = {}
        self.knn_imputer = KNNImputer(n_neighbors=5)
        
    def create_psychological_features(self, df):
        """心理学ドメイン特徴量の作成（Phase 3から継承）"""
        
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
        
        print(f"     心理学特徴量: 6個")
        return df_processed
    
    def create_outlier_features(self, df):
        """外れ値特徴量の作成（GM分析より）"""
        
        print("   外れ値特徴量作成中...")
        
        df_processed = df.copy()
        
        # 1. Time_spent_Alone の高値フラグ（94% Introvert精度）
        if 'Time_spent_Alone' in df_processed.columns:
            alone_mean = df_processed['Time_spent_Alone'].mean()
            alone_std = df_processed['Time_spent_Alone'].std()
            alone_threshold = alone_mean + 2 * alone_std
            df_processed['extreme_alone_flag'] = (df_processed['Time_spent_Alone'] > alone_threshold).astype(int)
        
        # 2. Stage_fear の欠損値フラグ（10.22%欠損率）
        if 'Stage_fear' in df.columns:
            df_processed['stage_fear_missing'] = df['Stage_fear'].isna().astype(int)
        
        # 3. 極端な内向的行動パターン
        if all(col in df_processed.columns for col in ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside']):
            alone_high = df_processed['Time_spent_Alone'] > df_processed['Time_spent_Alone'].quantile(0.95)
            social_low = df_processed['Social_event_attendance'] <= 1
            outside_low = df_processed['Going_outside'] <= 1
            df_processed['extreme_introvert_pattern'] = (alone_high & social_low & outside_low).astype(int)
        
        # 4. 極端な外向的行動パターン
        if all(col in df_processed.columns for col in ['Social_event_attendance', 'Friends_circle_size', 'Post_frequency']):
            social_high = df_processed['Social_event_attendance'] >= 2
            friends_high = df_processed['Friends_circle_size'] >= 2
            post_high = df_processed['Post_frequency'] >= 2
            df_processed['extreme_extrovert_pattern'] = (social_high & friends_high & post_high).astype(int)
        
        # 5. 両極端フラグ
        if all(col in df_processed.columns for col in ['extroversion_score', 'introversion_score']):
            ext_extreme = df_processed['extroversion_score'] > df_processed['extroversion_score'].quantile(0.9)
            int_extreme = df_processed['introversion_score'] > df_processed['introversion_score'].quantile(0.9)
            df_processed['personality_extreme_flag'] = (ext_extreme | int_extreme).astype(int)
        
        print(f"     外れ値特徴量: 5個")
        return df_processed
    
    def create_strategic_ngrams(self, df):
        """戦略的N-gram特徴量の作成（心理学的組み合わせ）"""
        
        print("   戦略的N-gram特徴量作成中...")
        
        df_processed = df.copy()
        
        # 心理学的に意味のある2-gram組み合わせ
        social_combos = [
            ('Social_event_attendance', 'Friends_circle_size'),
            ('Going_outside', 'Post_frequency'),
            ('Social_event_attendance', 'Going_outside'),
            ('Friends_circle_size', 'Post_frequency')
        ]
        
        introvert_combos = [
            ('Time_spent_Alone', 'Stage_fear'),
            ('Time_spent_Alone', 'Drained_after_socializing'),
            ('Stage_fear', 'Drained_after_socializing'),
            ('Time_spent_Alone', 'Social_event_attendance')  # 対照的組み合わせ
        ]
        
        # 2-gram特徴量作成
        for col1, col2 in social_combos + introvert_combos:
            if col1 in df_processed.columns and col2 in df_processed.columns:
                df_processed[f"{col1}_{col2}_combo"] = (
                    df_processed[col1].astype(str) + "_" + df_processed[col2].astype(str)
                )
        
        print(f"     戦略的N-gram特徴量: 8個")
        return df_processed
    
    def create_ambivert_features(self, df):
        """両向性特徴量の作成（バランス指標）"""
        
        print("   両向性特徴量作成中...")
        
        df_processed = df.copy()
        
        # 社会性・内向性スコアが必要
        if 'extroversion_score' in df_processed.columns and 'introversion_score' in df_processed.columns:
            
            # 1. 両向性スコア（バランス度）
            score_diff = abs(df_processed['extroversion_score'] - df_processed['introversion_score'])
            df_processed['ambivert_score'] = 1 / (1 + score_diff)
            
            # 2. 極端度スコア
            df_processed['extreme_score'] = score_diff
            
            # 3. 両向性フラグ（中程度のバランス）
            balance_threshold = 0.5
            df_processed['ambivert_flag'] = (df_processed['ambivert_score'] > balance_threshold).astype(int)
            
        print(f"     両向性特徴量: 3個")
        return df_processed
    
    def advanced_missing_value_handling(self, train_df, test_df):
        """高度欠損値処理（personality-aware + KNN）"""
        
        print("   高度欠損値処理中...")
        
        train_processed = train_df.copy()
        test_processed = test_df.copy()
        
        # 1. 数値特徴量のKNN補完
        numeric_cols = []
        for col in train_processed.columns:
            if col not in ['id', 'Personality', 'is_pseudo', 'confidence'] and pd.api.types.is_numeric_dtype(train_processed[col]):
                numeric_cols.append(col)
        
        if numeric_cols:
            # KNN補完
            train_numeric = train_processed[numeric_cols].copy()
            test_numeric = test_processed[numeric_cols].copy()
            
            # 欠損値がある場合のみKNN適用
            if train_numeric.isnull().sum().sum() > 0:
                knn_imputer = KNNImputer(n_neighbors=5)
                train_processed[numeric_cols] = knn_imputer.fit_transform(train_numeric)
                test_processed[numeric_cols] = knn_imputer.transform(test_numeric)
        
        # 2. カテゴリ特徴量の性格ベース補完
        if 'Personality' in train_processed.columns:
            categorical_cols = []
            for col in train_processed.columns:
                if col not in ['id', 'Personality', 'is_pseudo', 'confidence'] and train_processed[col].dtype == 'object':
                    categorical_cols.append(col)
            
            for col in categorical_cols:
                for personality in ['Extrovert', 'Introvert']:
                    # 性格別の最頻値で補完
                    personality_data = train_processed[train_processed['Personality'] == personality]
                    if len(personality_data) > 0:
                        mode_val = personality_data[col].mode()
                        if len(mode_val) > 0:
                            # 訓練データの補完
                            mask = (train_processed['Personality'] == personality) & (train_processed[col].isna())
                            train_processed.loc[mask, col] = mode_val.iloc[0]
                            
                            # テストデータの補完（全体のパターンを使用）
                            test_mask = test_processed[col].isna()
                            test_processed.loc[test_mask, col] = mode_val.iloc[0]
        
        # 3. 残りの欠損値を0で埋める
        train_processed = train_processed.fillna(0)
        test_processed = test_processed.fillna(0)
        
        print(f"     高度欠損値処理完了")
        return train_processed, test_processed
    
    def apply_target_encoding(self, train_df, test_df, target_col='Personality'):
        """Target Encodingの適用（Phase 3から継承）"""
        
        print("   Target Encoding適用中...")
        
        # カテゴリカル特徴量の特定
        categorical_features = []
        for col in train_df.columns:
            if col not in ['id', 'Personality', 'is_pseudo', 'confidence'] and train_df[col].dtype == 'object':
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
                train_fold_feature = train_df.iloc[train_idx][feature].reset_index(drop=True)
                train_fold_target = y_train.iloc[train_idx].reset_index(drop=True)
                
                # カテゴリ別平均計算
                encoding_dict = {}
                for cat_val in train_fold_feature.unique():
                    mask = (train_fold_feature == cat_val)
                    if mask.sum() > 0:
                        encoding_dict[cat_val] = train_fold_target[mask].mean()
                
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
        
        print(f"     Target Encoding特徴量: {len(categorical_features)}個")
        return train_encoded, test_encoded
    
    def create_statistical_features(self, df):
        """統計的特徴量の作成（Phase 3から継承）"""
        
        print("   統計的特徴量作成中...")
        
        df_processed = df.copy()
        
        # 数値特徴量の統計
        numeric_cols = []
        for col in df_processed.columns:
            if col not in ['id', 'Personality', 'is_pseudo', 'confidence'] and pd.api.types.is_numeric_dtype(df_processed[col]):
                numeric_cols.append(col)
        
        if len(numeric_cols) > 1:
            numeric_data = df_processed[numeric_cols]
            
            # 特徴量統計
            df_processed['feature_mean'] = numeric_data.mean(axis=1)
            df_processed['feature_std'] = numeric_data.std(axis=1)
            df_processed['feature_max'] = numeric_data.max(axis=1)
            df_processed['feature_min'] = numeric_data.min(axis=1)
            
            print(f"     統計的特徴量: 4個")
        else:
            print("     数値特徴量不足、統計的特徴量スキップ")
        
        return df_processed
    
    def create_enhanced_features(self, train_df, test_df):
        """拡張統合特徴量作成メイン関数"""
        
        print("=== Phase 4 拡張統合特徴量エンジニアリング実行 ===")
        print(f"元データ形状 - 訓練: {train_df.shape}, テスト: {test_df.shape}")
        
        # 1. 高度欠損値処理（最初に実行）
        train_processed, test_processed = self.advanced_missing_value_handling(train_df, test_df)
        
        # 2. 心理学ドメイン特徴量（Phase 3から）
        train_psych = self.create_psychological_features(train_processed)
        test_psych = self.create_psychological_features(test_processed)
        
        # 3. 外れ値特徴量（新規）
        train_outlier = self.create_outlier_features(train_psych)
        test_outlier = self.create_outlier_features(test_psych)
        
        # 4. 戦略的N-gram特徴量（新規）
        train_ngram = self.create_strategic_ngrams(train_outlier)
        test_ngram = self.create_strategic_ngrams(test_outlier)
        
        # 5. 両向性特徴量（新規）
        train_ambi = self.create_ambivert_features(train_ngram)
        test_ambi = self.create_ambivert_features(test_ngram)
        
        # 6. Target Encoding（Phase 3から）
        train_encoded, test_encoded = self.apply_target_encoding(train_ambi, test_ambi)
        
        # 7. 統計的特徴量（Phase 3から）
        train_final = self.create_statistical_features(train_encoded)
        test_final = self.create_statistical_features(test_encoded)
        
        print(f"最終データ形状 - 訓練: {train_final.shape}, テスト: {test_final.shape}")
        print(f"特徴量増加数: {train_final.shape[1] - train_df.shape[1]}個")
        
        # 特徴量カテゴリ別サマリー
        print(f"\\n🔧 特徴量カテゴリ別サマリー:")
        print(f"   Phase 3継承: 心理学(6) + Target Encoding + 統計(4) = 約17個")
        print(f"   Phase 4新規: 外れ値(5) + N-gram(8) + 両向性(3) = 16個")
        print(f"   合計予想: 約33個特徴量")
        
        return train_final, test_final

def create_pseudo_labeled_data(train_features, test_features, confidence_threshold=0.85):
    """擬似ラベリングデータの作成（引数変更）"""
    
    print("=== 擬似ラベリング実行 ===")
    
    # 前処理
    feature_cols = [col for col in train_features.columns if col not in ['id', 'Personality', 'is_pseudo', 'confidence']]
    
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
    
    print("=== Phase 4 拡張統合実装 ===")
    print("Phase 3の成功を基盤として、GM分析による改善を追加実装")
    
    # 1. 元データ読み込み
    print("\\n1. 元データ読み込み中...")
    train_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/raw/train.csv')
    test_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/raw/test.csv')
    
    print(f"   訓練データ: {train_data.shape}")
    print(f"   テストデータ: {test_data.shape}")
    
    # 2. 拡張統合特徴量エンジニアリング
    print("\\n2. 拡張統合特徴量エンジニアリング実行中...")
    feature_engineer = EnhancedHybridFeatureEngineer()
    train_features, test_features = feature_engineer.create_enhanced_features(train_data, test_data)
    
    # 3. 擬似ラベリング
    print("\\n3. 擬似ラベリング実行中...")
    augmented_train = create_pseudo_labeled_data(train_features, test_features)
    
    # 4. データ保存
    print("\\n4. データ保存中...")
    
    # 訓練データ保存
    train_path = '/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase4_train_features.csv'
    augmented_train.to_csv(train_path, index=False)
    
    # テストデータ保存
    test_path = '/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase4_test_features.csv'
    test_features.to_csv(test_path, index=False)
    
    print(f"✅ Phase 4 拡張データ作成完了!")
    print(f"   訓練データ: {augmented_train.shape} → {train_path}")
    print(f"   テストデータ: {test_features.shape} → {test_path}")
    
    # 5. 統計サマリー
    print(f"\\n📊 Phase 4 実装サマリー:")
    print(f"   Phase 3特徴量数: 約17個")
    print(f"   Phase 4特徴量数: {test_features.shape[1] - 1}個")  # idを除く
    print(f"   追加特徴量数: {test_features.shape[1] - 18}個")  # id + 17個を除く
    print(f"   擬似ラベル拡張率: {(len(augmented_train) - len(train_features))/len(train_features)*100:.1f}%")
    
    # 6. 改善要素詳細
    print(f"\\n🔧 Phase 4 改善要素:")
    print(f"   1. 高度欠損値処理: KNN + personality-aware補完")
    print(f"   2. 外れ値特徴量: 統計的閾値ベース 5個")
    print(f"   3. 戦略的N-gram: 心理学的組み合わせ 8個")
    print(f"   4. 両向性特徴量: バランス指標 3個")
    print(f"   5. Phase 3継承: 心理学+Target Encoding+統計+擬似ラベル")
    
    print(f"\\n🎯 Phase 4 期待効果:")
    print(f"   Phase 3 CVベースライン: 0.976404")
    print(f"   Phase 4 CV予想: 0.978404 (+0.002000)")
    print(f"   Phase 4 PB予想: 0.977000+ (GM超越)")
    
    print(f"\\n🚀 次のステップ:")
    print(f"   1. Phase 4 CV評価実行")
    print(f"   2. Phase 3 vs Phase 4 性能比較")
    print(f"   3. 最適手法の決定")
    
    return augmented_train, test_features

if __name__ == "__main__":
    main()