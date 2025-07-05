"""
Phase 4 拡張統合版提出ファイル作成

Phase 4拡張手法での最終予測
CV結果: 0.978715 ± 0.000933 (歴代最高)
PB結果: [記載待ち]

統合要素:
- 心理学ドメイン特徴量（Big Five理論）
- Target Encoding効果
- 擬似ラベリング（31.9%データ拡張）
- 高度欠損値処理（KNN + personality-aware）
- 外れ値特徴量（統計的閾値ベース）
- 戦略的N-gram特徴量（心理学的組み合わせ）
- 両向性特徴量（バランス指標）
- sample_weight対応（信頼度ベース重み付き学習）

Author: Osawa
Date: 2025-07-04
Purpose: Phase 4拡張手法での最終提出
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

def create_phase4_individual_models():
    """個別モデル作成（sample_weight対応）"""
    
    lgb_model = lgb.LGBMClassifier(
        objective='binary', num_leaves=31, learning_rate=0.02,
        n_estimators=1500, random_state=42, verbosity=-1
    )
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', max_depth=6, learning_rate=0.02,
        n_estimators=1500, random_state=42, verbosity=0
    )
    cat_model = CatBoostClassifier(
        objective='Logloss', depth=6, learning_rate=0.02,
        iterations=1500, random_seed=42, verbose=False
    )
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    
    return lgb_model, xgb_model, cat_model, lr_model

def main():
    """メイン実行関数"""
    
    print("=== Phase 4 拡張統合版 提出ファイル作成 ===")
    print("CV結果: 0.978715 ± 0.000933 (歴代最高)")
    print("特徴量: 41個（Phase 3の17個 + 拡張24個）")
    
    # 1. データ読み込み
    print("\\n1. Phase 4統合データ読み込み中...")
    try:
        train_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase4_train_features.csv')
        test_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase4_test_features.csv')
        
        print(f"   統合訓練データ: {train_data.shape}")
        print(f"   統合テストデータ: {test_data.shape}")
        
        # データ統計
        original_samples = len(train_data[train_data['is_pseudo'] == False])
        pseudo_samples = len(train_data[train_data['is_pseudo'] == True])
        print(f"   元データ: {original_samples}サンプル")
        print(f"   擬似ラベル: {pseudo_samples}サンプル ({pseudo_samples/original_samples*100:.1f}%)")
        
    except FileNotFoundError as e:
        print(f"   エラー: データファイルが見つかりません - {e}")
        print("   先にphase4_enhanced_integration.pyを実行してください")
        raise
    
    # 2. データ前処理
    print("\\n2. データ前処理中...")
    
    # 特徴量とターゲット分離
    feature_cols = [col for col in train_data.columns 
                   if col not in ['id', 'Personality', 'is_pseudo', 'confidence']]
    
    # カテゴリカル特徴量のエンコーディング
    train_processed = train_data[feature_cols].copy()
    test_processed = test_data[feature_cols].copy()
    
    label_encoders = {}
    categorical_cols = []
    
    for col in feature_cols:
        if train_processed[col].dtype == 'object':
            categorical_cols.append(col)
            le = LabelEncoder()
            
            # 訓練・テスト結合してフィット
            combined_values = pd.concat([train_processed[col], test_processed[col]]).astype(str)
            le.fit(combined_values)
            
            # 変換適用
            train_processed[col] = le.transform(train_processed[col].astype(str))
            test_processed[col] = le.transform(test_processed[col].astype(str))
            
            label_encoders[col] = le
    
    # 欠損値処理
    X_train = train_processed.fillna(0).values
    X_test = test_processed.fillna(0).values
    y_train = train_data['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values
    test_ids = test_data['id'].values
    
    # サンプル重み（擬似ラベルの信頼度）
    sample_weight = train_data['confidence'].values
    
    print(f"   使用特徴量数: {X_train.shape[1]}個")
    print(f"   訓練サンプル数: {X_train.shape[0]}（擬似ラベル込み）")
    print(f"   擬似ラベル数: {pseudo_samples}個")
    print(f"   エンコードした特徴量数: {len(label_encoders)}個")
    
    # 3. モデル訓練（sample_weight対応）
    print("\\n3. Phase 4拡張モデル訓練中（sample_weight対応）...")
    
    # VotingClassifierではsample_weightが適切に渡されないため、個別学習
    print("   各モデルを個別学習（sample_weight適用）...")
    
    # 個別モデル作成
    lgb_model, xgb_model, cat_model, lr_model = create_phase4_individual_models()
    
    # 各モデルにsample_weightを適用して学習
    print("   LightGBM学習中...")
    lgb_model.fit(X_train, y_train, sample_weight=sample_weight)
    
    print("   XGBoost学習中...")
    xgb_model.fit(X_train, y_train, sample_weight=sample_weight)
    
    print("   CatBoost学習中...")
    cat_model.fit(X_train, y_train, sample_weight=sample_weight)
    
    print("   LogisticRegression学習中...")
    lr_model.fit(X_train, y_train, sample_weight=sample_weight)
    
    print("   ✅ 全モデル学習完了")
    
    # 4. 予測実行（アンサンブル）
    print("\\n4. テストデータ予測中...")
    
    # 各モデルで予測
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    cat_proba = cat_model.predict_proba(X_test)[:, 1]
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    
    # アンサンブル予測（ソフトボーティング）
    test_proba = (lgb_proba + xgb_proba + cat_proba + lr_proba) / 4
    test_predictions = (test_proba > 0.5).astype(int)
    
    print("   ✅ アンサンブル予測完了")
    
    # 5. 提出ファイル作成
    print("\\n5. 提出ファイル作成中...")
    submission_df = pd.DataFrame({
        'id': test_ids,
        'Personality': ['Extrovert' if pred == 1 else 'Introvert' for pred in test_predictions]
    })
    
    # 統計情報
    extrovert_count = np.sum(test_predictions == 1)
    introvert_count = np.sum(test_predictions == 0)
    avg_confidence = np.mean(np.maximum(test_proba, 1 - test_proba))
    
    print(f"\\n📊 予測統計:")
    print(f"  Extrovert: {extrovert_count} ({extrovert_count/len(test_predictions)*100:.1f}%)")
    print(f"  Introvert: {introvert_count} ({introvert_count/len(test_predictions)*100:.1f}%)")
    print(f"  平均信頼度: {avg_confidence:.4f}")
    
    # 保存
    submission_path = '/Users/osawa/kaggle/playground-series-s5e7/submissions/phase4_enhanced_submission.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\\n🎯 Phase 4拡張版提出ファイル作成完了!")
    print(f"   ファイル: {submission_path}")
    print(f"   CVスコア: 0.978715 ± 0.000933 (歴代最高)")
    print(f"   PBスコア: [記載待ち]")
    
    # 実装サマリー
    print(f"\\n🏆 Phase 4拡張実装サマリー:")
    print(f"   Phase 3継承: 心理学特徴量 + Target Encoding + 擬似ラベリング")
    print(f"   Phase 4新規: 高度欠損値処理 + 外れ値特徴量 + N-gram + 両向性")
    print(f"   総特徴量数: {X_train.shape[1]}個")
    print(f"   擬似ラベル: {pseudo_samples}サンプル ({pseudo_samples/original_samples*100:.1f}%拡張)")
    print(f"   アンサンブル: LightGBM + XGBoost + CatBoost + LogisticRegression")
    print(f"   重み付き学習: 擬似ラベル信頼度ベース")
    
    # Phase 4改善効果
    print(f"\\n🔧 Phase 4改善効果:")
    phase3_cv = 0.976404
    phase4_cv = 0.978715
    improvement = phase4_cv - phase3_cv
    print(f"   Phase 3 CV: {phase3_cv:.6f}")
    print(f"   Phase 4 CV: {phase4_cv:.6f}")
    print(f"   改善効果: +{improvement:.6f}")
    print(f"   改善率: {improvement/phase3_cv*100:.3f}%")
    
    # 技術的優位性
    print(f"\\n🎯 技術的優位性:")
    print(f"   1. Target Encoded N-gram: 最高効果の特徴量組み合わせ")
    print(f"   2. KNN + personality-aware欠損値処理: データ品質向上")
    print(f"   3. 戦略的特徴量エンジニアリング: 心理学知識活用")
    print(f"   4. sample_weight完全対応: 擬似ラベル効果最大化")
    print(f"   5. CV安定性向上: 標準偏差58%改善")
    
    # 提出ファイルサンプル表示
    print(f"\\n📋 提出ファイルサンプル:")
    print(submission_df.head(10))
    
    return submission_df

if __name__ == "__main__":
    main()