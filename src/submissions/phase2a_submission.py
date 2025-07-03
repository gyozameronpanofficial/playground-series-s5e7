"""
Phase 2a提出ファイル作成

高次n-gram + TF-IDF特徴量での最終予測
CV結果: 0.968851 (期待値未達だが提出して実際の効果を確認)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def create_phase2a_final_model():
    """Phase 2a最終提出用モデル"""
    
    models = [
        ('lgb', lgb.LGBMClassifier(
            objective='binary', num_leaves=31, learning_rate=0.02,
            n_estimators=1500, random_state=42, verbosity=-1
        )),
        ('xgb', xgb.XGBClassifier(
            objective='binary:logistic', max_depth=6, learning_rate=0.02,
            n_estimators=1500, random_state=42, verbosity=0
        )),
        ('cat', CatBoostClassifier(
            objective='Logloss', depth=6, learning_rate=0.02,
            iterations=1500, random_seed=42, verbose=False
        )),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ]
    
    return VotingClassifier(estimators=models, voting='soft')

def main():
    """メイン実行関数"""
    print("=== Phase 2a 提出ファイル作成 ===")
    
    # 1. データ読み込み
    print("1. Phase 2a特徴量データ読み込み中...")
    try:
        train_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2a_train_features.csv')
        test_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2a_test_features.csv')
        
        print(f"   訓練データ形状: {train_data.shape}")
        print(f"   テストデータ形状: {test_data.shape}")
        
    except FileNotFoundError as e:
        print(f"   エラー: データファイルが見つかりません - {e}")
        return
    
    # 2. データ前処理
    print("2. データ前処理中...")
    
    # 特徴量とターゲット分離
    feature_cols = [col for col in train_data.columns if col not in ['id', 'Personality']]
    
    # カテゴリカル特徴量のエンコーディング
    train_processed = train_data[feature_cols].copy()
    test_processed = test_data[feature_cols].copy()
    
    label_encoders = {}
    for col in feature_cols:
        if train_processed[col].dtype == 'object':
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
    
    print(f"   使用特徴量数: {X_train.shape[1]}")
    print(f"   エンコードした特徴量数: {len(label_encoders)}")
    
    # 3. モデル訓練
    print("3. Phase 2aモデル訓練中...")
    final_model = create_phase2a_final_model()
    final_model.fit(X_train, y_train)
    
    # 4. 予測実行
    print("4. テストデータ予測中...")
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_predictions = final_model.predict(X_test)
    
    # 5. 提出ファイル作成
    print("5. 提出ファイル作成中...")
    submission_df = pd.DataFrame({
        'id': test_ids,
        'Personality': ['Extrovert' if pred == 1 else 'Introvert' for pred in test_predictions]
    })
    
    # 統計情報
    extrovert_count = np.sum(test_predictions == 1)
    introvert_count = np.sum(test_predictions == 0)
    avg_confidence = np.mean(np.maximum(test_proba, 1 - test_proba))
    
    print(f"\\n予測統計:")
    print(f"  Extrovert: {extrovert_count} ({extrovert_count/len(test_predictions)*100:.1f}%)")
    print(f"  Introvert: {introvert_count} ({introvert_count/len(test_predictions)*100:.1f}%)")
    print(f"  平均信頼度: {avg_confidence:.4f}")
    
    # 保存
    submission_path = '/Users/osawa/kaggle/playground-series-s5e7/submissions/phase2a_ngram_tfidf_submission.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\\n✅ Phase 2a提出ファイル作成完了!")
    print(f"   ファイル: {submission_path}")
    print(f"   CVスコア: 0.968851 (期待値未達)")
    print(f"   備考: CV-PB gapでの実際効果を確認")
    
    # 提出ファイルサンプル表示
    print(f"\\n提出ファイルサンプル:")
    print(submission_df.head(10))
    
    return submission_df

if __name__ == "__main__":
    main()