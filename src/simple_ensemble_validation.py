"""
シンプルなアンサンブル検証とCV結果取得

複雑な最適化の前に基本性能を確認
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

def create_simple_ensemble():
    """シンプルなアンサンブルモデル構築"""
    
    # ベースモデル（GMベースラインと同等構成）
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
    
    # Soft Voting（確率平均）
    ensemble = VotingClassifier(estimators=models, voting='soft')
    return ensemble

def evaluate_model_performance(X, y, model, model_name="Model"):
    """モデル性能のクロスバリデーション評価"""
    
    print(f"=== {model_name} 評価 ===")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.6f}")
    print(f"Std CV Score: {cv_scores.std():.6f}")
    print(f"95% CI: {cv_scores.mean():.6f} +/- {cv_scores.std()*2:.6f}")
    
    return cv_scores.mean(), cv_scores.std()

def main():
    """メイン実行関数"""
    print("=== シンプルアンサンブル検証 ===")
    
    # 擬似ラベル拡張データ読み込み
    print("1. データ読み込み中...")
    augmented_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/pseudo_labeled_train.csv')
    
    feature_cols = [col for col in augmented_data.columns 
                   if col not in ['Personality', 'sample_weight', 'is_pseudo_label']]
    
    X = augmented_data[feature_cols].fillna(0).values
    y = augmented_data['Personality'].values
    
    print(f"データ形状: {X.shape}")
    print(f"クラス分布: {np.bincount(y)}")
    
    # 元データのみでの評価
    print("\n2. 元データのみでの評価...")
    original_mask = augmented_data['is_pseudo_label'] == 0
    X_original = X[original_mask]
    y_original = y[original_mask]
    
    ensemble_original = create_simple_ensemble()
    original_mean, original_std = evaluate_model_performance(
        X_original, y_original, ensemble_original, "Original Data Only"
    )
    
    # 擬似ラベル込みでの評価
    print("\n3. 擬似ラベル込みでの評価...")
    ensemble_augmented = create_simple_ensemble()
    augmented_mean, augmented_std = evaluate_model_performance(
        X, y, ensemble_augmented, "Pseudo-labeled Augmented Data"
    )
    
    # 個別モデル評価
    print("\n4. 個別モデル評価...")
    
    individual_models = {
        'LightGBM': lgb.LGBMClassifier(
            objective='binary', num_leaves=31, learning_rate=0.02,
            n_estimators=1500, random_state=42, verbosity=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            objective='binary:logistic', max_depth=6, learning_rate=0.02,
            n_estimators=1500, random_state=42, verbosity=0
        ),
        'CatBoost': CatBoostClassifier(
            objective='Logloss', depth=6, learning_rate=0.02,
            iterations=1500, random_seed=42, verbose=False
        ),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    individual_results = {}
    for name, model in individual_models.items():
        mean_score, std_score = evaluate_model_performance(X, y, model, name)
        individual_results[name] = {'mean': mean_score, 'std': std_score}
    
    # 結果サマリー
    print("\n" + "="*60)
    print("結果サマリー")
    print("="*60)
    print(f"元データのみ:        {original_mean:.6f} +/- {original_std:.6f}")
    print(f"擬似ラベル込み:      {augmented_mean:.6f} +/- {augmented_std:.6f}")
    print(f"改善効果:           {augmented_mean - original_mean:+.6f}")
    print()
    print("個別モデル性能:")
    for name, result in individual_results.items():
        print(f"  {name:15}: {result['mean']:.6f} +/- {result['std']:.6f}")
    
    # CVスコア保存
    results = {
        'original_data_cv': original_mean,
        'augmented_data_cv': augmented_mean,
        'improvement': augmented_mean - original_mean,
        'individual_models': individual_results
    }
    
    import json
    with open('/Users/osawa/kaggle/playground-series-s5e7/data/processed/cv_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ CV結果保存完了: cv_results.json")
    
    return results

if __name__ == "__main__":
    main()