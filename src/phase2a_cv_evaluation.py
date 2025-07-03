"""
Phase 2a CV評価: 高次n-gram + TF-IDF特徴量の性能測定

Phase 2a実装効果の測定と既存手法との比較
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

def create_phase2a_ensemble():
    """Phase 2a用アンサンブルモデル"""
    
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

def evaluate_phase2a_performance():
    """Phase 2a特徴量の性能評価"""
    
    print("=== Phase 2a CV性能評価 ===")
    
    # 1. Phase 2a特徴量データ読み込み
    print("1. Phase 2a特徴量データ読み込み中...")
    try:
        phase2a_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2a_train_features.csv')
        print(f"   Phase 2a特徴量形状: {phase2a_data.shape}")
    except FileNotFoundError:
        print("   エラー: Phase 2a特徴量データが見つかりません")
        return None
    
    # 2. ベースライン（フェーズ1+2）データ読み込み
    print("2. ベースライン特徴量データ読み込み中...")
    try:
        baseline_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/pseudo_labeled_train.csv')
        print(f"   ベースライン特徴量形状: {baseline_data.shape}")
    except FileNotFoundError:
        print("   エラー: ベースライン特徴量データが見つかりません")
        return None
    
    # 3. データ準備
    print("3. 評価用データ準備中...")
    
    # Phase 2a特徴量準備
    phase2a_feature_cols = [col for col in phase2a_data.columns if col not in ['id', 'Personality']]
    
    # カテゴリカル特徴量のエンコーディング
    from sklearn.preprocessing import LabelEncoder
    phase2a_processed = phase2a_data[phase2a_feature_cols].copy()
    
    for col in phase2a_processed.columns:
        if phase2a_processed[col].dtype == 'object':
            le = LabelEncoder()
            phase2a_processed[col] = le.fit_transform(phase2a_processed[col].astype(str))
    
    X_phase2a = phase2a_processed.fillna(0).values
    y_phase2a = phase2a_data['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values
    
    # ベースライン特徴量準備（元データのみ）
    baseline_original_mask = baseline_data['is_pseudo_label'] == 0
    baseline_feature_cols = [col for col in baseline_data.columns 
                           if col not in ['Personality', 'sample_weight', 'is_pseudo_label']]
    X_baseline = baseline_data[baseline_original_mask][baseline_feature_cols].fillna(0).values
    y_baseline = baseline_data[baseline_original_mask]['Personality'].values
    
    print(f"   Phase 2a特徴量数: {X_phase2a.shape[1]}")
    print(f"   ベースライン特徴量数: {X_baseline.shape[1]}")
    print(f"   Phase 2a追加特徴量数: {X_phase2a.shape[1] - 7}")  # 元の7特徴量から増加分
    
    # 4. CV評価実行
    print("4. クロスバリデーション評価実行中...")
    
    ensemble_model = create_phase2a_ensemble()
    cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Phase 2a性能評価
    print("   Phase 2a特徴量でCV評価中...")
    cv_scores_phase2a = cross_val_score(
        ensemble_model, X_phase2a, y_phase2a, 
        cv=cv_folds, scoring='accuracy'
    )
    
    # ベースライン性能評価
    print("   ベースライン特徴量でCV評価中...")
    cv_scores_baseline = cross_val_score(
        ensemble_model, X_baseline, y_baseline, 
        cv=cv_folds, scoring='accuracy'
    )
    
    # 5. 結果分析
    print("\n" + "="*60)
    print("Phase 2a CV評価結果")
    print("="*60)
    
    phase2a_mean = cv_scores_phase2a.mean()
    phase2a_std = cv_scores_phase2a.std()
    baseline_mean = cv_scores_baseline.mean()
    baseline_std = cv_scores_baseline.std()
    improvement = phase2a_mean - baseline_mean
    
    print(f"ベースライン (元データのみ):     {baseline_mean:.6f} +/- {baseline_std:.6f}")
    print(f"Phase 2a (高次n-gram+TF-IDF): {phase2a_mean:.6f} +/- {phase2a_std:.6f}")
    print(f"改善効果:                    {improvement:+.6f}")
    print()
    print("詳細CVスコア:")
    print(f"  ベースライン: {cv_scores_baseline}")
    print(f"  Phase 2a:     {cv_scores_phase2a}")
    
    # 6. 目標達成判定
    print()
    print("目標達成判定:")
    target_improvement = 0.003  # 最低期待効果
    optimal_improvement = 0.005  # 理想期待効果
    
    if improvement >= optimal_improvement:
        print(f"✅ 理想目標達成! 改善効果 {improvement:.6f} >= {optimal_improvement:.6f}")
        status = "excellent"
    elif improvement >= target_improvement:
        print(f"✅ 最低目標達成! 改善効果 {improvement:.6f} >= {target_improvement:.6f}")
        status = "good"
    elif improvement > 0:
        print(f"⚠️ 改善効果あり（目標未達）: {improvement:.6f}")
        status = "partial"
    else:
        print(f"❌ 改善効果なし: {improvement:.6f}")
        status = "failed"
    
    # 7. GM比較
    gm_baseline = 0.975708
    if phase2a_mean > gm_baseline:
        print(f"🎯 GM超越! Phase 2a CV {phase2a_mean:.6f} > GM {gm_baseline:.6f}")
        gm_status = "exceeded"
    else:
        gap_to_gm = gm_baseline - phase2a_mean
        print(f"📊 GM未達: Phase 2a CV {phase2a_mean:.6f} < GM {gm_baseline:.6f} (差: {gap_to_gm:.6f})")
        gm_status = "not_reached"
    
    # 8. 結果保存
    results = {
        'phase2a_cv_mean': phase2a_mean,
        'phase2a_cv_std': phase2a_std,
        'baseline_cv_mean': baseline_mean,
        'baseline_cv_std': baseline_std,
        'improvement': improvement,
        'status': status,
        'gm_status': gm_status,
        'feature_count': X_phase2a.shape[1],
        'cv_scores_phase2a': cv_scores_phase2a.tolist(),
        'cv_scores_baseline': cv_scores_baseline.tolist()
    }
    
    import json
    with open('/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2a_cv_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 結果保存完了: phase2a_cv_results.json")
    
    return results

def main():
    """メイン実行関数"""
    results = evaluate_phase2a_performance()
    
    if results:
        print(f"\n🎯 Phase 2a実装効果: {results['improvement']:+.6f}")
        print(f"📊 次のステップ: 提出ファイル作成 → PB結果確認")
    
    return results

if __name__ == "__main__":
    main()