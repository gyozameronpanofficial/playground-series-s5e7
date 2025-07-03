"""
Phase 2b CV性能評価

高度Target Encodingの効果を検証
Phase 2aのTF-IDF失敗を受けて、シンプルなTarget Encodingの効果を測定

Author: Osawa
Date: 2025-07-03
Purpose: Phase 2b実装効果の定量評価
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

def create_phase2b_ensemble():
    """Phase 2b用アンサンブルモデル"""
    
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

def evaluate_phase2b_performance():
    """Phase 2b性能評価メイン関数"""
    
    print("=== Phase 2b CV性能評価 ===")
    
    # 1. データ読み込み
    print("1. データ読み込み中...")
    
    try:
        # Phase 2bデータ
        phase2b_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2b_train_features.csv')
        
        # ベースラインデータ（比較用）
        baseline_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/raw/train.csv')
        
        print(f"   Phase 2b データ: {phase2b_data.shape}")
        print(f"   ベースライン データ: {baseline_data.shape}")
        
    except FileNotFoundError as e:
        print(f"   エラー: ファイルが見つかりません - {e}")
        return None
    
    # 2. データ前処理
    print("2. データ前処理中...")
    
    # Phase 2bデータの前処理
    feature_cols_2b = [col for col in phase2b_data.columns if col not in ['id', 'Personality']]
    X_2b = phase2b_data[feature_cols_2b].copy()
    
    # カテゴリカル特徴量のエンコーディング
    for col in X_2b.columns:
        if X_2b[col].dtype == 'object':
            le = LabelEncoder()
            X_2b[col] = le.fit_transform(X_2b[col].astype(str))
    
    X_2b = X_2b.fillna(0).values
    y_2b = phase2b_data['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values
    
    # ベースラインデータの前処理
    feature_cols_base = [col for col in baseline_data.columns if col not in ['id', 'Personality']]
    X_base = baseline_data[feature_cols_base].copy()
    
    for col in X_base.columns:
        if X_base[col].dtype == 'object':
            le = LabelEncoder()
            X_base[col] = le.fit_transform(X_base[col].astype(str))
    
    X_base = X_base.fillna(0).values
    y_base = baseline_data['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values
    
    print(f"   Phase 2b 特徴量数: {X_2b.shape[1]}")
    print(f"   ベースライン 特徴量数: {X_base.shape[1]}")
    
    # 3. CV評価
    print("\\n3. クロスバリデーション評価中...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Phase 2b評価
    print("   Phase 2b評価中...")
    model_2b = create_phase2b_ensemble()
    cv_scores_2b = cross_val_score(model_2b, X_2b, y_2b, cv=cv, scoring='accuracy', n_jobs=-1)
    
    # ベースライン評価
    print("   ベースライン評価中...")
    model_base = create_phase2b_ensemble()
    cv_scores_base = cross_val_score(model_base, X_base, y_base, cv=cv, scoring='accuracy', n_jobs=-1)
    
    # 4. 結果分析
    print("\\n4. 結果分析")
    print("-" * 50)
    
    cv_mean_2b = cv_scores_2b.mean()
    cv_std_2b = cv_scores_2b.std()
    cv_mean_base = cv_scores_base.mean()
    cv_std_base = cv_scores_base.std()
    
    improvement = cv_mean_2b - cv_mean_base
    
    print(f"Phase 2b CV性能:")
    print(f"   平均スコア: {cv_mean_2b:.6f} ± {cv_std_2b:.6f}")
    print(f"   個別スコア: {[f'{score:.6f}' for score in cv_scores_2b]}")
    
    print(f"\\nベースライン CV性能:")
    print(f"   平均スコア: {cv_mean_base:.6f} ± {cv_std_base:.6f}")
    print(f"   個別スコア: {[f'{score:.6f}' for score in cv_scores_base]}")
    
    print(f"\\n改善効果:")
    print(f"   スコア改善: {improvement:+.6f}")
    print(f"   相対改善: {improvement/cv_mean_base*100:+.3f}%")
    
    # GM比較
    gm_baseline = 0.975708
    print(f"\\nGMベースライン比較:")
    print(f"   GMスコア: {gm_baseline:.6f}")
    print(f"   Phase 2bとの差: {cv_mean_2b - gm_baseline:+.6f}")
    
    status = "success" if improvement > 0 else "failed"
    gm_status = "reached" if cv_mean_2b > gm_baseline else "not_reached"
    
    if improvement > 0:
        print(f"   ✅ Phase 2b成功! スコア向上: {improvement:+.6f}")
    else:
        print(f"   ❌ Phase 2b失敗。スコア低下: {improvement:+.6f}")
    
    if cv_mean_2b > gm_baseline:
        print(f"   🎯 GM超越達成! 差分: {cv_mean_2b - gm_baseline:+.6f}")
    else:
        print(f"   ⚠️ GM未達。不足分: {gm_baseline - cv_mean_2b:.6f}")
    
    # 5. 結果保存
    print("\\n5. 結果保存中...")
    
    results = {
        'phase2b_cv_mean': cv_mean_2b,
        'phase2b_cv_std': cv_std_2b,
        'baseline_cv_mean': cv_mean_base,
        'baseline_cv_std': cv_std_base,
        'improvement': improvement,
        'relative_improvement_pct': improvement/cv_mean_base*100,
        'status': status,
        'gm_baseline': gm_baseline,
        'gm_status': gm_status,
        'gm_diff': cv_mean_2b - gm_baseline,
        'feature_count_2b': X_2b.shape[1],
        'feature_count_base': X_base.shape[1],
        'cv_scores_phase2b': cv_scores_2b.tolist(),
        'cv_scores_baseline': cv_scores_base.tolist()
    }
    
    # JSON保存
    results_path = '/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2b_cv_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   結果保存完了: {results_path}")
    
    # 6. Phase 2a比較（もしあれば）
    try:
        with open('/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2a_cv_results.json', 'r') as f:
            phase2a_results = json.load(f)
        
        print(f"\\n6. Phase 2a比較")
        print("-" * 30)
        phase2a_cv = phase2a_results['phase2a_cv_mean']
        phase2b_vs_2a = cv_mean_2b - phase2a_cv
        
        print(f"   Phase 2a CV: {phase2a_cv:.6f}")
        print(f"   Phase 2b CV: {cv_mean_2b:.6f}")
        print(f"   2b vs 2a: {phase2b_vs_2a:+.6f}")
        
        if phase2b_vs_2a > 0:
            print(f"   ✅ Phase 2b > Phase 2a")
        else:
            print(f"   ❌ Phase 2b < Phase 2a")
            
    except FileNotFoundError:
        print("\\n   Phase 2a結果が見つかりません")
    
    print(f"\\n" + "="*60)
    print("Phase 2b CV評価完了")
    print("="*60)
    
    return results

def main():
    """メイン実行関数"""
    
    results = evaluate_phase2b_performance()
    
    if results:
        if results['status'] == 'success':
            print("🎯 次のステップ: Phase 2b提出ファイル作成")
        else:
            print("📋 次のステップ: Phase 2b問題分析または Phase 2c実装")
    
    return results

if __name__ == "__main__":
    main()