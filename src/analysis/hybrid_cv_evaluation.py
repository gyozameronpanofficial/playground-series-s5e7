"""
統合版CV性能評価

Phase 2b + フェーズ1+2統合手法の性能評価
- 心理学ドメイン特徴量
- Target Encoding（実証済み有効性）  
- 擬似ラベリング（データ拡張効果）

期待: CV 0.975000+ & PB 0.976500+

Author: Osawa
Date: 2025-07-03
Purpose: GM超越確実な統合手法の性能検証
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

def create_hybrid_ensemble():
    """統合版アンサンブルモデル"""
    
    models = [
        ('lgb', lgb.LGBMClassifier(
            objective='binary', num_leaves=31, learning_rate=0.05,
            n_estimators=500, random_state=42, verbosity=-1
        )),
        ('xgb', xgb.XGBClassifier(
            objective='binary:logistic', max_depth=6, learning_rate=0.05,
            n_estimators=500, random_state=42, verbosity=0
        )),
        ('cat', CatBoostClassifier(
            objective='Logloss', depth=6, learning_rate=0.05,
            iterations=500, random_seed=42, verbose=False
        )),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ]
    
    return VotingClassifier(estimators=models, voting='soft')

def evaluate_hybrid_performance():
    """統合手法性能評価メイン関数"""
    
    print("=== 統合版CV性能評価 ===")
    
    # 1. データ読み込み
    print("1. データ読み込み中...")
    
    try:
        # 統合データ
        hybrid_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/hybrid_train_features.csv')
        
        # 比較用データ
        baseline_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/raw/train.csv')
        
        print(f"   統合データ: {hybrid_data.shape}")
        print(f"   ベースライン: {baseline_data.shape}")
        
    except FileNotFoundError as e:
        print(f"   エラー: ファイルが見つかりません - {e}")
        return None
    
    # 2. データ前処理
    print("2. データ前処理中...")
    
    # 統合データの前処理
    feature_cols_hybrid = [col for col in hybrid_data.columns 
                          if col not in ['id', 'Personality', 'is_pseudo', 'confidence']]
    
    # 擬似ラベル込みデータと元データのみの分離
    original_mask = hybrid_data['is_pseudo'] == False
    original_data = hybrid_data[original_mask].copy()
    
    print(f"   元データのみ: {original_data.shape[0]}サンプル")
    print(f"   擬似ラベル込み: {hybrid_data.shape[0]}サンプル")
    print(f"   擬似ラベル数: {hybrid_data.shape[0] - original_data.shape[0]}サンプル")
    
    # データセット準備
    datasets = {
        'hybrid_with_pseudo': hybrid_data,
        'hybrid_original_only': original_data,
        'baseline': baseline_data
    }
    
    results = {}
    
    for name, data in datasets.items():
        print(f"\\n   {name}データ前処理中...")
        
        if name == 'baseline':
            feature_cols = [col for col in data.columns if col not in ['id', 'Personality']]
        else:
            feature_cols = feature_cols_hybrid
        
        X = data[feature_cols].copy()
        
        # カテゴリカル特徴量のエンコーディング
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        X = X.fillna(0).values
        y = data['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values
        
        # サンプル重み（擬似ラベルの場合）
        if 'confidence' in data.columns:
            sample_weight = data['confidence'].values
        else:
            sample_weight = None
        
        results[name] = {
            'X': X,
            'y': y,
            'sample_weight': sample_weight,
            'feature_count': X.shape[1]
        }
        
        print(f"     特徴量数: {X.shape[1]}, サンプル数: {X.shape[0]}")
    
    # 3. CV評価
    print("\\n3. クロスバリデーション評価中...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    for name, data_info in results.items():
        print(f"   {name}評価中...")
        model = create_hybrid_ensemble()
        
        if data_info['sample_weight'] is not None:
            # 重み付きCV（手動実装が必要）
            cv_scores = []
            X, y = data_info['X'], data_info['y']
            
            for train_idx, valid_idx in cv.split(X, y):
                X_train, X_valid = X[train_idx], X[valid_idx]
                y_train, y_valid = y[train_idx], y[valid_idx]
                
                model_copy = create_hybrid_ensemble()
                model_copy.fit(X_train, y_train)
                valid_pred = model_copy.predict(X_valid)
                score = accuracy_score(y_valid, valid_pred)
                cv_scores.append(score)
            
            cv_scores = np.array(cv_scores)
        else:
            # 通常のCV
            cv_scores = cross_val_score(model, data_info['X'], data_info['y'], 
                                      cv=cv, scoring='accuracy', n_jobs=-1)
        
        cv_results[name] = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'feature_count': data_info['feature_count']
        }
    
    # 4. 結果分析
    print("\\n4. 結果分析")
    print("=" * 70)
    
    # 各手法の結果表示
    for name, result in cv_results.items():
        print(f"\\n【{name}】")
        print(f"   平均CVスコア: {result['mean']:.6f} ± {result['std']:.6f}")
        print(f"   個別スコア: {[f'{score:.6f}' for score in result['scores']]}")
        print(f"   特徴量数: {result['feature_count']}")
    
    # 改善効果分析
    print(f"\\n📊 改善効果分析")
    print("-" * 50)
    
    baseline_score = cv_results['baseline']['mean']
    hybrid_original_score = cv_results['hybrid_original_only']['mean']
    hybrid_pseudo_score = cv_results['hybrid_with_pseudo']['mean']
    
    print(f"ベースライン: {baseline_score:.6f}")
    print(f"統合版（元データのみ）: {hybrid_original_score:.6f} ({hybrid_original_score - baseline_score:+.6f})")
    print(f"統合版（擬似ラベル込み）: {hybrid_pseudo_score:.6f} ({hybrid_pseudo_score - baseline_score:+.6f})")
    
    # GM比較
    gm_baseline = 0.975708
    print(f"\\n🎯 GMベースライン比較")
    print("-" * 40)
    print(f"GMスコア: {gm_baseline:.6f}")
    
    for name, result in cv_results.items():
        score = result['mean']
        diff = score - gm_baseline
        status = "✅ 超越" if diff > 0 else "⚠️ 未達"
        print(f"{name}: {score:.6f} ({diff:+.6f}) {status}")
    
    # 最高性能確認
    best_method = max(cv_results.keys(), key=lambda x: cv_results[x]['mean'])
    best_score = cv_results[best_method]['mean']
    
    print(f"\\n🏆 最高性能")
    print(f"手法: {best_method}")
    print(f"CVスコア: {best_score:.6f}")
    
    if best_score > gm_baseline:
        print(f"✅ GM超越達成! 差分: {best_score - gm_baseline:+.6f}")
        status = "gm_exceeded"
    else:
        print(f"❌ GM未達。不足分: {gm_baseline - best_score:.6f}")
        status = "gm_not_reached"
    
    # 5. 過去実装比較
    print(f"\\n📈 過去実装比較")
    print("-" * 40)
    
    past_results = {
        'フェーズ1+2': 0.974211,
        'Phase 2a': 0.968851,
        'Phase 2b': 0.968905
    }
    
    for past_name, past_score in past_results.items():
        current_score = cv_results['hybrid_with_pseudo']['mean']
        diff = current_score - past_score
        print(f"{past_name}: {past_score:.6f} → 統合版: {current_score:.6f} ({diff:+.6f})")
    
    # 6. 結果保存
    print("\\n5. 結果保存中...")
    
    save_results = {
        'hybrid_with_pseudo_cv_mean': cv_results['hybrid_with_pseudo']['mean'],
        'hybrid_with_pseudo_cv_std': cv_results['hybrid_with_pseudo']['std'],
        'hybrid_original_only_cv_mean': cv_results['hybrid_original_only']['mean'],
        'hybrid_original_only_cv_std': cv_results['hybrid_original_only']['std'],
        'baseline_cv_mean': cv_results['baseline']['mean'],
        'baseline_cv_std': cv_results['baseline']['std'],
        'best_method': best_method,
        'best_score': best_score,
        'gm_baseline': gm_baseline,
        'gm_status': status,
        'gm_diff': best_score - gm_baseline,
        'cv_scores_details': {name: result['scores'].tolist() for name, result in cv_results.items()},
        'feature_counts': {name: result['feature_count'] for name, result in cv_results.items()}
    }
    
    results_path = '/Users/osawa/kaggle/playground-series-s5e7/data/processed/hybrid_cv_results.json'
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"   結果保存完了: {results_path}")
    
    print(f"\\n" + "="*70)
    print("統合版CV評価完了")
    print("="*70)
    
    return save_results

def main():
    """メイン実行関数"""
    
    results = evaluate_hybrid_performance()
    
    if results:
        if results['gm_status'] == 'gm_exceeded':
            print("🎯 次のステップ: GM超越確実！統合版提出ファイル作成")
        else:
            print("📋 次のステップ: 更なる改善策検討またはベスト手法での提出")
    
    return results

if __name__ == "__main__":
    main()