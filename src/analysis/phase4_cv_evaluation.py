"""
Phase 4 拡張版CV評価スクリプト

Phase 4（41特徴量）のクロスバリデーション性能評価
期待結果: CV 0.976404 → 0.978404 (+0.002000)

Author: Osawa
Date: 2025-07-04
Purpose: Phase 4拡張手法の性能検証
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import json
import warnings
warnings.filterwarnings('ignore')

def create_phase4_ensemble():
    """Phase 4用アンサンブルモデル作成（Phase 3と同じ構成）"""
    
    models = [
        ('lgb', lgb.LGBMClassifier(
            objective='binary',
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=500,
            random_state=42,
            verbosity=-1
        )),
        ('xgb', xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=6,
            learning_rate=0.05,
            n_estimators=500,
            random_state=42,
            verbosity=0
        )),
        ('cat', CatBoostClassifier(
            objective='Logloss',
            depth=6,
            learning_rate=0.05,
            iterations=500,
            random_seed=42,
            verbose=False
        )),
        ('lr', LogisticRegression(
            random_state=42,
            max_iter=1000
        ))
    ]
    
    return models

def preprocess_phase4_data(train_data):
    """Phase 4データの前処理"""
    
    print("   Phase 4データ前処理中...")
    
    # 特徴量とターゲット分離
    feature_cols = [col for col in train_data.columns 
                   if col not in ['id', 'Personality', 'is_pseudo', 'confidence']]
    
    print(f"     使用特徴量数: {len(feature_cols)}個")
    
    # カテゴリカル特徴量のエンコーディング
    train_processed = train_data[feature_cols].copy()
    
    label_encoders = {}
    categorical_cols = []
    
    for col in feature_cols:
        if train_processed[col].dtype == 'object':
            categorical_cols.append(col)
            le = LabelEncoder()
            train_processed[col] = le.fit_transform(train_processed[col].astype(str))
            label_encoders[col] = le
    
    print(f"     カテゴリカル特徴量: {len(categorical_cols)}個")
    
    # 欠損値処理
    X = train_processed.fillna(0).values
    y = train_data['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values
    
    # サンプル重み
    sample_weight = train_data['confidence'].values if 'confidence' in train_data.columns else None
    
    return X, y, sample_weight, feature_cols

def evaluate_phase4_performance():
    """Phase 4のCV性能評価"""
    
    print("=== Phase 4 CV性能評価 ===")
    
    # 1. データ読み込み
    print("1. Phase 4データ読み込み中...")
    try:
        train_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase4_train_features.csv')
        print(f"   データ形状: {train_data.shape}")
        
        # データ統計
        original_samples = len(train_data[train_data['is_pseudo'] == False])
        pseudo_samples = len(train_data[train_data['is_pseudo'] == True])
        print(f"   元データ: {original_samples}サンプル")
        print(f"   擬似ラベル: {pseudo_samples}サンプル ({pseudo_samples/original_samples*100:.1f}%)")
        
    except FileNotFoundError:
        print("   エラー: Phase 4データが見つかりません")
        print("   先にphase4_enhanced_integration.pyを実行してください")
        return None
    
    # 2. データ前処理
    print("\\n2. データ前処理実行中...")
    X, y, sample_weight, feature_cols = preprocess_phase4_data(train_data)
    
    print(f"   最終データ形状: {X.shape}")
    print(f"   クラス分布: Extrovert {np.sum(y==1)}, Introvert {np.sum(y==0)}")
    
    # 3. モデル別CV評価
    print("\\n3. モデル別CV評価実行中...")
    models = create_phase4_ensemble()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    individual_scores = {}
    
    for name, model in models:
        print(f"   {name}評価中...")
        
        if sample_weight is not None:
            # sample_weight対応の手動CV
            cv_scores = []
            for train_idx, valid_idx in cv.split(X, y):
                X_train, X_valid = X[train_idx], X[valid_idx]
                y_train, y_valid = y[train_idx], y[valid_idx]
                sw_train = sample_weight[train_idx]
                
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train, y_train, sample_weight=sw_train)
                valid_pred = model_copy.predict(X_valid)
                score = accuracy_score(y_valid, valid_pred)
                cv_scores.append(score)
            
            cv_scores = np.array(cv_scores)
        else:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        results[name] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        individual_scores[name] = cv_scores.mean()
        print(f"     {name}: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
    
    # 4. アンサンブルCV評価（sample_weight対応）
    print("\\n4. アンサンブルCV評価実行中...")
    
    ensemble_scores = []
    
    for train_idx, valid_idx in cv.split(X, y):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        if sample_weight is not None:
            sw_train = sample_weight[train_idx]
        else:
            sw_train = None
        
        # 各モデルで予測
        ensemble_preds = []
        
        for name, model in models:
            model_copy = type(model)(**model.get_params())
            
            if sw_train is not None:
                model_copy.fit(X_train, y_train, sample_weight=sw_train)
            else:
                model_copy.fit(X_train, y_train)
            
            pred_proba = model_copy.predict_proba(X_valid)[:, 1]
            ensemble_preds.append(pred_proba)
        
        # アンサンブル予測（ソフトボーティング）
        ensemble_proba = np.mean(ensemble_preds, axis=0)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        score = accuracy_score(y_valid, ensemble_pred)
        ensemble_scores.append(score)
    
    ensemble_scores = np.array(ensemble_scores)
    ensemble_mean = ensemble_scores.mean()
    ensemble_std = ensemble_scores.std()
    
    print(f"   アンサンブル: {ensemble_mean:.6f} ± {ensemble_std:.6f}")
    
    # 5. 結果まとめ
    print("\\n📊 Phase 4 CV評価結果:")
    print(f"   最終アンサンブルスコア: {ensemble_mean:.6f} ± {ensemble_std:.6f}")
    
    # ベストモデル特定
    best_model = max(individual_scores.keys(), key=lambda k: individual_scores[k])
    print(f"   ベスト単体モデル: {best_model} ({individual_scores[best_model]:.6f})")
    
    # Phase 3との比較予想
    phase3_baseline = 0.976404
    improvement = ensemble_mean - phase3_baseline
    print(f"\\n🎯 Phase 3比較:")
    print(f"   Phase 3ベースライン: {phase3_baseline:.6f}")
    print(f"   Phase 4スコア: {ensemble_mean:.6f}")
    print(f"   改善効果: {improvement:+.6f}")
    
    if improvement > 0:
        print(f"   ✅ 改善成功！ (+{improvement:.6f})")
    else:
        print(f"   ⚠️  改善なし ({improvement:.6f})")
    
    # 6. 結果保存
    evaluation_results = {
        'phase4_ensemble_cv': float(ensemble_mean),
        'phase4_ensemble_std': float(ensemble_std),
        'phase4_individual_models': individual_scores,
        'phase4_feature_count': len(feature_cols),
        'phase4_sample_count': len(train_data),
        'phase4_pseudo_ratio': float(pseudo_samples/original_samples),
        'phase3_baseline': phase3_baseline,
        'improvement_vs_phase3': float(improvement),
        'evaluation_date': '2025-07-04',
        'methodology': 'sample_weight_supported_cv'
    }
    
    results_path = '/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase4_cv_results.json'
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\\n💾 結果保存: {results_path}")
    
    return evaluation_results

def compare_feature_importance():
    """Phase 4特徴量重要度分析"""
    
    print("\\n=== Phase 4 特徴量重要度分析 ===")
    
    # データ読み込み
    train_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase4_train_features.csv')
    X, y, sample_weight, feature_cols = preprocess_phase4_data(train_data)
    
    # LightGBMで特徴量重要度計算
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=500,
        random_state=42,
        verbosity=-1
    )
    
    if sample_weight is not None:
        lgb_model.fit(X, y, sample_weight=sample_weight)
    else:
        lgb_model.fit(X, y)
    
    # 特徴量重要度取得
    feature_importance = lgb_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\\n🔝 Top 15 重要特徴量:")
    for i, row in importance_df.head(15).iterrows():
        print(f"   {row['feature']}: {row['importance']:.0f}")
    
    # 特徴量カテゴリ別分析
    print("\\n📋 特徴量カテゴリ別重要度:")
    
    categories = {
        'psychological': ['extroversion_score', 'introversion_score', 'social_balance', 
                         'social_fatigue', 'social_proactivity', 'solitude_preference'],
        'outlier': ['extreme_alone_flag', 'stage_fear_missing', 'extreme_introvert_pattern',
                   'extreme_extrovert_pattern', 'personality_extreme_flag'],
        'ngram': [col for col in feature_cols if '_combo' in col and '_target_encoded' not in col],
        'ambivert': ['ambivert_score', 'extreme_score', 'ambivert_flag'],
        'target_encoded': [col for col in feature_cols if '_target_encoded' in col],
        'statistical': ['feature_mean', 'feature_std', 'feature_max', 'feature_min']
    }
    
    for category, features in categories.items():
        category_features = [f for f in features if f in feature_cols]
        if category_features:
            category_importance = importance_df[importance_df['feature'].isin(category_features)]['importance'].sum()
            avg_importance = category_importance / len(category_features)
            print(f"   {category}: {category_importance:.0f} (平均 {avg_importance:.0f}, {len(category_features)}個)")
    
    return importance_df

def main():
    """メイン実行関数"""
    
    print("=== Phase 4 拡張版CV評価実行 ===")
    
    # 1. CV性能評価
    results = evaluate_phase4_performance()
    
    if results is None:
        return
    
    # 2. 特徴量重要度分析
    importance_df = compare_feature_importance()
    
    # 3. 最終サマリー
    print("\\n" + "="*50)
    print("📊 Phase 4 評価サマリー")
    print("="*50)
    print(f"CVスコア: {results['phase4_ensemble_cv']:.6f} ± {results['phase4_ensemble_std']:.6f}")
    print(f"特徴量数: {results['phase4_feature_count']}個")
    print(f"擬似ラベル拡張: {results['phase4_pseudo_ratio']*100:.1f}%")
    print(f"Phase 3比改善: {results['improvement_vs_phase3']:+.6f}")
    
    if results['improvement_vs_phase3'] > 0:
        print("\\n🎉 Phase 4 改善成功！")
        print("   Phase 3を上回る性能を達成しました")
    else:
        print("\\n📝 Phase 4 結果分析:")
        print("   追加特徴量の効果が限定的でした")
        print("   個別要素の検証が必要です")
    
    print(f"\\n🚀 次の推奨アクション:")
    if results['improvement_vs_phase3'] > 0.001:
        print("   1. Phase 4で提出ファイル作成")
        print("   2. Phase 5でさらなる最適化検討")
    else:
        print("   1. Phase 3との詳細比較分析")
        print("   2. 個別特徴量の効果検証")
        print("   3. Phase 4の部分的採用検討")

if __name__ == "__main__":
    main()