"""
動的重み付けによる高度アンサンブル手法

予測値レンジ別重み調整とメタ学習による最適化
GMベースラインの単純線形ブレンディングを超越

Author: Claude Code Team
Date: 2025-07-02
Target: アンサンブル多様性向上による +0.002-0.004 スコア向上
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import warnings
warnings.filterwarnings('ignore')

class DynamicEnsembleOptimizer:
    """動的重み付けアンサンブル最適化器"""
    
    def __init__(self, cv_folds=5):
        self.cv_folds = cv_folds
        self.base_models = {}
        self.meta_model = None
        self.optimal_weights = None
        self.prediction_ranges = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]
        
    def create_diverse_models(self):
        """多様性を重視したベースモデル群の構築"""
        
        models = {}
        
        # 1. Tree系モデル（異なるハイパーパラメータ）
        models['lgb_fast'] = lgb.LGBMClassifier(
            objective='binary', num_leaves=20, learning_rate=0.05,
            n_estimators=800, random_state=42, verbosity=-1
        )
        
        models['lgb_deep'] = lgb.LGBMClassifier(
            objective='binary', num_leaves=50, learning_rate=0.02,
            n_estimators=1500, random_state=43, verbosity=-1
        )
        
        models['xgb_conservative'] = xgb.XGBClassifier(
            objective='binary:logistic', max_depth=4, learning_rate=0.03,
            n_estimators=1000, random_state=44, verbosity=0
        )
        
        models['xgb_aggressive'] = xgb.XGBClassifier(
            objective='binary:logistic', max_depth=8, learning_rate=0.02,
            n_estimators=1200, random_state=45, verbosity=0
        )
        
        models['catboost'] = CatBoostClassifier(
            objective='Logloss', depth=6, learning_rate=0.02,
            iterations=1000, random_seed=46, verbose=False
        )
        
        # 2. 線形系モデル（正則化パラメータ違い）
        models['logistic_l1'] = LogisticRegression(
            penalty='l1', C=0.1, solver='liblinear', random_state=47
        )
        
        models['logistic_l2'] = LogisticRegression(
            penalty='l2', C=1.0, random_state=48
        )
        
        # 3. ニューラルネットワーク（異なる構造）
        models['mlp_small'] = MLPClassifier(
            hidden_layer_sizes=(50,), learning_rate_init=0.01,
            max_iter=500, random_state=49
        )
        
        models['mlp_deep'] = MLPClassifier(
            hidden_layer_sizes=(100, 50), learning_rate_init=0.001,
            max_iter=800, random_state=50
        )
        
        self.base_models = models
        return models
    
    def evaluate_base_models(self, X, y):
        """ベースモデルの個別性能評価"""
        
        print("=== ベースモデル評価 ===")
        model_scores = {}
        
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.base_models.items():
            try:
                scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
                model_scores[name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores
                }
                print(f"   {name:15}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
            except Exception as e:
                print(f"   {name:15}: エラー - {str(e)}")
                model_scores[name] = {'mean': 0.0, 'std': 1.0, 'scores': [0.0]}
        
        # 性能順でソート
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['mean'], reverse=True)
        print(f"\n   最高性能: {sorted_models[0][0]} ({sorted_models[0][1]['mean']:.4f})")
        
        return model_scores
    
    def create_stacked_ensemble(self, X, y):
        """多層スタッキングアンサンブルの構築"""
        
        print("\n=== スタッキングアンサンブル構築 ===")
        
        # Level 1: 多様なベースモデル
        level1_models = [
            ('lgb_fast', self.base_models['lgb_fast']),
            ('lgb_deep', self.base_models['lgb_deep']),
            ('xgb_conservative', self.base_models['xgb_conservative']),
            ('catboost', self.base_models['catboost']),
            ('mlp_small', self.base_models['mlp_small'])
        ]
        
        # Level 2: メタ学習器（複数試行）
        meta_models = {
            'logistic': LogisticRegression(random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=42)
        }
        
        best_stacking_score = 0
        best_stacking_model = None
        
        for meta_name, meta_model in meta_models.items():
            stacking_model = StackingClassifier(
                estimators=level1_models,
                final_estimator=meta_model,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                passthrough=False  # メタ特徴量のみ使用
            )
            
            # クロスバリデーション評価
            cv_scores = cross_val_score(
                stacking_model, X, y, 
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
                scoring='accuracy'
            )
            
            mean_score = cv_scores.mean()
            print(f"   Stacking + {meta_name:8}: {mean_score:.4f} (+/- {cv_scores.std()*2:.4f})")
            
            if mean_score > best_stacking_score:
                best_stacking_score = mean_score
                best_stacking_model = stacking_model
        
        print(f"   最高スタッキング性能: {best_stacking_score:.4f}")
        self.meta_model = best_stacking_model
        
        return best_stacking_model
    
    def optimize_ensemble_weights(self, X, y, trial_count=100):
        """Optunaによるアンサンブル重み最適化"""
        
        print(f"\n=== アンサンブル重み最適化 (試行回数: {trial_count}) ===")
        
        # ベースモデル予測値取得
        base_predictions = self._get_base_predictions(X, y)
        
        def objective(trial):
            # 重みをサンプリング
            weights = []
            for i in range(len(self.base_models)):
                weight = trial.suggest_float(f'weight_{i}', 0.0, 1.0)
                weights.append(weight)
            
            # 正規化
            weights = np.array(weights)
            if weights.sum() == 0:
                return 0.0
            weights = weights / weights.sum()
            
            # 重み付きアンサンブル予測
            ensemble_pred = np.zeros(len(y))
            for i, (model_name, model_preds) in enumerate(base_predictions.items()):
                ensemble_pred += weights[i] * model_preds
            
            # バイナリ予測に変換
            binary_pred = (ensemble_pred > 0.5).astype(int)
            
            # 精度計算
            accuracy = accuracy_score(y, binary_pred)
            return accuracy
        
        # 最適化実行
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=trial_count, show_progress_bar=False)
        
        # 最適重み取得
        best_weights = []
        for i in range(len(self.base_models)):
            best_weights.append(study.best_params[f'weight_{i}'])
        
        best_weights = np.array(best_weights)
        best_weights = best_weights / best_weights.sum()
        
        self.optimal_weights = best_weights
        
        print(f"   最適化完了! 最高スコア: {study.best_value:.4f}")
        print("   最適重み:")
        for i, (model_name, weight) in enumerate(zip(self.base_models.keys(), best_weights)):
            print(f"     {model_name:15}: {weight:.3f}")
        
        return best_weights
    
    def _get_base_predictions(self, X, y):
        """ベースモデルのクロスバリデーション予測値取得"""
        
        base_predictions = {}
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for model_name, model in self.base_models.items():
            cv_preds = np.zeros(len(y))
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # モデル訓練・予測
                model.fit(X_train, y_train)
                val_proba = model.predict_proba(X_val)[:, 1]  # 正クラス確率
                cv_preds[val_idx] = val_proba
            
            base_predictions[model_name] = cv_preds
        
        return base_predictions
    
    def create_dynamic_weighted_ensemble(self, X, y):
        """予測値レンジ別動的重み付けアンサンブル"""
        
        print("\n=== 動的重み付けアンサンブル ===")
        
        # レンジ別最適重みを計算
        range_weights = {}
        base_predictions = self._get_base_predictions(X, y)
        
        for range_name, (min_val, max_val) in zip(['low', 'mid', 'high'], self.prediction_ranges):
            print(f"   範囲 {range_name} ({min_val}-{max_val}) の重み最適化中...")
            
            # 該当範囲のサンプルを特定
            ensemble_pred = np.mean(list(base_predictions.values()), axis=0)
            range_mask = (ensemble_pred >= min_val) & (ensemble_pred <= max_val)
            
            if np.sum(range_mask) < 10:  # サンプル数が少ない場合はスキップ
                print(f"     サンプル数不足 ({np.sum(range_mask)}件) - デフォルト重み使用")
                range_weights[range_name] = np.ones(len(self.base_models)) / len(self.base_models)
                continue
            
            # 該当範囲でのみ重み最適化
            range_X = X[range_mask]
            range_y = y[range_mask]
            
            # 小規模最適化
            range_optimal_weights = self._optimize_weights_for_range(range_X, range_y, max_trials=50)
            range_weights[range_name] = range_optimal_weights
            
            print(f"     最適化完了: {len(range_X)}サンプル使用")
        
        self.range_weights = range_weights
        return range_weights
    
    def _optimize_weights_for_range(self, X, y, max_trials=50):
        """特定範囲での重み最適化"""
        
        def objective(trial):
            weights = []
            for i in range(len(self.base_models)):
                weight = trial.suggest_float(f'weight_{i}', 0.0, 1.0)
                weights.append(weight)
            
            weights = np.array(weights)
            if weights.sum() == 0:
                return 0.0
            weights = weights / weights.sum()
            
            # 簡易CV評価
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                ensemble_pred = np.zeros(len(y_val))
                for i, (model_name, model) in enumerate(self.base_models.items()):
                    model.fit(X_train, y_train)
                    pred_proba = model.predict_proba(X_val)[:, 1]
                    ensemble_pred += weights[i] * pred_proba
                
                binary_pred = (ensemble_pred > 0.5).astype(int)
                score = accuracy_score(y_val, binary_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=max_trials, show_progress_bar=False)
        
        best_weights = []
        for i in range(len(self.base_models)):
            best_weights.append(study.best_params[f'weight_{i}'])
        
        best_weights = np.array(best_weights)
        return best_weights / best_weights.sum()
    
    def predict_with_dynamic_weights(self, X_test, base_model_predictions):
        """動的重み付けによる最終予測"""
        
        final_predictions = np.zeros(len(X_test))
        
        # 全体予測で範囲を決定
        ensemble_pred = np.mean(list(base_model_predictions.values()), axis=0)
        
        for i, pred_val in enumerate(ensemble_pred):
            # 予測値に応じて重みを選択
            if pred_val <= 0.3:
                weights = self.range_weights['low']
            elif pred_val <= 0.7:
                weights = self.range_weights['mid']
            else:
                weights = self.range_weights['high']
            
            # 重み付き予測
            weighted_pred = 0
            for j, (model_name, model_preds) in enumerate(base_model_predictions.items()):
                weighted_pred += weights[j] * model_preds[i]
            
            final_predictions[i] = weighted_pred
        
        return final_predictions

def main():
    """メイン実行関数"""
    print("=== 動的重み付けアンサンブル最適化 ===")
    
    # 擬似ラベル拡張データ読み込み
    print("1. 拡張データ読み込み中...")
    try:
        augmented_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/pseudo_labeled_train.csv')
        print(f"   データ形状: {augmented_data.shape}")
        
        # 特徴量とラベル分離
        feature_cols = [col for col in augmented_data.columns 
                       if col not in ['Personality', 'sample_weight', 'is_pseudo_label']]
        
        X = augmented_data[feature_cols].values
        y = augmented_data['Personality'].values
        sample_weights = augmented_data['sample_weight'].values
        
        print(f"   特徴量数: {len(feature_cols)}")
        print(f"   サンプル数: {len(X)}")
        
    except FileNotFoundError:
        print("   エラー: 拡張データが見つかりません。")
        print("   先に pseudo_labeling.py を実行してください。")
        return
    
    # アンサンブル最適化実行
    print("\n2. アンサンブル最適化開始...")
    optimizer = DynamicEnsembleOptimizer(cv_folds=5)
    
    # ベースモデル構築・評価
    optimizer.create_diverse_models()
    model_scores = optimizer.evaluate_base_models(X, y)
    
    # スタッキングアンサンブル
    stacking_model = optimizer.create_stacked_ensemble(X, y)
    
    # 重み最適化
    optimal_weights = optimizer.optimize_ensemble_weights(X, y, trial_count=200)
    
    # 動的重み付け
    dynamic_weights = optimizer.create_dynamic_weighted_ensemble(X, y)
    
    # 結果保存
    print("\n3. 最適化結果保存中...")
    
    results = {
        'model_scores': model_scores,
        'optimal_weights': optimal_weights.tolist(),
        'dynamic_weights': dynamic_weights,
        'feature_columns': feature_cols
    }
    
    import json
    with open('/Users/osawa/kaggle/playground-series-s5e7/data/processed/ensemble_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ 動的アンサンブル最適化完了!")
    print(f"   最高ベースモデル性能: {max(model_scores.values(), key=lambda x: x['mean'])['mean']:.4f}")
    print(f"   最適化後性能向上期待: +0.002-0.004")
    
    return optimizer, results

if __name__ == "__main__":
    main()