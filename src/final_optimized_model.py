"""
最終最適化モデル
Optunaによるハイパーパラメータ最適化とベスト戦略の統合
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

class FinalOptimizedModel:
    """最終最適化モデルクラス"""
    
    def __init__(self):
        self.config = {
            'RANDOM_STATE': 42,
            'N_SPLITS': 5,
            'TARGET_COL': 'target',
            'OPTUNA_TRIALS': 100,
            'DATA_PATH': Path('/Users/osawa/kaggle/playground-series-s5e7/data/processed')
        }
        self.best_models = {}
        self.best_params = {}
        
    def load_data(self):
        """データ読み込み"""
        print("=== データ読み込み ===")
        
        train_df = pd.read_csv(self.config['DATA_PATH'] / 'enhanced_train.csv')
        test_df = pd.read_csv(self.config['DATA_PATH'] / 'enhanced_test.csv')
        
        X_train = train_df.drop(self.config['TARGET_COL'], axis=1)
        y_train = train_df[self.config['TARGET_COL']]
        X_test = test_df.copy()
        
        print(f"訓練データ形状: {X_train.shape}")
        print(f"テストデータ形状: {X_test.shape}")
        
        return X_train, y_train, X_test
    
    def optimize_xgboost(self, X_train, y_train):
        """XGBoostハイパーパラメータ最適化"""
        print("\n=== XGBoost最適化 ===")
        
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
                'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': self.config['RANDOM_STATE'],
                'verbosity': 0
            }
            
            model = Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', xgb.XGBClassifier(**params))
            ])
            
            cv = StratifiedKFold(n_splits=self.config['N_SPLITS'], shuffle=True, 
                               random_state=self.config['RANDOM_STATE'])
            
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', 
                                  sampler=optuna.samplers.TPESampler(seed=self.config['RANDOM_STATE']))
        study.optimize(objective, n_trials=self.config['OPTUNA_TRIALS'])
        
        print(f"最適XGBoostスコア: {study.best_value:.6f}")
        print(f"最適XGBoostパラメータ: {study.best_params}")
        
        self.best_params['XGBoost'] = study.best_params
        return study.best_value, study.best_params
    
    def optimize_lightgbm(self, X_train, y_train):
        """LightGBMハイパーパラメータ最適化"""
        print(f"\n=== LightGBM最適化 ===")
        
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'logloss',
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
                'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'random_state': self.config['RANDOM_STATE'],
                'verbosity': -1
            }
            
            model = Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', lgb.LGBMClassifier(**params))
            ])
            
            cv = StratifiedKFold(n_splits=self.config['N_SPLITS'], shuffle=True, 
                               random_state=self.config['RANDOM_STATE'])
            
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=self.config['RANDOM_STATE']))
        study.optimize(objective, n_trials=self.config['OPTUNA_TRIALS'])
        
        print(f"最適LightGBMスコア: {study.best_value:.6f}")
        print(f"最適LightGBMパラメータ: {study.best_params}")
        
        self.best_params['LightGBM'] = study.best_params
        return study.best_value, study.best_params
    
    def optimize_catboost(self, X_train, y_train):
        """CatBoostハイパーパラメータ最適化"""
        print(f"\n=== CatBoost最適化 ===")
        
        def objective(trial):
            params = {
                'loss_function': 'Logloss',
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
                'iterations': trial.suggest_int('iterations', 1000, 3000),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_state': self.config['RANDOM_STATE'],
                'verbose': False
            }
            
            model = Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', cb.CatBoostClassifier(**params))
            ])
            
            cv = StratifiedKFold(n_splits=self.config['N_SPLITS'], shuffle=True, 
                               random_state=self.config['RANDOM_STATE'])
            
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=self.config['RANDOM_STATE']))
        study.optimize(objective, n_trials=self.config['OPTUNA_TRIALS'])
        
        print(f"最適CatBoostスコア: {study.best_value:.6f}")
        print(f"最適CatBoostパラメータ: {study.best_params}")
        
        self.best_params['CatBoost'] = study.best_params
        return study.best_value, study.best_params
    
    def optimize_randomforest(self, X_train, y_train):
        """RandomForestハイパーパラメータ最適化"""
        print(f"\n=== RandomForest最適化 ===")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
                'random_state': self.config['RANDOM_STATE']
            }
            
            model = Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', RandomForestClassifier(**params))
            ])
            
            cv = StratifiedKFold(n_splits=self.config['N_SPLITS'], shuffle=True, 
                               random_state=self.config['RANDOM_STATE'])
            
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=self.config['RANDOM_STATE']))
        study.optimize(objective, n_trials=self.config['OPTUNA_TRIALS'])
        
        print(f"最適RandomForestスコア: {study.best_value:.6f}")
        print(f"最適RandomForestパラメータ: {study.best_params}")
        
        self.best_params['RandomForest'] = study.best_params
        return study.best_value, study.best_params
    
    def train_optimized_models(self, X_train, y_train):
        """最適化されたモデルの訓練"""
        print(f"\n=== 最適化モデル訓練 ===")
        
        cv = StratifiedKFold(n_splits=self.config['N_SPLITS'], shuffle=True, 
                           random_state=self.config['RANDOM_STATE'])
        
        # 各最適化モデルの構築
        optimized_models = {
            'XGBoost_Optimized': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    random_state=self.config['RANDOM_STATE'],
                    verbosity=0,
                    **self.best_params['XGBoost']
                ))
            ]),
            
            'LightGBM_Optimized': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', lgb.LGBMClassifier(
                    objective='binary',
                    metric='logloss',
                    random_state=self.config['RANDOM_STATE'],
                    verbosity=-1,
                    **self.best_params['LightGBM']
                ))
            ]),
            
            'CatBoost_Optimized': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', cb.CatBoostClassifier(
                    loss_function='Logloss',
                    random_state=self.config['RANDOM_STATE'],
                    verbose=False,
                    **self.best_params['CatBoost']
                ))
            ]),
            
            'RandomForest_Optimized': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', RandomForestClassifier(
                    random_state=self.config['RANDOM_STATE'],
                    **self.best_params['RandomForest']
                ))
            ])
        }
        
        # OOF予測の生成
        oof_predictions = {}
        model_scores = {}
        
        for name, model in optimized_models.items():
            print(f"\n--- {name} 訓練中 ---")
            
            oof_preds = np.zeros(len(X_train))
            fold_scores = []
            
            for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx]
                X_fold_valid = X_train.iloc[valid_idx]
                y_fold_train = y_train.iloc[train_idx]
                y_fold_valid = y_train.iloc[valid_idx]
                
                model.fit(X_fold_train, y_fold_train)
                valid_preds = model.predict_proba(X_fold_valid)[:, 1]
                oof_preds[valid_idx] = valid_preds
                
                fold_score = accuracy_score(y_fold_valid, (valid_preds >= 0.5).astype(int))
                fold_scores.append(fold_score)
                print(f"  Fold {fold+1}: {fold_score:.6f}")
            
            overall_score = accuracy_score(y_train, (oof_preds >= 0.5).astype(int))
            model_scores[name] = overall_score
            oof_predictions[name] = oof_preds
            
            print(f"  {name} 全体スコア: {overall_score:.6f} (±{np.std(fold_scores):.6f})")
        
        self.best_models = optimized_models
        return oof_predictions, model_scores
    
    def optimize_ensemble_weights(self, oof_predictions, y_train):
        """アンサンブル重みの最適化"""
        print(f"\n=== アンサンブル重み最適化 ===")
        
        def objective(trial):
            # 各モデルの重みを最適化
            weights = []
            for i, model_name in enumerate(oof_predictions.keys()):
                weight = trial.suggest_float(f'weight_{model_name}', 0.0, 1.0)
                weights.append(weight)
            
            # 重みの正規化
            weights = np.array(weights)
            if weights.sum() == 0:
                return 0.5  # 全て0の場合はランダムスコア
            weights = weights / weights.sum()
            
            # 重み付き平均予測
            oof_array = np.array(list(oof_predictions.values())).T
            weighted_preds = np.average(oof_array, axis=1, weights=weights)
            
            # スコア計算
            score = accuracy_score(y_train, (weighted_preds >= 0.5).astype(int))
            return score
        
        study = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=self.config['RANDOM_STATE']))
        study.optimize(objective, n_trials=200)  # 重み最適化は少ないトライアルで十分
        
        print(f"最適アンサンブルスコア: {study.best_value:.6f}")
        print(f"最適重み: {study.best_params}")
        
        return study.best_value, study.best_params
    
    def create_final_submission(self, X_train, y_train, X_test, optimal_weights):
        """最終提出ファイル作成"""
        print(f"\n=== 最終提出ファイル作成 ===")
        
        # 最適化モデルで全データ訓練
        final_predictions = []
        
        for name, model in self.best_models.items():
            print(f"全データで{name}訓練中...")
            model.fit(X_train, y_train)
            test_preds = model.predict_proba(X_test)[:, 1]
            final_predictions.append(test_preds)
        
        # 最適重みでアンサンブル
        weights = []
        for model_name in self.best_models.keys():
            weight_key = f'weight_{model_name}'
            weights.append(optimal_weights.get(weight_key, 0.25))  # デフォルト重み
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # 正規化
        
        final_predictions = np.array(final_predictions).T
        ensemble_predictions = np.average(final_predictions, axis=1, weights=weights)
        
        # 二値分類に変換
        binary_predictions = (ensemble_predictions >= 0.5).astype(int)
        string_predictions = ['Extrovert' if pred == 0 else 'Introvert' for pred in binary_predictions]
        
        # 提出ファイル作成
        submission = pd.DataFrame({
            'id': range(18524, 18524 + len(X_test)),  # テストデータのIDは18524から開始
            'Personality': string_predictions
        })
        
        submission_path = '/Users/osawa/kaggle/playground-series-s5e7/submissions/final_optimized_submission.csv'
        submission.to_csv(submission_path, index=False)
        
        print(f"提出ファイル保存: {submission_path}")
        print(f"提出データ形状: {submission.shape}")
        print(f"予測分布:")
        print(submission['Personality'].value_counts())
        
        return submission, ensemble_predictions
    
    def run_final_optimization(self):
        """最終最適化の実行"""
        print("=== 最終最適化モデル実行 ===\n")
        
        # 1. データ読み込み
        X_train, y_train, X_test = self.load_data()
        
        # 2. 各モデルのハイパーパラメータ最適化
        xgb_score, xgb_params = self.optimize_xgboost(X_train, y_train)
        lgb_score, lgb_params = self.optimize_lightgbm(X_train, y_train)
        cb_score, cb_params = self.optimize_catboost(X_train, y_train)
        rf_score, rf_params = self.optimize_randomforest(X_train, y_train)
        
        # 3. 最適化モデルの訓練
        oof_predictions, model_scores = self.train_optimized_models(X_train, y_train)
        
        # 4. アンサンブル重み最適化
        ensemble_score, optimal_weights = self.optimize_ensemble_weights(oof_predictions, y_train)
        
        # 5. 最終提出ファイル作成
        submission, final_preds = self.create_final_submission(
            X_train, y_train, X_test, optimal_weights
        )
        
        # 6. 結果まとめ
        print(f"\n=== 最終結果 ===")
        print(f"最良単一モデル: {max(model_scores.items(), key=lambda x: x[1])}")
        print(f"最適アンサンブルスコア: {ensemble_score:.6f}")
        print(f"目標0.975708まで: {0.975708 - ensemble_score:+.6f}")
        
        if ensemble_score >= 0.975708:
            print("🎉 目標スコア達成！")
        else:
            print(f"目標まで{0.975708 - ensemble_score:.6f}の改善が必要")
        
        return {
            'ensemble_score': ensemble_score,
            'model_scores': model_scores,
            'optimal_weights': optimal_weights,
            'submission': submission
        }

if __name__ == "__main__":
    optimizer = FinalOptimizedModel()
    results = optimizer.run_final_optimization()