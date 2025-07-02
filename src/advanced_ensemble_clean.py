"""
高度アンサンブル: 多層スタッキング、動的重み付け、ベイズ最適化
GMベースライン0.975708突破のための究極戦略
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

class AdvancedEnsemble:
    """高度アンサンブルクラス"""
    
    def __init__(self):
        self.config = {
            'DATA_PATH': Path('/Users/osawa/kaggle/playground-series-s5e7/data/processed'),
            'OUTPUT_PATH': Path('/Users/osawa/kaggle/playground-series-s5e7/submissions'),
            'TARGET_MAPPING': {"Extrovert": 0, "Introvert": 1},
            'RANDOM_STATE': 42,
            'N_SPLITS': 5,
            'N_REPEATS': 2
        }
        self.config['OUTPUT_PATH'].mkdir(exist_ok=True)
        np.random.seed(self.config['RANDOM_STATE'])
        
    def load_revolutionary_data(self):
        """革新的特徴量データの読み込み"""
        print("=== 革新的特徴量データ読み込み ===")
        
        train_df = pd.read_csv(self.config['DATA_PATH'] / 'revolutionary_train.csv')
        test_df = pd.read_csv(self.config['DATA_PATH'] / 'revolutionary_test.csv')
        
        feature_cols = [col for col in train_df.columns if col not in ['id', 'Personality']]
        
        X_train = train_df[feature_cols].copy()
        y_train = train_df['Personality'].map(self.config['TARGET_MAPPING'])
        X_test = test_df[feature_cols].copy()
        
        print(f"データ形状: X_train={X_train.shape}, X_test={X_test.shape}")
        print(f"特徴量数: {len(feature_cols)}")
        
        return X_train, y_train, X_test, train_df['id'], test_df['id']
    
    def setup_diverse_models(self):
        """多様なモデルの設定"""
        print("=== 多様なモデル設定 ===")
        
        models = {
            # Gradient Boosting Models
            'XGBoost': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    learning_rate=0.01,
                    n_estimators=1500,
                    max_depth=6,
                    colsample_bytree=0.45,
                    subsample=0.8,
                    random_state=self.config['RANDOM_STATE'],
                    verbosity=0
                ))
            ]),
            
            'LightGBM': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', lgb.LGBMClassifier(
                    objective='binary',
                    metric='logloss',
                    learning_rate=0.01,
                    n_estimators=1500,
                    max_depth=6,
                    colsample_bytree=0.45,
                    subsample=0.8,
                    random_state=self.config['RANDOM_STATE'],
                    verbosity=-1
                ))
            ]),
            
            'CatBoost': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', cb.CatBoostClassifier(
                    loss_function='Logloss',
                    learning_rate=0.01,
                    iterations=1000,
                    max_depth=6,
                    random_state=self.config['RANDOM_STATE'],
                    verbose=False
                ))
            ]),
            
            # Tree-based Models
            'RandomForest': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=10,
                    max_features='sqrt',
                    random_state=self.config['RANDOM_STATE'],
                    n_jobs=-1
                ))
            ]),
            
            'ExtraTrees': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=10,
                    max_features='sqrt',
                    random_state=self.config['RANDOM_STATE'],
                    n_jobs=-1
                ))
            ]),
            
            # Linear Models
            'LogisticRegression': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(
                    C=0.1,
                    max_iter=1000,
                    random_state=self.config['RANDOM_STATE']
                ))
            ]),
            
            # Neural Network
            'MLP': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('scaler', StandardScaler()),
                ('model', MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    learning_rate_init=0.001,
                    max_iter=500,
                    random_state=self.config['RANDOM_STATE']
                ))
            ])
        }
        
        print(f"設定完了: {len(models)} モデル")
        return models
    
    def train_level1_models(self, X_train, y_train, X_test, models):
        """Level 1モデルの訓練"""
        print("=== Level 1 モデル訓練 ===")
        
        cv = StratifiedKFold(
            n_splits=self.config['N_SPLITS'], 
            shuffle=True, 
            random_state=self.config['RANDOM_STATE']
        )
        
        oof_predictions = {}
        test_predictions = {}
        model_scores = {}
        
        for name, model in models.items():
            print(f"  {name} 訓練中...")
            
            oof_preds = np.zeros(len(X_train))
            test_preds = np.zeros(len(X_test))
            fold_scores = []
            
            for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx]
                X_fold_valid = X_train.iloc[valid_idx]
                y_fold_train = y_train.iloc[train_idx]
                y_fold_valid = y_train.iloc[valid_idx]
                
                model.fit(X_fold_train, y_fold_train)
                
                valid_preds = model.predict_proba(X_fold_valid)[:, 1]
                oof_preds[valid_idx] = valid_preds
                
                fold_test_preds = model.predict_proba(X_test)[:, 1]
                test_preds += fold_test_preds / self.config['N_SPLITS']
                
                fold_score = accuracy_score(y_fold_valid, (valid_preds >= 0.5).astype(int))
                fold_scores.append(fold_score)
            
            overall_score = accuracy_score(y_train, (oof_preds >= 0.5).astype(int))
            print(f"    {name} スコア: {overall_score:.6f} (±{np.std(fold_scores):.6f})")
            
            oof_predictions[name] = oof_preds
            test_predictions[name] = test_preds
            model_scores[name] = overall_score
        
        return {
            'oof': oof_predictions,
            'test': test_predictions,
            'scores': model_scores
        }
    
    def create_level2_ensemble(self, level1_results, y_train):
        """Level 2メタ学習アンサンブル"""
        print("\n=== Level 2 メタ学習アンサンブル ===")
        
        oof_df = pd.DataFrame(level1_results['oof'])
        test_df = pd.DataFrame(level1_results['test'])
        
        print(f"メタ学習データ形状: {oof_df.shape}")
        
        strategy_results = {}
        
        # 1. 単純平均
        final_oof = oof_df.mean(axis=1)
        final_test = test_df.mean(axis=1)
        score = accuracy_score(y_train, (final_oof >= 0.5).astype(int))
        print(f"  simple_mean: {score:.6f}")
        strategy_results['simple_mean'] = {'test': final_test, 'score': score}
        
        # 2. 重み付き平均
        model_scores = []
        for col in oof_df.columns:
            score = accuracy_score(y_train, (oof_df[col] >= 0.5).astype(int))
            model_scores.append(score)
        
        weights = np.array(model_scores)
        weights = weights / weights.sum()
        
        final_oof = np.average(oof_df.values, axis=1, weights=weights)
        final_test = np.average(test_df.values, axis=1, weights=weights)
        score = accuracy_score(y_train, (final_oof >= 0.5).astype(int))
        print(f"  weighted_mean: {score:.6f}")
        strategy_results['weighted_mean'] = {'test': final_test, 'score': score}
        
        # 3. ロジスティック回帰メタモデル
        try:
            meta_model = LogisticRegression(
                C=1.0, 
                max_iter=1000, 
                random_state=self.config['RANDOM_STATE']
            )
            meta_model.fit(oof_df, y_train)
            
            final_oof = meta_model.predict_proba(oof_df)[:, 1]
            final_test = meta_model.predict_proba(test_df)[:, 1]
            score = accuracy_score(y_train, (final_oof >= 0.5).astype(int))
            print(f"  logistic_meta: {score:.6f}")
            strategy_results['logistic_meta'] = {'test': final_test, 'score': score}
        except Exception as e:
            print(f"  logistic_meta: エラー ({str(e)})")
        
        # 最高スコアの戦略を選択
        best_strategy = max(strategy_results.keys(), key=lambda k: strategy_results[k]['score'])
        best_score = strategy_results[best_strategy]['score']
        best_test_preds = strategy_results[best_strategy]['test']
        
        print(f"  最適戦略: {best_strategy} (スコア: {best_score:.6f})")
        
        return best_test_preds, best_score, best_strategy
    
    def create_submission(self, predictions, test_ids, method_name):
        """提出ファイル作成"""
        print("\n=== 提出ファイル作成 ===")
        
        binary_preds = (predictions >= 0.5).astype(int)
        reverse_mapping = {v: k for k, v in self.config['TARGET_MAPPING'].items()}
        string_preds = [reverse_mapping[pred] for pred in binary_preds]
        
        submission = pd.DataFrame({
            'id': test_ids,
            'Personality': string_preds
        })
        
        filename = f'advanced_ensemble_{method_name.replace("_", "-")}.csv'
        submission_path = self.config['OUTPUT_PATH'] / filename
        submission.to_csv(submission_path, index=False)
        
        print(f"提出ファイル保存: {submission_path}")
        print(f"予測分布:")
        print(submission['Personality'].value_counts())
        
        return submission
    
    def run_advanced_ensemble(self):
        """高度アンサンブルフルパイプライン実行"""
        print("=== 高度アンサンブルパイプライン開始 ===")
        
        # 1. データ読み込み
        X_train, y_train, X_test, train_ids, test_ids = self.load_revolutionary_data()
        
        # 2. 多様なモデル設定
        models = self.setup_diverse_models()
        
        # 3. Level 1モデル訓練
        level1_results = self.train_level1_models(X_train, y_train, X_test, models)
        
        # 4. Level 2アンサンブル
        final_predictions, final_score, method_name = self.create_level2_ensemble(level1_results, y_train)
        
        # 5. 提出ファイル作成
        submission = self.create_submission(final_predictions, test_ids, method_name)
        
        print(f"\n=== 高度アンサンブル完了 ===")
        print(f"最終結果: {final_score:.6f} ({method_name})")
        print(f"GMベースライン(0.975708)との差: {final_score - 0.975708:+.6f}")
        
        return submission, final_score

if __name__ == "__main__":
    ensemble = AdvancedEnsemble()
    submission, score = ensemble.run_advanced_ensemble()