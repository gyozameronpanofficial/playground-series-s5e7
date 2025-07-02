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
from sklearn.linear_model import LogisticRegression, RidgeClassifier, ElasticNet
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Optimization
try:
    import optuna
except ImportError:
    optuna = None

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
        
        # シード設定
        np.random.seed(self.config['RANDOM_STATE'])
        
    def load_revolutionary_data(self):
        """革新的特徴量データの読み込み"""
        print("=== 革新的特徴量データ読み込み ===")
        
        train_df = pd.read_csv(self.config['DATA_PATH'] / 'revolutionary_train.csv')
        test_df = pd.read_csv(self.config['DATA_PATH'] / 'revolutionary_test.csv')
        
        # 特徴量とターゲットの分離
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
                    n_estimators=2000,
                    max_depth=6,
                    colsample_bytree=0.45,
                    subsample=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
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
                    n_estimators=2000,
                    max_depth=6,
                    colsample_bytree=0.45,
                    subsample=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=self.config['RANDOM_STATE'],
                    verbosity=-1
                ))
            ]),
            
            'CatBoost': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', cb.CatBoostClassifier(
                    loss_function='Logloss',
                    learning_rate=0.01,
                    iterations=1500,
                    max_depth=6,
                    random_state=self.config['RANDOM_STATE'],
                    verbose=False
                ))
            ]),
            
            # Tree-based Models
            'RandomForest': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', RandomForestClassifier(
                    n_estimators=500,
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
                    n_estimators=500,
                    max_depth=8,
                    min_samples_leaf=10,
                    max_features='sqrt',
                    random_state=self.config['RANDOM_STATE'],
                    n_jobs=-1
                ))
            ]),
            
            'HistGradientBoosting': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', HistGradientBoostingClassifier(
                    learning_rate=0.01,
                    max_iter=1000,
                    max_depth=6,
                    l2_regularization=0.1,
                    random_state=self.config['RANDOM_STATE']
                ))
            ]),
            
            # Linear Models (with scaling)
            'LogisticRegression': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(
                    C=0.1,
                    max_iter=1000,
                    random_state=self.config['RANDOM_STATE']
                ))
            ]),
            
            'Ridge': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('scaler', StandardScaler()),
                ('model', RidgeClassifier(
                    alpha=1.0,
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
            ]),
            
            # Distance-based Models
            'KNN': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('scaler', StandardScaler()),
                ('model', KNeighborsClassifier(
                    n_neighbors=15,
                    weights='distance'
                ))
            ]),
            
            # Naive Bayes
            'NaiveBayes': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('scaler', StandardScaler()),
                ('model', GaussianNB())
            ])
        }
        
        print(f"設定完了: {len(models)} モデル")
        return models
    
    def train_level1_models(self, X_train, y_train, X_test, models):
        """Level 1モデルの訓練"""
        print("=== Level 1 モデル訓練 ===")
        
        # 複数CV戦略
        cv_strategies = {
            'StratifiedKFold': StratifiedKFold(
                n_splits=self.config['N_SPLITS'], 
                shuffle=True, 
                random_state=self.config['RANDOM_STATE']
            ),
            'RepeatedStratifiedKFold': RepeatedStratifiedKFold(
                n_splits=self.config['N_SPLITS'], 
                n_repeats=self.config['N_REPEATS'],
                random_state=self.config['RANDOM_STATE']
            )
        }
        
        level1_results = {}
        
        for cv_name, cv in cv_strategies.items():
            print(f"\n--- {cv_name} による訓練 ---")
            
            oof_predictions = {}
            test_predictions = {}
            model_scores = {}
            
            for name, model in models.items():
                print(f"  {name} 訓練中...")
                
                # OOF予測とテスト予測の初期化
                oof_preds = np.zeros(len(X_train))
                test_preds = np.zeros(len(X_test))
                fold_scores = []
                
                # クロスバリデーション実行
                for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
                    # 分割
                    X_fold_train = X_train.iloc[train_idx]
                    X_fold_valid = X_train.iloc[valid_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    y_fold_valid = y_train.iloc[valid_idx]
                    
                    # 訓練
                    model.fit(X_fold_train, y_fold_train)
                    
                    # OOF予測
                    valid_preds = model.predict_proba(X_fold_valid)[:, 1]
                    oof_preds[valid_idx] = valid_preds
                    
                    # テスト予測（後で平均）
                    fold_test_preds = model.predict_proba(X_test)[:, 1]
                    test_preds += fold_test_preds / cv.get_n_splits()
                    
                    # スコア計算
                    fold_score = accuracy_score(y_fold_valid, (valid_preds >= 0.5).astype(int))
                    fold_scores.append(fold_score)
                
                # 全体スコア計算
                overall_score = accuracy_score(y_train, (oof_preds >= 0.5).astype(int))
                print(f"    {name} スコア: {overall_score:.6f} (±{np.std(fold_scores):.6f})")
                
                # 結果保存
                oof_predictions[name] = oof_preds
                test_predictions[name] = test_preds
                model_scores[name] = overall_score
            
            level1_results[cv_name] = {
                'oof': oof_predictions,
                'test': test_predictions,
                'scores': model_scores
            }
        
        return level1_results\n    \n    def create_level2_ensemble(self, level1_results, y_train):\n        \"\"\"Level 2メタ学習アンサンブル\"\"\"\n        print(\"\\n=== Level 2 メタ学習アンサンブル ===\")\n        \n        best_ensemble_results = {}\n        \n        for cv_name, results in level1_results.items():\n            print(f\"\\n--- {cv_name} によるLevel 2アンサンブル ---\")\n            \n            # OOF予測をDataFrameに変換\n            oof_df = pd.DataFrame(results['oof'])\n            test_df = pd.DataFrame(results['test'])\n            \n            print(f\"メタ学習データ形状: {oof_df.shape}\")\n            \n            # 複数のメタ学習戦略\n            meta_strategies = {\n                'simple_mean': lambda x: x.mean(axis=1),\n                'weighted_mean': self._create_weighted_ensemble,\n                'logistic_meta': self._create_logistic_meta_model,\n                'ridge_meta': self._create_ridge_meta_model\n            }\n            \n            strategy_results = {}\n            \n            for strategy_name, strategy_func in meta_strategies.items():\n                try:\n                    if strategy_name in ['simple_mean']:\n                        final_oof = strategy_func(oof_df)\n                        final_test = strategy_func(test_df)\n                    else:\n                        final_oof, final_test = strategy_func(oof_df, test_df, y_train)\n                    \n                    # スコア計算\n                    score = accuracy_score(y_train, (final_oof >= 0.5).astype(int))\n                    print(f\"  {strategy_name}: {score:.6f}\")\n                    \n                    strategy_results[strategy_name] = {\n                        'oof': final_oof,\n                        'test': final_test,\n                        'score': score\n                    }\n                except Exception as e:\n                    print(f\"  {strategy_name}: エラー ({str(e)})\")\n                    continue\n            \n            # 最高スコアの戦略を選択\n            if strategy_results:\n                best_strategy = max(strategy_results.keys(), key=lambda k: strategy_results[k]['score'])\n                best_score = strategy_results[best_strategy]['score']\n                best_test_preds = strategy_results[best_strategy]['test']\n                \n                print(f\"  最適戦略: {best_strategy} (スコア: {best_score:.6f})\")\n                \n                best_ensemble_results[cv_name] = {\n                    'strategy': best_strategy,\n                    'score': best_score,\n                    'test': best_test_preds\n                }\n        \n        return best_ensemble_results\n    \n    def _create_weighted_ensemble(self, oof_df, test_df, y_train):\n        \"\"\"重み付きアンサンブル\"\"\"\n        # 各モデルのスコアベースで重み計算\n        model_scores = []\n        for col in oof_df.columns:\n            score = accuracy_score(y_train, (oof_df[col] >= 0.5).astype(int))\n            model_scores.append(score)\n        \n        # スコアを重みに変換（正規化）\n        weights = np.array(model_scores)\n        weights = weights / weights.sum()\n        \n        # 重み付き平均\n        final_oof = np.average(oof_df.values, axis=1, weights=weights)\n        final_test = np.average(test_df.values, axis=1, weights=weights)\n        \n        return final_oof, final_test\n    \n    def _create_logistic_meta_model(self, oof_df, test_df, y_train):\n        \"\"\"ロジスティック回帰メタモデル\"\"\"\n        meta_model = LogisticRegression(\n            C=1.0, \n            max_iter=1000, \n            random_state=self.config['RANDOM_STATE']\n        )\n        \n        # メタモデル訓練\n        meta_model.fit(oof_df, y_train)\n        \n        # 最終予測\n        final_oof = meta_model.predict_proba(oof_df)[:, 1]\n        final_test = meta_model.predict_proba(test_df)[:, 1]\n        \n        return final_oof, final_test\n    \n    def _create_ridge_meta_model(self, oof_df, test_df, y_train):\n        \"\"\"リッジ回帰メタモデル\"\"\"\n        from sklearn.linear_model import Ridge\n        \n        meta_model = Ridge(\n            alpha=1.0,\n            random_state=self.config['RANDOM_STATE']\n        )\n        \n        # メタモデル訓練\n        meta_model.fit(oof_df, y_train)\n        \n        # 最終予測（確率に変換）\n        final_oof = meta_model.predict(oof_df)\n        final_test = meta_model.predict(test_df)\n        \n        # [0,1]に正規化\n        final_oof = np.clip(final_oof, 0, 1)\n        final_test = np.clip(final_test, 0, 1)\n        \n        return final_oof, final_test\n    \n    def select_final_ensemble(self, best_ensemble_results):\n        \"\"\"最終アンサンブルの選択\"\"\"\n        print(\"\\n=== 最終アンサンブル選択 ===\")\n        \n        # 全戦略から最高スコアを選択\n        best_overall = None\n        best_score = 0\n        best_predictions = None\n        \n        for cv_name, results in best_ensemble_results.items():\n            if results['score'] > best_score:\n                best_score = results['score']\n                best_overall = f\"{cv_name}_{results['strategy']}\"\n                best_predictions = results['test']\n        \n        print(f\"最終選択: {best_overall}\")\n        print(f\"最終スコア: {best_score:.6f}\")\n        print(f\"GMベースライン(0.975708)との差: {best_score - 0.975708:+.6f}\")\n        \n        return best_predictions, best_score, best_overall\n    \n    def create_submission(self, predictions, test_ids, method_name):\n        \"\"\"提出ファイル作成\"\"\"\n        print(\"\\n=== 提出ファイル作成 ===\")\n        \n        # 予測を二値分類に変換\n        binary_preds = (predictions >= 0.5).astype(int)\n        \n        # ラベルを文字列に変換\n        reverse_mapping = {v: k for k, v in self.config['TARGET_MAPPING'].items()}\n        string_preds = [reverse_mapping[pred] for pred in binary_preds]\n        \n        # 提出DataFrame作成\n        submission = pd.DataFrame({\n            'id': test_ids,\n            'Personality': string_preds\n        })\n        \n        # 保存\n        filename = f'advanced_ensemble_{method_name.replace(\"_\", \"-\")}.csv'\n        submission_path = self.config['OUTPUT_PATH'] / filename\n        submission.to_csv(submission_path, index=False)\n        \n        print(f\"提出ファイル保存: {submission_path}\")\n        print(f\"予測分布:\")\n        print(submission['Personality'].value_counts())\n        \n        return submission\n    \n    def run_advanced_ensemble(self):\n        \"\"\"高度アンサンブルフルパイプライン実行\"\"\"\n        print(\"=== 高度アンサンブルパイプライン開始 ===\")\n        \n        # 1. データ読み込み\n        X_train, y_train, X_test, train_ids, test_ids = self.load_revolutionary_data()\n        \n        # 2. 多様なモデル設定\n        models = self.setup_diverse_models()\n        \n        # 3. Level 1モデル訓練\n        level1_results = self.train_level1_models(X_train, y_train, X_test, models)\n        \n        # 4. Level 2アンサンブル\n        best_ensemble_results = self.create_level2_ensemble(level1_results, y_train)\n        \n        # 5. 最終アンサンブル選択\n        final_predictions, final_score, method_name = self.select_final_ensemble(best_ensemble_results)\n        \n        # 6. 提出ファイル作成\n        submission = self.create_submission(final_predictions, test_ids, method_name)\n        \n        print(f\"\\n=== 高度アンサンブル完了 ===\")\n        print(f\"最終結果: {final_score:.6f} ({method_name})\")\n        \n        return submission, final_score\n\nif __name__ == \"__main__\":\n    ensemble = AdvancedEnsemble()\n    submission, score = ensemble.run_advanced_ensemble()