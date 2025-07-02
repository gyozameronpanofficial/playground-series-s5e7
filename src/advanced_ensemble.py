"""
高度アンサンブル手法の実装
多層スタッキング、動的重み付け、多様なモデルの組み合わせ
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                             HistGradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression, ElasticNet, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from scipy import stats

class AdvancedEnsemble:
    """高度アンサンブル手法クラス"""
    
    def __init__(self):
        self.config = {
            'RANDOM_STATE': 42,
            'N_SPLITS': 5,
            'N_REPEATS': 2,
            'TARGET_COL': 'target',
            'DATA_PATH': Path('/Users/osawa/kaggle/playground-series-s5e7/data/processed')
        }
        self.level1_models = {}
        self.level2_models = {}
        self.oof_predictions = {}
        self.test_predictions = {}
        
    def load_enhanced_data(self):
        """拡張特徴量データの読み込み"""
        print("=== 拡張特徴量データ読み込み ===")
        
        train_df = pd.read_csv(self.config['DATA_PATH'] / 'enhanced_train.csv')
        test_df = pd.read_csv(self.config['DATA_PATH'] / 'enhanced_test.csv')
        
        # 特徴量とターゲットを分離
        X_train = train_df.drop(self.config['TARGET_COL'], axis=1)
        y_train = train_df[self.config['TARGET_COL']]
        X_test = test_df.copy()
        
        print(f"訓練データ形状: {X_train.shape}")
        print(f"テストデータ形状: {X_test.shape}")
        print(f"ターゲット分布: {y_train.value_counts().to_dict()}")
        
        return X_train, y_train, X_test
    
    def setup_diverse_models(self):
        """多様なモデル群の設定"""
        print("\n=== 多様なモデル群設定 ===")
        
        # レベル1モデル群（多様性を重視）
        self.level1_models = {
            # Gradient Boosting系
            'XGBoost': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', xgb.XGBClassifier(
                    objective='binary:logistic',
                    learning_rate=0.01,  # より慎重な学習
                    n_estimators=2000,
                    max_depth=6,
                    colsample_bytree=0.8,
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
                    learning_rate=0.01,
                    n_estimators=2000,
                    max_depth=6,
                    colsample_bytree=0.8,
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
                    iterations=2000,
                    max_depth=6,
                    reg_lambda=1.0,
                    random_state=self.config['RANDOM_STATE'],
                    verbose=False
                ))
            ]),
            
            # Tree系
            'RandomForest': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', RandomForestClassifier(
                    n_estimators=500,
                    max_depth=8,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=self.config['RANDOM_STATE']
                ))
            ]),
            
            'ExtraTrees': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=8,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=self.config['RANDOM_STATE']
                ))
            ]),
            
            'HistGradientBoosting': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', HistGradientBoostingClassifier(
                    learning_rate=0.02,
                    max_iter=1000,
                    max_depth=6,
                    min_samples_leaf=5,
                    l2_regularization=1.0,
                    random_state=self.config['RANDOM_STATE']
                ))
            ]),
            
            # 線形系
            'LogisticRegression': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(
                    C=0.1,
                    max_iter=2000,
                    random_state=self.config['RANDOM_STATE']
                ))
            ]),
            
            'ElasticNet': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('scaler', StandardScaler()),
                ('model', ElasticNet(
                    alpha=0.1,
                    l1_ratio=0.5,
                    max_iter=2000,
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
            
            # 距離系
            'KNN': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('scaler', StandardScaler()),
                ('model', KNeighborsClassifier(
                    n_neighbors=15,
                    weights='distance',
                    metric='minkowski'
                ))
            ]),
            
            'SVM_RBF': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('scaler', StandardScaler()),
                ('model', SVC(
                    C=1.0,
                    kernel='rbf',
                    probability=True,
                    random_state=self.config['RANDOM_STATE']
                ))
            ]),
            
            # ニューラルネットワーク
            'MLP': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('scaler', StandardScaler()),
                ('model', MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    learning_rate_init=0.001,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=self.config['RANDOM_STATE']
                ))
            ])
        }
        
        print(f"レベル1モデル数: {len(self.level1_models)}")
        
        # レベル2メタモデル群
        self.level2_models = {
            'LogisticMeta': LogisticRegression(
                C=0.1, 
                max_iter=1000, 
                random_state=self.config['RANDOM_STATE']
            ),
            'XGBoostMeta': xgb.XGBClassifier(
                objective='binary:logistic',
                learning_rate=0.05,
                n_estimators=100,
                max_depth=3,
                random_state=self.config['RANDOM_STATE'],
                verbosity=0
            ),
            'RidgeMeta': RidgeClassifier(
                alpha=1.0,
                random_state=self.config['RANDOM_STATE']
            )
        }
        
        print(f"レベル2メタモデル数: {len(self.level2_models)}")
    
    def train_level1_models(self, X_train, y_train):
        """レベル1モデルの訓練"""
        print("\n=== レベル1モデル訓練 ===")
        
        # クロスバリデーション設定
        cv = StratifiedKFold(
            n_splits=self.config['N_SPLITS'],
            shuffle=True,
            random_state=self.config['RANDOM_STATE']
        )
        
        # OOF予測とテスト予測の初期化
        n_models = len(self.level1_models)
        oof_predictions = np.zeros((len(X_train), n_models))
        test_predictions = np.zeros((len(X_train), n_models))  # テストデータサイズに合わせて後で修正
        
        model_scores = {}
        
        for model_idx, (name, model) in enumerate(self.level1_models.items()):
            print(f"\n--- {name} 訓練中 ---")
            
            # モデル別のOOF予測
            oof_preds = np.zeros(len(X_train))
            fold_scores = []
            
            for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
                # データ分割
                X_fold_train = X_train.iloc[train_idx]
                X_fold_valid = X_train.iloc[valid_idx]
                y_fold_train = y_train.iloc[train_idx]
                y_fold_valid = y_train.iloc[valid_idx]
                
                # モデル訓練
                try:
                    model.fit(X_fold_train, y_fold_train)
                    
                    # 予測
                    if hasattr(model, 'predict_proba'):
                        valid_preds = model.predict_proba(X_fold_valid)[:, 1]
                    else:
                        # ElasticNetなど、predict_probaがないモデル
                        valid_preds = model.predict(X_fold_valid)
                    
                    oof_preds[valid_idx] = valid_preds
                    
                    # スコア計算
                    fold_score = accuracy_score(y_fold_valid, (valid_preds >= 0.5).astype(int))
                    fold_scores.append(fold_score)
                    print(f"  Fold {fold+1}: {fold_score:.6f}")
                    
                except Exception as e:
                    print(f"  エラー in Fold {fold+1}: {str(e)}")
                    fold_scores.append(0.5)  # デフォルトスコア
            
            # モデルの全体スコア
            if len(fold_scores) > 0:
                model_score = np.mean(fold_scores)
                model_scores[name] = {
                    'score': model_score,
                    'std': np.std(fold_scores)
                }
                print(f"  {name} 全体スコア: {model_score:.6f} (±{np.std(fold_scores):.6f})")
            
            # OOF予測を保存
            oof_predictions[:, model_idx] = oof_preds
            self.oof_predictions[name] = oof_preds
        
        # レベル1結果サマリー
        print(f"\n=== レベル1モデル結果サマリー ===")
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        for name, scores in sorted_models:
            print(f"{name}: {scores['score']:.6f} (±{scores['std']:.6f})")
        
        return oof_predictions, model_scores
    
    def train_level2_ensemble(self, oof_predictions, y_train):
        """レベル2メタモデルの訓練"""
        print(f"\n=== レベル2アンサンブル訓練 ===")
        
        # OOF予測をDataFrameに変換
        level1_features = pd.DataFrame(
            oof_predictions, 
            columns=list(self.level1_models.keys())
        )
        
        print(f"レベル1特徴量形状: {level1_features.shape}")
        
        # クロスバリデーション
        cv = StratifiedKFold(
            n_splits=self.config['N_SPLITS'],
            shuffle=True,
            random_state=self.config['RANDOM_STATE']
        )
        
        meta_scores = {}
        
        for name, meta_model in self.level2_models.items():
            print(f"\n--- {name} メタモデル訓練 ---")
            
            fold_scores = []
            for fold, (train_idx, valid_idx) in enumerate(cv.split(level1_features, y_train)):
                X_meta_train = level1_features.iloc[train_idx]
                X_meta_valid = level1_features.iloc[valid_idx]
                y_meta_train = y_train.iloc[train_idx]
                y_meta_valid = y_train.iloc[valid_idx]
                
                # メタモデル訓練
                meta_model.fit(X_meta_train, y_meta_train)
                
                # 予測
                if hasattr(meta_model, 'predict_proba'):
                    meta_preds = meta_model.predict_proba(X_meta_valid)[:, 1]
                else:
                    meta_preds = meta_model.predict(X_meta_valid)
                
                # スコア計算
                fold_score = accuracy_score(y_meta_valid, (meta_preds >= 0.5).astype(int))
                fold_scores.append(fold_score)
                print(f"  Fold {fold+1}: {fold_score:.6f}")
            
            # メタモデルのスコア
            meta_score = np.mean(fold_scores)
            meta_scores[name] = {
                'score': meta_score,
                'std': np.std(fold_scores)
            }
            print(f"  {name} 全体スコア: {meta_score:.6f} (±{np.std(fold_scores):.6f})")
        
        # 最良のメタモデルを選択
        best_meta = max(meta_scores.items(), key=lambda x: x[1]['score'])
        print(f"\n最良メタモデル: {best_meta[0]} ({best_meta[1]['score']:.6f})")
        
        # 最良メタモデルで全データを訓練
        best_meta_model = self.level2_models[best_meta[0]]
        best_meta_model.fit(level1_features, y_train)
        
        return best_meta_model, meta_scores, best_meta[0]
    
    def advanced_ensemble_techniques(self, oof_predictions, y_train):
        """高度アンサンブル技術"""
        print(f"\n=== 高度アンサンブル技術 ===")
        
        results = {}
        
        # 1. 単純平均
        simple_avg = np.mean(oof_predictions, axis=1)
        simple_score = accuracy_score(y_train, (simple_avg >= 0.5).astype(int))
        results['simple_average'] = simple_score
        print(f"単純平均: {simple_score:.6f}")
        
        # 2. 重み付き平均（性能ベース）
        model_names = list(self.level1_models.keys())
        weights = []
        for name in model_names:
            if name in self.oof_predictions:
                pred = self.oof_predictions[name]
                score = accuracy_score(y_train, (pred >= 0.5).astype(int))
                weights.append(score)
            else:
                weights.append(0.5)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # 正規化
        
        weighted_avg = np.average(oof_predictions, axis=1, weights=weights)
        weighted_score = accuracy_score(y_train, (weighted_avg >= 0.5).astype(int))
        results['weighted_average'] = weighted_score
        print(f"重み付き平均: {weighted_score:.6f}")
        
        # 3. ランクアベレージング
        rank_avg = np.mean(stats.rankdata(oof_predictions, axis=0), axis=1)
        rank_avg = (rank_avg - rank_avg.min()) / (rank_avg.max() - rank_avg.min())  # 正規化
        rank_score = accuracy_score(y_train, (rank_avg >= 0.5).astype(int))
        results['rank_average'] = rank_score
        print(f"ランク平均: {rank_score:.6f}")
        
        # 4. 動的重み付け（予測値に応じて重みを調整）
        dynamic_preds = np.zeros(len(oof_predictions))
        for i in range(len(oof_predictions)):
            row_preds = oof_predictions[i, :]
            # 予測値の分散を重みとする
            pred_std = np.std(row_preds)
            if pred_std > 0:
                # 分散が大きい場合は単純平均
                dynamic_preds[i] = np.mean(row_preds)
            else:
                # 分散が小さい場合は最高性能モデルの予測を重視
                best_idx = np.argmax(weights)
                dynamic_preds[i] = row_preds[best_idx]
        
        dynamic_score = accuracy_score(y_train, (dynamic_preds >= 0.5).astype(int))
        results['dynamic_weighting'] = dynamic_score
        print(f"動的重み付け: {dynamic_score:.6f}")
        
        return results
    
    def run_advanced_ensemble(self):
        """高度アンサンブル全体実行"""
        print("=== 高度アンサンブル手法実行 ===\n")
        
        # 1. データ読み込み
        X_train, y_train, X_test = self.load_enhanced_data()
        
        # 2. 多様なモデル設定
        self.setup_diverse_models()
        
        # 3. レベル1モデル訓練
        oof_predictions, level1_scores = self.train_level1_models(X_train, y_train)
        
        # 4. レベル2アンサンブル
        best_meta_model, meta_scores, best_meta_name = self.train_level2_ensemble(
            oof_predictions, y_train
        )
        
        # 5. 高度アンサンブル技術
        ensemble_results = self.advanced_ensemble_techniques(oof_predictions, y_train)
        
        # 6. 結果まとめ
        print(f"\n=== 最終結果まとめ ===")
        
        # 最良の単一モデル
        best_single = max(level1_scores.items(), key=lambda x: x[1]['score'])
        print(f"最良単一モデル: {best_single[0]} ({best_single[1]['score']:.6f})")
        
        # 最良メタモデル
        best_meta_score = meta_scores[best_meta_name]['score']
        print(f"最良メタモデル: {best_meta_name} ({best_meta_score:.6f})")
        
        # 最良アンサンブル手法
        best_ensemble = max(ensemble_results.items(), key=lambda x: x[1])
        print(f"最良アンサンブル: {best_ensemble[0]} ({best_ensemble[1]:.6f})")
        
        # 改善効果
        baseline_score = best_single[1]['score']
        final_score = max(best_meta_score, best_ensemble[1])
        improvement = final_score - baseline_score
        
        print(f"\n最終スコア: {final_score:.6f}")
        print(f"ベースラインからの改善: {improvement:+.6f}")
        print(f"目標0.975708まで: {0.975708 - final_score:+.6f}")
        
        return {
            'level1_scores': level1_scores,
            'meta_scores': meta_scores,
            'ensemble_results': ensemble_results,
            'final_score': final_score,
            'improvement': improvement
        }

if __name__ == "__main__":
    ensemble = AdvancedEnsemble()
    results = ensemble.run_advanced_ensemble()