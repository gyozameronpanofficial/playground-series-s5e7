"""
最終提出モデル（クリーン版）
これまでの分析結果を統合した実用的なアプローチ
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

class FinalSubmissionModel:
    """最終提出モデルクラス"""
    
    def __init__(self):
        self.config = {
            'RANDOM_STATE': 42,
            'N_SPLITS': 5,
            'TARGET_COL': 'Personality',
            'TARGET_MAPPING': {"Extrovert": 0, "Introvert": 1},
            'DATA_PATH': Path('/Users/osawa/kaggle/playground-series-s5e7/data/raw')
        }
        
    def load_and_preprocess_data(self):
        """データ読み込みと前処理"""
        print("=== データ読み込み・前処理 ===")
        
        # データ読み込み
        train_df = pd.read_csv(self.config['DATA_PATH'] / 'train.csv')
        test_df = pd.read_csv(self.config['DATA_PATH'] / 'test.csv')
        
        # 特徴量抽出
        feature_cols = [col for col in train_df.columns 
                       if col not in ['id', self.config['TARGET_COL']]]
        
        X_train = train_df[feature_cols].copy()
        y_train = train_df[self.config['TARGET_COL']].map(self.config['TARGET_MAPPING'])
        X_test = test_df[feature_cols].copy()
        
        print(f"元データ形状: X_train={X_train.shape}, X_test={X_test.shape}")
        
        # GMベースライン前処理
        # 数値特徴量処理
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train[numeric_cols] = X_train[numeric_cols].fillna(-1).astype(np.int16).astype(str)
        X_test[numeric_cols] = X_test[numeric_cols].fillna(-1).astype(np.int16).astype(str)
        
        # カテゴリカル特徴量処理
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        X_train[categorical_cols] = X_train[categorical_cols].astype(str).fillna("missing")
        X_test[categorical_cols] = X_test[categorical_cols].astype(str).fillna("missing")
        
        print(f"前処理後形状: X_train={X_train.shape}, X_test={X_test.shape}")
        
        return X_train, y_train, X_test
    
    def create_advanced_features(self, X_train, X_test):
        """高度特徴量生成"""
        print("=== 高度特徴量生成 ===")
        
        from itertools import combinations
        
        original_cols = list(X_train.columns)
        print(f"元の特徴量数: {len(original_cols)}")
        
        # 2-gram特徴量
        print("2-gram特徴量生成...")
        for col1, col2 in combinations(original_cols, 2):
            feature_name = f"{col1}-{col2}"
            X_train[feature_name] = X_train[col1] + "-" + X_train[col2]
            X_test[feature_name] = X_test[col1] + "-" + X_test[col2]
        
        print(f"2-gram後: {X_train.shape[1]} 特徴量")
        
        # 3-gram特徴量
        print("3-gram特徴量生成...")
        for col1, col2, col3 in combinations(original_cols, 3):
            feature_name = f"{col1}-{col2}-{col3}"
            X_train[feature_name] = (X_train[col1] + "-" + 
                                    X_train[col2] + "-" + 
                                    X_train[col3])
            X_test[feature_name] = (X_test[col1] + "-" + 
                                   X_test[col2] + "-" + 
                                   X_test[col3])
        
        print(f"最終特徴量数: {X_train.shape[1]}")
        
        return X_train, X_test
    
    def setup_models(self):
        """最適化されたモデル群設定"""
        print("=== 最適化モデル群設定 ===")
        
        models = {
            # 高性能XGBoost
            'XGBoost_Optimized': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    learning_rate=0.01,
                    n_estimators=2000,
                    max_depth=6,
                    colsample_bytree=0.7,
                    subsample=0.8,
                    reg_alpha=0.3,
                    reg_lambda=1.0,
                    min_child_weight=4,
                    random_state=self.config['RANDOM_STATE'],
                    verbosity=0
                ))
            ]),
            
            # 高性能LightGBM
            'LightGBM_Optimized': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', lgb.LGBMClassifier(
                    objective='binary',
                    metric='logloss',
                    learning_rate=0.01,
                    n_estimators=2000,
                    max_depth=6,
                    colsample_bytree=0.7,
                    subsample=0.8,
                    reg_alpha=0.3,
                    reg_lambda=1.0,
                    min_child_samples=10,
                    num_leaves=50,
                    random_state=self.config['RANDOM_STATE'],
                    verbosity=-1
                ))
            ]),
            
            # 高性能CatBoost
            'CatBoost_Optimized': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', cb.CatBoostClassifier(
                    loss_function='Logloss',
                    learning_rate=0.01,
                    iterations=2000,
                    max_depth=6,
                    reg_lambda=1.0,
                    subsample=0.8,
                    random_strength=0.5,
                    bagging_temperature=0.5,
                    random_state=self.config['RANDOM_STATE'],
                    verbose=False
                ))
            ]),
            
            # 最適化RandomForest
            'RandomForest_Optimized': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', RandomForestClassifier(
                    n_estimators=500,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    max_samples=0.8,
                    random_state=self.config['RANDOM_STATE']
                ))
            ])
        }
        
        print(f"設定完了: {len(models)} モデル")
        return models
    
    def train_models_with_oof(self, models, X_train, y_train):
        """モデル訓練とOOF予測生成"""
        print("=== モデル訓練・OOF予測生成 ===")
        
        cv = StratifiedKFold(
            n_splits=self.config['N_SPLITS'],
            shuffle=True,
            random_state=self.config['RANDOM_STATE']
        )
        
        oof_predictions = {}
        model_scores = {}
        
        for name, model in models.items():
            print(f"\n--- {name} 訓練中 ---")
            
            oof_preds = np.zeros(len(X_train))
            fold_scores = []
            
            for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx]
                X_fold_valid = X_train.iloc[valid_idx]
                y_fold_train = y_train.iloc[train_idx]
                y_fold_valid = y_train.iloc[valid_idx]
                
                # モデル訓練
                model.fit(X_fold_train, y_fold_train)
                
                # OOF予測
                valid_preds = model.predict_proba(X_fold_valid)[:, 1]
                oof_preds[valid_idx] = valid_preds
                
                # フォールドスコア
                fold_score = accuracy_score(y_fold_valid, (valid_preds >= 0.5).astype(int))
                fold_scores.append(fold_score)
                print(f"  Fold {fold+1}: {fold_score:.6f}")
            
            # 全体スコア
            overall_score = accuracy_score(y_train, (oof_preds >= 0.5).astype(int))
            model_scores[name] = {
                'score': overall_score,
                'std': np.std(fold_scores)
            }
            oof_predictions[name] = oof_preds
            
            print(f"  {name} 全体スコア: {overall_score:.6f} (±{np.std(fold_scores):.6f})")
        
        return oof_predictions, model_scores
    
    def create_optimized_ensemble(self, oof_predictions, y_train):
        """最適化アンサンブル"""
        print("=== 最適化アンサンブル ===")
        
        # OOF予測をDataFrameに変換
        oof_df = pd.DataFrame(oof_predictions)
        
        # 複数のアンサンブル手法を試行
        ensemble_results = {}
        
        # 1. 単純平均
        simple_avg = oof_df.mean(axis=1)
        simple_score = accuracy_score(y_train, (simple_avg >= 0.5).astype(int))
        ensemble_results['simple_average'] = simple_score
        print(f"単純平均: {simple_score:.6f}")
        
        # 2. 性能ベース重み付き平均
        weights = []
        for col in oof_df.columns:
            pred = oof_df[col]
            score = accuracy_score(y_train, (pred >= 0.5).astype(int))
            weights.append(score)
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # 正規化
        
        weighted_avg = np.average(oof_df.values, axis=1, weights=weights)
        weighted_score = accuracy_score(y_train, (weighted_avg >= 0.5).astype(int))
        ensemble_results['weighted_average'] = weighted_score
        print(f"重み付き平均: {weighted_score:.6f}")
        print(f"重み: {dict(zip(oof_df.columns, weights))}")
        
        # 3. LogisticRegression メタモデル
        meta_model = LogisticRegression(
            C=0.1, 
            max_iter=1000, 
            random_state=self.config['RANDOM_STATE']
        )
        meta_model.fit(oof_df, y_train)
        meta_preds = meta_model.predict_proba(oof_df)[:, 1]
        meta_score = accuracy_score(y_train, (meta_preds >= 0.5).astype(int))
        ensemble_results['meta_model'] = meta_score
        print(f"メタモデル: {meta_score:.6f}")
        
        # 最良手法を選択
        best_method = max(ensemble_results.items(), key=lambda x: x[1])
        print(f"\n最良アンサンブル: {best_method[0]} ({best_method[1]:.6f})")
        
        return best_method, weights, meta_model
    
    def generate_test_predictions(self, models, X_train, y_train, X_test, 
                                 best_method, weights, meta_model):
        """テスト予測生成"""
        print("=== テスト予測生成 ===")
        
        # 全データでモデル再訓練
        test_predictions = {}
        for name, model in models.items():
            print(f"{name} 全データで再訓練中...")
            model.fit(X_train, y_train)
            test_preds = model.predict_proba(X_test)[:, 1]
            test_predictions[name] = test_preds
        
        # 最良アンサンブル手法でテスト予測を統合
        test_df = pd.DataFrame(test_predictions)
        
        if best_method[0] == 'simple_average':
            final_preds = test_df.mean(axis=1)
        elif best_method[0] == 'weighted_average':
            final_preds = np.average(test_df.values, axis=1, weights=weights)
        else:  # meta_model
            final_preds = meta_model.predict_proba(test_df)[:, 1]
        
        return final_preds
    
    def create_submission_file(self, test_predictions, test_df_original):
        """提出ファイル作成"""
        print("=== 提出ファイル作成 ===")
        
        # 二値分類に変換
        binary_preds = (test_predictions >= 0.5).astype(int)
        
        # ラベルを文字列に戻す
        reverse_mapping = {v: k for k, v in self.config['TARGET_MAPPING'].items()}
        string_preds = [reverse_mapping[pred] for pred in binary_preds]
        
        # 提出DataFrame作成
        submission = pd.DataFrame({
            'id': test_df_original['id'],
            self.config['TARGET_COL']: string_preds
        })
        
        # 保存
        submission_path = '/Users/osawa/kaggle/playground-series-s5e7/submissions/final_submission.csv'
        submission.to_csv(submission_path, index=False)
        
        print(f"提出ファイル保存: {submission_path}")
        print(f"提出データ形状: {submission.shape}")
        print(f"予測分布:")
        print(submission[self.config['TARGET_COL']].value_counts())
        
        return submission
    
    def run_final_submission(self):
        """最終提出パイプライン実行"""
        print("=== 最終提出モデル実行 ===\n")
        
        # 1. データ読み込み・前処理
        X_train, y_train, X_test = self.load_and_preprocess_data()
        test_df_original = pd.read_csv(self.config['DATA_PATH'] / 'test.csv')
        
        # 2. 高度特徴量生成
        X_train_enhanced, X_test_enhanced = self.create_advanced_features(X_train, X_test)
        
        # 3. 最適化モデル設定
        models = self.setup_models()
        
        # 4. モデル訓練・OOF予測
        oof_predictions, model_scores = self.train_models_with_oof(
            models, X_train_enhanced, y_train
        )
        
        # 5. 最適化アンサンブル
        best_method, weights, meta_model = self.create_optimized_ensemble(
            oof_predictions, y_train
        )
        
        # 6. テスト予測生成
        test_predictions = self.generate_test_predictions(
            models, X_train_enhanced, y_train, X_test_enhanced,
            best_method, weights, meta_model
        )
        
        # 7. 提出ファイル作成
        submission = self.create_submission_file(test_predictions, test_df_original)
        
        # 8. 結果サマリー
        print(f"\n=== 最終結果 ===")
        best_single_model = max(model_scores.items(), key=lambda x: x[1]['score'])
        print(f"最良単一モデル: {best_single_model[0]} ({best_single_model[1]['score']:.6f})")
        print(f"最良アンサンブル: {best_method[0]} ({best_method[1]:.6f})")
        print(f"目標0.975708との差: {0.975708 - best_method[1]:+.6f}")
        
        if best_method[1] >= 0.975708:
            print("🎉 目標スコア達成！")
        else:
            gap = 0.975708 - best_method[1]
            print(f"目標まで{gap:.6f}の改善が必要")
        
        return {
            'final_score': best_method[1],
            'model_scores': model_scores,
            'submission': submission
        }

if __name__ == "__main__":
    final_model = FinalSubmissionModel()
    results = final_model.run_final_submission()