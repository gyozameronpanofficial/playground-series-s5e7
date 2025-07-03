"""
GMベースライン完全再現版
GMの手法を100%忠実に再現し、0.975708を目指す
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from itertools import combinations

class GMExactReproduction:
    """GM手法の完全再現クラス"""
    
    def __init__(self):
        self.config = {
            'DATA_PATH': Path('/Users/osawa/kaggle/playground-series-s5e7/data/raw'),
            'OUTPUT_PATH': Path('/Users/osawa/kaggle/playground-series-s5e7/submissions'),
            'TARGET_MAPPING': {"Extrovert": 0, "Introvert": 1},
            'RANDOM_STATE': 42,
            'N_SPLITS': 5
        }
        self.config['OUTPUT_PATH'].mkdir(exist_ok=True)
        
    def load_data(self):
        """データ読み込み"""
        print("=== データ読み込み ===")
        
        self.train_df = pd.read_csv(self.config['DATA_PATH'] / 'train.csv')
        self.test_df = pd.read_csv(self.config['DATA_PATH'] / 'test.csv')
        
        self.feature_cols = [col for col in self.train_df.columns 
                           if col not in ['id', 'Personality']]
        
        print(f"データ形状: train={self.train_df.shape}, test={self.test_df.shape}")
        print(f"特徴量: {self.feature_cols}")
        
    def gm_preprocessing(self):
        """GM式前処理の完全再現"""
        print("\n=== GM式前処理 ===")
        
        X_train = self.train_df[self.feature_cols].copy()
        y_train = self.train_df['Personality'].copy()
        X_test = self.test_df[self.feature_cols].copy()
        
        # 数値特徴量処理（GMと同じ方式）
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        print(f"数値特徴量: {numeric_cols}")
        
        # 数値→文字列変換（欠損値は-1）
        X_train[numeric_cols] = X_train[numeric_cols].fillna(-1).astype(np.int16).astype(str)
        X_test[numeric_cols] = X_test[numeric_cols].fillna(-1).astype(np.int16).astype(str)
        
        # カテゴリカル特徴量処理
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        print(f"カテゴリカル特徴量: {categorical_cols}")
        
        X_train[categorical_cols] = X_train[categorical_cols].astype(str).fillna("missing")
        X_test[categorical_cols] = X_test[categorical_cols].astype(str).fillna("missing")
        
        # ターゲットマッピング
        y_train = y_train.map(self.config['TARGET_MAPPING'])
        
        print(f"前処理後形状: X_train={X_train.shape}, X_test={X_test.shape}")
        
        return X_train, y_train, X_test
    
    def gm_ngram_features(self, X_train, X_test):
        """GM式n-gram特徴量生成の完全再現"""
        print("\n=== GM式n-gram特徴量生成 ===")
        
        original_cols = list(X_train.columns)
        print(f"元の特徴量数: {len(original_cols)}")
        
        # 2-gram特徴量生成（GMと完全同一）
        print("2-gram特徴量生成中...")
        for col1, col2 in combinations(original_cols, 2):
            feature_name = f"{col1}-{col2}"
            X_train[feature_name] = X_train[col1] + "-" + X_train[col2]
            X_test[feature_name] = X_test[col1] + "-" + X_test[col2]
        
        print(f"2-gram後の特徴量数: {X_train.shape[1]}")
        
        # 3-gram特徴量生成（GMと完全同一）
        print("3-gram特徴量生成中...")
        for col1, col2, col3 in combinations(original_cols, 3):
            feature_name = f"{col1}-{col2}-{col3}"
            X_train[feature_name] = (X_train[col1] + "-" + 
                                   X_train[col2] + "-" + 
                                   X_train[col3])
            X_test[feature_name] = (X_test[col1] + "-" + 
                                  X_test[col2] + "-" + 
                                  X_test[col3])
        
        print(f"3-gram後の最終特徴量数: {X_train.shape[1]}")
        
        return X_train, X_test
    
    def gm_models_setup(self):
        """GM式モデル設定の完全再現"""
        print("\n=== GM式モデル設定 ===")
        
        # GMが使用した5つのモデルを完全再現
        self.models = {
            'XGBoost': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    learning_rate=0.02,  # GMのパラメータ
                    n_estimators=1500,
                    max_depth=5,
                    colsample_bytree=0.45,
                    random_state=self.config['RANDOM_STATE'],
                    verbosity=0,
                    enable_categorical=True
                ))
            ]),
            
            'CatBoost': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', cb.CatBoostClassifier(
                    loss_function='Logloss',
                    learning_rate=0.02,  # GMのパラメータ
                    iterations=1500,
                    max_depth=5,
                    random_state=self.config['RANDOM_STATE'],
                    verbose=False
                ))
            ]),
            
            'LightGBM': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', lgb.LGBMClassifier(
                    objective='binary',
                    metric='logloss',
                    learning_rate=0.02,  # GMのパラメータ
                    n_estimators=1500,
                    max_depth=5,
                    colsample_bytree=0.45,
                    reg_lambda=1.50,
                    random_state=self.config['RANDOM_STATE'],
                    verbosity=-1
                ))
            ]),
            
            'RandomForest': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', RandomForestClassifier(
                    n_estimators=100,  # GMのパラメータ
                    max_depth=6,
                    min_samples_leaf=16,
                    random_state=self.config['RANDOM_STATE']
                ))
            ]),
            
            'HistGradientBoosting': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', HistGradientBoostingClassifier(
                    learning_rate=0.03,  # GMのパラメータ
                    min_samples_leaf=12,
                    max_iter=500,
                    max_depth=5,
                    l2_regularization=0.75,
                    random_state=self.config['RANDOM_STATE']
                ))
            ])
        }
        
        print(f"設定完了: {len(self.models)} モデル（GM仕様）")
    
    def gm_training(self, X_train, y_train, X_test):
        """GM式訓練プロセスの完全再現"""
        print("\n=== GM式モデル訓練 ===")
        
        # GMと同じクロスバリデーション設定
        cv = StratifiedKFold(
            n_splits=self.config['N_SPLITS'], 
            shuffle=True, 
            random_state=self.config['RANDOM_STATE']
        )
        
        self.oof_predictions = {}
        self.test_predictions = {}
        
        for name, model in self.models.items():
            print(f"\n--- {name} 訓練中 ---")
            
            oof_preds = np.zeros(len(X_train))
            test_preds = np.zeros(len(X_test))
            fold_scores = []
            
            for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx]
                X_fold_valid = X_train.iloc[valid_idx]
                y_fold_train = y_train.iloc[train_idx]
                y_fold_valid = y_train.iloc[valid_idx]
                
                # 訓練
                model.fit(X_fold_train, y_fold_train)
                
                # OOF予測
                valid_preds = model.predict_proba(X_fold_valid)[:, 1]
                oof_preds[valid_idx] = valid_preds
                
                # テスト予測
                test_preds += model.predict_proba(X_test)[:, 1] / self.config['N_SPLITS']
                
                # スコア計算
                fold_score = accuracy_score(y_fold_valid, (valid_preds >= 0.5).astype(int))
                fold_scores.append(fold_score)
                print(f"  Fold {fold+1}: {fold_score:.6f}")
            
            # 全体スコア
            overall_score = accuracy_score(y_train, (oof_preds >= 0.5).astype(int))
            print(f"  {name} 全体スコア: {overall_score:.6f} (±{np.std(fold_scores):.6f})")
            
            self.oof_predictions[name] = oof_preds
            self.test_predictions[name] = test_preds
    
    def gm_ensemble(self, y_train):
        """GM式アンサンブルの完全再現"""
        print("\n=== GM式アンサンブル ===")
        
        # OOF予測をDataFrameに変換
        oof_df = pd.DataFrame(self.oof_predictions)
        test_df = pd.DataFrame(self.test_predictions)
        
        print(f"アンサンブル用データ形状: OOF={oof_df.shape}, Test={test_df.shape}")
        
        # GMが使用したLogisticRegressionブレンディング
        meta_model = LogisticRegression(
            C=0.01,  # GMの設定
            max_iter=10000, 
            random_state=self.config['RANDOM_STATE']
        )
        
        # メタモデル訓練
        meta_model.fit(oof_df, y_train)
        
        # 最終予測
        final_oof = meta_model.predict_proba(oof_df)[:, 1]
        final_test = meta_model.predict_proba(test_df)[:, 1]
        
        # アンサンブルスコア
        ensemble_score = accuracy_score(y_train, (final_oof >= 0.5).astype(int))
        print(f"GM式アンサンブルスコア: {ensemble_score:.6f}")
        
        # モデル重み表示
        print("\nGM式モデル重み:")
        for i, (name, weight) in enumerate(zip(oof_df.columns, meta_model.coef_[0])):
            print(f"  {name}: {weight:.4f}")
        
        return final_test, ensemble_score
    
    def create_submission(self, predictions, filename_suffix=""):
        """提出ファイル作成"""
        print("\n=== 提出ファイル作成 ===")
        
        # 予測を二値分類に変換
        binary_preds = (predictions >= 0.5).astype(int)
        
        # ラベルを文字列に戻す
        reverse_mapping = {v: k for k, v in self.config['TARGET_MAPPING'].items()}
        string_preds = [reverse_mapping[pred] for pred in binary_preds]
        
        # 提出DataFrame作成
        submission = pd.DataFrame({
            'id': self.test_df['id'],
            'Personality': string_preds
        })
        
        # 保存
        submission_path = self.config['OUTPUT_PATH'] / f'gm_exact_reproduction{filename_suffix}.csv'
        submission.to_csv(submission_path, index=False)
        
        print(f"提出ファイル保存: {submission_path}")
        print(f"提出データ形状: {submission.shape}")
        print(f"予測分布:")
        print(submission['Personality'].value_counts())
        
        return submission
    
    def run_gm_reproduction(self):
        """GM手法完全再現パイプライン実行"""
        print("=== GM手法完全再現パイプライン開始 ===\n")
        
        # 1. データ読み込み
        self.load_data()
        
        # 2. GM式前処理
        X_train, y_train, X_test = self.gm_preprocessing()
        
        # 3. GM式n-gram特徴量生成
        X_train, X_test = self.gm_ngram_features(X_train, X_test)
        
        # 4. GM式モデル設定
        self.gm_models_setup()
        
        # 5. GM式モデル訓練
        self.gm_training(X_train, y_train, X_test)
        
        # 6. GM式アンサンブル
        final_predictions, ensemble_score = self.gm_ensemble(y_train)
        
        # 7. 提出ファイル作成
        submission = self.create_submission(final_predictions)
        
        print(f"\n=== GM手法完全再現完了 ===")
        print(f"最終アンサンブルスコア: {ensemble_score:.6f}")
        print(f"GMベースライン(0.975708)との差: {ensemble_score - 0.975708:+.6f}")
        
        if abs(ensemble_score - 0.975708) < 0.001:
            print("✅ GMベースライン完全再現成功！")
        else:
            print("⚠️ GMベースラインとの差が発生。設定を再確認が必要。")
        
        return submission, ensemble_score

if __name__ == "__main__":
    gm = GMExactReproduction()
    submission, score = gm.run_gm_reproduction()