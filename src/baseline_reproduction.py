"""
GMベースライン再現・改良版
スコア 0.975708 を再現し、さらなる改良を加える
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# 設定
class Config:
    RANDOM_STATE = 42
    N_SPLITS = 5
    TARGET_COL = 'Personality'
    TARGET_MAPPING = {"Extrovert": 0, "Introvert": 1}
    DATA_PATH = Path('/Users/osawa/kaggle/playground-series-s5e7/data/raw')

class BaselineReproduction:
    """GMベースライン再現クラス"""
    
    def __init__(self):
        self.config = Config()
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.models = {}
        self.oof_predictions = {}
        self.test_predictions = {}
        
    def load_data(self):
        """データ読み込み"""
        print("=== データ読み込み ===")
        self.train_df = pd.read_csv(self.config.DATA_PATH / 'train.csv')
        self.test_df = pd.read_csv(self.config.DATA_PATH / 'test.csv')
        
        print(f"訓練データ形状: {self.train_df.shape}")
        print(f"テストデータ形状: {self.test_df.shape}")
        
        # ターゲット分布確認
        target_dist = self.train_df[self.config.TARGET_COL].value_counts()
        print(f"ターゲット分布:\n{target_dist}")
        
    def preprocess_data(self):
        """GMベースライン前処理の再現"""
        print("\n=== 前処理（GMベースライン再現） ===")
        
        # 特徴量列を取得（id, Personalityを除く）
        feature_cols = [col for col in self.train_df.columns 
                       if col not in ['id', self.config.TARGET_COL]]
        
        # X, y分割
        X_train = self.train_df[feature_cols].copy()
        y_train = self.train_df[self.config.TARGET_COL].copy()
        X_test = self.test_df[feature_cols].copy()
        
        # 数値特徴量処理（GMベースライン方式）
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        print(f"数値特徴量: {numeric_cols}")
        
        # 数値を文字列に変換（欠損値は-1）
        X_train[numeric_cols] = X_train[numeric_cols].fillna(-1).astype(np.int16).astype(str)
        X_test[numeric_cols] = X_test[numeric_cols].fillna(-1).astype(np.int16).astype(str)
        
        # カテゴリカル特徴量処理
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        print(f"カテゴリカル特徴量: {categorical_cols}")
        
        X_train[categorical_cols] = X_train[categorical_cols].astype(str).fillna("missing")
        X_test[categorical_cols] = X_test[categorical_cols].astype(str).fillna("missing")
        
        print(f"前処理後形状: X_train={X_train.shape}, X_test={X_test.shape}")
        
        self.X_train = X_train
        self.y_train = y_train.map(self.config.TARGET_MAPPING)
        self.X_test = X_test
        
    def create_ngram_features(self):
        """N-gram特徴量生成（GMベースライン再現）"""
        print("\n=== N-gram特徴量生成 ===")
        
        # 元の特徴量列
        original_cols = list(self.X_train.columns)
        print(f"元の特徴量数: {len(original_cols)}")
        
        # 2-gram特徴量生成
        print("2-gram特徴量生成中...")
        for col1, col2 in combinations(original_cols, 2):
            feature_name = f"{col1}-{col2}"
            self.X_train[feature_name] = self.X_train[col1] + "-" + self.X_train[col2]
            self.X_test[feature_name] = self.X_test[col1] + "-" + self.X_test[col2]
        
        print(f"2-gram後の特徴量数: {self.X_train.shape[1]}")
        
        # 3-gram特徴量生成
        print("3-gram特徴量生成中...")
        for col1, col2, col3 in combinations(original_cols, 3):
            feature_name = f"{col1}-{col2}-{col3}"
            self.X_train[feature_name] = (self.X_train[col1] + "-" + 
                                        self.X_train[col2] + "-" + 
                                        self.X_train[col3])
            self.X_test[feature_name] = (self.X_test[col1] + "-" + 
                                       self.X_test[col2] + "-" + 
                                       self.X_test[col3])
        
        print(f"3-gram後の最終特徴量数: {self.X_train.shape[1]}")
        
    def setup_models(self):
        """GMベースラインモデル設定"""
        print("\n=== モデル設定 ===")
        
        self.models = {
            'XGBoost': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config.RANDOM_STATE)),
                ('model', xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    learning_rate=0.02,
                    n_estimators=1500,
                    max_depth=5,
                    colsample_bytree=0.45,
                    random_state=self.config.RANDOM_STATE,
                    verbosity=0,
                    enable_categorical=True
                ))
            ]),
            
            'CatBoost': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config.RANDOM_STATE)),
                ('model', cb.CatBoostClassifier(
                    loss_function='Logloss',
                    learning_rate=0.02,
                    iterations=1500,
                    max_depth=5,
                    random_state=self.config.RANDOM_STATE,
                    verbose=False
                ))
            ]),
            
            'LightGBM': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config.RANDOM_STATE)),
                ('model', lgb.LGBMClassifier(
                    objective='binary',
                    metric='logloss',
                    learning_rate=0.02,
                    n_estimators=1500,
                    max_depth=5,
                    colsample_bytree=0.45,
                    reg_lambda=1.50,
                    random_state=self.config.RANDOM_STATE,
                    verbosity=-1
                ))
            ]),
            
            'RandomForest': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config.RANDOM_STATE)),
                ('model', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=16,
                    random_state=self.config.RANDOM_STATE
                ))
            ]),
            
            'HistGradientBoosting': Pipeline([
                ('encoder', TargetEncoder(random_state=self.config.RANDOM_STATE)),
                ('model', HistGradientBoostingClassifier(
                    learning_rate=0.03,
                    min_samples_leaf=12,
                    max_iter=500,
                    max_depth=5,
                    l2_regularization=0.75,
                    random_state=self.config.RANDOM_STATE
                ))
            ])
        }
        
        print(f"設定完了: {len(self.models)} モデル")
        
    def train_models(self):
        """モデル訓練とOOF予測生成"""
        print("\n=== モデル訓練 ===")
        
        # クロスバリデーション設定
        cv = StratifiedKFold(
            n_splits=self.config.N_SPLITS, 
            shuffle=True, 
            random_state=self.config.RANDOM_STATE
        )
        
        # 各モデルの訓練
        for name, model in self.models.items():
            print(f"\n--- {name} 訓練中 ---")
            
            # OOF予測とテスト予測の初期化
            oof_preds = np.zeros(len(self.X_train))
            test_preds = np.zeros(len(self.X_test))
            fold_scores = []
            
            # クロスバリデーション実行
            for fold, (train_idx, valid_idx) in enumerate(cv.split(self.X_train, self.y_train)):
                # 分割
                X_fold_train = self.X_train.iloc[train_idx]
                X_fold_valid = self.X_train.iloc[valid_idx]
                y_fold_train = self.y_train.iloc[train_idx]
                y_fold_valid = self.y_train.iloc[valid_idx]
                
                # 訓練
                model.fit(X_fold_train, y_fold_train)
                
                # OOF予測
                valid_preds = model.predict_proba(X_fold_valid)[:, 1]
                oof_preds[valid_idx] = valid_preds
                
                # テスト予測（後で平均）
                test_preds += model.predict_proba(self.X_test)[:, 1] / self.config.N_SPLITS
                
                # スコア計算
                fold_score = accuracy_score(y_fold_valid, (valid_preds >= 0.5).astype(int))
                fold_scores.append(fold_score)
                print(f"  Fold {fold+1}: {fold_score:.6f}")
            
            # 全体スコア計算
            overall_score = accuracy_score(self.y_train, (oof_preds >= 0.5).astype(int))
            print(f"  {name} 全体スコア: {overall_score:.6f} (±{np.std(fold_scores):.6f})")
            
            # 結果保存
            self.oof_predictions[name] = oof_preds
            self.test_predictions[name] = test_preds
            
    def ensemble_predictions(self):
        """アンサンブル予測（GMベースライン方式）"""
        print("\n=== アンサンブル ===")
        
        # OOF予測をDataFrameに変換
        oof_df = pd.DataFrame(self.oof_predictions)
        test_df = pd.DataFrame(self.test_predictions)
        
        print(f"アンサンブル用データ形状: OOF={oof_df.shape}, Test={test_df.shape}")
        
        # LogisticRegressionでブレンディング
        meta_model = LogisticRegression(
            C=0.01, 
            max_iter=10000, 
            random_state=self.config.RANDOM_STATE
        )
        
        # メタモデル訓練
        meta_model.fit(oof_df, self.y_train)
        
        # 最終予測
        final_oof = meta_model.predict_proba(oof_df)[:, 1]
        final_test = meta_model.predict_proba(test_df)[:, 1]
        
        # アンサンブルスコア
        ensemble_score = accuracy_score(self.y_train, (final_oof >= 0.5).astype(int))
        print(f"アンサンブルスコア: {ensemble_score:.6f}")
        
        # モデル重み表示
        print("\nモデル重み:")
        for i, (name, weight) in enumerate(zip(oof_df.columns, meta_model.coef_[0])):
            print(f"  {name}: {weight:.4f}")
        
        return final_test, ensemble_score
        
    def create_submission(self, predictions):
        """提出ファイル作成"""
        print("\n=== 提出ファイル作成 ===")
        
        # 予測を二値分類に変換
        binary_preds = (predictions >= 0.5).astype(int)
        
        # ラベルを文字列に戻す
        reverse_mapping = {v: k for k, v in self.config.TARGET_MAPPING.items()}
        string_preds = [reverse_mapping[pred] for pred in binary_preds]
        
        # 提出DataFrame作成
        submission = pd.DataFrame({
            'id': self.test_df['id'],
            self.config.TARGET_COL: string_preds
        })
        
        # 保存
        submission_path = '/Users/osawa/kaggle/playground-series-s5e7/submissions/baseline_reproduction.csv'
        submission.to_csv(submission_path, index=False)
        
        print(f"提出ファイル保存: {submission_path}")
        print(f"提出データ形状: {submission.shape}")
        print(f"予測分布:\n{submission[self.config.TARGET_COL].value_counts()}")
        
        return submission
        
    def run_full_pipeline(self):
        """フルパイプライン実行"""
        print("=== GMベースライン再現・改良版 実行開始 ===\n")
        
        # 1. データ読み込み
        self.load_data()
        
        # 2. 前処理
        self.preprocess_data()
        
        # 3. N-gram特徴量生成
        self.create_ngram_features()
        
        # 4. モデル設定
        self.setup_models()
        
        # 5. モデル訓練
        self.train_models()
        
        # 6. アンサンブル
        final_predictions, ensemble_score = self.ensemble_predictions()
        
        # 7. 提出ファイル作成
        submission = self.create_submission(final_predictions)
        
        print(f"\n=== 完了 ===")
        print(f"最終アンサンブルスコア: {ensemble_score:.6f}")
        print(f"目標スコア 0.975708 との差: {ensemble_score - 0.975708:+.6f}")
        
        return submission, ensemble_score

if __name__ == "__main__":
    # ベースライン実行
    baseline = BaselineReproduction()
    submission, score = baseline.run_full_pipeline()