"""
Deep Learning手法: TabNet、Neural Networks、AutoEncoderベース特徴量
GMベースライン0.975708突破のための高度モデリング
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep Learning libraries
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_tabnet.tab_model import TabNetClassifier

# Traditional ML for comparison
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import lightgbm as lgb

# Neural Network frameworks
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DeepLearningPipeline:
    """Deep Learning統合パイプラインクラス"""
    
    def __init__(self):
        self.config = {
            'DATA_PATH': Path('/Users/osawa/kaggle/playground-series-s5e7/data/processed'),
            'OUTPUT_PATH': Path('/Users/osawa/kaggle/playground-series-s5e7/submissions'),
            'TARGET_MAPPING': {"Extrovert": 0, "Introvert": 1},
            'RANDOM_STATE': 42,
            'N_SPLITS': 5
        }
        self.config['OUTPUT_PATH'].mkdir(exist_ok=True)
        
        # Set seeds for reproducibility
        np.random.seed(self.config['RANDOM_STATE'])
        torch.manual_seed(self.config['RANDOM_STATE'])
        tf.random.set_seed(self.config['RANDOM_STATE'])
        
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
    
    def preprocess_for_deep_learning(self, X_train, X_test):
        """Deep Learning用前処理"""
        print("=== Deep Learning用前処理 ===")
        
        # カテゴリカル特徴量（文字列）のエンコーディング
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        print(f"カテゴリカル特徴量: {len(categorical_cols)}")
        print(f"数値特徴量: {len(numeric_cols)}")
        
        # カテゴリカル特徴量のラベルエンコーディング
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            
            # 訓練とテストを結合してエンコーディング
            combined_values = pd.concat([X_train_processed[col], X_test_processed[col]])
            le.fit(combined_values.astype(str))
            
            X_train_processed[col] = le.transform(X_train_processed[col].astype(str))
            X_test_processed[col] = le.transform(X_test_processed[col].astype(str))
            
            label_encoders[col] = le
        
        # 数値特徴量の欠損値処理
        for col in numeric_cols:
            X_train_processed[col] = X_train_processed[col].fillna(-1)
            X_test_processed[col] = X_test_processed[col].fillna(-1)
        
        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_processed)
        X_test_scaled = scaler.transform(X_test_processed)
        
        print(f"前処理完了: {X_train_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, scaler, label_encoders
    
    def create_autoencoder_features(self, X_train, X_test, encoding_dim=50):
        """AutoEncoderによる特徴量学習"""
        print(f"=== AutoEncoder特徴量学習 (次元数: {encoding_dim}) ===")
        
        input_dim = X_train.shape[1]
        
        # AutoEncoderモデル定義
        input_layer = keras.Input(shape=(input_dim,))
        
        # Encoder
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu', name='encoding')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # AutoEncoderモデル
        autoencoder = keras.Model(input_layer, decoded)
        encoder = keras.Model(input_layer, encoded)
        
        # コンパイル
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # 訓練
        autoencoder.fit(
            X_train, X_train,
            epochs=50,
            batch_size=256,
            validation_split=0.2,
            verbose=0
        )
        
        # エンコードされた特徴量を取得
        train_encoded = encoder.predict(X_train, verbose=0)
        test_encoded = encoder.predict(X_test, verbose=0)
        
        print(f"AutoEncoder特徴量生成完了: {train_encoded.shape}")
        
        return train_encoded, test_encoded
    
    def train_enhanced_traditional_models(self, X_train, y_train, X_test):
        """拡張された従来モデルの訓練"""
        print("=== 拡張従来モデル訓練 ===")
        
        models = {
            'XGBoost': xgb.XGBClassifier(
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
            ),
            'LightGBM': lgb.LGBMClassifier(
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
            )
        }
        
        cv = StratifiedKFold(n_splits=self.config['N_SPLITS'], shuffle=True, 
                           random_state=self.config['RANDOM_STATE'])
        
        results = {}
        
        for name, model in models.items():
            print(f"  {name} 訓練中...")
            
            oof_predictions = np.zeros(len(X_train))
            test_predictions = np.zeros(len(X_test))
            
            for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
                X_fold_train = X_train[train_idx]
                X_fold_valid = X_train[valid_idx]
                y_fold_train = y_train.iloc[train_idx]
                y_fold_valid = y_train.iloc[valid_idx]
                
                # 訓練
                model.fit(X_fold_train, y_fold_train)
                
                # 予測
                valid_preds = model.predict_proba(X_fold_valid)[:, 1]
                oof_predictions[valid_idx] = valid_preds
                
                test_preds = model.predict_proba(X_test)[:, 1]
                test_predictions += test_preds / self.config['N_SPLITS']
            
            # スコア計算
            overall_score = accuracy_score(y_train, (oof_predictions >= 0.5).astype(int))
            print(f"    {name} 全体スコア: {overall_score:.6f}")
            
            results[name] = {
                'oof': oof_predictions,
                'test': test_predictions,
                'score': overall_score
            }
        
        return results
    
    def create_meta_ensemble(self, predictions_dict, y_train):
        """メタアンサンブルの作成"""
        print("=== メタアンサンブル作成 ===")
        
        # OOF予測を結合
        oof_features = []
        test_features = []
        model_names = []
        
        for name, preds in predictions_dict.items():
            if isinstance(preds, dict):
                oof_features.append(preds['oof'])
                test_features.append(preds['test'])
            else:
                oof_features.append(preds[0])  # oof
                test_features.append(preds[1])  # test
            model_names.append(name)
        
        # OOFとテスト予測をDataFrameに変換
        oof_df = pd.DataFrame(np.column_stack(oof_features), columns=model_names)
        test_df = pd.DataFrame(np.column_stack(test_features), columns=model_names)
        
        print(f"メタ学習データ形状: {oof_df.shape}")
        
        # 単純平均アンサンブル
        final_oof = oof_df.mean(axis=1)
        final_test = test_df.mean(axis=1)
        
        # スコア計算
        score = accuracy_score(y_train, (final_oof >= 0.5).astype(int))
        print(f"  アンサンブルスコア: {score:.6f}")
        
        return final_test, score
    
    def create_submission(self, predictions, test_ids, filename_suffix=""):
        """提出ファイル作成"""
        print("=== 提出ファイル作成 ===")
        
        # 予測を二値分類に変換
        binary_preds = (predictions >= 0.5).astype(int)
        
        # ラベルを文字列に変換
        reverse_mapping = {v: k for k, v in self.config['TARGET_MAPPING'].items()}
        string_preds = [reverse_mapping[pred] for pred in binary_preds]
        
        # 提出DataFrame作成
        submission = pd.DataFrame({
            'id': test_ids,
            'Personality': string_preds
        })
        
        # 保存
        filename = f'deep_learning_submission{filename_suffix}.csv'
        submission_path = self.config['OUTPUT_PATH'] / filename
        submission.to_csv(submission_path, index=False)
        
        print(f"提出ファイル保存: {submission_path}")
        print(f"予測分布:")
        print(submission['Personality'].value_counts())
        
        return submission
    
    def run_full_pipeline(self):
        """Deep Learningフルパイプライン実行"""
        print("=== Deep Learning フルパイプライン開始 ===")
        
        # 1. データ読み込み
        X_train, y_train, X_test, train_ids, test_ids = self.load_revolutionary_data()
        
        # 2. 前処理
        X_train_processed, X_test_processed, scaler, label_encoders = self.preprocess_for_deep_learning(X_train, X_test)
        
        # 3. AutoEncoder特徴量生成
        ae_train, ae_test = self.create_autoencoder_features(X_train_processed, X_test_processed)
        
        # 4. AutoEncoder特徴量を使った従来モデル
        predictions = self.train_enhanced_traditional_models(ae_train, y_train, ae_test)
        
        # 5. メタアンサンブル
        final_predictions, final_score = self.create_meta_ensemble(predictions, y_train)
        
        # 6. 提出ファイル作成
        submission = self.create_submission(final_predictions, test_ids, "_autoencoder")
        
        print(f"\n=== Deep Learning パイプライン完了 ===")
        print(f"最終スコア: {final_score:.6f}")
        print(f"GMベースライン(0.975708)との差: {final_score - 0.975708:+.6f}")
        
        return submission, final_score

if __name__ == "__main__":
    pipeline = DeepLearningPipeline()
    submission, score = pipeline.run_full_pipeline()