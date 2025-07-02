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
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer

# Traditional ML for comparison
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
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
        
        return X_train_scaled, X_test_scaled, scaler, label_encoders\n    \n    def create_autoencoder_features(self, X_train, X_test, encoding_dim=50):\n        \"\"\"AutoEncoderによる特徴量学習\"\"\"\n        print(f\"=== AutoEncoder特徴量学習 (次元数: {encoding_dim}) ===\")\n        \n        input_dim = X_train.shape[1]\n        \n        # AutoEncoderモデル定義\n        input_layer = keras.Input(shape=(input_dim,))\n        \n        # Encoder\n        encoded = layers.Dense(128, activation='relu')(input_layer)\n        encoded = layers.Dropout(0.2)(encoded)\n        encoded = layers.Dense(64, activation='relu')(encoded)\n        encoded = layers.Dropout(0.2)(encoded)\n        encoded = layers.Dense(encoding_dim, activation='relu', name='encoding')(encoded)\n        \n        # Decoder\n        decoded = layers.Dense(64, activation='relu')(encoded)\n        decoded = layers.Dropout(0.2)(decoded)\n        decoded = layers.Dense(128, activation='relu')(decoded)\n        decoded = layers.Dropout(0.2)(decoded)\n        decoded = layers.Dense(input_dim, activation='linear')(decoded)\n        \n        # AutoEncoderモデル\n        autoencoder = keras.Model(input_layer, decoded)\n        encoder = keras.Model(input_layer, encoded)\n        \n        # コンパイル\n        autoencoder.compile(optimizer='adam', loss='mse')\n        \n        # 訓練\n        autoencoder.fit(\n            X_train, X_train,\n            epochs=50,\n            batch_size=256,\n            validation_split=0.2,\n            verbose=0\n        )\n        \n        # エンコードされた特徴量を取得\n        train_encoded = encoder.predict(X_train)\n        test_encoded = encoder.predict(X_test)\n        \n        print(f\"AutoEncoder特徴量生成完了: {train_encoded.shape}\")\n        \n        return train_encoded, test_encoded\n    \n    def train_tabnet_model(self, X_train, y_train, X_test):\n        \"\"\"TabNetモデルの訓練\"\"\"\n        print(\"=== TabNet モデル訓練 ===\")\n        \n        # TabNet用にデータ型を調整\n        X_train_np = X_train.astype(np.float32)\n        X_test_np = X_test.astype(np.float32)\n        y_train_np = y_train.values.astype(np.int64)\n        \n        # クロスバリデーション設定\n        cv = StratifiedKFold(n_splits=self.config['N_SPLITS'], shuffle=True, \n                           random_state=self.config['RANDOM_STATE'])\n        \n        oof_predictions = np.zeros(len(X_train))\n        test_predictions = np.zeros(len(X_test))\n        \n        for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):\n            print(f\"  Fold {fold+1}/{self.config['N_SPLITS']}\")\n            \n            X_fold_train = X_train_np[train_idx]\n            X_fold_valid = X_train_np[valid_idx]\n            y_fold_train = y_train_np[train_idx]\n            y_fold_valid = y_train_np[valid_idx]\n            \n            # TabNetモデル\n            tabnet = TabNetClassifier(\n                n_d=64, n_a=64,\n                n_steps=5,\n                gamma=1.5,\n                n_independent=2,\n                n_shared=2,\n                lambda_sparse=1e-4,\n                optimizer_fn=torch.optim.Adam,\n                optimizer_params=dict(lr=2e-2),\n                mask_type='entmax',\n                scheduler_params=dict(step_size=50, gamma=0.9),\n                scheduler_fn=torch.optim.lr_scheduler.StepLR,\n                verbose=0,\n                seed=self.config['RANDOM_STATE']\n            )\n            \n            # 訓練\n            tabnet.fit(\n                X_fold_train, y_fold_train,\n                eval_set=[(X_fold_valid, y_fold_valid)],\n                eval_metric=['accuracy'],\n                max_epochs=100,\n                patience=20,\n                batch_size=1024,\n                virtual_batch_size=256\n            )\n            \n            # 予測\n            valid_preds = tabnet.predict_proba(X_fold_valid)[:, 1]\n            oof_predictions[valid_idx] = valid_preds\n            \n            test_preds = tabnet.predict_proba(X_test_np)[:, 1]\n            test_predictions += test_preds / self.config['N_SPLITS']\n            \n            # スコア計算\n            fold_score = accuracy_score(y_fold_valid, (valid_preds >= 0.5).astype(int))\n            print(f\"    Accuracy: {fold_score:.6f}\")\n        \n        # 全体スコア\n        overall_score = accuracy_score(y_train, (oof_predictions >= 0.5).astype(int))\n        print(f\"TabNet 全体スコア: {overall_score:.6f}\")\n        \n        return oof_predictions, test_predictions\n    \n    def train_neural_network(self, X_train, y_train, X_test):\n        \"\"\"Neural Networkモデルの訓練\"\"\"\n        print(\"=== Neural Network モデル訓練 ===\")\n        \n        cv = StratifiedKFold(n_splits=self.config['N_SPLITS'], shuffle=True, \n                           random_state=self.config['RANDOM_STATE'])\n        \n        oof_predictions = np.zeros(len(X_train))\n        test_predictions = np.zeros(len(X_test))\n        \n        for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):\n            print(f\"  Fold {fold+1}/{self.config['N_SPLITS']}\")\n            \n            X_fold_train = X_train[train_idx]\n            X_fold_valid = X_train[valid_idx]\n            y_fold_train = y_train.iloc[train_idx]\n            y_fold_valid = y_train.iloc[valid_idx]\n            \n            # Neural Networkモデル構築\n            model = keras.Sequential([\n                layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),\n                layers.BatchNormalization(),\n                layers.Dropout(0.3),\n                layers.Dense(128, activation='relu'),\n                layers.BatchNormalization(),\n                layers.Dropout(0.3),\n                layers.Dense(64, activation='relu'),\n                layers.BatchNormalization(),\n                layers.Dropout(0.2),\n                layers.Dense(32, activation='relu'),\n                layers.Dropout(0.2),\n                layers.Dense(1, activation='sigmoid')\n            ])\n            \n            model.compile(\n                optimizer=keras.optimizers.Adam(learning_rate=1e-3),\n                loss='binary_crossentropy',\n                metrics=['accuracy']\n            )\n            \n            # Early stopping\n            early_stopping = keras.callbacks.EarlyStopping(\n                monitor='val_accuracy',\n                patience=15,\n                restore_best_weights=True\n            )\n            \n            # 訓練\n            model.fit(\n                X_fold_train, y_fold_train,\n                validation_data=(X_fold_valid, y_fold_valid),\n                epochs=100,\n                batch_size=256,\n                callbacks=[early_stopping],\n                verbose=0\n            )\n            \n            # 予測\n            valid_preds = model.predict(X_fold_valid).flatten()\n            oof_predictions[valid_idx] = valid_preds\n            \n            test_preds = model.predict(X_test).flatten()\n            test_predictions += test_preds / self.config['N_SPLITS']\n            \n            # スコア計算\n            fold_score = accuracy_score(y_fold_valid, (valid_preds >= 0.5).astype(int))\n            print(f\"    Accuracy: {fold_score:.6f}\")\n        \n        # 全体スコア\n        overall_score = accuracy_score(y_train, (oof_predictions >= 0.5).astype(int))\n        print(f\"Neural Network 全体スコア: {overall_score:.6f}\")\n        \n        return oof_predictions, test_predictions\n    \n    def train_enhanced_traditional_models(self, X_train, y_train, X_test):\n        \"\"\"拡張された従来モデルの訓練\"\"\"\n        print(\"=== 拡張従来モデル訓練 ===\")\n        \n        models = {\n            'XGBoost': xgb.XGBClassifier(\n                objective='binary:logistic',\n                eval_metric='logloss',\n                learning_rate=0.01,\n                n_estimators=2000,\n                max_depth=6,\n                colsample_bytree=0.45,\n                subsample=0.8,\n                reg_alpha=0.1,\n                reg_lambda=1.0,\n                random_state=self.config['RANDOM_STATE'],\n                verbosity=0\n            ),\n            'LightGBM': lgb.LGBMClassifier(\n                objective='binary',\n                metric='logloss',\n                learning_rate=0.01,\n                n_estimators=2000,\n                max_depth=6,\n                colsample_bytree=0.45,\n                subsample=0.8,\n                reg_alpha=0.1,\n                reg_lambda=1.0,\n                random_state=self.config['RANDOM_STATE'],\n                verbosity=-1\n            )\n        }\n        \n        cv = StratifiedKFold(n_splits=self.config['N_SPLITS'], shuffle=True, \n                           random_state=self.config['RANDOM_STATE'])\n        \n        results = {}\n        \n        for name, model in models.items():\n            print(f\"  {name} 訓練中...\")\n            \n            oof_predictions = np.zeros(len(X_train))\n            test_predictions = np.zeros(len(X_test))\n            \n            for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):\n                X_fold_train = X_train[train_idx]\n                X_fold_valid = X_train[valid_idx]\n                y_fold_train = y_train.iloc[train_idx]\n                y_fold_valid = y_train.iloc[valid_idx]\n                \n                # 訓練\n                model.fit(X_fold_train, y_fold_train)\n                \n                # 予測\n                valid_preds = model.predict_proba(X_fold_valid)[:, 1]\n                oof_predictions[valid_idx] = valid_preds\n                \n                test_preds = model.predict_proba(X_test)[:, 1]\n                test_predictions += test_preds / self.config['N_SPLITS']\n            \n            # スコア計算\n            overall_score = accuracy_score(y_train, (oof_predictions >= 0.5).astype(int))\n            print(f\"    {name} 全体スコア: {overall_score:.6f}\")\n            \n            results[name] = {\n                'oof': oof_predictions,\n                'test': test_predictions,\n                'score': overall_score\n            }\n        \n        return results\n    \n    def create_meta_ensemble(self, predictions_dict, y_train, test_ids):\n        \"\"\"メタアンサンブルの作成\"\"\"\n        print(\"=== メタアンサンブル作成 ===\")\n        \n        # OOF予測を結合\n        oof_features = []\n        test_features = []\n        model_names = []\n        \n        for name, preds in predictions_dict.items():\n            if isinstance(preds, dict):\n                oof_features.append(preds['oof'])\n                test_features.append(preds['test'])\n            else:\n                oof_features.append(preds[0])  # oof\n                test_features.append(preds[1])  # test\n            model_names.append(name)\n        \n        # OOFとテスト予測をDataFrameに変換\n        oof_df = pd.DataFrame(np.column_stack(oof_features), columns=model_names)\n        test_df = pd.DataFrame(np.column_stack(test_features), columns=model_names)\n        \n        print(f\"メタ学習データ形状: {oof_df.shape}\")\n        \n        # 複数のメタモデルでアンサンブル\n        ensemble_methods = {\n            'simple_mean': np.mean,\n            'weighted_mean': lambda x: np.average(x, weights=[0.3, 0.25, 0.25, 0.2]),  # 手動重み\n        }\n        \n        results = {}\n        \n        for method_name, method_func in ensemble_methods.items():\n            if method_name == 'simple_mean':\n                final_oof = oof_df.mean(axis=1)\n                final_test = test_df.mean(axis=1)\n            elif method_name == 'weighted_mean' and len(model_names) == 4:\n                final_oof = test_df.apply(method_func, axis=1)\n                final_test = test_df.apply(method_func, axis=1)\n            else:\n                continue\n            \n            # スコア計算\n            score = accuracy_score(y_train, (final_oof >= 0.5).astype(int))\n            print(f\"  {method_name}: {score:.6f}\")\n            \n            results[method_name] = {\n                'oof': final_oof,\n                'test': final_test,\n                'score': score\n            }\n        \n        # 最高スコアの手法を選択\n        best_method = max(results.keys(), key=lambda k: results[k]['score'])\n        best_score = results[best_method]['score']\n        best_test_preds = results[best_method]['test']\n        \n        print(f\"最適アンサンブル: {best_method} (スコア: {best_score:.6f})\")\n        \n        return best_test_preds, best_score\n    \n    def create_submission(self, predictions, test_ids, filename_suffix=\"\"):\n        \"\"\"提出ファイル作成\"\"\"\n        print(\"=== 提出ファイル作成 ===\")\n        \n        # 予測を二値分類に変換\n        binary_preds = (predictions >= 0.5).astype(int)\n        \n        # ラベルを文字列に変換\n        reverse_mapping = {v: k for k, v in self.config['TARGET_MAPPING'].items()}\n        string_preds = [reverse_mapping[pred] for pred in binary_preds]\n        \n        # 提出DataFrame作成\n        submission = pd.DataFrame({\n            'id': test_ids,\n            'Personality': string_preds\n        })\n        \n        # 保存\n        filename = f'deep_learning_submission{filename_suffix}.csv'\n        submission_path = self.config['OUTPUT_PATH'] / filename\n        submission.to_csv(submission_path, index=False)\n        \n        print(f\"提出ファイル保存: {submission_path}\")\n        print(f\"予測分布:\\n{submission['Personality'].value_counts()}\")\n        \n        return submission\n    \n    def run_full_pipeline(self):\n        \"\"\"Deep Learningフルパイプライン実行\"\"\"\n        print(\"=== Deep Learning フルパイプライン開始 ===\")\n        \n        # 1. データ読み込み\n        X_train, y_train, X_test, train_ids, test_ids = self.load_revolutionary_data()\n        \n        # 2. 前処理\n        X_train_processed, X_test_processed, scaler, label_encoders = self.preprocess_for_deep_learning(X_train, X_test)\n        \n        # 3. AutoEncoder特徴量生成\n        ae_train, ae_test = self.create_autoencoder_features(X_train_processed, X_test_processed)\n        \n        # 4. 各種モデルの訓練\n        predictions = {}\n        \n        # TabNet\n        predictions['TabNet'] = self.train_tabnet_model(X_train_processed, y_train, X_test_processed)\n        \n        # Neural Network\n        predictions['NeuralNet'] = self.train_neural_network(X_train_processed, y_train, X_test_processed)\n        \n        # AutoEncoder特徴量を使った従来モデル\n        ae_traditional = self.train_enhanced_traditional_models(ae_train, y_train, ae_test)\n        predictions.update(ae_traditional)\n        \n        # 5. メタアンサンブル\n        final_predictions, final_score = self.create_meta_ensemble(predictions, y_train, test_ids)\n        \n        # 6. 提出ファイル作成\n        submission = self.create_submission(final_predictions, test_ids, \"_meta_ensemble\")\n        \n        print(f\"\\n=== Deep Learning パイプライン完了 ===\")\n        print(f\"最終スコア: {final_score:.6f}\")\n        print(f\"GMベースライン(0.975708)との差: {final_score - 0.975708:+.6f}\")\n        \n        return submission, final_score\n\nif __name__ == \"__main__\":\n    pipeline = DeepLearningPipeline()\n    submission, score = pipeline.run_full_pipeline()