"""
Advanced Models for Personality Prediction
Neural Networks, Stacking, and Optimization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import optuna
from optuna.integration import TFKerasPruningCallback
import warnings
warnings.filterwarnings('ignore')

class NeuralNetworkModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        tf.random.set_seed(random_state)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        
    def create_model(self, input_dim, hidden_units=[256, 128, 64], dropout_rate=0.3):
        """Neural Networkモデル作成"""
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(hidden_units[0], activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(hidden_units[1], activation='relu'),
            BatchNormalization(), 
            Dropout(dropout_rate),
            
            Dense(hidden_units[2], activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100):
        """モデル訓練"""
        # スケーリング
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # ラベルエンコーディング
        if y_train.dtype == 'object':
            y_train_encoded = self.label_encoder.fit_transform(y_train)
        else:
            y_train_encoded = y_train
            
        if y_val is not None and y_val.dtype == 'object':
            y_val_encoded = self.label_encoder.transform(y_val)
        elif y_val is not None:
            y_val_encoded = y_val
        else:
            y_val_encoded = None
            
        # モデル作成
        self.model = self.create_model(X_train.shape[1])
        
        # コールバック設定
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=7, factor=0.5)
        ]
        
        # 訓練
        if X_val is not None:
            history = self.model.fit(
                X_train_scaled, y_train_encoded,
                validation_data=(X_val_scaled, y_val_encoded),
                epochs=epochs,
                batch_size=64,
                callbacks=callbacks,
                verbose=0
            )
        else:
            history = self.model.fit(
                X_train_scaled, y_train_encoded,
                epochs=epochs,
                batch_size=64,
                callbacks=callbacks,
                verbose=0
            )
            
        return history
    
    def predict_proba(self, X):
        """確率予測"""
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict(X_scaled, verbose=0)
        return np.column_stack([1 - proba.flatten(), proba.flatten()])
    
    def predict(self, X):
        """クラス予測"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

class AdvancedStackingModel:
    def __init__(self, base_models, meta_model=None, cv=5, random_state=42):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(random_state=random_state)
        self.cv = cv
        self.random_state = random_state
        self.stacking_model = None
        
    def fit(self, X, y):
        """スタッキングモデル訓練"""
        self.stacking_model = StackingClassifier(
            estimators=self.base_models,
            final_estimator=self.meta_model,
            cv=StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state),
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        self.stacking_model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """確率予測"""
        return self.stacking_model.predict_proba(X)
    
    def predict(self, X):
        """クラス予測"""
        return self.stacking_model.predict(X)

class OptunaOptimizer:
    def __init__(self, model_type='lightgbm', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.best_params = None
        
    def objective_lightgbm(self, trial, X_train, y_train, cv_folds):
        """LightGBM最適化目的関数"""
        import lightgbm as lgb
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'verbosity': -1,
            'random_state': self.random_state
        }
        
        cv_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            y_pred = model.predict(X_val)
            score = log_loss(y_val, y_pred)
            cv_scores.append(score)
            
        return np.mean(cv_scores)
    
    def objective_xgboost(self, trial, X_train, y_train, cv_folds):
        """XGBoost最適化目的関数"""
        import xgboost as xgb
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': self.random_state
        }
        
        cv_scores = []
        
        for train_idx, val_idx in cv_folds:
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred = model.predict_proba(X_val)[:, 1]
            score = log_loss(y_val, y_pred)
            cv_scores.append(score)
            
        return np.mean(cv_scores)
    
    def optimize(self, X_train, y_train, cv_folds, n_trials=100):
        """ハイパーパラメータ最適化実行"""
        study = optuna.create_study(direction='minimize')
        
        if self.model_type == 'lightgbm':
            objective = lambda trial: self.objective_lightgbm(trial, X_train, y_train, cv_folds)
        elif self.model_type == 'xgboost':
            objective = lambda trial: self.objective_xgboost(trial, X_train, y_train, cv_folds)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        return study.best_params, study.best_value

class AdvancedEnsemble:
    def __init__(self, models, weights=None, method='weighted_average'):
        self.models = models
        self.weights = weights
        self.method = method
        self.optimal_weights = None
        
    def find_optimal_weights(self, predictions, y_true):
        """最適重みを探索"""
        from scipy.optimize import minimize
        
        def objective(weights):
            weights = weights / np.sum(weights)  # 正規化
            ensemble_pred = np.average(predictions, weights=weights, axis=0)
            return log_loss(y_true, ensemble_pred)
        
        # 初期重み
        initial_weights = np.ones(len(self.models)) / len(self.models)
        
        # 制約（重みの合計=1, 重み>=0）
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        result = minimize(objective, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.optimal_weights = result.x
        return self.optimal_weights
    
    def predict_proba(self, predictions):
        """アンサンブル予測"""
        if self.method == 'weighted_average':
            weights = self.optimal_weights if self.optimal_weights is not None else self.weights
            if weights is None:
                weights = np.ones(len(self.models)) / len(self.models)
            return np.average(predictions, weights=weights, axis=0)
        
        elif self.method == 'rank_average':
            # ランク平均
            ranks = np.zeros_like(predictions[0])
            for pred in predictions:
                ranks += np.argsort(np.argsort(pred))
            return ranks / len(predictions)
        
        else:
            # 単純平均
            return np.mean(predictions, axis=0)