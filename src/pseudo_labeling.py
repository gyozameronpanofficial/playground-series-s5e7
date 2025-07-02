"""
擬似ラベリング手法による訓練データ拡張

高信頼度テストデータを活用してGMベースラインを超越
Semi-supervised Learning アプローチ

Author: Claude Code Team
Date: 2025-07-02
Target: 訓練データ1.3倍拡張によるスコア向上 (+0.004-0.007)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class PseudoLabelingEngine:
    """擬似ラベリングによる半教師あり学習"""
    
    def __init__(self, confidence_threshold=0.85, max_pseudo_ratio=0.3):
        """
        Args:
            confidence_threshold: 擬似ラベル採用の信頼度閾値
            max_pseudo_ratio: 元データに対する擬似ラベルの最大比率
        """
        self.confidence_threshold = confidence_threshold
        self.max_pseudo_ratio = max_pseudo_ratio
        self.base_models = None
        self.pseudo_labels = None
        
    def create_base_models(self):
        """擬似ラベル生成用のベースモデル群"""
        
        # LightGBM（高速・高精度）
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.02,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42,
            n_estimators=1500
        )
        
        # XGBoost（頑健性）
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_estimators=1500,
            verbosity=0
        )
        
        # CatBoost（カテゴリカル特徴量に強い）
        cat_model = CatBoostClassifier(
            objective='Logloss',
            learning_rate=0.02,
            depth=6,
            l2_leaf_reg=3,
            bootstrap_type='Bernoulli',
            random_seed=42,
            iterations=1500,
            verbose=False
        )
        
        # アンサンブル（投票による安定性向上）
        ensemble = VotingClassifier(
            estimators=[
                ('lgb', lgb_model),
                ('xgb', xgb_model),
                ('cat', cat_model)
            ],
            voting='soft'  # 確率による投票
        )
        
        self.base_models = ensemble
        return ensemble
    
    def generate_pseudo_labels(self, X_train, y_train, X_test, feature_names=None):
        """
        高信頼度擬似ラベルの生成
        
        Args:
            X_train: 訓練特徴量
            y_train: 訓練ラベル
            X_test: テスト特徴量
            feature_names: 特徴量名リスト
            
        Returns:
            pseudo_data: 擬似ラベル付きデータ
            confidence_scores: 信頼度スコア
        """
        
        print("=== 擬似ラベル生成 ===")
        
        # ベースモデル構築
        if self.base_models is None:
            self.base_models = self.create_base_models()
        
        # クロスバリデーションによる性能評価
        print("1. ベースモデルのCV評価中...")
        cv_scores = self._cross_validate_models(X_train, y_train)
        print(f"   CV平均スコア: {cv_scores:.4f}")
        
        # 全データでモデル訓練
        print("2. 全データでモデル訓練中...")
        self.base_models.fit(X_train, y_train)
        
        # テストデータの予測確率
        print("3. テストデータ予測中...")
        test_proba = self.base_models.predict_proba(X_test)
        
        # 信頼度計算（最大確率値）
        confidence_scores = np.max(test_proba, axis=1)
        
        # 擬似ラベル決定
        pseudo_labels = self.base_models.predict(X_test)
        
        # 高信頼度サンプル選択
        high_confidence_mask = confidence_scores >= self.confidence_threshold
        
        # 最大数制限
        max_pseudo_samples = int(len(X_train) * self.max_pseudo_ratio)
        if np.sum(high_confidence_mask) > max_pseudo_samples:
            # 信頼度上位N件を選択
            top_indices = np.argsort(confidence_scores)[-max_pseudo_samples:]
            high_confidence_mask = np.zeros(len(X_test), dtype=bool)
            high_confidence_mask[top_indices] = True
        
        # 擬似ラベルデータ構築
        pseudo_X = X_test[high_confidence_mask]
        pseudo_y = pseudo_labels[high_confidence_mask]
        pseudo_confidence = confidence_scores[high_confidence_mask]
        
        print(f"4. 擬似ラベル統計:")
        print(f"   総テストサンプル数: {len(X_test)}")
        print(f"   高信頼度サンプル数: {len(pseudo_X)} ({len(pseudo_X)/len(X_test)*100:.1f}%)")
        print(f"   平均信頼度: {pseudo_confidence.mean():.4f}")
        print(f"   擬似ラベル分布: Extrovert={np.sum(pseudo_y==1)}, Introvert={np.sum(pseudo_y==0)}")
        
        return pseudo_X, pseudo_y, pseudo_confidence, high_confidence_mask
    
    def create_augmented_dataset(self, X_train, y_train, X_test, feature_names=None):
        """
        擬似ラベルで拡張された訓練データセットの作成
        
        Returns:
            X_augmented: 拡張された特徴量
            y_augmented: 拡張されたラベル
            sample_weights: サンプル重み（擬似ラベルは低重み）
        """
        
        # 擬似ラベル生成
        pseudo_X, pseudo_y, pseudo_confidence, _ = self.generate_pseudo_labels(
            X_train, y_train, X_test, feature_names
        )
        
        # データ結合
        X_augmented = np.vstack([X_train, pseudo_X])
        y_augmented = np.hstack([y_train, pseudo_y])
        
        # サンプル重み設定
        # 元データ: 重み1.0、擬似ラベル: 信頼度に比例した重み
        original_weights = np.ones(len(y_train))
        pseudo_weights = pseudo_confidence * 0.8  # 最大重み0.8に制限
        sample_weights = np.hstack([original_weights, pseudo_weights])
        
        print(f"\n=== 拡張データセット ===")
        print(f"元訓練データ: {len(X_train)} samples")
        print(f"擬似ラベル: {len(pseudo_X)} samples")
        print(f"拡張後総数: {len(X_augmented)} samples (拡張率: {len(X_augmented)/len(X_train):.2f}x)")
        
        return X_augmented, y_augmented, sample_weights
    
    def _cross_validate_models(self, X, y, cv_folds=5):
        """クロスバリデーションによるモデル性能評価"""
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # モデル訓練
            fold_model = self.create_base_models()
            fold_model.fit(X_fold_train, y_fold_train)
            
            # 予測・評価
            y_pred = fold_model.predict(X_fold_val)
            fold_score = accuracy_score(y_fold_val, y_pred)
            cv_scores.append(fold_score)
            
            if fold == 0:  # 最初のfoldのみ詳細表示
                print(f"   Fold {fold+1} Score: {fold_score:.4f}")
        
        return np.mean(cv_scores)
    
    def iterative_pseudo_labeling(self, X_train, y_train, X_test, iterations=3):
        """
        反復的擬似ラベリング（段階的データ拡張）
        
        Args:
            iterations: 反復回数
            
        Returns:
            最終的な拡張データセット
        """
        
        print("=== 反復的擬似ラベリング ===")
        
        current_X_train = X_train.copy()
        current_y_train = y_train.copy()
        
        for iteration in range(iterations):
            print(f"\n--- Iteration {iteration + 1}/{iterations} ---")
            
            # 現在のデータでベースモデル更新
            self.base_models = self.create_base_models()
            
            # 擬似ラベル生成（段階的に閾値を下げる）
            current_threshold = self.confidence_threshold - (iteration * 0.05)
            current_threshold = max(current_threshold, 0.75)  # 最低閾値
            
            temp_engine = PseudoLabelingEngine(
                confidence_threshold=current_threshold,
                max_pseudo_ratio=0.1  # 各反復で10%ずつ追加
            )
            temp_engine.base_models = self.base_models
            
            pseudo_X, pseudo_y, pseudo_confidence, _ = temp_engine.generate_pseudo_labels(
                current_X_train, current_y_train, X_test
            )
            
            if len(pseudo_X) == 0:
                print(f"   反復 {iteration + 1}: 信頼度の高い擬似ラベルなし")
                break
            
            # データ追加
            current_X_train = np.vstack([current_X_train, pseudo_X])
            current_y_train = np.hstack([current_y_train, pseudo_y])
            
            print(f"   追加サンプル数: {len(pseudo_X)}")
            print(f"   現在の総サンプル数: {len(current_X_train)}")
        
        # 最終的な重み計算
        original_size = len(X_train)
        sample_weights = np.ones(len(current_X_train))
        
        # 擬似ラベル部分の重みを調整
        if len(current_X_train) > original_size:
            pseudo_start_idx = original_size
            sample_weights[pseudo_start_idx:] = 0.7  # 擬似ラベルの重み
        
        return current_X_train, current_y_train, sample_weights

def main():
    """メイン実行関数"""
    print("=== 擬似ラベリングによるデータ拡張 ===")
    
    # 処理済み特徴量データ読み込み
    print("1. 処理済みデータ読み込み中...")
    try:
        train_features = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/psychological_train_features.csv')
        test_features = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/psychological_test_features.csv')
        
        train_ngrams = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/psychological_train_ngrams.csv')
        test_ngrams = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/psychological_test_ngrams.csv')
        
        print(f"   特徴量データ: train {train_features.shape}, test {test_features.shape}")
        print(f"   n-gramデータ: train {train_ngrams.shape}, test {test_ngrams.shape}")
        
    except FileNotFoundError:
        print("   エラー: 処理済みデータが見つかりません。")
        print("   先に psychological_features.py を実行してください。")
        return
    
    # データ準備
    print("\n2. 擬似ラベリング用データ準備中...")
    
    # 特徴量結合（数値特徴量のみ使用）
    numeric_cols = [col for col in train_features.columns if col not in ['id', 'Personality'] and train_features[col].dtype in ['int64', 'float64']]
    
    X_train = train_features[numeric_cols].fillna(0).values
    y_train = train_features['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values
    X_test = test_features[numeric_cols].fillna(0).values
    
    print(f"   使用特徴量数: {len(numeric_cols)}")
    print(f"   訓練データ形状: {X_train.shape}")
    print(f"   テストデータ形状: {X_test.shape}")
    
    # 擬似ラベリング実行
    print("\n3. 擬似ラベリング実行中...")
    engine = PseudoLabelingEngine(confidence_threshold=0.9, max_pseudo_ratio=0.3)
    
    # 反復的擬似ラベリング
    X_augmented, y_augmented, sample_weights = engine.iterative_pseudo_labeling(
        X_train, y_train, X_test, iterations=2
    )
    
    # 結果保存
    print("\n4. 拡張データセット保存中...")
    
    # 拡張データをDataFrameに変換
    augmented_df = pd.DataFrame(X_augmented, columns=numeric_cols)
    augmented_df['Personality'] = y_augmented
    augmented_df['sample_weight'] = sample_weights
    
    # 元データと擬似ラベルを区別するフラグ
    is_pseudo = np.zeros(len(X_augmented))
    is_pseudo[len(X_train):] = 1
    augmented_df['is_pseudo_label'] = is_pseudo
    
    augmented_df.to_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/pseudo_labeled_train.csv', index=False)
    
    print("✅ 擬似ラベリング完了!")
    print(f"   最終データサイズ: {augmented_df.shape}")
    print(f"   拡張率: {len(X_augmented) / len(X_train):.2f}x")
    print(f"   擬似ラベル数: {np.sum(is_pseudo)} samples")
    
    return augmented_df

if __name__ == "__main__":
    main()