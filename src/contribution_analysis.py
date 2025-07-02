"""
GMベースライン技術要素の貢献度分析
各技術要素がスコアに与える影響を定量化
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import xgboost as xgb

class ContributionAnalysis:
    """技術要素の貢献度分析クラス"""
    
    def __init__(self):
        self.config = {
            'RANDOM_STATE': 42,
            'N_SPLITS': 5,
            'TARGET_COL': 'Personality',
            'TARGET_MAPPING': {"Extrovert": 0, "Introvert": 1},
            'DATA_PATH': Path('/Users/osawa/kaggle/playground-series-s5e7/data/raw')
        }
        self.results = {}
        
    def load_data(self):
        """データ読み込み"""
        train_df = pd.read_csv(self.config['DATA_PATH'] / 'train.csv')
        test_df = pd.read_csv(self.config['DATA_PATH'] / 'test.csv')
        
        feature_cols = [col for col in train_df.columns 
                       if col not in ['id', self.config['TARGET_COL']]]
        
        X_train = train_df[feature_cols].copy()
        y_train = train_df[self.config['TARGET_COL']].map(self.config['TARGET_MAPPING'])
        X_test = test_df[feature_cols].copy()
        
        return X_train, y_train, X_test
    
    def preprocess_baseline(self, X_train, X_test):
        """ベースライン前処理"""
        # 数値特徴量処理
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train[numeric_cols] = X_train[numeric_cols].fillna(-1).astype(np.int16).astype(str)
        X_test[numeric_cols] = X_test[numeric_cols].fillna(-1).astype(np.int16).astype(str)
        
        # カテゴリカル特徴量処理
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        X_train[categorical_cols] = X_train[categorical_cols].astype(str).fillna("missing")
        X_test[categorical_cols] = X_test[categorical_cols].astype(str).fillna("missing")
        
        return X_train, X_test
        
    def add_ngram_features(self, X_train, X_test, max_n=3):
        """N-gram特徴量追加"""
        original_cols = list(X_train.columns)
        
        if max_n >= 2:
            # 2-gram
            for col1, col2 in combinations(original_cols, 2):
                feature_name = f"{col1}-{col2}"
                X_train[feature_name] = X_train[col1] + "-" + X_train[col2]
                X_test[feature_name] = X_test[col1] + "-" + X_test[col2]
        
        if max_n >= 3:
            # 3-gram
            for col1, col2, col3 in combinations(original_cols, 3):
                feature_name = f"{col1}-{col2}-{col3}"
                X_train[feature_name] = (X_train[col1] + "-" + 
                                        X_train[col2] + "-" + 
                                        X_train[col3])
                X_test[feature_name] = (X_test[col1] + "-" + 
                                       X_test[col2] + "-" + 
                                       X_test[col3])
        
        return X_train, X_test
    
    def evaluate_model(self, X_train, y_train, model_type='simple'):
        """モデル評価"""
        if model_type == 'simple':
            # シンプルなXGBoostモデル
            model = Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.config['RANDOM_STATE'],
                    verbosity=0
                ))
            ])
        else:
            # GMベースライン相当モデル
            model = Pipeline([
                ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
                ('model', xgb.XGBClassifier(
                    objective='binary:logistic',
                    learning_rate=0.02,
                    n_estimators=500,  # 高速化のため削減
                    max_depth=5,
                    colsample_bytree=0.45,
                    random_state=self.config['RANDOM_STATE'],
                    verbosity=0
                ))
            ])
        
        # クロスバリデーション
        cv = StratifiedKFold(
            n_splits=self.config['N_SPLITS'], 
            shuffle=True, 
            random_state=self.config['RANDOM_STATE']
        )
        
        scores = []
        for train_idx, valid_idx in cv.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_valid = X_train.iloc[valid_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_valid = y_train.iloc[valid_idx]
            
            model.fit(X_fold_train, y_fold_train)
            preds = model.predict(X_fold_valid)
            score = accuracy_score(y_fold_valid, preds)
            scores.append(score)
        
        return np.mean(scores), np.std(scores)
    
    def analyze_contributions(self):
        """各技術要素の貢献度分析"""
        print("=== 技術要素貢献度分析 ===\n")
        
        # データ読み込み
        X_train, y_train, X_test = self.load_data()
        
        # 1. ベースライン（前処理なし）
        print("1. ベースライン（前処理なし）")
        X_raw = X_train.copy()
        # 欠損値を0で埋める
        X_raw = X_raw.fillna(0)
        # オブジェクト型をカテゴリコードに変換
        for col in X_raw.select_dtypes(include=['object']).columns:
            X_raw[col] = pd.Categorical(X_raw[col]).codes
        
        score_raw, std_raw = self.evaluate_model(X_raw, y_train, 'simple')
        self.results['raw'] = {'score': score_raw, 'std': std_raw, 'features': X_raw.shape[1]}
        print(f"  スコア: {score_raw:.6f} (±{std_raw:.6f}), 特徴量数: {X_raw.shape[1]}")
        
        # 2. GM前処理のみ
        print("\n2. GMベースライン前処理のみ")
        X_prep, X_test_prep = self.preprocess_baseline(X_train.copy(), X_test.copy())
        score_prep, std_prep = self.evaluate_model(X_prep, y_train, 'simple')
        self.results['preprocessing'] = {'score': score_prep, 'std': std_prep, 'features': X_prep.shape[1]}
        print(f"  スコア: {score_prep:.6f} (±{std_prep:.6f}), 特徴量数: {X_prep.shape[1]}")
        print(f"  改善: {score_prep - score_raw:+.6f}")
        
        # 3. GM前処理 + 2-gram
        print("\n3. GMベースライン前処理 + 2-gram")
        X_2gram, X_test_2gram = self.add_ngram_features(X_prep.copy(), X_test_prep.copy(), max_n=2)
        score_2gram, std_2gram = self.evaluate_model(X_2gram, y_train, 'simple')
        self.results['2gram'] = {'score': score_2gram, 'std': std_2gram, 'features': X_2gram.shape[1]}
        print(f"  スコア: {score_2gram:.6f} (±{std_2gram:.6f}), 特徴量数: {X_2gram.shape[1]}")
        print(f"  改善: {score_2gram - score_prep:+.6f}")
        
        # 4. GM前処理 + 2-gram + 3-gram
        print("\n4. GMベースライン前処理 + 2-gram + 3-gram")
        X_3gram, X_test_3gram = self.add_ngram_features(X_prep.copy(), X_test_prep.copy(), max_n=3)
        score_3gram, std_3gram = self.evaluate_model(X_3gram, y_train, 'simple')
        self.results['3gram'] = {'score': score_3gram, 'std': std_3gram, 'features': X_3gram.shape[1]}
        print(f"  スコア: {score_3gram:.6f} (±{std_3gram:.6f}), 特徴量数: {X_3gram.shape[1]}")
        print(f"  改善: {score_3gram - score_2gram:+.6f}")
        
        # 5. GM前処理 + 3-gram + 高度モデル
        print("\n5. GMベースライン前処理 + 3-gram + 高度モデル")
        score_advanced, std_advanced = self.evaluate_model(X_3gram, y_train, 'advanced')
        self.results['advanced'] = {'score': score_advanced, 'std': std_advanced, 'features': X_3gram.shape[1]}
        print(f"  スコア: {score_advanced:.6f} (±{std_advanced:.6f}), 特徴量数: {X_3gram.shape[1]}")
        print(f"  改善: {score_advanced - score_3gram:+.6f}")
        
    def analyze_feature_importance(self):
        """特徴量重要度分析"""
        print("\n=== 特徴量重要度分析 ===\n")
        
        # データ準備
        X_train, y_train, X_test = self.load_data()
        X_train, X_test = self.preprocess_baseline(X_train, X_test)
        X_train, X_test = self.add_ngram_features(X_train, X_test, max_n=3)
        
        # モデル訓練
        model = Pipeline([
            ('encoder', TargetEncoder(random_state=self.config['RANDOM_STATE'])),
            ('model', xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.config['RANDOM_STATE'],
                verbosity=0
            ))
        ])
        
        model.fit(X_train, y_train)
        
        # 特徴量重要度取得
        feature_importance = model.named_steps['model'].feature_importances_
        
        # エンコード後の特徴量名は取得困難なため、代替手法を使用
        # 元の特徴量、2-gram、3-gramのグループ別重要度を計算
        original_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                        'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
                        'Post_frequency']
        
        # 特徴量グループ別分析
        n_original = len(original_cols)
        n_2gram = len(list(combinations(original_cols, 2)))
        n_3gram = len(list(combinations(original_cols, 3)))
        
        importance_original = np.mean(feature_importance[:n_original])
        importance_2gram = np.mean(feature_importance[n_original:n_original+n_2gram])
        importance_3gram = np.mean(feature_importance[n_original+n_2gram:])
        
        print(f"元の特徴量平均重要度: {importance_original:.4f}")
        print(f"2-gram特徴量平均重要度: {importance_2gram:.4f}")
        print(f"3-gram特徴量平均重要度: {importance_3gram:.4f}")
        
        # 上位重要度特徴量のインデックス
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        print(f"\n上位10特徴量のインデックス: {top_indices}")
        print(f"上位10特徴量の重要度: {feature_importance[top_indices]}")
        
    def generate_improvement_suggestions(self):
        """改善提案生成"""
        print("\n=== 改善提案 ===\n")
        
        # 各技術要素の貢献度を分析
        improvements = []
        
        # 前処理の効果
        prep_gain = self.results['preprocessing']['score'] - self.results['raw']['score']
        print(f"1. 前処理の効果: +{prep_gain:.6f}")
        if prep_gain > 0.01:
            improvements.append("前処理戦略はかなり効果的")
        
        # 2-gramの効果
        gram2_gain = self.results['2gram']['score'] - self.results['preprocessing']['score']
        print(f"2. 2-gramの効果: +{gram2_gain:.6f}")
        if gram2_gain > 0.005:
            improvements.append("2-gram特徴量は有効、さらなる拡張を検討")
        
        # 3-gramの効果
        gram3_gain = self.results['3gram']['score'] - self.results['2gram']['score']
        print(f"3. 3-gramの効果: +{gram3_gain:.6f}")
        if gram3_gain > 0.001:
            improvements.append("3-gram特徴量は有効、4-gram/5-gramを検討")
        else:
            improvements.append("3-gramの効果は限定的、特徴量選択を検討")
        
        # 高度モデルの効果
        advanced_gain = self.results['advanced']['score'] - self.results['3gram']['score']
        print(f"4. 高度モデルの効果: +{advanced_gain:.6f}")
        if advanced_gain > 0.005:
            improvements.append("モデル複雑化は有効")
        
        print(f"\n改善提案:")
        for i, suggestion in enumerate(improvements, 1):
            print(f"  {i}. {suggestion}")
        
        # 目標到達のための追加改善
        current_best = max([result['score'] for result in self.results.values()])
        target_score = 0.975708
        gap = target_score - current_best
        
        print(f"\n現在の最高スコア: {current_best:.6f}")
        print(f"目標スコア: {target_score:.6f}")
        print(f"必要な改善: +{gap:.6f}")
        
        if gap > 0:
            print("\n目標達成のための推奨改善:")
            print("  1. 4-gram/5-gram特徴量の追加")
            print("  2. 統計的特徴量の追加")
            print("  3. Target Encodingの改良")
            print("  4. アンサンブル手法の高度化")
            print("  5. ハイパーパラメータ最適化")
    
    def run_analysis(self):
        """全分析実行"""
        print("=== GMベースライン技術要素貢献度分析 ===\n")
        
        # 1. 各技術要素の貢献度分析
        self.analyze_contributions()
        
        # 2. 特徴量重要度分析
        self.analyze_feature_importance()
        
        # 3. 改善提案生成
        self.generate_improvement_suggestions()
        
        return self.results

if __name__ == "__main__":
    analyzer = ContributionAnalysis()
    results = analyzer.run_analysis()