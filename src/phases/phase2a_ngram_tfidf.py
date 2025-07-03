"""
Phase 2a: 高次n-gram + TF-IDF重み付け特徴量実装

4-gram/5-gram特徴量とTF-IDF重み付けによるGMベースライン超越
improvement_strategy.md と GM_Differentiation_Strategy.md 統合アプローチ

Author: Osawa
Date: 2025-07-02
Target: CV 0.974211 → 0.977211+ (期待効果 +0.003-0.005)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import itertools
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

class AdvancedNgramFeatureEngineer:
    """高次n-gram + TF-IDF重み付け特徴量エンジニア"""
    
    def __init__(self, max_ngram=5, tfidf_max_features=1000):
        self.max_ngram = max_ngram
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_vectorizers = {}
        self.important_ngram_combinations = {}
        
    def create_advanced_ngram_features(self, df, target=None):
        """
        高次n-gram + TF-IDF重み付け特徴量の生成
        
        Args:
            df: 入力データフレーム
            target: ターゲット変数 (学習時のみ)
            
        Returns:
            拡張された特徴量データフレーム
        """
        
        print("=== Phase 2a: 高次n-gram + TF-IDF特徴量生成 ===")
        
        # 1. 基本前処理
        df_processed = self._preprocess_data(df)
        
        # 2. 高次n-gram特徴量生成
        df_with_ngrams = self._create_high_order_ngrams(df_processed)
        
        # 3. TF-IDF重み付け特徴量生成
        df_with_tfidf = self._create_tfidf_features(df_with_ngrams, target)
        
        # 4. 重要特徴量選択 (メモリ効率化)
        df_optimized = self._optimize_features(df_with_tfidf, target)
        
        return df_optimized
    
    def _preprocess_data(self, df):
        """データ前処理"""
        
        print("1. データ前処理中...")
        df_processed = df.copy()
        
        # 数値特徴量の文字列変換 (GMベースライン準拠)
        for col in df_processed.columns:
            if col not in ['id', 'Personality']:
                df_processed[col] = df_processed[col].fillna(-1).astype(str)
        
        print(f"   変換済み特徴量数: {len([c for c in df_processed.columns if c not in ['id', 'Personality']])}")
        
        return df_processed
    
    def _create_high_order_ngrams(self, df):
        """4-gram/5-gram高次特徴量生成"""
        
        print("2. 高次n-gram特徴量生成中...")
        df_ngrams = df.copy()
        
        # 基本特徴量リスト
        base_features = [col for col in df.columns if col not in ['id', 'Personality']]
        
        # 4-gram特徴量生成 (重要な組み合わせのみ)
        print("   4-gram特徴量生成中...")
        important_4grams = self._get_important_4gram_combinations(base_features)
        
        for i, combo in enumerate(important_4grams):
            if len(combo) == 4:
                feature_name = f"{'_'.join(combo)}_4gram"
                df_ngrams[feature_name] = (
                    df_ngrams[combo[0]] + "_" + 
                    df_ngrams[combo[1]] + "_" + 
                    df_ngrams[combo[2]] + "_" + 
                    df_ngrams[combo[3]]
                )
        
        # 5-gram特徴量生成 (最重要な組み合わせのみ)
        print("   5-gram特徴量生成中...")
        important_5grams = self._get_important_5gram_combinations(base_features)
        
        for i, combo in enumerate(important_5grams):
            if len(combo) == 5:
                feature_name = f"{'_'.join(combo)}_5gram"
                df_ngrams[feature_name] = (
                    df_ngrams[combo[0]] + "_" + 
                    df_ngrams[combo[1]] + "_" + 
                    df_ngrams[combo[2]] + "_" + 
                    df_ngrams[combo[3]] + "_" + 
                    df_ngrams[combo[4]]
                )
        
        ngram_count = len([c for c in df_ngrams.columns if 'gram' in c])
        print(f"   生成された高次n-gram特徴量数: {ngram_count}")
        
        return df_ngrams
    
    def _get_important_4gram_combinations(self, features):
        """重要な4-gram組み合わせの選択"""
        
        # 心理学的に意味のある4-gram組み合わせ
        important_combinations = [
            # 社交活動関連
            ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size'],
            ['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency'],
            
            # 心理状態関連
            ['Stage_fear', 'Drained_after_socializing', 'Time_spent_Alone', 'Social_event_attendance'],
            
            # バランス関連
            ['Time_spent_Alone', 'Social_event_attendance', 'Stage_fear', 'Drained_after_socializing'],
            
            # 外向性指標
            ['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Stage_fear'],
            
            # 内向性指標
            ['Time_spent_Alone', 'Stage_fear', 'Drained_after_socializing', 'Post_frequency']
        ]
        
        # 既存特徴量との組み合わせのみ返す
        valid_combinations = []
        for combo in important_combinations:
            if all(feature in features for feature in combo):
                valid_combinations.append(combo)
        
        return valid_combinations
    
    def _get_important_5gram_combinations(self, features):
        """重要な5-gram組み合わせの選択 (計算コスト考慮で最小限)"""
        
        # 最も重要な5-gram組み合わせのみ
        important_combinations = [
            # 完全な社交プロファイル
            ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency'],
            
            # 心理プロファイル
            ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Drained_after_socializing', 'Going_outside'],
            
            # 外向性完全プロファイル
            ['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency', 'Stage_fear']
        ]
        
        # 既存特徴量との組み合わせのみ返す
        valid_combinations = []
        for combo in important_combinations:
            if all(feature in features for feature in combo):
                valid_combinations.append(combo)
        
        return valid_combinations
    
    def _create_tfidf_features(self, df, target=None):
        """TF-IDF重み付け特徴量生成"""
        
        print("3. TF-IDF重み付け特徴量生成中...")
        df_tfidf = df.copy()
        
        # n-gram特徴量を特定
        ngram_features = [col for col in df.columns if 'gram' in col]
        
        if not ngram_features:
            print("   警告: n-gram特徴量が見つかりません")
            return df_tfidf
        
        # 各n-gram特徴量にTF-IDF適用
        tfidf_features_added = 0
        
        for feature in ngram_features:
            try:
                # TF-IDFベクトル化
                tfidf_vectorizer = TfidfVectorizer(
                    max_features=100,  # メモリ効率のため制限
                    ngram_range=(1, 1),  # 単語レベル
                    min_df=2,  # 最低出現回数
                    token_pattern=r'[^_]+',  # アンダースコア区切り
                    lowercase=False
                )
                
                # 文字列データでTF-IDF計算
                tfidf_matrix = tfidf_vectorizer.fit_transform(df[feature].astype(str))
                
                # 主要成分のみ追加 (メモリ効率化)
                if tfidf_matrix.shape[1] > 0:
                    # 最も重要な5成分のみ追加
                    top_n = min(5, tfidf_matrix.shape[1])
                    feature_names = tfidf_vectorizer.get_feature_names_out()[:top_n]
                    
                    for i, fname in enumerate(feature_names):
                        tfidf_feature_name = f"{feature}_tfidf_{fname}"
                        if tfidf_matrix.shape[1] > i:
                            df_tfidf[tfidf_feature_name] = tfidf_matrix[:, i].toarray().flatten()
                            tfidf_features_added += 1
                
                # ベクトル化器を保存
                self.tfidf_vectorizers[feature] = tfidf_vectorizer
                
            except Exception as e:
                print(f"   警告: {feature}のTF-IDF処理でエラー: {str(e)}")
                continue
        
        print(f"   生成されたTF-IDF特徴量数: {tfidf_features_added}")
        
        return df_tfidf
    
    def _optimize_features(self, df, target=None):
        """メモリ効率化のための特徴量最適化"""
        
        print("4. 特徴量最適化中...")
        
        # 基本特徴量は保持
        base_columns = ['id']
        if 'Personality' in df.columns:
            base_columns.append('Personality')
        
        # 元の特徴量も保持
        original_features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                           'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']
        base_columns.extend([col for col in original_features if col in df.columns])
        
        # 新しく生成された特徴量
        new_features = [col for col in df.columns if col not in base_columns]
        
        # 分散フィルタリング (分散が極めて小さい特徴量を除外)
        if target is not None:
            print("   分散ベース特徴量フィルタリング中...")
            filtered_features = []
            
            for feature in new_features:
                try:
                    if df[feature].dtype in ['object', 'string']:
                        # カテゴリカル特徴量の場合はユニーク値数で判断
                        unique_ratio = df[feature].nunique() / len(df)
                        if unique_ratio > 0.01:  # 1%以上のユニーク値比率
                            filtered_features.append(feature)
                    else:
                        # 数値特徴量の場合は分散で判断
                        if df[feature].var() > 1e-6:
                            filtered_features.append(feature)
                except:
                    continue
            
            final_columns = base_columns + filtered_features
            print(f"   フィルタリング前: {len(new_features)} → フィルタリング後: {len(filtered_features)}")
        else:
            final_columns = base_columns + new_features
        
        df_optimized = df[final_columns].copy()
        
        print(f"   最終特徴量数: {len(final_columns)}")
        
        return df_optimized

def main():
    """メイン実行関数"""
    print("=== Phase 2a: 高次n-gram + TF-IDF特徴量エンジニアリング実行 ===")
    
    # 1. データ読み込み
    print("\n📁 データ読み込み中...")
    try:
        train_df = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/raw/train.csv')
        test_df = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/raw/test.csv')
        
        print(f"   訓練データ: {train_df.shape}")
        print(f"   テストデータ: {test_df.shape}")
        
        # ターゲット変数準備
        y_train = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0})
        
    except FileNotFoundError as e:
        print(f"   エラー: データファイルが見つかりません - {e}")
        return
    
    # 2. 高次n-gram + TF-IDF特徴量生成
    print("\n🔧 特徴量エンジニアリング実行中...")
    engineer = AdvancedNgramFeatureEngineer(max_ngram=5, tfidf_max_features=1000)
    
    # 訓練データ
    train_features = engineer.create_advanced_ngram_features(train_df, target=y_train)
    
    # テストデータ (同じ変換を適用)
    test_features = engineer.create_advanced_ngram_features(test_df, target=None)
    
    # 3. 特徴量数レポート
    original_count = len([c for c in train_df.columns if c not in ['id', 'Personality']])
    new_count = len([c for c in train_features.columns if c not in ['id', 'Personality']])
    added_count = new_count - original_count
    
    print(f"\n📊 特徴量生成結果:")
    print(f"   元の特徴量数: {original_count}")
    print(f"   新しい特徴量数: {new_count}")
    print(f"   追加された特徴量数: {added_count}")
    
    # 4. データ保存
    print("\n💾 処理済みデータ保存中...")
    train_output_path = '/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2a_train_features.csv'
    test_output_path = '/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2a_test_features.csv'
    
    train_features.to_csv(train_output_path, index=False)
    test_features.to_csv(test_output_path, index=False)
    
    print(f"   訓練データ保存: {train_output_path}")
    print(f"   テストデータ保存: {test_output_path}")
    
    # 5. 成功レポート
    print(f"\n✅ Phase 2a実装完了!")
    print(f"   期待CV改善: +0.003-0.005")
    print(f"   目標CVスコア: 0.977211+")
    print(f"   次のステップ: CV評価実行")
    
    return train_features, test_features

if __name__ == "__main__":
    main()