"""
心理学ドメイン特化特徴量エンジニアリング

Big Five理論に基づく外向性/内向性予測のための専門特徴量を生成
GM ベースラインを超越するための革新的アプローチ

Author: Claude Code Team
Date: 2025-07-02
Target: GM Baseline (0.975708) 超越
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class PsychologicalFeatureEngineer:
    """心理学理論に基づく特徴量エンジニアリング"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        
    def create_psychological_features(self, df):
        """Big Five理論に基づく心理学的特徴量の生成"""
        df_features = df.copy()
        
        # 1. 基本的な心理学的スコア
        df_features = self._create_basic_psychological_scores(df_features)
        
        # 2. 相互作用特徴量（社交行動パターン）
        df_features = self._create_interaction_features(df_features)
        
        # 3. 統計的変換（分布正規化）
        df_features = self._create_statistical_transformations(df_features)
        
        # 4. 欠損値パターン特徴量
        df_features = self._create_missing_pattern_features(df_features)
        
        # 5. クラスタリング特徴量（潜在的人格タイプ）
        df_features = self._create_clustering_features(df_features)
        
        return df_features
    
    def _create_basic_psychological_scores(self, df):
        """基本的な心理学的スコア計算"""
        
        # 数値変換（欠損値は中央値で補完）
        numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                       'Friends_circle_size', 'Post_frequency']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
        
        # カテゴリカル変数の数値化
        df['Stage_fear_numeric'] = df['Stage_fear'].map({'Yes': 1, 'No': 0}).fillna(0.5)
        df['Drained_numeric'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0}).fillna(0.5)
        
        # 1. 外向性スコア（社交的活動の総合指標）
        df['extroversion_score'] = (
            df['Social_event_attendance'] + 
            df['Going_outside'] + 
            df['Friends_circle_size'] + 
            df['Post_frequency']
        ) / 4
        
        # 2. 内向性スコア（内向的特性の総合指標）
        df['introversion_score'] = (
            df['Time_spent_Alone'] + 
            df['Stage_fear_numeric'] * 10 + 
            df['Drained_numeric'] * 10
        ) / 3
        
        # 3. 社交バランス指標
        df['social_balance'] = df['extroversion_score'] - df['introversion_score']
        df['social_balance_abs'] = np.abs(df['social_balance'])
        
        # 4. 人格一貫性スコア（回答の一貫性を測定）
        social_features = ['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
        df['social_consistency'] = df[social_features].std(axis=1)
        
        return df
    
    def _create_interaction_features(self, df):
        """相互作用特徴量（心理学的に意味のある組み合わせ）"""
        
        # 1. 社交疲労度（社交活動 × 疲労感）
        df['social_fatigue'] = df['Social_event_attendance'] * df['Drained_numeric']
        
        # 2. 社交積極度（社交活動 × 恐怖の克服）
        df['social_proactivity'] = df['Social_event_attendance'] * (1 - df['Stage_fear_numeric'])
        
        # 3. デジタル社交度（SNS × 友人関係）
        df['digital_social'] = df['Post_frequency'] * df['Friends_circle_size']
        
        # 4. 外出社交度（外出 × 社交イベント）
        df['outdoor_social'] = df['Going_outside'] * df['Social_event_attendance']
        
        # 5. 孤独嗜好度（一人時間 / 社交時間）
        denominator = df['Social_event_attendance'] + df['Going_outside'] + 0.01
        df['solitude_preference'] = df['Time_spent_Alone'] / denominator
        
        # 6. 社交効率（友人数 / 社交活動）
        df['social_efficiency'] = df['Friends_circle_size'] / (df['Social_event_attendance'] + 0.01)
        
        # 7. コミュニケーション比率（デジタル / リアル）
        real_communication = df['Social_event_attendance'] + df['Going_outside'] + 0.01
        df['communication_ratio'] = df['Post_frequency'] / real_communication
        
        return df
    
    def _create_statistical_transformations(self, df):
        """統計的変換による特徴量正規化"""
        
        numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                           'Friends_circle_size', 'Post_frequency']
        
        for col in numeric_features:
            # 対数変換（右裾分布対応）
            df[f'{col}_log'] = np.log1p(df[col])
            
            # 平方根変換
            df[f'{col}_sqrt'] = np.sqrt(df[col])
            
            # Box-Cox風変換
            df[f'{col}_boxcox'] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
            
            # Z-score標準化
            df[f'{col}_zscore'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            
            # ランク変換
            df[f'{col}_rank'] = df[col].rank(pct=True)
        
        return df
    
    def _create_missing_pattern_features(self, df):
        """欠損値パターンを新特徴量として活用"""
        
        # 元の特徴量リスト
        original_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                        'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']
        
        # 1. 各特徴量の欠損フラグ
        for col in original_cols:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
        
        # 2. 総欠損数
        df['total_missing'] = df[[f'{col}_missing' for col in original_cols]].sum(axis=1)
        
        # 3. 欠損率
        df['missing_ratio'] = df['total_missing'] / len(original_cols)
        
        # 4. 欠損パターンハッシュ（2^7 = 128通り）
        missing_pattern = ''
        for col in original_cols:
            missing_pattern += df[f'{col}_missing'].astype(str)
        df['missing_pattern_hash'] = pd.Categorical(missing_pattern).codes
        
        # 5. 特定の欠損組み合わせ（心理学的に意味のある組み合わせ）
        df['social_missing'] = (df['Social_event_attendance_missing'] + 
                               df['Going_outside_missing'] + 
                               df['Friends_circle_size_missing'])
        
        df['anxiety_missing'] = (df['Stage_fear_missing'] + 
                                df['Drained_after_socializing_missing'])
        
        return df
    
    def _create_clustering_features(self, df):
        """クラスタリングによる潜在的人格タイプの発見"""
        
        # クラスタリング用特徴量（欠損値補完済み）
        cluster_features = ['extroversion_score', 'introversion_score', 'social_balance',
                           'social_consistency', 'social_fatigue', 'social_proactivity']
        
        # 欠損値処理
        cluster_data = df[cluster_features].fillna(df[cluster_features].median())
        
        # 標準化
        cluster_data_scaled = self.scaler.fit_transform(cluster_data)
        
        # K-Means クラスタリング（5つの人格タイプ）
        df['personality_cluster'] = self.kmeans.fit_predict(cluster_data_scaled)
        
        # クラスタ中心からの距離
        cluster_distances = self.kmeans.transform(cluster_data_scaled)
        for i in range(5):
            df[f'cluster_{i}_distance'] = cluster_distances[:, i]
        
        # PCA変換（次元削減）
        pca_features = self.pca.fit_transform(cluster_data_scaled)
        for i in range(3):
            df[f'pca_component_{i}'] = pca_features[:, i]
        
        return df
    
    def create_ngram_features(self, df):
        """GMベースラインの n-gram 特徴量を拡張"""
        
        # 全特徴量を文字列に変換（GMと同じ手法）
        string_features = df.copy()
        for col in df.columns:
            if col != 'id':
                string_features[col] = df[col].fillna(-1).astype(str)
        
        # 2-gram特徴量
        ngram_features = pd.DataFrame()
        ngram_features['id'] = df['id']
        
        feature_combinations = []
        cols = [col for col in string_features.columns if col != 'id']
        
        # 2-gram
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                col1, col2 = cols[i], cols[j]
                feature_name = f"{col1}_{col2}_2gram"
                ngram_features[feature_name] = string_features[col1] + "_" + string_features[col2]
                feature_combinations.append(feature_name)
        
        # 3-gram（選択的に重要な組み合わせのみ）
        important_3grams = [
            ('Time_spent_Alone', 'Social_event_attendance', 'extroversion_score'),
            ('Stage_fear', 'Drained_after_socializing', 'introversion_score'),
            ('Going_outside', 'Friends_circle_size', 'Post_frequency'),
            ('social_balance', 'social_consistency', 'personality_cluster')
        ]
        
        for col1, col2, col3 in important_3grams:
            if all(col in string_features.columns for col in [col1, col2, col3]):
                feature_name = f"{col1}_{col2}_{col3}_3gram"
                ngram_features[feature_name] = (string_features[col1] + "_" + 
                                              string_features[col2] + "_" + 
                                              string_features[col3])
                feature_combinations.append(feature_name)
        
        return ngram_features, feature_combinations

def main():
    """メイン実行関数"""
    print("=== 心理学ドメイン特化特徴量エンジニアリング ===")
    
    # データ読み込み
    print("1. データ読み込み中...")
    train_df = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/raw/train.csv')
    test_df = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/raw/test.csv')
    
    print(f"   訓練データ: {train_df.shape}")
    print(f"   テストデータ: {test_df.shape}")
    
    # 特徴量エンジニアリング実行
    print("\n2. 心理学的特徴量生成中...")
    engineer = PsychologicalFeatureEngineer()
    
    train_features = engineer.create_psychological_features(train_df)
    test_features = engineer.create_psychological_features(test_df)
    
    print(f"   生成された特徴量数: {train_features.shape[1] - train_df.shape[1]}")
    
    # n-gram特徴量生成
    print("\n3. n-gram特徴量生成中...")
    train_ngrams, ngram_cols = engineer.create_ngram_features(train_features)
    test_ngrams, _ = engineer.create_ngram_features(test_features)
    
    print(f"   n-gram特徴量数: {len(ngram_cols)}")
    
    # 保存
    print("\n4. 処理済みデータ保存中...")
    train_features.to_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/psychological_train_features.csv', index=False)
    test_features.to_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/psychological_test_features.csv', index=False)
    
    train_ngrams.to_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/psychological_train_ngrams.csv', index=False)
    test_ngrams.to_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/psychological_test_ngrams.csv', index=False)
    
    print("✅ 心理学的特徴量エンジニアリング完了!")
    print(f"   最終特徴量数: {train_features.shape[1]} + {len(ngram_cols)} n-gram特徴量")
    
    return train_features, test_features, train_ngrams, test_ngrams

if __name__ == "__main__":
    main()