"""
高度データ分析: 心理学的知見を活用したスコア改善戦略
GMベースライン0.975708を超えるための革新的アプローチ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import itertools

class AdvancedPersonalityAnalysis:
    """心理学的知見を活用した高度分析クラス"""
    
    def __init__(self):
        self.config = {
            'DATA_PATH': Path('/Users/osawa/kaggle/playground-series-s5e7/data/raw'),
            'OUTPUT_PATH': Path('/Users/osawa/kaggle/playground-series-s5e7/analysis'),
            'TARGET_MAPPING': {"Extrovert": 0, "Introvert": 1}
        }
        self.config['OUTPUT_PATH'].mkdir(exist_ok=True)
        
    def load_data(self):
        """データ読み込み"""
        print("=== データ読み込み ===")
        self.train_df = pd.read_csv(self.config['DATA_PATH'] / 'train.csv')
        self.test_df = pd.read_csv(self.config['DATA_PATH'] / 'test.csv')
        
        # 特徴量列を取得
        self.feature_cols = [col for col in self.train_df.columns 
                           if col not in ['id', 'Personality']]
        
        print(f"データ形状: train={self.train_df.shape}, test={self.test_df.shape}")
        print(f"特徴量数: {len(self.feature_cols)}")
        
    def psychological_pattern_analysis(self):
        """心理学的パターン分析"""
        print("\n=== 心理学的パターン分析 ===")
        
        X = self.train_df[self.feature_cols].copy()
        y = self.train_df['Personality'].map(self.config['TARGET_MAPPING'])
        
        # 1. Big Five因子構造の探索
        print("1. Big Five因子構造分析...")
        
        # 数値データのみを取得
        numeric_data = X.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            # 欠損値処理
            numeric_data_filled = numeric_data.fillna(numeric_data.median())
            
            # 因子分析（5因子構造を仮定）
            if len(numeric_data.columns) >= 5:
                fa = FactorAnalysis(n_components=5, random_state=42)
                factor_scores = fa.fit_transform(StandardScaler().fit_transform(numeric_data_filled))
                
                # 因子負荷量の分析
                factor_loadings = pd.DataFrame(
                    fa.components_.T,
                    columns=[f'Factor_{i+1}' for i in range(5)],
                    index=numeric_data.columns
                )
                
                print(f"因子分析完了: 因子数={fa.n_components}")
                print(f"説明可能分散比: {fa.noise_variance_}")
                
                # 各因子と外向性の関係
                for i in range(5):
                    corr = np.corrcoef(factor_scores[:, i], y)[0, 1]
                    print(f"  Factor_{i+1} vs Extroversion: r={corr:.4f}")
        
        # 2. 回答パターン分析（バイアス検出）
        print("\n2. 回答パターンバイアス分析...")
        
        # 極端回答バイアス（Extreme Response Style）
        if not numeric_data.empty:
            # 各回答者の回答分散（低分散 = 極端回答）
            response_variance = numeric_data_filled.var(axis=1)
            extreme_responders = response_variance < np.percentile(response_variance, 10)
            
            print(f"極端回答者の割合: {extreme_responders.mean():.3f}")
            print(f"極端回答者の外向性率: {y[extreme_responders].mean():.3f}")
            print(f"通常回答者の外向性率: {y[~extreme_responders].mean():.3f}")
            
            # 同意偏向（Acquiescence Bias）- 高得点に偏る傾向
            response_mean = numeric_data_filled.mean(axis=1)
            high_scorers = response_mean > np.percentile(response_mean, 80)
            
            print(f"高得点偏向者の割合: {high_scorers.mean():.3f}")
            print(f"高得点偏向者の外向性率: {y[high_scorers].mean():.3f}")
        
        # 3. 相互情報量による特徴量重要度
        print("\n3. 相互情報量分析...")
        
        # カテゴリカル特徴量の処理
        categorical_cols = X.select_dtypes(include=['object']).columns
        X_encoded = X.copy()
        
        for col in categorical_cols:
            X_encoded[col] = pd.Categorical(X_encoded[col].fillna('missing')).codes
        
        # 数値特徴量の欠損値処理
        numeric_cols = X_encoded.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            X_encoded[col] = X_encoded[col].fillna(-1)
        
        # 相互情報量計算
        mi_scores = mutual_info_classif(X_encoded, y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': self.feature_cols,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)
        
        print("上位10特徴量（相互情報量）:")
        print(mi_df.head(10))
        
        return {
            'mutual_info_ranking': mi_df,
            'extreme_responders': extreme_responders if not numeric_data.empty else None,
            'high_scorers': high_scorers if not numeric_data.empty else None
        }
    
    def discover_nonlinear_relationships(self):
        """非線形関係の発見"""
        print("\n=== 非線形関係分析 ===")
        
        X = self.train_df[self.feature_cols].copy()
        y = self.train_df['Personality'].map(self.config['TARGET_MAPPING'])
        
        # 数値特徴量で非線形パターンを探索
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            print(f"数値特徴量数: {len(numeric_cols)}")
            
            # 特徴量間の非線形相互作用を探索
            interaction_scores = []
            
            # 上位の特徴量ペアを選択（計算量削減）
            top_features = numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
            
            for col1, col2 in itertools.combinations(top_features, 2):
                # 欠損値処理
                data1 = X[col1].fillna(-1)
                data2 = X[col2].fillna(-1)
                
                # 相互作用項の生成
                interaction_features = {
                    f'{col1}*{col2}': data1 * data2,
                    f'{col1}+{col2}': data1 + data2,
                    f'{col1}-{col2}': data1 - data2,
                    f'abs({col1}-{col2})': np.abs(data1 - data2),
                    f'{col1}/{col2}': data1 / (data2 + 1e-8),  # ゼロ除算回避
                    f'max({col1},{col2})': np.maximum(data1, data2),
                    f'min({col1},{col2})': np.minimum(data1, data2)
                }
                
                # 各相互作用項の相互情報量を計算
                for feature_name, feature_values in interaction_features.items():
                    try:
                        mi_score = mutual_info_classif(
                            feature_values.values.reshape(-1, 1), 
                            y, 
                            random_state=42
                        )[0]
                        
                        interaction_scores.append({
                            'feature': feature_name,
                            'mutual_info': mi_score,
                            'base_features': f'{col1}, {col2}'
                        })
                    except:
                        continue
            
            # 相互作用項のランキング
            interaction_df = pd.DataFrame(interaction_scores)
            if not interaction_df.empty:
                interaction_df = interaction_df.sort_values('mutual_info', ascending=False)
                
                print("上位10相互作用項（相互情報量）:")
                print(interaction_df.head(10))
                
                return interaction_df
        
        return pd.DataFrame()
    
    def personality_clustering_analysis(self):
        """性格クラスタリング分析"""
        print("\n=== 性格クラスタリング分析 ===")
        
        X = self.train_df[self.feature_cols].copy()
        y = self.train_df['Personality'].map(self.config['TARGET_MAPPING'])
        
        # 数値データの準備
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_data = X[numeric_cols].fillna(-1)
            
            # 標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(numeric_data)
            
            # K-means クラスタリング（複数のk値で試行）
            cluster_results = {}
            
            for n_clusters in [2, 3, 4, 5, 6]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                # 各クラスタの外向性率
                cluster_extroversion = []
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    if cluster_mask.sum() > 0:
                        extroversion_rate = (1 - y[cluster_mask]).mean()  # 外向性率
                        cluster_extroversion.append(extroversion_rate)
                    else:
                        cluster_extroversion.append(0)
                
                cluster_results[n_clusters] = {
                    'labels': cluster_labels,
                    'extroversion_rates': cluster_extroversion,
                    'silhouette_score': None  # 必要に応じて計算
                }
                
                print(f"K={n_clusters}: クラスタ外向性率 = {cluster_extroversion}")
            
            # 最適クラスタ数の選択（外向性率の分散が最大）
            best_k = max(cluster_results.keys(), 
                        key=lambda k: np.var(cluster_results[k]['extroversion_rates']))
            
            print(f"最適クラスタ数: {best_k}")
            
            return cluster_results[best_k]['labels']
        
        return None
    
    def missing_pattern_analysis(self):
        """欠損値パターン分析"""
        print("\n=== 欠損値パターン分析 ===")
        
        X = self.train_df[self.feature_cols].copy()
        y = self.train_df['Personality'].map(self.config['TARGET_MAPPING'])
        
        # 欠損値パターンの生成
        missing_patterns = X.isnull()
        
        if missing_patterns.any().any():
            # 欠損値の総数
            missing_counts = missing_patterns.sum(axis=1)
            
            print(f"平均欠損数: {missing_counts.mean():.2f}")
            print(f"最大欠損数: {missing_counts.max()}")
            
            # 欠損数と外向性の関係
            correlation = np.corrcoef(missing_counts, y)[0, 1]
            print(f"欠損数 vs 外向性の相関: {correlation:.4f}")
            
            # 特定の欠損パターンの分析
            pattern_features = []
            
            for col in self.feature_cols:
                if missing_patterns[col].any():
                    # この特徴量が欠損している人の外向性率
                    missing_mask = missing_patterns[col]
                    if missing_mask.sum() > 10:  # 十分なサンプル数
                        extroversion_rate = (1 - y[missing_mask]).mean()
                        overall_rate = (1 - y).mean()
                        
                        pattern_features.append({
                            'feature': f'{col}_missing',
                            'missing_count': missing_mask.sum(),
                            'extroversion_rate': extroversion_rate,
                            'difference': extroversion_rate - overall_rate
                        })
            
            if pattern_features:
                pattern_df = pd.DataFrame(pattern_features)
                pattern_df = pattern_df.sort_values('difference', key=abs, ascending=False)
                
                print("欠損パターンと外向性の関係:")
                print(pattern_df.head(10))
                
                return pattern_df
        
        print("欠損値が見つかりませんでした")
        return pd.DataFrame()
    
    def generate_psychological_features(self):
        """心理学的特徴量の生成"""
        print("\n=== 心理学的特徴量生成 ===")
        
        X_train = self.train_df[self.feature_cols].copy()
        X_test = self.test_df[self.feature_cols].copy()
        
        new_features_train = pd.DataFrame(index=X_train.index)
        new_features_test = pd.DataFrame(index=X_test.index)
        
        # 数値特徴量で心理学的指標を計算
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            # 欠損値処理
            X_train_num = X_train[numeric_cols].fillna(-1)
            X_test_num = X_test[numeric_cols].fillna(-1)
            
            # 1. 回答一貫性（標準偏差）
            new_features_train['response_consistency'] = X_train_num.std(axis=1)
            new_features_test['response_consistency'] = X_test_num.std(axis=1)
            
            # 2. 極端回答傾向（最大-最小）
            new_features_train['extreme_tendency'] = X_train_num.max(axis=1) - X_train_num.min(axis=1)
            new_features_test['extreme_tendency'] = X_test_num.max(axis=1) - X_test_num.min(axis=1)
            
            # 3. 中央値からの偏差
            median_values = X_train_num.median()
            new_features_train['median_deviation'] = (X_train_num - median_values).abs().mean(axis=1)
            new_features_test['median_deviation'] = (X_test_num - median_values).abs().mean(axis=1)
            
            # 4. 正の回答率（閾値を超える回答の割合）
            threshold = X_train_num.median().median()
            new_features_train['positive_response_rate'] = (X_train_num > threshold).mean(axis=1)
            new_features_test['positive_response_rate'] = (X_test_num > threshold).mean(axis=1)
            
            # 5. 歪度（回答分布の非対称性）
            new_features_train['response_skewness'] = X_train_num.apply(lambda row: stats.skew(row), axis=1)
            new_features_test['response_skewness'] = X_test_num.apply(lambda row: stats.skew(row), axis=1)
            
            # 6. 尖度（回答分布の尖り具合）
            new_features_train['response_kurtosis'] = X_train_num.apply(lambda row: stats.kurtosis(row), axis=1)
            new_features_test['response_kurtosis'] = X_test_num.apply(lambda row: stats.kurtosis(row), axis=1)
            
            print(f"生成された心理学的特徴量: {len(new_features_train.columns)}")
            
            # 特徴量と目標変数の相関
            y = self.train_df['Personality'].map(self.config['TARGET_MAPPING'])
            
            print("心理学的特徴量と外向性の相関:")
            for col in new_features_train.columns:
                if not new_features_train[col].isnull().all():
                    correlation = np.corrcoef(new_features_train[col].fillna(0), y)[0, 1]
                    print(f"  {col}: {correlation:.4f}")
        
        return new_features_train, new_features_test
    
    def run_comprehensive_analysis(self):
        """包括的分析の実行"""
        print("=== 高度データ分析開始 ===")
        
        # データ読み込み
        self.load_data()
        
        # 各種分析の実行
        results = {}
        
        # 1. 心理学的パターン分析
        results['psychological_patterns'] = self.psychological_pattern_analysis()
        
        # 2. 非線形関係の発見
        results['nonlinear_relationships'] = self.discover_nonlinear_relationships()
        
        # 3. クラスタリング分析
        results['cluster_labels'] = self.personality_clustering_analysis()
        
        # 4. 欠損値パターン分析
        results['missing_patterns'] = self.missing_pattern_analysis()
        
        # 5. 心理学的特徴量生成
        results['psychological_features'] = self.generate_psychological_features()
        
        print("\n=== 分析完了 ===")
        print("発見された改善ポイント:")
        
        # 結果のサマリー
        if not results['nonlinear_relationships'].empty:
            print(f"- 有望な相互作用項: {len(results['nonlinear_relationships'])} 個")
        
        if results['cluster_labels'] is not None:
            print(f"- 性格クラスタ特徴量: 追加可能")
        
        if not results['missing_patterns'].empty:
            print(f"- 欠損パターン特徴量: {len(results['missing_patterns'])} 個")
        
        if results['psychological_features'][0] is not None:
            print(f"- 心理学的特徴量: {len(results['psychological_features'][0].columns)} 個")
        
        return results

if __name__ == "__main__":
    analyzer = AdvancedPersonalityAnalysis()
    results = analyzer.run_comprehensive_analysis()