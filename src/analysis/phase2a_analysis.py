"""
Phase 2a 詳細問題分析

高次n-gram + TF-IDF特徴量が期待効果を発揮しなかった原因を徹底分析
CV -0.000162の要因特定と改善策の検討

Author: Osawa  
Date: 2025-07-02
Purpose: Phase 2a失敗原因の特定と学習
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class Phase2aAnalyzer:
    """Phase 2a詳細分析器"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def comprehensive_analysis(self):
        """包括的分析実行"""
        
        print("=== Phase 2a 詳細問題分析 ===")
        
        # 1. データ品質分析
        self._analyze_data_quality()
        
        # 2. 特徴量重要度分析
        self._analyze_feature_importance()
        
        # 3. 特徴量分布分析
        self._analyze_feature_distributions()
        
        # 4. 相関分析
        self._analyze_correlations()
        
        # 5. TF-IDF品質分析
        self._analyze_tfidf_quality()
        
        # 6. 問題要因特定
        self._identify_root_causes()
        
        # 7. 改善提案
        self._propose_improvements()
        
        return self.analysis_results
    
    def _analyze_data_quality(self):
        """データ品質分析"""
        
        print("\n1. データ品質分析")
        print("-" * 40)
        
        try:
            # Phase 2aデータ読み込み
            phase2a_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2a_train_features.csv')
            baseline_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/raw/train.csv')
            
            print(f"Phase 2a特徴量数: {phase2a_data.shape[1] - 2}")  # id, Personalityを除く
            print(f"ベースライン特徴量数: {baseline_data.shape[1] - 2}")
            
            # 欠損値分析
            phase2a_missing = phase2a_data.isnull().sum()
            missing_features = phase2a_missing[phase2a_missing > 0]
            
            if len(missing_features) > 0:
                print(f"\n❌ 欠損値問題発見:")
                for feature, count in missing_features.items():
                    print(f"   {feature}: {count}個 ({count/len(phase2a_data)*100:.2f}%)")
            else:
                print("✅ 欠損値なし")
            
            # データ型分析
            object_features = phase2a_data.select_dtypes(include=['object']).columns
            if len(object_features) > 2:  # id, Personality以外
                print(f"\n⚠️ カテゴリカル特徴量多数: {len(object_features)}個")
                print(f"   問題: TF-IDF特徴量が文字列のまま → モデル学習困難")
                self.analysis_results['categorical_issue'] = True
            else:
                print("✅ データ型適切")
                self.analysis_results['categorical_issue'] = False
            
            # 重複値分析
            feature_cols = [col for col in phase2a_data.columns if col not in ['id', 'Personality']]
            duplicate_count = phase2a_data[feature_cols].duplicated().sum()
            print(f"\n重複行数: {duplicate_count}")
            
            self.analysis_results['data_quality'] = {
                'missing_features': len(missing_features),
                'categorical_features': len(object_features),
                'duplicate_rows': duplicate_count,
                'total_features': len(feature_cols)
            }
            
        except Exception as e:
            print(f"データ品質分析エラー: {e}")
    
    def _analyze_feature_importance(self):
        """特徴量重要度分析"""
        
        print("\n2. 特徴量重要度分析")
        print("-" * 40)
        
        try:
            # データ準備
            phase2a_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2a_train_features.csv')
            feature_cols = [col for col in phase2a_data.columns if col not in ['id', 'Personality']]
            
            # カテゴリカル特徴量をエンコード
            X_encoded = phase2a_data[feature_cols].copy()
            for col in X_encoded.columns:
                if X_encoded[col].dtype == 'object':
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            
            X = X_encoded.fillna(0).values
            y = phase2a_data['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values
            
            # RandomForest重要度
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_importance = rf.feature_importances_
            
            # 重要度上位・下位特徴量
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_importance
            }).sort_values('importance', ascending=False)
            
            print("特徴量重要度 Top 10:")
            for i, row in importance_df.head(10).iterrows():
                feature_type = self._classify_feature_type(row['feature'])
                print(f"   {row['feature'][:40]:40} {row['importance']:.6f} [{feature_type}]")
            
            print("\n特徴量重要度 Bottom 10:")
            for i, row in importance_df.tail(10).iterrows():
                feature_type = self._classify_feature_type(row['feature'])
                print(f"   {row['feature'][:40]:40} {row['importance']:.6f} [{feature_type}]")
            
            # 特徴量タイプ別重要度
            feature_type_importance = {}
            for _, row in importance_df.iterrows():
                ftype = self._classify_feature_type(row['feature'])
                if ftype not in feature_type_importance:
                    feature_type_importance[ftype] = []
                feature_type_importance[ftype].append(row['importance'])
            
            print(f"\n特徴量タイプ別平均重要度:")
            for ftype, importances in feature_type_importance.items():
                avg_importance = np.mean(importances)
                count = len(importances)
                print(f"   {ftype:15}: {avg_importance:.6f} (数: {count})")
            
            # 低重要度特徴量の比率
            low_importance_count = len(importance_df[importance_df['importance'] < 0.001])
            low_ratio = low_importance_count / len(importance_df)
            
            if low_ratio > 0.3:
                print(f"\n❌ 低重要度特徴量が多数: {low_importance_count}/{len(importance_df)} ({low_ratio*100:.1f}%)")
                print("   問題: ノイズ特徴量が多く、モデル性能を阻害している可能性")
                self.analysis_results['noise_features_issue'] = True
            else:
                print(f"✅ 低重要度特徴量の比率は適切: {low_ratio*100:.1f}%")
                self.analysis_results['noise_features_issue'] = False
            
            self.analysis_results['feature_importance'] = {
                'top_features': importance_df.head(10).to_dict('records'),
                'bottom_features': importance_df.tail(10).to_dict('records'),
                'type_importance': {k: np.mean(v) for k, v in feature_type_importance.items()},
                'low_importance_ratio': low_ratio
            }
            
        except Exception as e:
            print(f"特徴量重要度分析エラー: {e}")
    
    def _classify_feature_type(self, feature_name):
        """特徴量名から特徴量タイプを分類"""
        
        if 'tfidf' in feature_name.lower():
            return 'TF-IDF'
        elif '4gram' in feature_name.lower():
            return '4-gram'
        elif '5gram' in feature_name.lower():
            return '5-gram'
        elif any(orig in feature_name for orig in ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                                                  'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']):
            return 'Original'
        else:
            return 'Other'
    
    def _analyze_feature_distributions(self):
        """特徴量分布分析"""
        
        print("\n3. 特徴量分布分析")
        print("-" * 40)
        
        try:
            phase2a_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2a_train_features.csv')
            feature_cols = [col for col in phase2a_data.columns if col not in ['id', 'Personality']]
            
            # TF-IDF特徴量の分布分析
            tfidf_features = [col for col in feature_cols if 'tfidf' in col.lower()]
            print(f"TF-IDF特徴量数: {len(tfidf_features)}")
            
            if len(tfidf_features) > 0:
                # TF-IDF値の統計
                tfidf_data = phase2a_data[tfidf_features]
                
                # 数値変換が必要かチェック
                tfidf_numeric = tfidf_data.copy()
                for col in tfidf_numeric.columns:
                    if tfidf_numeric[col].dtype == 'object':
                        try:
                            tfidf_numeric[col] = pd.to_numeric(tfidf_numeric[col], errors='coerce')
                        except:
                            pass
                
                if tfidf_numeric.select_dtypes(include=[np.number]).shape[1] > 0:
                    numeric_tfidf = tfidf_numeric.select_dtypes(include=[np.number])
                    
                    print(f"TF-IDF統計:")
                    print(f"   平均値: {numeric_tfidf.mean().mean():.6f}")
                    print(f"   標準偏差: {numeric_tfidf.std().mean():.6f}")
                    print(f"   最大値: {numeric_tfidf.max().max():.6f}")
                    print(f"   最小値: {numeric_tfidf.min().min():.6f}")
                    
                    # ゼロ値の比率
                    zero_ratio = (numeric_tfidf == 0).sum().sum() / (numeric_tfidf.shape[0] * numeric_tfidf.shape[1])
                    print(f"   ゼロ値比率: {zero_ratio*100:.1f}%")
                    
                    if zero_ratio > 0.8:
                        print("   ❌ TF-IDF特徴量の大部分がゼロ → 情報量が少ない")
                        self.analysis_results['tfidf_sparsity_issue'] = True
                    else:
                        print("   ✅ TF-IDF特徴量の密度は適切")
                        self.analysis_results['tfidf_sparsity_issue'] = False
                else:
                    print("   ❌ TF-IDF特徴量が全て非数値 → エンコーディング問題")
                    self.analysis_results['tfidf_encoding_issue'] = True
            
            # n-gram特徴量のユニーク値分析
            ngram_features = [col for col in feature_cols if 'gram' in col.lower() and 'tfidf' not in col.lower()]
            print(f"\nn-gram特徴量数: {len(ngram_features)}")
            
            if len(ngram_features) > 0:
                for feature in ngram_features[:5]:  # 最初の5個をサンプル
                    unique_count = phase2a_data[feature].nunique()
                    total_count = len(phase2a_data[feature])
                    unique_ratio = unique_count / total_count
                    print(f"   {feature[:40]:40}: ユニーク値 {unique_count}/{total_count} ({unique_ratio:.3f})")
                    
                    if unique_ratio > 0.95:
                        print(f"     ⚠️ ユニーク値比率が高すぎる → 過度に詳細な特徴量")
                        self.analysis_results['high_cardinality_issue'] = True
            
        except Exception as e:
            print(f"特徴量分布分析エラー: {e}")
    
    def _analyze_correlations(self):
        """相関分析"""
        
        print("\n4. 相関分析")
        print("-" * 40)
        
        try:
            phase2a_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2a_train_features.csv')
            feature_cols = [col for col in phase2a_data.columns if col not in ['id', 'Personality']]
            
            # 数値特徴量のみ抽出
            numeric_data = phase2a_data[feature_cols].copy()
            for col in numeric_data.columns:
                if numeric_data[col].dtype == 'object':
                    try:
                        numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
                    except:
                        numeric_data[col] = np.nan
            
            numeric_data = numeric_data.select_dtypes(include=[np.number]).fillna(0)
            
            if numeric_data.shape[1] > 1:
                # 相関行列計算
                corr_matrix = numeric_data.corr().abs()
                
                # 高相関ペアの特定 (対角線除く)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                corr_matrix_masked = corr_matrix.mask(mask)
                
                # 高相関ペア (>0.9)
                high_corr_pairs = []
                for i in range(len(corr_matrix_masked.columns)):
                    for j in range(len(corr_matrix_masked.columns)):
                        if not pd.isna(corr_matrix_masked.iloc[i, j]) and corr_matrix_masked.iloc[i, j] > 0.9:
                            high_corr_pairs.append((
                                corr_matrix_masked.columns[i],
                                corr_matrix_masked.columns[j],
                                corr_matrix_masked.iloc[i, j]
                            ))
                
                print(f"高相関ペア数 (>0.9): {len(high_corr_pairs)}")
                
                if len(high_corr_pairs) > 10:
                    print("❌ 高相関特徴量が多数存在 → 冗長性問題")
                    print("   上位5ペア:")
                    sorted_pairs = sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)
                    for feat1, feat2, corr in sorted_pairs[:5]:
                        print(f"     {feat1[:20]} - {feat2[:20]}: {corr:.3f}")
                    self.analysis_results['multicollinearity_issue'] = True
                else:
                    print("✅ 高相関特徴量は適切な範囲")
                    self.analysis_results['multicollinearity_issue'] = False
                
                # 平均相関
                avg_corr = corr_matrix_masked.stack().mean()
                print(f"平均相関: {avg_corr:.3f}")
                
                self.analysis_results['correlation_analysis'] = {
                    'high_corr_pairs': len(high_corr_pairs),
                    'avg_correlation': avg_corr
                }
            
        except Exception as e:
            print(f"相関分析エラー: {e}")
    
    def _analyze_tfidf_quality(self):
        """TF-IDF品質分析"""
        
        print("\n5. TF-IDF品質分析")
        print("-" * 40)
        
        try:
            phase2a_data = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2a_train_features.csv')
            
            # TF-IDF特徴量を特定
            tfidf_features = [col for col in phase2a_data.columns if 'tfidf' in col.lower()]
            
            if len(tfidf_features) == 0:
                print("❌ TF-IDF特徴量が見つからない")
                return
            
            print(f"TF-IDF特徴量数: {len(tfidf_features)}")
            
            # サンプルTF-IDF特徴量の値を確認
            sample_tfidf = phase2a_data[tfidf_features[:5]]
            print(f"\nサンプルTF-IDF値:")
            for col in sample_tfidf.columns:
                values = sample_tfidf[col].head(10).tolist()
                print(f"   {col[:50]:50}: {values}")
            
            # TF-IDF特徴量のデータ型確認
            tfidf_dtypes = phase2a_data[tfidf_features].dtypes
            object_tfidf = [col for col in tfidf_features if tfidf_dtypes[col] == 'object']
            
            if len(object_tfidf) > 0:
                print(f"\n❌ 文字列型TF-IDF特徴量: {len(object_tfidf)}個")
                print("   問題: TF-IDF値が数値化されていない")
                print(f"   例: {object_tfidf[:3]}")
                self.analysis_results['tfidf_type_issue'] = True
            else:
                print("✅ TF-IDF特徴量は全て数値型")
                self.analysis_results['tfidf_type_issue'] = False
            
            # TF-IDF実装の問題点チェック
            print(f"\nTF-IDF実装品質チェック:")
            
            # 1. 命名パターンチェック
            naming_patterns = {}
            for col in tfidf_features:
                parts = col.split('_')
                if len(parts) >= 3 and 'tfidf' in parts:
                    pattern = '_'.join(parts[:-1])  # 最後の要素（単語）を除いた部分
                    if pattern not in naming_patterns:
                        naming_patterns[pattern] = 0
                    naming_patterns[pattern] += 1
            
            print(f"   TF-IDF命名パターン数: {len(naming_patterns)}")
            for pattern, count in list(naming_patterns.items())[:5]:
                print(f"     {pattern}: {count}個")
            
            if len(naming_patterns) < 5:
                print("   ⚠️ TF-IDF命名パターンが少ない → 生成バリエーション不足")
            
        except Exception as e:
            print(f"TF-IDF品質分析エラー: {e}")
    
    def _identify_root_causes(self):
        """根本原因特定"""
        
        print("\n6. 根本原因特定")
        print("-" * 40)
        
        root_causes = []
        
        # 各問題をチェック
        if self.analysis_results.get('categorical_issue', False):
            root_causes.append("❌ カテゴリカル特徴量エンコーディング問題")
        
        if self.analysis_results.get('noise_features_issue', False):
            root_causes.append("❌ ノイズ特徴量大量生成")
        
        if self.analysis_results.get('tfidf_sparsity_issue', False):
            root_causes.append("❌ TF-IDF特徴量の極度な希薄性")
        
        if self.analysis_results.get('tfidf_encoding_issue', False):
            root_causes.append("❌ TF-IDF特徴量の数値化失敗")
        
        if self.analysis_results.get('high_cardinality_issue', False):
            root_causes.append("❌ 高カーディナリティ特徴量問題")
        
        if self.analysis_results.get('multicollinearity_issue', False):
            root_causes.append("❌ 多重共線性問題")
        
        if self.analysis_results.get('tfidf_type_issue', False):
            root_causes.append("❌ TF-IDF実装の根本的問題")
        
        print("特定された根本原因:")
        for i, cause in enumerate(root_causes, 1):
            print(f"   {i}. {cause}")
        
        if not root_causes:
            print("✅ 明確な技術的問題は特定されませんでした")
            print("   → 手法自体の有効性の問題の可能性")
        
        self.analysis_results['root_causes'] = root_causes
    
    def _propose_improvements(self):
        """改善提案"""
        
        print("\n7. 改善提案")
        print("-" * 40)
        
        improvements = []
        
        # 根本原因に基づく改善提案
        if self.analysis_results.get('categorical_issue', False):
            improvements.append("🔧 全特徴量の適切な数値エンコーディング実装")
        
        if self.analysis_results.get('noise_features_issue', False):
            improvements.append("🔧 特徴量選択アルゴリズムによるノイズ除去")
        
        if self.analysis_results.get('tfidf_sparsity_issue', False):
            improvements.append("🔧 TF-IDF最小文書頻度(min_df)の調整")
        
        if self.analysis_results.get('high_cardinality_issue', False):
            improvements.append("🔧 n-gram組み合わせの更なる厳選")
        
        if self.analysis_results.get('multicollinearity_issue', False):
            improvements.append("🔧 相関フィルタリングによる冗長性除去")
        
        if self.analysis_results.get('tfidf_type_issue', False):
            improvements.append("🔧 TF-IDF実装の根本的見直し")
        
        # 一般的改善提案
        improvements.extend([
            "🎯 より selective な4-gram/5-gram選択",
            "📊 特徴量重要度に基づく段階的追加",
            "🧪 単一特徴量タイプでの効果検証",
            "⚡ より軽量な特徴量エンジニアリング手法の検討"
        ])
        
        print("推奨改善策:")
        for i, improvement in enumerate(improvements, 1):
            print(f"   {i}. {improvement}")
        
        # Phase 2b への提言
        print(f"\nPhase 2b への提言:")
        print("   📍 高次n-gram + TF-IDFは効果限定的")
        print("   📍 Target Encoding等のよりシンプルな手法を優先")
        print("   📍 特徴量品質 > 特徴量数量")
        
        self.analysis_results['improvements'] = improvements
        
        return improvements

def main():
    """メイン実行関数"""
    
    analyzer = Phase2aAnalyzer()
    results = analyzer.comprehensive_analysis()
    
    # 結果保存 (JSON serializable形式に変換)
    import json
    
    # Numpy型をPython型に変換
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy_types(results)
    
    with open('/Users/osawa/kaggle/playground-series-s5e7/data/processed/phase2a_analysis_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n" + "="*60)
    print("Phase 2a 分析完了")
    print("="*60)
    print("✅ 分析結果保存: phase2a_analysis_results.json")
    print("📋 主要問題: TF-IDF実装品質とノイズ特徴量")
    print("🎯 Phase 2b: よりシンプルで効果的な手法に注力")
    
    return results

if __name__ == "__main__":
    main()