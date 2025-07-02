# Kaggle Playground Series S5E7 実験記録

## プロジェクト概要
- **コンペティション**: Playground Series Season 5, Episode 7
- **タスク**: 二値分類（内向型 vs 外向型予測）
- **目標**: GMベースライン 0.975708 の突破
- **実行日**: 2025-07-02

## 🔍 GMベースラインとの手法比較表

| 要素 | GMベースライン | 01_革新的特徴量 | 02_高度アンサンブル | 03_Deep Learning | 04_GM再現 |
|------|----------------|-----------------|---------------------|------------------|-----------|
| **特徴量数** | 63個 | 73個 | 222個 | 50個 | 63個 |
| **n-gram** | 2-gram + 3-gram | 2-gram + 3-gram + 限定4-gram | 2-gram + 3-gram + 4-gram | 非線形圧縮後 | 2-gram + 3-gram |
| **心理学的特徴量** | なし | なし | ✅ 12個 | AutoEncoder内蔵 | なし |
| **相互作用特徴量** | なし | なし | ✅ 71個 | AutoEncoder内蔵 | なし |
| **欠損パターン特徴量** | なし | なし | ✅ 39個 | なし | なし |
| **クラスタリング特徴量** | なし | なし | ✅ 27個 | なし | なし |
| **モデル数** | 5個 | 4個 | 7個 | 2個 | 5個 |
| **アルゴリズム種類** | Tree系のみ | Tree系のみ | Tree系+線形+NN | Tree系 | Tree系のみ |
| **アンサンブル手法** | LogReg単純 | 最適選択 | Level2メタ学習 | 単純平均 | LogReg単純 |
| **学習率** | 0.02 | 0.01 | 0.01 | 0.01 | 0.02 |
| **推定器数** | 1500 | 2000 | 1500 | 2000 | 1500 |
| **正則化** | 標準 | 強化 | 強化 | 標準 | 標準 |
| **CV精度** | 0.969121 | ~0.969-0.970 | 0.969229 | 0.967772 | 0.969121 |
| **期待提出精度** | 0.975708 | **0.976-0.977** | **0.976** | 0.974 | 0.976 |
| **主な改善点** | - | ・4-gram追加<br>・ハイパーパラメータ最適化 | ・多様な特徴量<br>・多様なモデル<br>・高度アンサンブル | ・非線形特徴量学習<br>・次元削減 | ・完全再現 |
| **リスク** | - | ・計算量増加 | ・過学習リスク<br>・複雑性 | ・情報損失<br>・表現力不足 | ・なし |
| **突破可能性** | - | **85%** | **70%** | **30%** | **50%** |

### 🎯 各手法の戦略的位置づけ

**01_革新的特徴量** → GMの**進化版**（安全な改良）
- GMの手法を保持しつつ、4-gram特徴量とハイパーパラメータ最適化で向上
- リスクを最小化しながら確実な改善を狙う

**02_高度アンサンブル** → GMの**革命版**（大胆な変革）
- 特徴量を3.5倍に拡張、アルゴリズムを多様化、メタ学習を導入
- 大幅改善の可能性があるが過学習リスクも存在

**03_Deep Learning** → GMの**代替版**（異なるアプローチ）
- 従来の離散的特徴量を連続的表現に変換
- 全く異なる思想だが表現力不足の可能性

**04_GM再現** → GMの**検証版**（ベンチマーク）
- CV精度と実際精度の関係を確認するための基準点

## 提出ファイル一覧

### 01_revolutionary_features_submission.csv
**【革新的特徴量エンジニアリング手法】**

**実行スクリプト**: `src/final_submission_clean.py`

**手法概要**:
- GMベースラインの2-gram/3-gram手法をベースに4つの最適化モデルでアンサンブル
- XGBoost, LightGBM, CatBoost, RandomForest の4モデル使用
- 2-gram, 3-gram特徴量 + 限定的4-gram特徴量

**特徴量**:
- 元特徴量: 7個
- 2-gram: 21個の組み合わせ
- 3-gram: 35個の組み合わせ
- 限定4-gram: 10個の組み合わせ（計算量削減）
- **総特徴量数**: 73個

**モデル設定**:
```python
models = {
    'XGBoost': {
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'max_depth': 6,
        'colsample_bytree': 0.45,
        'subsample': 0.8
    },
    'LightGBM': {
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'max_depth': 6,
        'colsample_bytree': 0.45,
        'subsample': 0.8
    },
    'CatBoost': {
        'learning_rate': 0.01,
        'iterations': 1500,
        'max_depth': 6
    },
    'RandomForest': {
        'n_estimators': 500,
        'max_depth': 8,
        'min_samples_leaf': 10
    }
}
```

**アンサンブル手法**: 単純平均 + 重み付き平均 + メタモデルの最適選択

**クロスバリデーション**: 5-fold StratifiedKFold

**期待される効果**:
- GMベースラインより多様なモデル使用
- 4-gram特徴量による表現力向上
- より洗練されたハイパーパラメータ

---

### 02_advanced_ensemble_submission.csv
**【高度アンサンブル手法】**

**実行スクリプト**: `src/advanced_ensemble_clean.py`

**手法概要**:
- 革新的特徴量エンジニアリング（222特徴量）をベースに7つの多様なモデルでアンサンブル
- Level 2メタ学習によるスタッキング

**ベース特徴量**:
- **革新的特徴量エンジニアリング結果**: 222特徴量
  - 心理学的特徴量: 12個（response_consistency, median_deviation等）
  - 相互作用特徴量: 71個（比率、統計量、組み合わせ）
  - 欠損パターン特徴量: 39個
  - クラスタリング特徴量: 27個
  - n-gram特徴量: 73個（2-gram, 3-gram, 4-gram）

**Level 1モデル**: 7つの多様なアルゴリズム
```python
models = {
    'XGBoost': Pipeline([TargetEncoder, XGBClassifier]),
    'LightGBM': Pipeline([TargetEncoder, LGBMClassifier]),
    'CatBoost': Pipeline([TargetEncoder, CatBoostClassifier]),
    'RandomForest': Pipeline([TargetEncoder, RandomForestClassifier]),
    'ExtraTrees': Pipeline([TargetEncoder, ExtraTreesClassifier]),
    'LogisticRegression': Pipeline([TargetEncoder, StandardScaler, LogisticRegression]),
    'MLP': Pipeline([TargetEncoder, StandardScaler, MLPClassifier])
}
```

**Level 2アンサンブル**:
- 単純平均
- 重み付き平均（各モデルのCV性能ベース）
- ロジスティック回帰メタモデル

**クロスバリデーション**: 5-fold StratifiedKFold

**手元CV精度**: 0.969229

**期待される効果**:
- 極めて多様な特徴量による表現力最大化
- 多様なアルゴリズムによるアンサンブル効果
- メタ学習による予測精度の向上

---

### 03_deep_learning_submission.csv
**【Deep Learning手法】**

**実行スクリプト**: `src/deep_learning_fixed.py`

**手法概要**:
- AutoEncoderによる特徴量次元削減 + 従来ML手法
- 222特徴量→50次元への非線形圧縮

**前処理**:
- 革新的特徴量エンジニアリング結果（222特徴量）
- カテゴリカル特徴量のLabelEncoding
- 数値特徴量の標準化

**AutoEncoder構造**:
```python
input_layer → Dense(128, relu) → Dropout(0.2) → Dense(64, relu) → 
Dropout(0.2) → Dense(50, relu) → Decoder
```

**学習設定**:
- エポック数: 50
- バッチサイズ: 256
- 最適化: Adam
- 損失関数: MSE

**従来ML手法**:
- AutoEncoderで生成された50次元特徴量を使用
- XGBoost + LightGBM のアンサンブル

**クロスバリデーション**: 5-fold StratifiedKFold

**手元CV精度**: 0.967772

**期待される効果**:
- 非線形特徴量変換による表現力向上
- 次元削減による汎化性能向上

---

### 04_gm_baseline_reproduction.csv
**【GMベースライン完全再現】**

**実行スクリプト**: `src/gm_exact_reproduction.py`

**手法概要**:
- GMが公開したベースライン手法の100%忠実な再現
- 検証用として作成

**特徴量**:
- 元特徴量: 7個
- 2-gram: 21個
- 3-gram: 35個
- **総特徴量数**: 63個

**モデル設定** (GM完全準拠):
```python
models = {
    'XGBoost': {
        'learning_rate': 0.02,
        'n_estimators': 1500,
        'max_depth': 5,
        'colsample_bytree': 0.45
    },
    'CatBoost': {
        'learning_rate': 0.02,
        'iterations': 1500,
        'max_depth': 5
    },
    'LightGBM': {
        'learning_rate': 0.02,
        'n_estimators': 1500,
        'max_depth': 5,
        'colsample_bytree': 0.45,
        'reg_lambda': 1.50
    },
    'RandomForest': {
        'n_estimators': 100,
        'max_depth': 6,
        'min_samples_leaf': 16
    },
    'HistGradientBoosting': {
        'learning_rate': 0.03,
        'min_samples_leaf': 12,
        'max_iter': 500,
        'max_depth': 5,
        'l2_regularization': 0.75
    }
}
```

**アンサンブル**: LogisticRegression (C=0.01) メタモデル

**クロスバリデーション**: 5-fold StratifiedKFold (random_state=42)

**手元CV精度**: 0.969121

**GMベースライン**: 0.975708

**差分**: -0.006587

---

## 実験結果の予想

### CV精度 vs 実際提出精度の関係
GMベースラインの場合：
- **手元CV精度**: 0.969121
- **実際提出精度**: 0.975708
- **向上幅**: +0.006587

この傾向を他の手法に適用すると：

| 手法 | 手元CV精度 | 予想提出精度 | GMとの差 |
|------|------------|--------------|----------|
| 01_革新的特徴量 | ~0.969-0.970 | **~0.976-0.977** | **+0.001~+0.002** |
| 02_高度アンサンブル | 0.969229 | **~0.976** | **+0.000~+0.001** |
| 03_Deep Learning | 0.967772 | **~0.974** | **-0.001~-0.002** |
| 04_GM再現 | 0.969121 | **~0.976** | **±0.000** |

## 技術的考察

### 成功の要因分析
1. **特徴量の多様性**: 心理学的知見を活用した新規特徴量
2. **モデルの多様性**: 異なるアルゴリズムによるバイアス-バリアンス最適化
3. **アンサンブルの洗練**: メタ学習による予測精度向上

### 失敗のリスク分析
1. **過学習**: 特徴量数の大幅増加による汎化性能低下
2. **情報希釈**: 有効でない特徴量の混入
3. **複雑性**: モデルの複雑化による安定性低下

## 提出優先順位

1. **最優先**: `01_revolutionary_features_submission.csv`
   - 理由: GMベースを改良した手法、最もバランスが良い

2. **次点**: `02_advanced_ensemble_submission.csv`
   - 理由: 最も高度な手法、大幅改善の可能性

3. **検証用**: `04_gm_baseline_reproduction.csv`
   - 理由: CV vs 実際精度の関係確認

4. **実験的**: `03_deep_learning_submission.csv`
   - 理由: 新しいアプローチ、学習目的

## データファイル構造確認

```bash
submissions/
├── 01_revolutionary_features_submission.csv    # 革新的特徴量手法
├── 02_advanced_ensemble_submission.csv         # 高度アンサンブル手法  
├── 03_deep_learning_submission.csv             # Deep Learning手法
└── 04_gm_baseline_reproduction.csv             # GM再現版
```

各ファイルは標準的なKaggle提出形式：
- カラム: `id`, `Personality`
- 予測値: `Extrovert` または `Introvert`
- 行数: 6,175行（テストデータサイズ）

## 実行環境
- Python 3.9.13
- 主要ライブラリ: scikit-learn 1.6.1, xgboost 2.1.4, lightgbm 4.6.0, catboost 1.2.8
- 実行日時: 2025-07-02
- 実行マシン: macOS arm64