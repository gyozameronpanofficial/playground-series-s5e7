# 📁 プロジェクト構成ガイド

## 🎯 整理済みフォルダ構成

```
/Users/osawa/kaggle/playground-series-s5e7/
├── 📋 プロジェクト文書
│   ├── CLAUDE.md                    # プロジェクト設定・指示書
│   ├── IMPLEMENTATION_REPORT.md     # 詳細実装レポート
│   ├── README.md                    # プロジェクト概要
│   └── PROJECT_STRUCTURE.md         # このファイル
│
├── 📊 分析・戦略文書
│   └── analysis/
│       ├── EDA_Report.md            # 探索的データ分析
│       ├── GM_Differentiation_Strategy.md  # GM差別化戦略
│       └── improvement_strategy.md   # 改善戦略
│
├── 💾 データファイル
│   ├── data/
│   │   ├── raw/                     # 元データ
│   │   │   ├── train.csv
│   │   │   ├── test.csv
│   │   │   └── sample_submission.csv
│   │   └── processed/               # 処理済み特徴量データ
│   │       ├── psychological_*.csv   # Phase 1: 心理学特徴量
│   │       ├── phase2a_*.csv        # Phase 2a: n-gram + TF-IDF
│   │       ├── phase2b_*.csv        # Phase 2b: Target Encoding
│   │       ├── hybrid_*.csv         # Phase 3: 統合版
│   │       └── pseudo_labeled_train.csv  # 擬似ラベル拡張データ
│
├── 🧪 ソースコード
│   └── src/
│       ├── phases/                  # フェーズ別実装
│       │   ├── phase1_psychological_features.py  # 心理学特徴量
│       │   ├── phase2_pseudo_labeling.py         # 擬似ラベリング
│       │   ├── phase2a_ngram_tfidf.py           # n-gram + TF-IDF
│       │   ├── phase2b_target_encoding.py       # Target Encoding
│       │   └── phase3_hybrid_integration.py     # 統合実装
│       │
│       ├── analysis/                # 評価・分析スクリプト
│       │   ├── phase2a_cv_evaluation.py         # Phase 2a CV評価
│       │   ├── phase2b_cv_evaluation.py         # Phase 2b CV評価
│       │   ├── hybrid_cv_evaluation.py          # 統合版CV評価
│       │   ├── phase2a_analysis.py              # Phase 2a失敗分析
│       │   └── simple_ensemble_validation.py    # アンサンブル検証
│       │
│       ├── submissions/             # 提出ファイル作成
│       │   ├── phase1_2_submission.py           # Phase 1+2提出
│       │   ├── phase2a_submission.py            # Phase 2a提出
│       │   ├── phase2b_submission.py            # Phase 2b提出
│       │   └── hybrid_submission.py             # 統合版提出
│       │
│       └── utils/                   # ユーティリティ
│           ├── baseline_reproduction.py         # ベースライン再現
│           ├── gm_exact_reproduction.py         # GM完全再現
│           └── advanced_ensemble.py             # 高度アンサンブル
│
├── 📊 結果・モデル
│   └── results/
│       ├── cv_results/              # CV評価結果
│       │   ├── phase2a_cv_results.json
│       │   ├── phase2b_cv_results.json
│       │   ├── hybrid_cv_results.json
│       │   └── phase2a_analysis_results.json
│       └── models/                  # 訓練済みモデル（予約）
│
├── 🚀 提出ファイル
│   └── submissions/
│       ├── psychological_pseudo_submission.csv      # Phase 1+2
│       ├── phase2a_ngram_tfidf_submission.csv      # Phase 2a
│       ├── phase2b_target_encoding_submission.csv  # Phase 2b ⭐GM同値
│       └── gm_exceed_hybrid_submission.csv         # 統合版 ⭐GM同値
│
├── 📓 ノートブック
│   └── notebooks/
│       ├── EDA_Analysis.ipynb                       # 探索的データ分析
│       └── playgrounds5e7-public-baseline-v1.ipynb # GM公開ベースライン
│
└── 🔧 環境設定
    ├── requirements.txt             # Python依存関係
    └── venv/                        # 仮想環境
```

## 🎯 主要ファイルの実行順序

### Phase 1+2: 心理学特徴量 + 擬似ラベリング
```bash
cd /Users/osawa/kaggle/playground-series-s5e7
python src/phases/phase1_psychological_features.py
python src/phases/phase2_pseudo_labeling.py
python src/analysis/simple_ensemble_validation.py
python src/submissions/phase1_2_submission.py
```

### Phase 2a: 高次n-gram + TF-IDF
```bash
python src/phases/phase2a_ngram_tfidf.py
python src/analysis/phase2a_cv_evaluation.py
python src/analysis/phase2a_analysis.py
python src/submissions/phase2a_submission.py
```

### Phase 2b: 高度Target Encoding ⭐推奨
```bash
python src/phases/phase2b_target_encoding.py
python src/analysis/phase2b_cv_evaluation.py
python src/submissions/phase2b_submission.py
```

### Phase 3: 統合版
```bash
python src/phases/phase3_hybrid_integration.py
python src/analysis/hybrid_cv_evaluation.py
python src/submissions/hybrid_submission.py
```

## 🏆 最高性能手法

**推奨**: Phase 2b（高度Target Encoding）
- **PBスコア**: 0.975708（GM同値達成）
- **実装**: `src/phases/phase2b_target_encoding.py`
- **提出**: `submissions/phase2b_target_encoding_submission.csv`

## 📊 結果確認

各フェーズの結果は `results/cv_results/` で確認可能：
- CV性能、PB性能、改善効果の詳細分析
- 特徴量重要度、失敗原因分析も含む

## 🔧 環境設定

```bash
cd /Users/osawa/kaggle/playground-series-s5e7
pip install -r requirements.txt
```

---

*Last Updated: 2025-07-03*
*整理により、各フェーズの実装が明確に分離され、チーム開発に最適化されました*