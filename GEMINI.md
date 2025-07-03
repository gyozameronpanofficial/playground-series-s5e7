# Gemini プロジェクト概要

## 🎯 プロジェクトの目的

このプロジェクトは、KaggleのPlayground Series S5E7「Predict the Introverts from the Extroverts」コンペティションにおいて、Grandmaster (GM) の公開ベースラインスコア（PB: 0.975708）を超えることを目的としています。

## 📁 プロジェクト構成

プロジェクトは、データ処理、特徴量エンジニアリング、モデリング、評価の各フェーズごとに整理されています。

- `data/`: 元データと処理済みデータを格納
- `src/`: Pythonスクリプトを格納
    - `phases/`: 各特徴量エンジニアリングフェーズの実装
    - `analysis/`: モデル評価と分析スクリプト
    - `submissions/`: 提出ファイル作成スクリプト
    - `utils/`: 共通関数や再現コード
- `notebooks/`: 探索的データ分析（EDA）やベースラインモデルのノートブック
- `results/`: CV（交差検証）結果やモデルを保存
- `submissions/`: 生成された提出ファイル

## 🚀 主要なアプローチと結果

このプロジェクトでは、複数の特徴量エンジニアリング手法が試行されました。`IMPLEMENTATION_REPORT.md`によると、主要な結果は以下の通りです。

| フェーズ | 実装内容 | CVスコア | PBスコア | GM比較 |
|---|---|---|---|---|
| **Phase 2b** | 高度Target Encoding | 0.968905 | **0.975708** | ✅ **GM同値** |
| **統合版** | Phase 2b + 心理学特徴量 | **0.976404** | **0.975708** | ✅ **GM同値** |

- **最高のCVスコア**: 「統合版」が最高のCV（Cross-Validation）スコアを達成しました。
- **性能上限**: しかし、最終的なPublic Boardスコアは「Phase 2b」と「統合版」で同じであり、`0.975708` がこのデータセットの実質的な性能上限である可能性が高いと分析されています。

## 🏆 最終推奨手法

**Phase 2b: 高度なTarget Encoding**

`IMPLEMENTATION_REPORT.md`に基づき、本プロジェクトの最終的な推奨手法は **Phase 2b** です。

### 推奨理由
1.  **最高のPBスコア達成**: 複雑な「統合版」と同じ最高のPBスコアを達成。
2.  **シンプルさ**: より少ない特徴量（15個）で構成されており、モデルがシンプル。
3.  **効率性と再現性**: 計算コストが低く、再現性が高い。
4.  **過学習リスクの低減**: CVスコアとPBスコアの乖離が大きく、これはむしろ汎化性能の高さを示唆していると分析されています。一方、「統合版」はCVスコアに対してPBスコアが伸びず、過学習の傾向が見られます。

### 実行コマンド
```bash
# 1. 特徴量生成
python src/phases/phase2b_target_encoding.py

# 2. CV評価
python src/analysis/phase2b_cv_evaluation.py

# 3. 提出ファイルの作成
python src/submissions/phase2b_submission.py
```

## 🔧 環境設定

必要なPythonライブラリは`requirements.txt`に記載されています。以下のコマンドでインストールできます。

```bash
pip install -r requirements.txt
```