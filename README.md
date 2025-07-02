# 🧠 Playground Series S5E7 - Predict the Introverts from the Extroverts

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)

**個人の性格特性データから内向型・外向型を予測する機械学習コンペティション**

[📊 Competition Page](https://www.kaggle.com/competitions/playground-series-s5e7) • [🚀 Quick Start](#-クイックスタート) • [📈 Results](#-結果)

</div>

---

## 📋 コンペティション概要

<table>
<tr>
<td><b>🏆 コンペ名</b></td>
<td>Predict the Introverts from the Extroverts</td>
</tr>
<tr>
<td><b>📝 シリーズ</b></td>
<td>Playground Series - Season 5, Episode 7</td>
</tr>
<tr>
<td><b>🎯 問題設定</b></td>
<td>人格特性データに基づいて内向型(Introvert)か外向型(Extrovert)かを予測する二項分類問題</td>
</tr>
<tr>
<td><b>📊 評価指標</b></td>
<td>Accuracy（予測精度）</td>
</tr>
<tr>
<td><b>📈 データセット</b></td>
<td>人格特性・行動パターンに関する特徴量</td>
</tr>
</table>

### 🎯 目標
個人の行動パターンや性格特性のデータから、その人が**内向型**か**外向型**かを機械学習で予測すること。

---

## 📊 データセット構成

```
📂 data/
├── 📄 train.csv              # 学習用データ（目的変数あり）
├── 📄 test.csv               # テスト用データ（目的変数なし）
└── 📄 sample_submission.csv  # 提出ファイルのサンプル
```

### 🔍 主要な特徴量（推定）

| カテゴリ | 内容 | 説明 |
|----------|------|------|
| 🎭 **行動パターン** | 日常の行動傾向 | 社交的な活動への参加頻度など |
| 🤝 **社交性指標** | 人との関わり方 | グループ活動 vs 個人活動の選好 |
| 💬 **コミュニケーションスタイル** | 意思疎通の方法 | 積極性、発言頻度、聞き手傾向 |
| 🎪 **活動傾向・選好** | 好みの活動タイプ | エネルギッシュ vs 静的な活動 |

---

## 🚀 アプローチ戦略

### 🔍 1. データ探索・前処理

<details>
<summary><b>詳細な前処理ステップ</b></summary>

- [ ] **📊 欠損値の確認と処理**
  - 欠損パターンの分析
  - 適切な補完方法の選択
  
- [ ] **📈 データ分布の可視化**
  - ヒストグラム、箱ひげ図による分布確認
  - ターゲット変数のバランス確認
  
- [ ] **🎯 外れ値の検出・処理**
  - IQR法、Z-score法による検出
  - 外れ値の処理方針決定
  
- [ ] **🔗 特徴量間の相関分析**
  - 相関行列の作成
  - 多重共線性の確認

</details>

### ⚡ 2. 特徴量エンジニアリング

<details>
<summary><b>特徴量作成戦略</b></summary>

- [ ] **🏷 カテゴリ変数のエンコーディング**
  - Label Encoding
  - One-Hot Encoding
  - Target Encoding
  
- [ ] **📏 数値変数の標準化・正規化**
  - StandardScaler
  - MinMaxScaler
  - RobustScaler
  
- [ ] **🆕 新規特徴量の作成**
  - 特徴量の組み合わせ
  - 統計的特徴量
  - ドメイン知識に基づく特徴量
  
- [ ] **🎯 特徴量選択**
  - 重要度ベース選択
  - 統計的検定
  - 再帰的特徴量削除

</details>

### 🤖 3. モデリング

#### 📊 ベースライン戦略
```mermaid
graph LR
    A[データ] --> B[Logistic Regression]
    A --> C[Random Forest]
    B --> D[ベースライン比較]
    C --> D
```

<details>
<summary><b>モデル選択戦略</b></summary>

- [ ] **🔤 ベースラインモデル**
  - Logistic Regression
  - 解釈しやすく高速
  
- [ ] **🌳 ツリー系モデル**
  - Random Forest
  - XGBoost
  - LightGBM
  
- [ ] **🧠 ニューラルネットワーク**
  - 深層学習アプローチ
  - 複雑なパターン学習
  
- [ ] **🎯 アンサンブル手法**
  - Voting Classifier
  - Stacking
  - Blending

</details>

### 📈 4. モデル評価・改善

<details>
<summary><b>評価・最適化プロセス</b></summary>

- [ ] **✅ 交差検証による性能評価**
  - StratifiedKFold
  - 時系列考慮（必要に応じて）
  
- [ ] **🔧 ハイパーパラメータ調整**
  - Grid Search
  - Random Search
  - Bayesian Optimization
  
- [ ] **📊 特徴量重要度の分析**
  - SHAP値
  - Permutation Importance
  
- [ ] **🔍 モデル解釈性の向上**
  - LIME
  - 特徴量の寄与度分析

</details>

---

## 🚀 チーム向けセットアップガイド

### 🎯 初回セットアップ（Git未インストールの場合）

<details>
<summary><b>🔧 Step 1: 必要なツールのインストール</b></summary>

#### 1️⃣ Git のインストール

**Windows の場合:**
1. [Git for Windows](https://gitforwindows.org/) にアクセス
2. "Download" ボタンをクリック
3. ダウンロードした `.exe` ファイルを実行
4. インストール中は基本的に「Next」で進む
5. インストール完了後、コマンドプロンプトまたはPowerShellを開く
6. 確認: `git --version` と入力してバージョンが表示されればOK

**macOS の場合:**
```bash
# Homebrewでインストール（推奨）
brew install git

# または、公式インストーラーを使用
# https://git-scm.com/download/mac からダウンロード
```

**Linux (Ubuntu/Debian) の場合:**
```bash
sudo apt update
sudo apt install git
```

#### 2️⃣ Python 3.9+ のインストール確認

```bash
# バージョン確認
python --version
# または
python3 --version

# 3.9以上でない場合は公式サイトからインストール
# https://www.python.org/downloads/
```

#### 3️⃣ Kaggle API の設定

1. [Kaggle](https://www.kaggle.com) にログイン
2. アカウント設定 → API → "Create New API Token"
3. `kaggle.json` ファイルがダウンロードされる
4. ファイルを適切な場所に配置:
   - **Windows**: `C:\Users\{username}\.kaggle\kaggle.json`
   - **macOS/Linux**: `~/.kaggle/kaggle.json`

```bash
# Windows (PowerShell)
mkdir $env:USERPROFILE\.kaggle
move .\kaggle.json $env:USERPROFILE\.kaggle\

# macOS/Linux
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

</details>

<details>
<summary><b>📂 Step 2: リポジトリのクローンと初期設定</b></summary>

#### 1️⃣ Git の初期設定（初回のみ）

```bash
# ユーザー名とメールアドレスの設定
git config --global user.name "あなたの名前"
git config --global user.email "your.email@example.com"

# 設定確認
git config --global --list
```

#### 2️⃣ プロジェクトをクローン

```bash
# プロジェクトをクローン（HTTPS方式・推奨）
git clone https://github.com/YOUR_USERNAME/kaggle-playground-s5e7.git

# プロジェクトディレクトリに移動
cd kaggle-playground-s5e7

# リモートリポジトリの確認
git remote -v
```

#### 3️⃣ 作業ブランチの作成

```bash
# 最新の変更を取得
git pull origin main

# 自分専用の作業ブランチを作成
git checkout -b feature/your-name-setup

# 例:
# git checkout -b feature/tanaka-setup
# git checkout -b feature/yamada-analysis
```

</details>

<details>
<summary><b>🐍 Step 3: Python環境のセットアップ</b></summary>

#### 1️⃣ 仮想環境の作成とアクティベート

```bash
# 仮想環境を作成
python -m venv venv

# 仮想環境をアクティベート
# Windows (コマンドプロンプト)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate

# アクティベート確認（プロンプトに(venv)が表示される）
```

#### 2️⃣ 依存関係のインストール

```bash
# pip をアップデート
python -m pip install --upgrade pip

# プロジェクトの依存関係をインストール
pip install -r requirements.txt

# インストール確認
pip list
```

#### 3️⃣ Jupyter Lab のセットアップ（オプション）

```bash
# Jupyter Lab を起動
jupyter lab

# ブラウザで http://localhost:8888 が開けばOK
```

</details>

<details>
<summary><b>📊 Step 4: データの取得と配置</b></summary>

#### 1️⃣ Kaggle API の動作確認

```bash
# Kaggle API の動作テスト
kaggle competitions list

# エラーが出る場合は Step 1 の Kaggle API 設定を再確認
```

#### 2️⃣ コンペティションデータのダウンロード

```bash
# コンペティションデータをダウンロード
kaggle competitions download -c playground-series-s5e7

# Windows の場合
Expand-Archive playground-series-s5e7.zip -DestinationPath data\raw\

# macOS/Linux の場合
unzip playground-series-s5e7.zip -d data/raw/

# ファイル確認
ls data/raw/
# 以下のファイルがあることを確認:
# - train.csv
# - test.csv
# - sample_submission.csv
```

</details>

<details>
<summary><b>✅ Step 5: セットアップの確認</b></summary>

#### 1️⃣ 環境の動作確認

```bash
# Python とライブラリの確認
python -c "import pandas as pd; import numpy as np; import sklearn; print('✅ All libraries loaded successfully!')"

# Git の動作確認
git status

# Kaggle API の確認
kaggle competitions list -s playground-series-s5e7
```

#### 2️⃣ 初回コミット（確認用）

```bash
# セットアップ完了を記録
echo "Setup completed by [あなたの名前] on $(date)" >> setup_log.txt

# 変更をステージング
git add setup_log.txt

# コミット
git commit -m "setup: Complete initial setup by [あなたの名前]"

# リモートにプッシュ
git push origin feature/your-name-setup
```

#### 3️⃣ 動作テスト

```bash
# 簡単なデータ読み込みテスト
python -c "
import pandas as pd
train = pd.read_csv('data/raw/train.csv')
print(f'✅ Training data loaded: {train.shape}')
print(f'✅ Columns: {list(train.columns)}')
"
```

</details>

### 🚀 クイックスタート（既にセットアップ済みの場合）

```bash
# 1️⃣ リポジトリのクローン
git clone https://github.com/YOUR_USERNAME/kaggle-playground-s5e7.git
cd kaggle-playground-s5e7

# 2️⃣ 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3️⃣ 依存関係のインストール
pip install -r requirements.txt

# 4️⃣ データのダウンロード
kaggle competitions download -c playground-series-s5e7
unzip playground-series-s5e7.zip -d data/raw/
```

### 🆘 よくある問題と解決方法

<details>
<summary><b>❌ トラブルシューティング</b></summary>

#### Git関連の問題

**問題**: `git: command not found`
```bash
# 解決: Gitを再インストール
# Windowsの場合は https://gitforwindows.org/ から
# macOSの場合は brew install git
```

**問題**: `Permission denied (publickey)`
```bash
# 解決: HTTPS でクローンし直す
git clone https://github.com/USERNAME/REPO.git
```

#### Python関連の問題

**問題**: `python: command not found`
```bash
# 解決: python3 を試す
python3 --version
python3 -m venv venv
```

**問題**: `pip install` でエラー
```bash
# 解決: pip をアップデート
python -m pip install --upgrade pip
```

#### Kaggle API関連の問題

**問題**: `OSError: Could not find kaggle.json`
```bash
# 解決: kaggle.json の配置場所を確認
# Windows: C:\Users\{username}\.kaggle\kaggle.json
# macOS/Linux: ~/.kaggle/kaggle.json
```

**問題**: `403 Forbidden` エラー
```bash
# 解決: コンペティションのルールに同意
# ブラウザでコンペティションページにアクセスして「Join Competition」
```

</details>

---

## 📁 プロジェクト構成

<details>
<summary>🗂 <b>フォルダ構成を表示</b></summary>

```
📦 kaggle-playground-s5e7/
├── 📂 data/
│   ├── 📂 raw/                     # 🔒 元データ
│   ├── 📂 processed/               # ⚙️ 前処理済み
│   └── 📂 external/                # 🌐 外部データ
├── 📂 notebooks/
│   ├── 📓 01_eda.ipynb            # 🔍 探索的データ分析
│   ├── 📓 02_preprocessing.ipynb   # 🧹 データ前処理
│   ├── 📓 03_modeling.ipynb       # 🤖 モデル構築
│   └── 📓 04_ensemble.ipynb       # 🎯 アンサンブル
├── 📂 src/
│   ├── 📄 data_preprocessing.py    # 🛠 前処理関数
│   ├── 📄 feature_engineering.py  # ⚡ 特徴量エンジニアリング
│   ├── 📄 models.py               # 🧠 モデル定義
│   └── 📄 utils.py                # 🔧 ユーティリティ
├── 📂 scripts/
│   ├── 📄 train.py                # 🏃‍♂️ 学習スクリプト
│   ├── 📄 predict.py              # 🔮 予測スクリプト
│   └── 📄 create_submission.py    # 📤 提出ファイル作成
├── 📂 submissions/                 # 📋 提出ファイル
├── 📂 models/                     # 💾 学習済みモデル
└── 📂 experiments/                # 📊 実験記録
```

</details>

---

## 📈 結果

### 🏆 現在の成績
| 指標 | スコア | 順位 | 備考 |
|------|--------|------|------|
| **CV Score** | `TBD` | - | 5-Fold Cross Validation |
| **Public LB** | `TBD` | `TBD` | Public Leaderboard |
| **Private LB** | `TBD` | `TBD` | Private Leaderboard |

### 📊 実験履歴
| 実験ID | モデル | 特徴量 | CV Score | Public LB | 備考 |
|--------|--------|--------|----------|-----------|------|
| `exp_001` | Logistic Regression | ベースライン | - | - | 初期ベースライン |
| `exp_002` | Random Forest | 基本特徴量 | - | - | ツリー系ベースライン |
| `exp_003` | XGBoost | 特徴量エンジニアリング後 | - | - | 勾配ブースティング |

---

## 🔧 使用方法

### 🏃‍♂️ 学習の実行
```bash
# 基本的な学習
python scripts/train.py --model xgboost

# ハイパーパラメータ最適化付き
python scripts/train.py --model lightgbm --optimize

# 交差検証の設定
python scripts/train.py --model ensemble --cv 10
```

### 🔮 予測の実行
```bash
# 予測の実行
python scripts/predict.py --model best_model

# 提出ファイルの作成
python scripts/create_submission.py
```

---

## 💡 重要な洞察

### 🔍 データの特徴
- **データバランス**: 内向型・外向型の分布を確認中
- **重要特徴量**: 分析中
- **欠損パターン**: 調査中

### 🧠 モデルの洞察
- **最良モデル**: 検証中
- **アンサンブル効果**: 検証中
- **解釈性**: SHAP値による分析実施予定

---

## 📚 参考資料

| リソース | リンク | 説明 |
|----------|------|------|
| 🏆 **コンペページ** | [Kaggle](https://www.kaggle.com/competitions/playground-series-s5e7) | 公式コンペティションページ |
| 📊 **Playground Series** | [Overview](https://www.kaggle.com/competitions/playground-series) | シリーズ概要 |
| 🧠 **MBTI Dataset** | [Kaggle](https://www.kaggle.com/datasets/datasnaek/mbti-type) | 関連する性格データ |
| 📈 **Big Five Test** | [Kaggle](https://www.kaggle.com/datasets/tunguz/big-five-personality-test) | 性格研究データ |

---

## 🤝 コントリビューション

プロジェクトへの貢献を歓迎します！

### 📝 開発ガイドライン
1. **ブランチ命名**: `feature/説明` または `fix/説明`
2. **コミットメッセージ**: [Conventional Commits](https://conventionalcommits.org/) に従う
3. **コードスタイル**: PEP 8 に準拠
4. **ドキュメント**: 重要な変更時はREADME更新

### 🔄 ワークフロー
```bash
# 1️⃣ 最新の変更を取得
git pull origin main

# 2️⃣ フィーチャーブランチを作成
git checkout -b feature/your-feature

# 3️⃣ 変更をコミット
git commit -m "feat: 新機能の説明"

# 4️⃣ プッシュしてPR作成
git push origin feature/your-feature
```

---

<div align="center">

## 🏆 Happy Kaggling! 🏆

**個性を予測する機械学習の旅へ出発！**

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)](https://www.kaggle.com/competitions/playground-series-s5e7)

**Created by**: [Your Team Name]  
**Date**: 2025年7月  
**Competition**: Playground Series S5E7

</div>