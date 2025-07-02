import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

data_path = Path('../data/raw')
output_path = Path('../analysis')
output_path.mkdir(exist_ok=True)

print("=== Kaggle Playground Series S5E7: 内向型・外向型予測 EDA ===\n")

# データ読み込み
train_df = pd.read_csv(data_path / 'train.csv')
test_df = pd.read_csv(data_path / 'test.csv')
sample_sub = pd.read_csv(data_path / 'sample_submission.csv')

print("1. データセット基本情報")
print("=" * 50)
print(f"訓練データ形状: {train_df.shape}")
print(f"テストデータ形状: {test_df.shape}")
print(f"サンプル提出ファイル形状: {sample_sub.shape}")

print(f"\n訓練データ列: {list(train_df.columns)}")
print(f"テストデータ列: {list(test_df.columns)}")

# データ型確認
print(f"\n訓練データ型:\n{train_df.dtypes}")

# 欠損値確認
print(f"\n2. 欠損値分析")
print("=" * 50)
print("訓練データ欠損値:")
train_missing = train_df.isnull().sum()
train_missing_pct = (train_missing / len(train_df)) * 100
missing_train = pd.DataFrame({
    '欠損数': train_missing,
    '欠損率(%)': train_missing_pct
}).round(2)
print(missing_train[missing_train['欠損数'] > 0])

print("\nテストデータ欠損値:")
test_missing = test_df.isnull().sum()
test_missing_pct = (test_missing / len(test_df)) * 100
missing_test = pd.DataFrame({
    '欠損数': test_missing,
    '欠損率(%)': test_missing_pct
}).round(2)
print(missing_test[missing_test['欠損数'] > 0])

# ターゲット変数分析
print(f"\n3. ターゲット変数分析")
print("=" * 50)
target_dist = train_df['Personality'].value_counts()
target_pct = train_df['Personality'].value_counts(normalize=True) * 100

print("ターゲット分布:")
for personality, count in target_dist.items():
    pct = target_pct[personality]
    print(f"  {personality}: {count} ({pct:.1f}%)")

# 特徴量基本統計
print(f"\n4. 特徴量基本統計")
print("=" * 50)
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'id']

print("数値特徴量基本統計:")
print(train_df[numeric_cols].describe().round(2))

# カテゴリカル特徴量分析
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col != 'Personality']

print(f"\n5. カテゴリカル特徴量分析")
print("=" * 50)
for col in categorical_cols:
    print(f"\n{col}の分布:")
    value_counts = train_df[col].value_counts(dropna=False)
    value_pct = train_df[col].value_counts(normalize=True, dropna=False) * 100
    for value, count in value_counts.items():
        pct = value_pct[value]
        print(f"  {value}: {count} ({pct:.1f}%)")

print(f"\n6. データ品質チェック")
print("=" * 50)

# 重複行チェック
train_duplicates = train_df.duplicated().sum()
test_duplicates = test_df.duplicated().sum()
print(f"訓練データ重複行数: {train_duplicates}")
print(f"テストデータ重複行数: {test_duplicates}")

# ID重複チェック
train_id_duplicates = train_df['id'].duplicated().sum()
test_id_duplicates = test_df['id'].duplicated().sum()
print(f"訓練データID重複数: {train_id_duplicates}")
print(f"テストデータID重複数: {test_id_duplicates}")

# 数値範囲チェック
print(f"\n数値特徴量範囲チェック:")
for col in numeric_cols:
    min_val = train_df[col].min()
    max_val = train_df[col].max()
    print(f"  {col}: [{min_val}, {max_val}]")

print("\n=== EDA完了 ===")