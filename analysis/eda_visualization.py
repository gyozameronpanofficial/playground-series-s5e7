import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

data_path = Path('../data/raw')
output_path = Path('../analysis')
output_path.mkdir(exist_ok=True)

# データ読み込み
train_df = pd.read_csv(data_path / 'train.csv')

print("=== 特徴量分布と相関分析 ===\n")

# 1. ターゲット変数の可視化
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# ターゲット分布（カウント）
target_counts = train_df['Personality'].value_counts()
ax[0].bar(target_counts.index, target_counts.values, color=['skyblue', 'lightcoral'])
ax[0].set_title('ターゲット変数分布（カウント）', fontsize=14)
ax[0].set_ylabel('件数')
for i, v in enumerate(target_counts.values):
    ax[0].text(i, v + 100, str(v), ha='center', va='bottom', fontweight='bold')

# ターゲット分布（割合）
target_pct = train_df['Personality'].value_counts(normalize=True) * 100
colors = ['skyblue', 'lightcoral']
wedges, texts, autotexts = ax[1].pie(target_pct.values, labels=target_pct.index, 
                                    autopct='%1.1f%%', startangle=90, colors=colors)
ax[1].set_title('ターゲット変数分布（割合）', fontsize=14)

plt.tight_layout()
plt.savefig(output_path / 'target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 数値特徴量分布の可視化
numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']

fig, axes = plt.subplots(3, 2, figsize=(15, 18))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    # ヒストグラム（全体）
    axes[i].hist(train_df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[i].set_title(f'{col}の分布', fontsize=12)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('頻度')
    axes[i].grid(True, alpha=0.3)

# 6番目のサブプロットを削除
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig(output_path / 'numeric_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. カテゴリカル特徴量の可視化
categorical_cols = ['Stage_fear', 'Drained_after_socializing']

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for i, col in enumerate(categorical_cols):
    value_counts = train_df[col].value_counts(dropna=False)
    bars = axes[i].bar(range(len(value_counts)), value_counts.values, 
                      color=['lightgreen', 'lightcoral', 'lightgray'])
    axes[i].set_title(f'{col}の分布', fontsize=12)
    axes[i].set_xticks(range(len(value_counts)))
    axes[i].set_xticklabels(value_counts.index, rotation=45)
    axes[i].set_ylabel('件数')
    
    # 値をバーの上に表示
    for j, v in enumerate(value_counts.values):
        axes[i].text(j, v + 50, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(output_path / 'categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. ターゲット別特徴量分布
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    # ターゲット別ヒストグラム
    extrovert_data = train_df[train_df['Personality'] == 'Extrovert'][col].dropna()
    introvert_data = train_df[train_df['Personality'] == 'Introvert'][col].dropna()
    
    axes[i].hist(extrovert_data, bins=20, alpha=0.7, label='Extrovert', color='skyblue')
    axes[i].hist(introvert_data, bins=20, alpha=0.7, label='Introvert', color='lightcoral')
    axes[i].set_title(f'{col}のターゲット別分布', fontsize=12)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('頻度')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

fig.delaxes(axes[5])
plt.tight_layout()
plt.savefig(output_path / 'target_by_features.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 特徴量相関分析
numeric_df = train_df[numeric_cols + ['id']].copy()
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('特徴量間相関マトリックス', fontsize=14)
plt.tight_layout()
plt.savefig(output_path / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 欠損値パターン分析
missing_data = train_df.isnull()
missing_cols = missing_data.columns[missing_data.sum() > 0].tolist()

if missing_cols:
    plt.figure(figsize=(12, 8))
    sns.heatmap(train_df[missing_cols].isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('欠損値パターン', fontsize=14)
    plt.xlabel('特徴量')
    plt.tight_layout()
    plt.savefig(output_path / 'missing_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()

# 7. 外れ値検出（箱ひげ図）
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    box_plot = axes[i].boxplot(train_df[col].dropna(), patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    axes[i].set_title(f'{col}の箱ひげ図', fontsize=12)
    axes[i].set_ylabel(col)
    axes[i].grid(True, alpha=0.3)

fig.delaxes(axes[5])
plt.tight_layout()
plt.savefig(output_path / 'outlier_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

print("すべての可視化が完了しました。")
print(f"画像ファイルは {output_path} に保存されました。")