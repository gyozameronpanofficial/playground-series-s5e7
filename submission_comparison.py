import pandas as pd

# Load submission files
baseline = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/submissions/baseline_reproduction.csv')
final = pd.read_csv('/Users/osawa/kaggle/playground-series-s5e7/submissions/final_submission.csv')

print("=== Submission Comparison Analysis ===")
print(f"Baseline shape: {baseline.shape}")
print(f"Final shape: {final.shape}")

# Check if they're identical
differences = baseline.compare(final)
print(f"\nNumber of differences: {len(differences)}")

if len(differences) > 0:
    print("\nDifferences found:")
    print(differences)
    
    # Count prediction differences
    baseline_counts = baseline['Personality'].value_counts()
    final_counts = final['Personality'].value_counts()
    
    print(f"\nBaseline predictions:")
    print(baseline_counts)
    print(f"\nFinal predictions:")
    print(final_counts)
    
    print(f"\nPrediction difference:")
    for label in ['Extrovert', 'Introvert']:
        diff = final_counts.get(label, 0) - baseline_counts.get(label, 0)
        print(f"  {label}: {diff:+d}")
        
    # Calculate the exact ID where difference occurs
    diff_mask = baseline['Personality'] != final['Personality']
    diff_ids = baseline.loc[diff_mask, 'id'].tolist()
    print(f"\nDifferent predictions at IDs: {diff_ids}")
    
    # Show the specific difference
    for idx in diff_ids:
        baseline_pred = baseline.loc[baseline['id'] == idx, 'Personality'].iloc[0]
        final_pred = final.loc[final['id'] == idx, 'Personality'].iloc[0]
        print(f"  ID {idx}: Baseline='{baseline_pred}' -> Final='{final_pred}'")
        
else:
    print("Submissions are identical!")