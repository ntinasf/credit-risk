"""
Data Splitting Script for German Credit Dataset

This script reads the german_credit.csv and creates train_data.csv and test_data.csv
using a stratified hash-based split. Hash-based splitting ensures reproducibility
without relying on random seeds - the same row will always end up in the same split.

Usage:
    python scripts/split_data.py --test-size 0.15
    python scripts/split_data.py --test-size 0.2 --hash-column checking_account_status
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def hash_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    hash_column: str = None,
    target_column: str = "class",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified hash-based split on the dataframe.
    
    Hash-based splitting ensures:
    - Reproducibility without random seeds
    - Same row always ends up in same split
    - Deterministic behavior across runs
    
    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe to split
    test_size : float
        Proportion of data to use for test set (0.0 to 1.0)
    hash_column : str, optional
        Column to use for hashing. If None, uses row index.
    target_column : str
        Column to stratify on (default: "class")
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    # Create a hash value for each row
    def get_hash_bucket(row_id: str, n_buckets: int = 1000) -> int:
        """Convert row identifier to a hash bucket (0 to n_buckets-1)."""
        hash_digest = hashlib.md5(str(row_id).encode()).hexdigest()
        return int(hash_digest, 16) % n_buckets
    
    # Determine what to hash
    if hash_column and hash_column in df.columns:
        # Combine hash_column with index for uniqueness
        hash_values = df[hash_column].astype(str) + "_" + df.index.astype(str)
    else:
        # Use index as hash source - convert to Series for .apply()
        hash_values = pd.Series(df.index.astype(str), index=df.index)
    
    # Calculate hash buckets
    n_buckets = 1000
    test_threshold = int(test_size * n_buckets)
    
    hash_buckets = hash_values.apply(lambda x: get_hash_bucket(x, n_buckets))
    
    # Perform stratified split
    train_indices = []
    test_indices = []
    
    for class_value in df[target_column].unique():
        class_mask = df[target_column] == class_value
        class_indices = df[class_mask].index
        class_buckets = hash_buckets[class_mask]
        
        # Split based on hash bucket threshold
        class_test_mask = class_buckets < test_threshold
        class_train_mask = ~class_test_mask
        
        test_indices.extend(class_indices[class_test_mask].tolist())
        train_indices.extend(class_indices[class_train_mask].tolist())
    
    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    return train_df, test_df


def validate_split(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                   target_column: str = "class") -> dict:
    """
    Validate the split and return statistics.
    
    Returns
    -------
    dict
        Split statistics including sizes and class distributions
    """
    total = len(train_df) + len(test_df)
    
    train_dist = train_df[target_column].value_counts(normalize=True)
    test_dist = test_df[target_column].value_counts(normalize=True)
    
    stats = {
        "total_samples": total,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_proportion": len(train_df) / total,
        "test_proportion": len(test_df) / total,
        "train_class_distribution": train_dist.to_dict(),
        "test_class_distribution": test_dist.to_dict(),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Split German Credit data into train and test sets using stratified hash split."
    )
    parser.add_argument(
        "--test-size", "-t",
        type=float,
        default=0.15,
        help="Proportion of data for test set (default: 0.15)"
    )
    parser.add_argument(
        "--hash-column", "-c",
        type=str,
        default=None,
        help="Column to use for hashing (default: uses row index)"
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="class",
        help="Target column for stratification (default: 'class')"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Path to input CSV (default: data/processed/german_credit.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input)"
    )
    
    args = parser.parse_args()
    
    # Determine paths
    if args.input_path:
        input_path = Path(args.input_path)
    else:
        # Default: look for data relative to script location
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        input_path = project_root / "data" / "processed" / "german_credit.csv"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent
    
    # Read data
    print(f"📂 Reading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Total samples: {len(df)}")
    print(f"   Features: {len(df.columns) - 1}")
    print(f"   Target column: '{args.target_column}'")
    
    # Show original class distribution
    original_dist = df[args.target_column].value_counts(normalize=True)
    print("\n📊 Original class distribution:")
    for cls, prop in original_dist.items():
        print(f"   Class {cls}: {prop:.2%}")
    
    # Perform split
    print(f"\n🔀 Performing stratified hash split (test_size={args.test_size:.0%})...")
    train_df, test_df = hash_split(
        df, 
        test_size=args.test_size,
        hash_column=args.hash_column,
        target_column=args.target_column
    )
    
    # Validate and print stats
    stats = validate_split(train_df, test_df, args.target_column)
    
    print("\n✅ Split complete:")
    print(f"   Train set: {stats['train_samples']} samples ({stats['train_proportion']:.1%})")
    print(f"   Test set:  {stats['test_samples']} samples ({stats['test_proportion']:.1%})")
    
    print("\n📊 Train class distribution:")
    for cls, prop in stats['train_class_distribution'].items():
        print(f"   Class {cls}: {prop:.2%}")
    
    print("\n📊 Test class distribution:")
    for cls, prop in stats['test_class_distribution'].items():
        print(f"   Class {cls}: {prop:.2%}")
    
    # Save files
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_data.csv"
    test_path = output_dir / "test_data.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Also write a copy for the Streamlit app (used as sample data)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    app_data_dir = project_root / "app" / "data"
    app_data_dir.mkdir(parents=True, exist_ok=True)
    sample_path = app_data_dir / "sample_data.csv"
    test_df.to_csv(sample_path, index=False)
    
    print("\n💾 Files saved:")
    print(f"   {train_path}")
    print(f"   {test_path}")
    print(f"   {sample_path}  (Streamlit app sample data)")


if __name__ == "__main__":
    main()
