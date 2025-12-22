"""
Driver script for testing qqq_project locally.
Mimics the Colab workflow for data loading, preprocessing, and feature building.
"""
from data_completeness import check_data_completeness
from data_preprocessing import preprocessor
from features import build_time_features, build_volume_oi_features, build_vol_features

import sys
import importlib
from pathlib import Path
import pandas as pd

import data_preprocessing as dp
import data_completeness as dc
import features as tf
import vol_helpers as vh

# Reload modules to ensure latest code is used
tf = importlib.reload(tf)
dp = importlib.reload(dp)
dc = importlib.reload(dc)
vh = importlib.reload(vh)


# ============================================================================
# Configuration
# ============================================================================

# Current script location (inside qqq_pipeline)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Data path - using CSV files in the same directory as this script
DATA_DIR = SCRIPT_DIR
QQQ_PATH = DATA_DIR / "options_eod_QQQ.csv"


# ============================================================================
# Main Driver Function
# ============================================================================

def main():
    """Main driver function to load, preprocess, and feature engineer data."""
    
    print("=" * 80)
    print("QQQ Project - Local Driver")
    print("=" * 80)
    
    # ========================================================================
    # 1. Load Raw Data
    # ========================================================================
    print("\n[1/5] Loading raw options data...")
    if not QQQ_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {QQQ_PATH}")
    
    QQQ = pd.read_csv(QQQ_PATH)
    print(f"    ✓ Loaded {len(QQQ)} rows from {QQQ_PATH.name}")
    print(f"    Columns: {list(QQQ.columns)}")
    print(f"    Shape: {QQQ.shape}")
    
    # ========================================================================
    # 2. Preprocess Data
    # ========================================================================
    print("\n[2/5] Preprocessing data...")
    QQQ = preprocessor(QQQ)
    print("    ✓ Preprocessing complete")
    print(f"    Shape after preprocessing: {QQQ.shape}")
    
    # ========================================================================
    # 3. Check Data Completeness
    # ========================================================================
    print("\n[3/5] Checking data completeness...")
    QQQ = check_data_completeness(QQQ)
    print("    ✓ Data completeness check complete")
    
    # ========================================================================
    # 4. Build Time-Based Features
    # ========================================================================
    print("\n[4/5] Building time-based features...")
    daily = build_time_features(QQQ)
    print("    ✓ Time features complete")
    print(f"    Daily data shape: {daily.shape}")
    print(f"    Daily data columns: {list(daily.columns)}")
    
    # ========================================================================
    # 5. Build Volume/OI and IV Features
    # ========================================================================
    print("\n[5/5] Building volume/OI and IV features...")
    QQQ, daily = build_volume_oi_features(QQQ, daily)
    print("    ✓ Volume/OI features complete")
    print(f"    QQQ shape: {QQQ.shape}")
    print(f"    Daily shape: {daily.shape}")

    daily = build_vol_features(QQQ, daily)
    print("    ✓ IV features complete")
    print(f"    QQQ shape: {QQQ.shape}")
    print(f"    Daily shape: {daily.shape}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print("\nFinal Data Shapes:")
    print(f"  QQQ (options):  {QQQ.shape}")
    print(f"  daily (targets): {daily.shape}")
    print("\nDaily Data Preview:")
    print(daily.head())
    print("\nDaily Data Info:")
    print(daily.info())
    
    return QQQ, daily

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        # Run the main pipeline
        QQQ, daily = main()
        
        # Optional: run additional tests
        print("\n" + "=" * 80)
        print("Available for further testing:")
        print("  - inspect_data(QQQ, daily)")
        print("  - test_iv_calculation(QQQ, daily, target_dte=30)")
        print("=" * 80)
        
    except Exception as e:
        print("\n❌ Error during pipeline execution:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
