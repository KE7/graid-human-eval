#!/usr/bin/env python3
"""
Test script to compare non-streaming vs streaming dataset loading performance.
"""

import logging
import time
from dataset_sampler import DatasetSampler

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_non_streaming():
    """Test non-streaming dataset loading (downloads all files)."""
    print("🧪 Testing NON-STREAMING Dataset Loading (downloads all 130+ files)")
    print("=" * 70)
    
    # Initialize sampler
    print("1. Initializing DatasetSampler...")
    sampler = DatasetSampler("kd7/graid-bdd100k-ground-truth")
    
    # Load dataset with non-streaming mode
    print("2. Loading dataset with NON-STREAMING mode (this will take a while)...")
    start_time = time.time()
    
    try:
        sampler.load_dataset(streaming=False)  # This downloads ALL files
        load_time = time.time() - start_time
        print(f"✅ Dataset loaded successfully in {load_time:.1f} seconds!")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False
    
    # Get dataset info
    print("3. Getting dataset information...")
    info = sampler.get_dataset_info()
    print(f"   Dataset: {info['dataset_name']}")
    print(f"   Split: {info['split_used']}")
    print(f"   Total samples: {info['total_samples']}")
    print(f"   Question types: {len(info['question_types'])}")
    
    return True

if __name__ == "__main__":
    print("⚠️  WARNING: This will download 130+ parquet files (~20GB)")
    print("⚠️  This is for comparison purposes only - normally use streaming=True")
    print()
    
    response = input("Continue with full download? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled. Use streaming mode instead!")
        exit(0)
    
    success = test_non_streaming()
    exit(0 if success else 1)
