#!/usr/bin/env python3
"""
Quick test script to verify dataset loading and sampling functionality.
"""

import logging
from dataset_sampler import DatasetSampler

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_dataset_loading():
    """Test basic dataset loading functionality."""
    print("🧪 Testing GRAID Human Evaluation Dataset Loading")
    print("=" * 60)
    
    # Initialize sampler
    print("1. Initializing DatasetSampler...")
    sampler = DatasetSampler("kd7/graid-bdd100k-ground-truth")
    
    # Load dataset
    print("2. Loading dataset from HuggingFace Hub...")
    try:
        sampler.load_dataset()
        print("✅ Dataset loaded successfully!")
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
    print(f"   Features: {info['features']}")
    
    # Show question types
    question_types = sampler.get_question_types()
    print(f"\n4. Discovered question types ({len(question_types)}):")
    for i, qtype in enumerate(question_types, 1):
        print(f"   {i:2d}. {qtype}")
    
    # Test sampling
    print("\n5. Testing question sampling...")
    try:
        sampled = sampler.sample_questions("test_user", n_per_type=2)
        print(f"✅ Sampled {len(sampled)} questions successfully!")
        
        # Show sample distribution
        type_counts = {}
        for _, sample in sampled:
            qtype = sample.get('question_type', 'unknown')
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
        
        print("   Sample distribution:")
        for qtype, count in sorted(type_counts.items()):
            print(f"     {qtype}: {count} questions")
            
        # Show first sample
        if sampled:
            idx, sample = sampled[0]
            print(f"\n   First sample (index {idx}):")
            print(f"     Question Type: {sample.get('question_type', 'N/A')}")
            print(f"     Question: {sample.get('question', 'N/A')[:100]}...")
            print(f"     Answer: {sample.get('answer', 'N/A')}")
            print(f"     Source ID: {sample.get('source_id', 'N/A')}")
            print(f"     Annotations: {len(sample.get('annotations', []))} objects")
            
    except Exception as e:
        print(f"❌ Sampling failed: {e}")
        return False
    
    print("\n🎉 All tests passed! The dataset is ready for human evaluation.")
    return True

if __name__ == "__main__":
    success = test_dataset_loading()
    exit(0 if success else 1)
