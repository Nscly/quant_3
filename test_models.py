#!/usr/bin/env python3
"""
Test script to verify all models work correctly
"""
import os
import sys

def test_data_preprocessing():
    """Test data preprocessing"""
    print("=" * 60)
    print("Testing Data Preprocessing...")
    print("=" * 60)
    
    required_files = [
        'stock_data_processed.pickle',
        'inner_edge.npy',
        'inner10_edge.npy',
        'inner20_edge.npy',
        'outer_edge.npy',
        'category_mapping.pickle'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        print("Please run: python clean_data.py")
        return False
    else:
        print("✓ All preprocessed data files exist")
        return True


def test_model(model_name):
    """Test a specific model"""
    print(f"\nTesting model: {model_name}")
    print("-" * 60)
    
    import subprocess
    cmd = [
        'python3', 'train.py',
        '--model', model_name,
        '--epochs', '1',
        '--device', 'cpu'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            # Check if training completed
            if 'Training completed!' in result.stdout:
                print(f"✓ Model {model_name} completed successfully")
                # Extract MAE from output
                for line in result.stdout.split('\n'):
                    if '[BEST MRR] MAE:' in line:
                        print(f"  Result: {line.strip()}")
                return True
            else:
                print(f"❌ Model {model_name} did not complete properly")
                return False
        else:
            print(f"❌ Model {model_name} failed with error:")
            print(result.stderr[:500])
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ Model {model_name} timed out")
        return False
    except Exception as e:
        print(f"❌ Model {model_name} error: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("FinGAT Model Testing Suite")
    print("="*60 + "\n")
    
    # Test data preprocessing
    if not test_data_preprocessing():
        print("\n⚠️  Please run 'python clean_data.py' first")
        sys.exit(1)
    
    # Test all models
    models = ['CG', 'CAT', 'CPool']
    results = {}
    
    for model in models:
        results[model] = test_model(model)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for model, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{model:10s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! The code is ready for training.")
        print("\nTo train a model, run:")
        print("  python train.py --model CAT --epochs 20 --device cpu")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
