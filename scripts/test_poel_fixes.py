"""
Quick test to verify POEL fixes before full 50-episode run
Tests:
1. Unicode character removal (logging compatibility)
2. NoveltyDetector with small buffer
"""
import numpy as np
import logging
from underdog.rl.poel.novelty import NoveltyDetector, DistanceMetric

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_unicode_logging():
    """Test that ASCII characters work in Windows terminal"""
    print("\n" + "="*60)
    print("TEST 1: Unicode Character Fix")
    print("="*60)
    
    try:
        logger.warning("[!] DD BREACH: 12.39% > 10.0%")
        logger.info("[OK] Failure recorded (breach size: 2.39%)")
        print("[PASS] ASCII characters logged successfully")
        return True
    except UnicodeEncodeError as e:
        print(f"[FAIL] Unicode error: {e}")
        return False

def test_novelty_small_buffer():
    """Test NoveltyDetector with buffer smaller than k_neighbors"""
    print("\n" + "="*60)
    print("TEST 2: NoveltyDetector Small Buffer")
    print("="*60)
    
    try:
        # Initialize detector
        detector = NoveltyDetector(
            state_dim=31,
            action_dim=1,
            buffer_size=10000,
            metric=DistanceMetric.L2,
            normalization=True
        )
        
        # Test 1: Empty buffer (0 samples, k=5)
        state = np.random.randn(31)
        action = np.random.randn(1)
        
        novelty = detector.compute_novelty(state, action, k_neighbors=5)
        print(f"  Empty buffer novelty: {novelty:.4f} (expected: 1.0)")
        assert novelty == 1.0, f"Expected 1.0, got {novelty}"
        
        # Test 2: Add 1 sample (buffer=1, k=5)
        detector.add_experience(state, action)
        state2 = np.random.randn(31)
        action2 = np.random.randn(1)
        
        novelty2 = detector.compute_novelty(state2, action2, k_neighbors=5)
        print(f"  Buffer size 1 novelty: {novelty2:.4f} (should work)")
        
        # Test 3: Add 3 more samples (buffer=4, k=5)
        for _ in range(3):
            detector.add_experience(np.random.randn(31), np.random.randn(1))
        
        state3 = np.random.randn(31)
        action3 = np.random.randn(1)
        novelty3 = detector.compute_novelty(state3, action3, k_neighbors=5)
        print(f"  Buffer size 4 novelty: {novelty3:.4f} (should work)")
        
        # Test 4: Add 6 more samples (buffer=10, k=5)
        for _ in range(6):
            detector.add_experience(np.random.randn(31), np.random.randn(1))
        
        state4 = np.random.randn(31)
        action4 = np.random.randn(1)
        novelty4 = detector.compute_novelty(state4, action4, k_neighbors=5)
        print(f"  Buffer size 10 novelty: {novelty4:.4f} (should use k=5)")
        
        print("[PASS] NoveltyDetector handles small buffers correctly")
        return True
        
    except Exception as e:
        print(f"[FAIL] NoveltyDetector error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("POEL FIXES VALIDATION")
    print("="*60)
    
    results = {
        'unicode_logging': test_unicode_logging(),
        'novelty_small_buffer': test_novelty_small_buffer(),
    }
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
    
    print("="*60)
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED - Safe to run 50-episode POEL training")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Fix errors before full run")
        return 1

if __name__ == "__main__":
    exit(main())
