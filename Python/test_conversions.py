"""
Test script for converted Python modules.

This script verifies that the converted modules work correctly.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from Core.post_processor import PostProcessor
        print("  ✓ PostProcessor imported")
    except Exception as e:
        print(f"  ✗ PostProcessor failed: {e}")
        return False
    
    try:
        from Utils.visualizer import Visualizer
        print("  ✓ Visualizer imported")
    except Exception as e:
        print(f"  ✗ Visualizer failed: {e}")
        return False
    
    try:
        from Utils.kmz_mask_generator import KMZMaskGenerator
        print("  ✓ KMZMaskGenerator imported")
    except Exception as e:
        print(f"  ✗ KMZMaskGenerator failed: {e}")
        return False
    
    return True


def test_visualizer():
    """Test Visualizer with dummy data."""
    print("\nTesting Visualizer...")
    from Utils.visualizer import Visualizer
    import tempfile
    
    try:
        # Create dummy data
        h, w = 100, 100
        f_map = np.random.rand(h, w)
        delta_red = np.random.randn(h, w) * 10
        moran = np.random.rand(h, w)
        mask = np.random.rand(h, w) > 0.5
        depth = np.random.rand(h, w) * 2000
        grad_p = np.random.rand(h, w) * 40
        freq = np.random.rand(h, w) * 90 + 10
        rgb = np.random.rand(h, w, 3)
        lon_grid = np.linspace(100, 101, w)
        lat_grid = np.linspace(30, 31, h)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            Visualizer.run_resonance(
                f_map, delta_red, moran, mask, depth, grad_p, freq, rgb,
                tmpdir, np.tile(lon_grid, (h, 1)), np.tile(lat_grid[:, None], (1, w))
            )
            
            # Check if file was created
            if os.path.exists(os.path.join(tmpdir, '01_共振参数综合图.png')):
                print("  ✓ run_resonance works")
            else:
                print("  ✗ run_resonance failed to create output")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Visualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kmz_generator():
    """Test KMZMaskGenerator initialization."""
    print("\nTesting KMZMaskGenerator...")
    from Utils.kmz_mask_generator import KMZMaskGenerator
    
    try:
        # Just test initialization (don't run without real files)
        gen = KMZMaskGenerator(
            'dummy.kml',
            'dummy.tif',
            ['test'],
            3
        )
        print(f"  ✓ KMZMaskGenerator initialization works")
        print(f"    - Keywords: {gen.target_keywords}")
        print(f"    - Point radius: {gen.point_radius_pixel}")
        return True
    except Exception as e:
        print(f"  ✗ KMZMaskGenerator test failed: {e}")
        return False


def test_post_processor():
    """Test PostProcessor safe_get method."""
    print("\nTesting PostProcessor...")
    from Core.post_processor import PostProcessor
    
    try:
        # Test safe_get with mock engine
        class MockEngine:
            def __init__(self):
                self.results = {}
            
            def get_result(self, name):
                return self.results[name]
        
        engine = MockEngine()
        in_roi = np.ones((10, 10), dtype=bool)
        
        # Test missing result
        result = PostProcessor.safe_get(engine, 'NonExistent', in_roi)
        assert 'mask' in result
        assert 'debug' in result
        print("  ✓ safe_get handles missing results")
        
        # Test existing result
        engine.results['Test'] = {
            'mask': np.zeros((10, 10)),
            'debug': {'F_map': np.zeros((10, 10))}
        }
        result = PostProcessor.safe_get(engine, 'Test', in_roi)
        assert result['mask'].shape == (10, 10)
        print("  ✓ safe_get handles existing results")
        
        return True
    except Exception as e:
        print(f"  ✗ PostProcessor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Converted Python Modules")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("PostProcessor", test_post_processor()))
    results.append(("Visualizer", test_visualizer()))
    results.append(("KMZMaskGenerator", test_kmz_generator()))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print("=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
