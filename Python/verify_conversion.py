#!/usr/bin/env python3
"""
Syntax and import verification script for the Python conversion.
Checks that all Python modules can be imported successfully.
"""

import sys
import os
from pathlib import Path

# Add Python directory to path
python_dir = Path(__file__).parent
sys.path.insert(0, str(python_dir))


def test_syntax():
    """Test that all Python files have valid syntax."""
    print("=" * 60)
    print("Testing Python Syntax")
    print("=" * 60)
    
    python_files = []
    for root, dirs, files in os.walk(python_dir):
        # Skip __pycache__ and test files
        if '__pycache__' in root or 'test' in root:
            continue
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                python_files.append(os.path.join(root, file))
    
    errors = []
    for filepath in python_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                compile(f.read(), filepath, 'exec')
            print(f"✅ {os.path.relpath(filepath, python_dir)}")
        except SyntaxError as e:
            errors.append((filepath, str(e)))
            print(f"❌ {os.path.relpath(filepath, python_dir)}: {e}")
    
    print()
    if errors:
        print(f"❌ Found {len(errors)} syntax errors")
        return False
    else:
        print(f"✅ All {len(python_files)} Python files have valid syntax")
        return True


def test_imports():
    """Test that core modules can be imported."""
    print("\n" + "=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    
    modules_to_test = [
        ('Core.geo_data_context', 'GeoDataContext'),
        ('Core.fusion_engine', 'FusionEngine'),
        ('Core.post_processor', 'PostProcessor'),
        ('Detectors.anomaly_detector', 'AnomalyDetector'),
        ('Detectors.red_edge_detector', 'RedEdgeDetector'),
        ('Detectors.intrinsic_detector', 'IntrinsicDetector'),
        ('Detectors.slow_vars_detector', 'SlowVarsDetector'),
        ('Detectors.known_anomaly_detector', 'KnownAnomalyDetector'),
        ('Utils.geo_utils', 'GeoUtils'),
        ('Utils.visualizer', 'Visualizer'),
        ('Utils.kmz_mask_generator', 'KMZMaskGenerator'),
    ]
    
    errors = []
    success_count = 0
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
            success_count += 1
        except Exception as e:
            errors.append((module_name, class_name, str(e)))
            print(f"❌ {module_name}.{class_name}: {e}")
    
    print()
    if errors:
        print(f"❌ Failed to import {len(errors)} modules")
        return False
    else:
        print(f"✅ Successfully imported all {success_count} modules")
        return True


def test_package_structure():
    """Test that package structure is correct."""
    print("\n" + "=" * 60)
    print("Testing Package Structure")
    print("=" * 60)
    
    required_files = [
        '__init__.py',
        'requirements.txt',
        'README.md',
        'main.py',
        'Core/__init__.py',
        'Core/geo_data_context.py',
        'Core/fusion_engine.py',
        'Core/post_processor.py',
        'Detectors/__init__.py',
        'Detectors/anomaly_detector.py',
        'Detectors/red_edge_detector.py',
        'Detectors/intrinsic_detector.py',
        'Detectors/slow_vars_detector.py',
        'Detectors/known_anomaly_detector.py',
        'Utils/__init__.py',
        'Utils/geo_utils.py',
        'Utils/visualizer.py',
        'Utils/kmz_mask_generator.py',
    ]
    
    missing = []
    for file in required_files:
        filepath = python_dir / file
        if filepath.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing.append(file)
    
    print()
    if missing:
        print(f"❌ Missing {len(missing)} required files")
        return False
    else:
        print(f"✅ All {len(required_files)} required files present")
        return True


def main():
    """Run all tests."""
    print("\n🔍 Python Conversion Verification\n")
    
    results = []
    
    # Test 1: Package Structure
    results.append(('Package Structure', test_package_structure()))
    
    # Test 2: Syntax
    results.append(('Python Syntax', test_syntax()))
    
    # Test 3: Imports
    results.append(('Module Imports', test_imports()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print()
    if all_passed:
        print("🎉 All verification tests passed!")
        print("✅ Python conversion is complete and functional")
        return 0
    else:
        print("⚠️ Some tests failed - please review the errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
