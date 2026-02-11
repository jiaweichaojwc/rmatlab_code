"""
KMZ export module for calling the Python KMZ generation script.

This module wraps the chengjie_matlab_code.py script to generate KMZ files
from MATLAB-exported prediction data.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_kmz(data_file: str, out_dir: str, python_exe: Optional[str] = None) -> Tuple[int, str]:
    """
    Export KMZ file by calling the Python chengjie_matlab_code.py script.
    
    This function wraps the existing KMZ generation script, handling path
    resolution and subprocess execution with proper error handling.
    
    Args:
        data_file: Full path to the .mat file containing prediction data
        out_dir: Output directory path for the KMZ file
        python_exe: Optional path to Python interpreter. If None, uses sys.executable
    
    Returns:
        Tuple of (status_code, message):
            - status_code: 0 for success, non-zero for failure
            - message: Status message or error details
    
    Raises:
        FileNotFoundError: If the Python script or data file is not found
        subprocess.SubprocessError: If the subprocess execution fails
    
    Example:
        >>> status, msg = export_kmz('data/prediction.mat', 'output/')
        >>> if status == 0:
        ...     print(f"Success: {msg}")
    """
    logger.info(">>> [KMZ Export] Starting KMZ generation...")
    
    try:
        # ================= 1. Locate Python script =================
        # Get the directory of this file (utils/)
        current_file_dir = Path(__file__).parent.resolve()
        
        # Navigate to Python/ directory (utils/../)
        python_dir = current_file_dir.parent
        
        # Construct path to chengjie_matlab_code.py
        py_script = python_dir / 'chengjie_matlab_code.py'
        
        # Verify script exists
        if not py_script.exists():
            error_msg = (
                f"Python script not found!\n"
                f"Expected path: {py_script}\n"
                f"Please ensure chengjie_matlab_code.py is in the Python folder."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Found Python script: {py_script}")
        
        # ================= 2. Verify data file exists =================
        data_file_path = Path(data_file).resolve()
        if not data_file_path.exists():
            error_msg = f"Data file not found: {data_file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Found data file: {data_file_path}")
        
        # ================= 3. Ensure output directory exists =================
        out_dir_path = Path(out_dir).resolve()
        out_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {out_dir_path}")
        
        # ================= 4. Set Python interpreter =================
        if python_exe is None:
            python_exe = sys.executable
        
        python_exe_path = Path(python_exe)
        if not python_exe_path.exists():
            error_msg = f"Python interpreter not found: {python_exe_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Using Python interpreter: {python_exe_path}")
        
        # ================= 5. Construct and execute command =================
        cmd = [
            str(python_exe_path),
            str(py_script),
            str(data_file_path),
            str(out_dir_path)
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Execute subprocess with proper encoding
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace undecodable characters
            timeout=300  # 5 minute timeout
        )
        
        # ================= 6. Process results =================
        if result.returncode == 0:
            success_msg = (
                f"✅ KMZ export successful!\n"
                f"📂 Output location: {out_dir_path}\n"
            )
            if result.stdout:
                success_msg += f"\nOutput:\n{result.stdout}"
            
            logger.info(success_msg)
            return 0, success_msg
        else:
            error_msg = (
                f"❌ KMZ generation failed!\n"
                f"Return code: {result.returncode}\n"
            )
            
            if result.stderr:
                error_msg += f"\nError output:\n{'-' * 50}\n{result.stderr}\n{'-' * 50}\n"
            
            if result.stdout:
                error_msg += f"\nStandard output:\n{'-' * 50}\n{result.stdout}\n{'-' * 50}\n"
            
            error_msg += (
                f"\nTroubleshooting:\n"
                f"  1. Verify Python interpreter: {python_exe_path}\n"
                f"  2. Verify Python script: {py_script}\n"
                f"  3. Check required packages: pip install simplekml pyproj scipy numpy matplotlib\n"
                f"  4. Verify data file format and contents\n"
            )
            
            logger.error(error_msg)
            return result.returncode, error_msg
            
    except subprocess.TimeoutExpired:
        error_msg = "❌ KMZ generation timed out (exceeded 5 minutes)"
        logger.error(error_msg)
        return -1, error_msg
        
    except FileNotFoundError as e:
        error_msg = f"❌ File not found error: {e}"
        logger.error(error_msg)
        return -1, error_msg
        
    except Exception as e:
        error_msg = f"❌ Unexpected error during KMZ export: {type(e).__name__}: {e}"
        logger.error(error_msg, exc_info=True)
        return -1, error_msg


def main():
    """
    Command-line interface for export_kmz function.
    
    Usage:
        python export_kmz.py <data_file> <out_dir> [python_exe]
    """
    if len(sys.argv) < 3:
        print("Usage: python export_kmz.py <data_file> <out_dir> [python_exe]")
        print("\nArguments:")
        print("  data_file  : Path to .mat file with prediction data")
        print("  out_dir    : Output directory for KMZ file")
        print("  python_exe : (Optional) Path to Python interpreter")
        sys.exit(1)
    
    data_file = sys.argv[1]
    out_dir = sys.argv[2]
    python_exe = sys.argv[3] if len(sys.argv) > 3 else None
    
    status, message = export_kmz(data_file, out_dir, python_exe)
    
    print(message)
    sys.exit(status)


if __name__ == "__main__":
    main()
