#!/usr/bin/env python3
"""
Main script for Schumann Resonance Remote Sensing Analysis
舒曼波共振遥感 - Python主程序

This is the Python equivalent of Main.m, providing an interactive workflow
for mineral exploration using multiple anomaly detection methods.
"""

import os
import sys
from datetime import datetime
from typing import List, Optional
import tkinter as tk
from tkinter import filedialog, messagebox

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Core.geo_data_context import GeoDataContext
from Core.fusion_engine import FusionEngine
from Core.post_processor import PostProcessor
from Detectors.red_edge_detector import RedEdgeDetector
from Detectors.intrinsic_detector import IntrinsicDetector
from Detectors.slow_vars_detector import SlowVarsDetector
from Detectors.known_anomaly_detector import KnownAnomalyDetector


def select_file(title: str, filetypes: List[tuple]) -> Optional[str]:
    """
    Open file selection dialog.
    
    Args:
        title: Dialog window title
        filetypes: List of (description, pattern) tuples
        
    Returns:
        Selected file path or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()  # Hide main window
    filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return filepath if filepath else None


def select_directory(title: str) -> Optional[str]:
    """
    Open directory selection dialog.
    
    Args:
        title: Dialog window title
        
    Returns:
        Selected directory path or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()
    dirpath = filedialog.askdirectory(title=title)
    root.destroy()
    return dirpath if dirpath else None


def ask_yes_no(title: str, message: str) -> bool:
    """
    Show yes/no dialog.
    
    Args:
        title: Dialog title
        message: Dialog message
        
    Returns:
        True if yes, False if no
    """
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno(title, message)
    root.destroy()
    return result


def main():
    """Main execution function."""
    print("=" * 60)
    print("  Schumann Resonance Remote Sensing - Python Version")
    print("  舒曼波共振遥感 - Python版本")
    print("=" * 60)
    print()
    
    # ================= 1. Basic Configuration =================
    config = {
        'mineral_type': 'gold',
        'region_type': '',  # Empty triggers interactive selection
        'levashov_mode': True,
        'fusion_mode': True
    }
    
    # ================= 2. Interactive Data Selection =================
    print(">>> [Interactive Mode] Please select data folder...")
    data_dir = select_directory("Select Data Folder")
    if not data_dir:
        print("User cancelled folder selection. Exiting.")
        return
    
    print(f"✅ Selected data directory: {data_dir}")
    config['data_dir'] = data_dir
    
    print("\n>>> [Interactive Mode] Please select coordinates file (Excel/CSV)...")
    roi_file = select_file(
        "Select Coordinates File",
        [("Excel Files", "*.xlsx *.xls"), ("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not roi_file:
        print("User cancelled file selection. Exiting.")
        return
    
    print(f"✅ Selected coordinates file: {roi_file}")
    config['roi_file'] = roi_file
    
    # ================= 3. KML Configuration =================
    print("\n>>> [Interactive Mode] Import KML/KMZ known anomalies (4th mask)?")
    use_kml = ask_yes_no("KML Configuration", "Import KML/KMZ file with known anomalies?")
    
    # Define base detector list
    detectors_to_use = ['RedEdge', 'Intrinsic']
    
    if use_kml:
        kmz_file = select_file(
            "Select KML/KMZ File",
            [("Google Earth Files", "*.kml *.kmz"), ("All Files", "*.*")]
        )
        if kmz_file:
            config['kmz_path'] = kmz_file
            config['kmz_keywords'] = ['矿体投影', 'Object ID', 'ZK', '异常', '已知矿点']
            print(f"✅ Selected KML file: {kmz_file}")
            detectors_to_use.append('KnownAnomaly')
        else:
            print("⚠️ User cancelled KML selection, skipping this step.")
            config['kmz_path'] = ''
    else:
        print(">>> Skipping KML import.")
        config['kmz_path'] = ''
    
    # ================= 4. Initialize Data Context =================
    print("\n>>> Initializing data context (GeoDataContext)...")
    try:
        data_ctx = GeoDataContext(config)
        print("✅ Data context initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing data context: {str(e)}")
        return
    
    # ================= 5. Setup Output Directory =================
    # Create folder name from detector types
    types_str = '_'.join(detectors_to_use)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    folder_name = f"{types_str}_Result_{config['mineral_type']}_{timestamp}"
    out_dir = os.path.join(data_ctx.data_dir, folder_name)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    config['out_dir'] = out_dir
    print(f"📂 Results will be saved to: {out_dir}")
    
    # ================= 6. Initialize Fusion Engine =================
    print("\n>>> Initializing fusion engine...")
    engine = FusionEngine()
    
    # Register detectors
    engine.add_detector('RedEdge', RedEdgeDetector())
    engine.add_detector('Intrinsic', IntrinsicDetector())
    engine.add_detector('SlowVars', SlowVarsDetector())
    
    # Only register KnownAnomaly if KML is used
    if 'KnownAnomaly' in detectors_to_use:
        engine.add_detector('KnownAnomaly', KnownAnomalyDetector())
    
    print("✅ Fusion engine initialized")
    
    # ================= 7. Execute Computation =================
    print("\n>>> Starting parallel computation of anomaly layers...")
    try:
        engine.compute_all(data_ctx)
        print("✅ All detectors computed successfully")
    except Exception as e:
        print(f"❌ Error during computation: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # ================= 8. Fuse Results =================
    print("\n>>> Fusing detection results...")
    try:
        final_mask = engine.get_fused_mask(detectors_to_use)
        print("✅ Fusion complete")
    except Exception as e:
        print(f"❌ Error during fusion: {str(e)}")
        return
    
    # ================= 9. Post-Processing & Visualization =================
    print("\n>>> Running post-processing and visualization...")
    try:
        PostProcessor.run(data_ctx, engine, final_mask, out_dir)
        print("✅ Post-processing complete")
    except Exception as e:
        print(f"❌ Error during post-processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # ================= 10. Complete =================
    print("\n" + "=" * 60)
    print("✅ All processes complete!")
    print(f"📂 Results saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
