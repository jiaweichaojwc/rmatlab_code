"""
Slow Variables Detector for thermodynamic instability detection.
Equivalent to SlowVarsDetector.m

This detector implements thermodynamic instability detection based on
cubic equation discriminant analysis of 7 slow geological variables.
"""
import numpy as np
from typing import Dict, Any
from scipy.ndimage import binary_opening, binary_dilation, label, gaussian_filter
from .anomaly_detector import AnomalyDetector


class SlowVarsDetector(AnomalyDetector):
    """
    Slow variables thermodynamic instability detector.
    
    Detects anomalies by computing 7 slow geological variables:
    1. Stress gradient (from DEM)
    2. Redox gradient (from iron oxide and oxygen fugacity)
    3. Fluid overpressure (from TIR and NDVI)
    4. Fault activity (from edge detection)
    5. Carbonate cover (from ASTER bands)
    6. Temperature gradient (from TIR)
    7. Chemical potential gradient
    
    Uses cubic equation discriminant Delta = b² + (8/27)a³ to identify
    thermodynamic instability regions where Delta < 0.
    """
    
    def calculate(self, ctx: Any) -> Dict[str, Any]:
        """
        Calculate thermodynamic instability-based anomaly detection.
        
        This method computes 7 slow geological variables, standardizes them
        via z-score normalization, and identifies regions of thermodynamic
        instability using cubic equation discriminant analysis.
        
        Args:
            ctx: Data context object containing:
                - dem: Digital Elevation Model (H x W)
                - lan: Landsat data cube (H x W x bands)
                - s2: Sentinel-2 data cube (H x W x bands)
                - ast: ASTER data cube (H x W x 14)
                - NIR: Near-infrared band (H x W)
                - Red: Red band (H x W)
                - inROI: Boolean ROI mask (H x W)
        
        Returns:
            Dictionary with:
                - mask: Binary detection mask (H x W)
                - debug: Dictionary containing:
                    - Delta: Discriminant values
        """
        inROI = ctx.inROI
        eps = np.finfo(float).eps
        
        # 1. Stress gradient (地应力梯度)
        # Compute gradient magnitude from DEM
        gy, gx = np.gradient(ctx.dem)
        stress_grad = np.sqrt(gx**2 + gy**2)
        
        # 2. Redox gradient (氧化还原梯度)
        # Iron oxide ratio from Landsat bands
        iron_oxide = ctx.lan[:, :, 2] / (ctx.lan[:, :, 1] + eps)
        
        # Oxygen fugacity from Sentinel-2 Red Edge bands
        # Note: S2 band order is B02,B03,B04,B08,B11,B12,B05,B06,B07
        # So index 7 = B06 (Red Edge 2), index 8 = B07 (Red Edge 3)
        re2 = ctx.s2[:, :, 7]  # Band 6 (B06 - Red Edge 2)
        re3 = ctx.s2[:, :, 8]  # Band 7 (B07 - Red Edge 3)
        re_mean = np.mean(np.stack([re2, re3], axis=2), axis=2)
        oxy_fug = re3 - re_mean
        
        # Handle invalid values
        oxy_fug[np.isnan(oxy_fug) | np.isinf(oxy_fug)] = 0
        
        # Combine iron oxide and oxygen fugacity deviations
        iron_mean = np.nanmean(iron_oxide[inROI])
        oxy_mean = np.nanmean(oxy_fug[inROI])
        redox_grad = (np.abs(iron_oxide - iron_mean) + 
                     np.abs(oxy_fug - oxy_mean))
        
        # 3. Fluid overpressure (流体超压)
        # Mean of ASTER TIR bands (10-14 in MATLAB = indices 9-13 in Python)
        tir_mean = np.mean(ctx.ast[:, :, 9:14], axis=2)
        
        # NDVI calculation
        ndvi = (ctx.NIR - ctx.Red) / (ctx.NIR + ctx.Red + eps)
        
        # Fluid overpressure indicator
        fluid_over = tir_mean + 3 * (1 - ndvi)
        
        # 4. Fault activity (断裂活动性)
        # Edge detection using Canny algorithm
        edges = self._canny_edge_detection(stress_grad, 
                                           low_threshold=0.05, 
                                           high_threshold=0.25)
        
        # Remove small regions (< 50 pixels)
        edges = self._remove_small_objects(edges, 50)
        
        # Weight edges by stress gradient
        fault_activity = edges.astype(float) * stress_grad
        
        # 5. Carbonate cover (盖层)
        # Ratio of carbonate-sensitive ASTER bands
        carbonate = ((ctx.ast[:, :, 5] + ctx.ast[:, :, 7]) / 
                    (ctx.ast[:, :, 6] + eps))
        
        # 6. Temperature gradient (温度梯度)
        gty, gtx = np.gradient(tir_mean)
        temp_grad = np.sqrt(gtx**2 + gty**2)
        
        # 7. Chemical potential gradient (化学势梯度)
        chem_field = iron_oxide + oxy_fug
        gcy, gcx = np.gradient(chem_field)
        chem_grad = np.sqrt(gcx**2 + gcy**2)
        
        # Z-score standardization helper function
        def zscore(x: np.ndarray) -> np.ndarray:
            """
            Compute z-score normalization within ROI.
            
            Args:
                x: Input array
                
            Returns:
                Z-score normalized array
            """
            x_roi = x[inROI]
            mean_val = np.nanmean(x_roi)
            std_val = np.nanstd(x_roi)
            
            if std_val == 0 or np.isnan(std_val):
                std_val = eps
                
            return (x - mean_val) / std_val
        
        # Compute cubic equation coefficients
        # a coefficient: barrier terms (negative of cover and temperature gradient)
        a = -(0.5 * zscore(carbonate) + 0.5 * zscore(temp_grad))
        
        # b coefficient: weighted sum of driving forces
        b = (0.25 * zscore(stress_grad) + 
             0.20 * zscore(redox_grad) + 
             0.25 * zscore(fluid_over) + 
             0.15 * zscore(fault_activity) + 
             0.15 * zscore(chem_grad))
        
        # Compute discriminant: Delta = b² + (8/27)a³
        # Negative discriminant indicates thermodynamic instability
        Delta = b**2 + (8/27) * a**3
        
        # Generate mask: instability regions within ROI
        mask = (Delta < 0) & inROI
        
        # Remove small regions (< 100 pixels)
        mask = self._remove_small_objects(mask, 100)
        
        # Apply dilation with disk structuring element (radius 8)
        mask = self._disk_dilation(mask, 8)
        
        # Convert to double precision
        mask = mask.astype(np.float64)
        
        # Return results
        res = {
            'mask': mask,
            'debug': {
                'Delta': Delta
            }
        }
        
        return res
    
    @staticmethod
    def _canny_edge_detection(image: np.ndarray, 
                              low_threshold: float = 0.05, 
                              high_threshold: float = 0.25,
                              sigma: float = 1.0) -> np.ndarray:
        """
        Canny edge detection algorithm.
        Equivalent to MATLAB's edge(image, 'canny', [low, high]).
        
        Args:
            image: Input grayscale image
            low_threshold: Low threshold for hysteresis (0-1 range)
            high_threshold: High threshold for hysteresis (0-1 range)
            sigma: Standard deviation for Gaussian filter
            
        Returns:
            Binary edge map
        """
        # Step 1: Smooth with Gaussian filter
        smoothed = gaussian_filter(image, sigma=sigma)
        
        # Step 2: Compute gradients
        gy, gx = np.gradient(smoothed)
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Normalize magnitude to 0-1 range
        mag_max = np.max(magnitude)
        if mag_max > 0:
            magnitude = magnitude / mag_max
        
        # Step 3: Non-maximum suppression
        angle = np.arctan2(gy, gx) * 180.0 / np.pi
        angle[angle < 0] += 180
        
        nms = np.zeros_like(magnitude)
        h, w = magnitude.shape
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                q = 255
                r = 255
                
                # Angle 0 degree
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j+1]
                    r = magnitude[i, j-1]
                # Angle 45 degree
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i+1, j-1]
                    r = magnitude[i-1, j+1]
                # Angle 90 degree
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i+1, j]
                    r = magnitude[i-1, j]
                # Angle 135 degree
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i-1, j-1]
                    r = magnitude[i+1, j+1]
                
                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    nms[i, j] = magnitude[i, j]
        
        # Step 4: Double threshold
        high_threshold_val = high_threshold
        low_threshold_val = low_threshold
        
        strong_edges = nms >= high_threshold_val
        weak_edges = (nms >= low_threshold_val) & (nms < high_threshold_val)
        
        # Step 5: Edge tracking by hysteresis
        edges = strong_edges.copy()
        
        # Connect weak edges to strong edges
        changed = True
        while changed:
            changed = False
            for i in range(1, h-1):
                for j in range(1, w-1):
                    if weak_edges[i, j]:
                        # Check if any neighbor is a strong edge
                        if np.any(edges[i-1:i+2, j-1:j+2]):
                            edges[i, j] = True
                            weak_edges[i, j] = False
                            changed = True
        
        return edges
    
    @staticmethod
    def _remove_small_objects(binary_img: np.ndarray, min_size: int) -> np.ndarray:
        """
        Remove connected components smaller than min_size.
        Equivalent to MATLAB's bwareaopen.
        
        Args:
            binary_img: Binary image
            min_size: Minimum size threshold
            
        Returns:
            Cleaned binary image
        """
        labeled, num_features = label(binary_img)
        
        if num_features == 0:
            return binary_img
        
        # Compute sizes of each component
        sizes = np.bincount(labeled.ravel())
        
        # Create mask of components to keep
        mask_sizes = sizes >= min_size
        mask_sizes[0] = 0  # Background always removed
        
        # Apply mask
        result = mask_sizes[labeled]
        
        return result
    
    @staticmethod
    def _disk_dilation(binary_img: np.ndarray, radius: int) -> np.ndarray:
        """
        Apply binary dilation with disk structuring element.
        Equivalent to MATLAB's imdilate with strel('disk', radius).
        
        Args:
            binary_img: Binary image
            radius: Disk radius
            
        Returns:
            Dilated binary image
        """
        # Create disk structuring element
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        disk = x**2 + y**2 <= radius**2
        
        # Apply dilation
        result = binary_dilation(binary_img, structure=disk)
        
        return result
