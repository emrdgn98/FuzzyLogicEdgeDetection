#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fuzzy Edge Detection Methods - Complete Evaluation (FIXED VERSION)
Methods: Type-1, Type-2, Hybrid-1, Hybrid-2
Datasets: STARE, BIPED, BSDS500
Metrics: MSE, PSNR, SSIM, ODS, OIS, AP

✅ FIXED: Distance tolerance (0.0075) for ODS/OIS calculation
"""

import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm
import json
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import average_precision_score
import scipy.io as sio
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("\n❌ ERROR: scikit-fuzzy not installed!")
    print("   Install: pip install scikit-fuzzy --break-system-packages\n")
    exit(1)


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def sobel_operation(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sobel operation with high-pass and mean filters"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    x = gray.astype(np.float64)
    
    k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    hHP = np.array([[-1/16, -1/8, -1/16], [-1/8, 3/4, -1/8], [-1/16, -1/8, -1/16]], dtype=np.float64)
    hMF = np.ones((5, 5), dtype=np.float64) / 25.0
    
    Gx = ndimage.convolve(x, k)
    Gy = ndimage.convolve(x, k.T)
    G = np.sqrt(Gx**2 + Gy**2)
    HP = ndimage.convolve(x, hHP)
    M = ndimage.convolve(x, hMF)
    
    return Gx, Gy, G, HP, M

def build_type1_lut(fis, sim, levels=16):
    """Build LUT for Type-1 fuzzy inference"""
    grid = np.linspace(0, 255, levels)
    lut = np.zeros((levels, levels, levels, levels), dtype=np.uint8)

    for i, dh in enumerate(tqdm(grid, desc="Type-1 LUT", unit="level")):
        for j, dv in enumerate(grid):
            for k, hp in enumerate(grid):
                for l, m in enumerate(grid):
                    sim.input['DH'] = dh
                    sim.input['DV'] = dv
                    sim.input['HP'] = hp
                    sim.input['M']  = m
                    sim.compute()
                    lut[i, j, k, l] = np.clip(sim.output['Iout'], 0, 255)
    return lut

def robinson_compass_operation(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Robinson Compass Operation (8 directions, return 4)"""
    G_0 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    G_45 = np.array([[-2, -2, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float64)
    G_90 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    G_135 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float64)
    
    g1 = ndimage.convolve(image, G_0)
    g2 = ndimage.convolve(image, G_45)
    g3 = ndimage.convolve(image, G_90)
    g4 = ndimage.convolve(image, G_135)
    
    return np.abs(g1), np.abs(g2), np.abs(g3), np.abs(g4)


# ==================================================
# NORMALIZATION
# ==================================================

def normalize(img):
    """Normalize to 0-1 range"""
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


# ==================================================
# FUZZY EDGE DETECTORS (Same as yours)
# ==================================================

class FuzzyEdgeDetectors:
    """Fuzzy edge detection methods"""
    
    def detect_type1_fast(self, image: np.ndarray, return_soft: bool = False) -> np.ndarray:
        """Type-1 Fuzzy Edge Detection"""
        print("    Type-1 FAST: Extracting features...")
        Gx, Gy, G, HP, M = sobel_operation(image)

        def norm(x):
            return np.clip((x - x.min()) / (x.max() - x.min() + 1e-8) * 255, 0, 255).astype(np.uint8)

        Gx_n = norm(Gx)
        Gy_n = norm(Gy)
        HP_n = norm(HP)
        M_n  = norm(M)

        universe = np.linspace(0, 255, 100)
        DH = ctrl.Antecedent(universe, 'DH')
        DV = ctrl.Antecedent(universe, 'DV')
        HPF = ctrl.Antecedent(universe, 'HP')
        MF = ctrl.Antecedent(universe, 'M')
        Iout = ctrl.Consequent(universe, 'Iout')

        for var in [DH, DV, HPF, MF]:
            var['low'] = fuzz.gaussmf(var.universe, 0, 43)
            var['medium'] = fuzz.gaussmf(var.universe, 127, 43)
            var['high'] = fuzz.gaussmf(var.universe, 255, 43)

        Iout['low'] = fuzz.gaussmf(Iout.universe, 0, 43)
        Iout['medium'] = fuzz.gaussmf(Iout.universe, 127, 43)
        Iout['high'] = fuzz.gaussmf(Iout.universe, 255, 43)

        rules = [
            ctrl.Rule(DH['low'] & DV['low'], Iout['low']),
            ctrl.Rule(DH['medium'] & DV['medium'], Iout['high']),
            ctrl.Rule(DH['high'] & DV['high'], Iout['high']),
            ctrl.Rule(DH['medium'] & HPF['low'], Iout['high']),
            ctrl.Rule(DV['medium'] & HPF['low'], Iout['high']),
            ctrl.Rule(MF['low'] & DV['medium'], Iout['low']),
            ctrl.Rule(MF['low'] & DH['medium'], Iout['low'])
        ]

        fis = ctrl.ControlSystem(rules)
        sim = ctrl.ControlSystemSimulation(fis)

        print("    Type-1 FAST: Building LUT...")
        levels = 16
        lut = build_type1_lut(fis, sim, levels)

        idx = lambda x: (x * (levels - 1) // 255).astype(np.int32)

        output = lut[
            idx(Gx_n),
            idx(Gy_n),
            idx(HP_n),
            idx(M_n)
        ]

        if return_soft:
            print("    Type-1 FAST: Done! (soft edges)")
            return output
        else:
            threshold = np.mean(output) + 2 * np.std(output)
            result = ((output > threshold).astype(np.uint8) * 255)
            print("    Type-1 FAST: Done! (binary edges)")
            return result
    
    def detect_type2(self, image: np.ndarray, return_soft: bool = False) -> np.ndarray:
        """Type-2 Fuzzy Edge Detection"""
        # print("    Type-2: Preprocessing...")
        
        if len(image.shape) == 3:
            img_G = image[:, :, 1]
        else:
            img_G = image
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_contrasted = clahe.apply(img_G)
        img_median = cv2.medianBlur(img_contrasted, 25)
        img_wbg = cv2.subtract(img_median, img_contrasted)
        img_blurred = cv2.GaussianBlur(img_wbg, (3, 3), 0)
        
        # print("    Type-2: Robinson compass operation...")
        Ix, Iy, Iz, Ik = robinson_compass_operation(img_blurred.astype(np.float64))
        
        # print("    Type-2: Type-2 fuzzy inference...")
        
        if return_soft:
            G_combined = np.maximum.reduce([Ix, Iy, Iz, Ik])
            G_norm = ((G_combined - G_combined.min()) / 
                    (G_combined.max() - G_combined.min() + 1e-8) * 255).astype(np.uint8)
            
            lower = np.minimum.reduce([Ix, Iy, Iz, Ik])
            upper = np.maximum.reduce([Ix, Iy, Iz, Ik])
            
            lower_norm = ((lower - lower.min()) / (lower.max() - lower.min() + 1e-8) * 255).astype(np.uint8)
            upper_norm = ((upper - upper.min()) / (upper.max() - upper.min() + 1e-8) * 255).astype(np.uint8)
            
            soft_output = ((lower_norm.astype(np.float32) + upper_norm.astype(np.float32)) / 2).astype(np.uint8)
            
            if soft_output.mean() > 127:
                soft_output = 255 - soft_output
            
            result = cv2.GaussianBlur(soft_output, (3, 3), 0.5)
            
            # print(f"    Type-2: Done! (soft edges, range: [{result.min()}, {result.max()}])")
            return result
        
        else:
            def get_threshold(img):
                img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                _, binary = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return binary
            
            thresh_x = get_threshold(Ix)
            thresh_y = get_threshold(Iy)
            thresh_z = get_threshold(Iz)
            thresh_k = get_threshold(Ik)
            
            lower = np.minimum.reduce([thresh_x, thresh_y, thresh_z, thresh_k])
            upper = np.maximum.reduce([thresh_x, thresh_y, thresh_z, thresh_k])
            
            output = ((lower.astype(np.float32) + upper.astype(np.float32)) / 2).astype(np.uint8)
            
            if output.mean() > 127:
                output = 255 - output
            
            threshold = np.mean(output) + 1.5 * np.std(output)
            fuzzy2_edge = ((output > threshold).astype(np.uint8) * 255)
            
            result = cv2.medianBlur(fuzzy2_edge, 3)
            
            # print(f"    Type-2: Done! (binary edges)")
            return result
    
    def detect_hybrid1(self, image: np.ndarray, return_soft: bool = False) -> np.ndarray:
        """Hybrid-1: AND operation between Type-1 and Type-2"""
        print("  Hybrid-1: Running Type-1...")
        out1 = self.detect_type1_fast(image, return_soft=return_soft)
        
        print("  Hybrid-1: Running Type-2...")
        out2 = self.detect_type2(image, return_soft=return_soft)
        
        print("  Hybrid-1: AND operation...")
        if return_soft:
            result = np.minimum(out1, out2)
        else:
            result = cv2.bitwise_and(out1, out2)
        
        print("  Hybrid-1: Done!")
        return result
    
    def detect_hybrid2(self, image: np.ndarray, return_soft: bool = False) -> np.ndarray:
        """Hybrid-2: OR operation between Type-1 and Type-2"""
        print("  Hybrid-2: Running Type-1...")
        out1 = self.detect_type1_fast(image, return_soft=return_soft)
        
        print("  Hybrid-2: Running Type-2...")
        out2 = self.detect_type2(image, return_soft=return_soft)
        
        print("  Hybrid-2: OR operation...")
        if return_soft:
            result = np.maximum(out1, out2)
        else:
            result = cv2.bitwise_or(out1, out2)
        
        print("  Hybrid-2: Done!")
        return result


# ==================================================
# ✅ FIXED: Distance Tolerance Metrics
# ==================================================

def compute_f_score(precision, recall):
    """F-score hesapla"""
    with np.errstate(divide='ignore', invalid='ignore'):
        f_score = 2 * (precision * recall) / (precision + recall)
        f_score = np.nan_to_num(f_score)
    return f_score


def correspondence_with_tolerance(pred_binary, gt_binary, max_dist=0.0075):
    """
    ✅ FIXED: Distance tolerance ile TP/FP/FN hesapla
    """
    diag = np.sqrt(pred_binary.shape[0]**2 + pred_binary.shape[1]**2)
    max_dist_px = max_dist * diag
    
    if np.sum(gt_binary) > 0:
        dist_gt = distance_transform_edt(1 - gt_binary)
    else:
        dist_gt = np.ones_like(gt_binary) * np.inf
        
    if np.sum(pred_binary) > 0:
        dist_pred = distance_transform_edt(1 - pred_binary)
    else:
        dist_pred = np.ones_like(pred_binary) * np.inf
    
    tp = np.sum((pred_binary == 1) & (dist_gt <= max_dist_px))
    fp = np.sum((pred_binary == 1) & (dist_gt > max_dist_px))
    fn = np.sum((gt_binary == 1) & (dist_pred > max_dist_px))
    
    return tp, fp, fn


def calculate_ois_with_tolerance(pred, gt, thresholds=99, max_dist=0.0075):
    """
    ✅ FIXED: OIS hesapla - distance tolerance ile
    """
    pred = normalize(pred)
    gt = normalize(gt)
    
    gt_binary = (gt > 0.5).astype(np.uint8)
    
    threshold_values = np.linspace(0, 1, thresholds)
    f_scores = []
    precisions = []
    recalls = []
    
    for thresh in threshold_values:
        pred_binary = (pred >= thresh).astype(np.uint8)
        
        tp, fp, fn = correspondence_with_tolerance(pred_binary, gt_binary, max_dist)
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f = compute_f_score(precision, recall)
        
        f_scores.append(f)
        precisions.append(precision)
        recalls.append(recall)
    
    f_scores = np.array(f_scores)
    best_idx = np.argmax(f_scores)
    
    # AP calculation
    try:
        pred_flat = pred.flatten()
        gt_flat = (gt > 0.5).astype(int).flatten()
        ap = average_precision_score(gt_flat, pred_flat)
    except:
        ap = 0.0
    
    return float(f_scores[best_idx]), float(threshold_values[best_idx]), float(ap), f_scores.tolist()


def calculate_ods_with_tolerance(all_predictions, all_ground_truths, thresholds=99, max_dist=0.0075):
    """
    ✅ FIXED: ODS hesapla - distance tolerance ile
    """
    threshold_values = np.linspace(0, 1, thresholds)
    dataset_f_scores = []
    
    for thresh in threshold_values:
        total_tp, total_fp, total_fn = 0, 0, 0
        
        for pred, gt in zip(all_predictions, all_ground_truths):
            pred = normalize(pred)
            gt = normalize(gt)
            
            pred_binary = (pred >= thresh).astype(np.uint8)
            gt_binary = (gt > 0.5).astype(np.uint8)
            
            tp, fp, fn = correspondence_with_tolerance(pred_binary, gt_binary, max_dist)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f = compute_f_score(precision, recall)
        dataset_f_scores.append(f)
    
    dataset_f_scores = np.array(dataset_f_scores)
    best_idx = np.argmax(dataset_f_scores)
    
    return float(dataset_f_scores[best_idx]), float(threshold_values[best_idx])


# ==================================================
# Basic Metrics (Same as yours)
# ==================================================

def calculate_basic_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict:
    """Calculate MSE, PSNR, SSIM on binary images (0/255)"""
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    
    if pred.dtype != np.uint8:
        pred = pred.astype(np.uint8)
    if gt.dtype != np.uint8:
        gt = gt.astype(np.uint8)
    
    mse_val = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    
    if mse_val == 0:
        psnr_val = float('inf')
    else:
        psnr_val = 10 * np.log10((255.0 ** 2) / mse_val)
    
    ssim_val = ssim(gt, pred, data_range=255)
    
    return {
        'mse': float(mse_val),
        'psnr': float(psnr_val),
        'ssim': float(ssim_val)
    }


# ==================================================
# Dataset Loaders (Same as yours)
# ==================================================

def read_mat_ground_truth(mat_path: str) -> np.ndarray:
    """Read BSDS500 .mat ground truth"""
    try:
        mat = sio.loadmat(mat_path)
        
        if 'groundTruth' in mat:
            gt_cell = mat['groundTruth']
            if gt_cell.size > 0:
                boundaries_list = []
                
                for i in range(gt_cell.shape[1]):
                    gt_struct = gt_cell[0, i]
                    
                    if 'Boundaries' in gt_struct.dtype.names:
                        boundary = gt_struct['Boundaries'][0, 0]
                        boundary = np.array(boundary, dtype=np.float32)
                        boundaries_list.append(boundary)
                    elif 'Segmentation' in gt_struct.dtype.names:
                        seg = gt_struct['Segmentation'][0, 0]
                        seg = np.array(seg, dtype=np.float32)
                        seg_uint8 = (seg / (seg.max() + 1e-8) * 255).astype(np.uint8)
                        boundary = cv2.Canny(seg_uint8, 50, 150).astype(np.float32) / 255.0
                        boundaries_list.append(boundary)
                
                if boundaries_list:
                    avg_boundary = np.mean(boundaries_list, axis=0).astype(np.float32)
                    avg_boundary = np.clip(avg_boundary, 0.0, 1.0)
                    return avg_boundary
        
        for key in ['bw', 'boundary', 'edge']:
            if key in mat:
                data = np.array(mat[key], dtype=np.float32)
                if data.max() > 1.0:
                    data = data / 255.0
                return np.clip(data, 0.0, 1.0)
        
        return None
        
    except Exception as e:
        print(f"Error reading {mat_path}: {e}")
        return None


def get_stare_pairs(stare_root: str) -> List[Tuple[str, str]]:
    raw_dir = os.path.join(stare_root, "stare_raw")
    gt_dir = os.path.join(stare_root, "stare_vk")
    
    if not os.path.exists(raw_dir) or not os.path.exists(gt_dir):
        return []
    
    raw_images = glob(os.path.join(raw_dir, "*.ppm"))
    gt_images = glob(os.path.join(gt_dir, "*.ppm"))
    
    raw_dict = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in raw_images}
    
    pairs = []
    for gt_path in gt_images:
        gt_name = os.path.basename(gt_path).lower()
        base = gt_name.replace(".vk", "").replace(".ppm", "")
        
        if base in raw_dict:
            pairs.append((raw_dict[base], gt_path))
    
    return pairs


def get_miedt_pairs(miedt_root: str) -> List[Tuple[str, str]]:
    """
    Get image-groundtruth pairs for MIEDT dataset
    
    Args:
        miedt_root: Root directory of MIEDT dataset
        
    Expected structure:
        miedt_root/
            Original/
                IMG-001.png, IMG-002.png, ...
            Ground Truth/
                GT-001.png, GT-002.png, ...
    
    Returns:
        List of (image_path, ground_truth_path) tuples
    """
    img_dir = os.path.join(miedt_root, 'ct_brain_original')
    gt_dir = os.path.join(miedt_root, 'ct_brain_ground_truth')
    
    print(f"Checking MIEDT directories:")
    print(f"  Image dir: {img_dir} - Exists: {os.path.exists(img_dir)}")
    print(f"  GT dir: {gt_dir} - Exists: {os.path.exists(gt_dir)}")
    
    if not os.path.exists(img_dir):
        print(f"Warning: Image directory not found: {img_dir}")
        return []
    
    if not os.path.exists(gt_dir):
        print(f"Warning: Ground Truth directory not found: {gt_dir}")
        return []
    
    # Get image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']:
        found = glob(os.path.join(img_dir, ext))
        if found:
            image_files.extend(found)
    
    if not image_files:
        print(f"Warning: No images found in {img_dir}")
        return []
    
    print(f"Total images found: {len(image_files)}")
    
    pairs = []
    unmatched = []
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        
        # Extract number from IMG-XXX.ext
        # Expected format: IMG-001.png -> 001
        if img_name.startswith('IMG-'):
            # Get the number part and extension
            parts = img_name[4:]  # Remove 'IMG-'
            number_part = os.path.splitext(parts)[0]  # Remove extension
            img_ext = os.path.splitext(parts)[1]  # Get extension
            
            # Construct GT filename: GT-XXX.png
            gt_name = f"GT-{number_part}.png"
            gt_path = os.path.join(gt_dir, gt_name)
            
            # Also try with same extension as image
            if not os.path.exists(gt_path):
                gt_name = f"GT-{number_part}{img_ext}"
                gt_path = os.path.join(gt_dir, gt_name)
            
            if os.path.exists(gt_path):
                pairs.append((img_path, gt_path))
            else:
                unmatched.append(img_name)
        else:
            print(f"  Warning: Unexpected filename format: {img_name}")
            unmatched.append(img_name)
    
    print(f"\nMatching results:")
    print(f"  Matched pairs: {len(pairs)}")
    print(f"  Unmatched images: {len(unmatched)}")
    
    if unmatched and len(unmatched) <= 10:
        print(f"\nUnmatched images:")
        for img in unmatched:
            print(f"  {img}")
    elif unmatched:
        print(f"\nFirst 10 unmatched images:")
        for img in unmatched[:10]:
            print(f"  {img}")
    
    return pairs



def preprocess_miedt_ground_truth(gt_image: np.ndarray) -> np.ndarray:
    """
    Preprocess MIEDT ground truth
    - Convert segmentation mask to edge map if needed
    - Normalize to 0-255 range
    
    Args:
        gt_image: Ground truth image (can be mask or edge map)
        
    Returns:
        Processed ground truth as uint8
    """
    if len(gt_image.shape) == 3:
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
    
    # Normalize to 0-1
    gt_normalized = normalize(gt_image)
    
    # If it's a binary mask (not edge map), extract edges
    unique_values = np.unique(gt_normalized)
    if len(unique_values) <= 2:  # Binary mask
        # Convert mask to edge using Canny
        gt_uint8 = (gt_normalized * 255).astype(np.uint8)
        edges = cv2.Canny(gt_uint8, 50, 150)
        return edges
    else:
        # Already an edge map, just normalize to 0-255
        return (gt_normalized * 255).astype(np.uint8)
    

def get_biped_pairs(biped_root: str) -> List[Tuple[str, str]]:
    test_list_file = os.path.join(biped_root, 'test_rgb.lst')
    
    if not os.path.exists(test_list_file):
        return []
    
    pairs = []
    with open(test_list_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            img_rel, gt_rel = parts
            
            img_path = os.path.join(biped_root, 'imgs', 'test', img_rel)
            gt_path = os.path.join(biped_root, 'edge_maps', 'test', gt_rel)
            
            if os.path.exists(img_path) and os.path.exists(gt_path):
                pairs.append((img_path, gt_path))
    
    return pairs


def get_bsds500_pairs(bsds_root: str, split: str = 'test') -> List[Tuple[str, str]]:
    images_dir = os.path.join(bsds_root, "images", split)
    gt_dir = os.path.join(bsds_root, "ground_truth", split)
    
    if not os.path.exists(images_dir) or not os.path.exists(gt_dir):
        return []
    
    image_files = glob(os.path.join(images_dir, "*.jpg"))
    if not image_files:
        image_files = glob(os.path.join(images_dir, "*.png"))
    
    pairs = []
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mat_path = os.path.join(gt_dir, f"{base_name}.mat")
        
        if os.path.exists(mat_path):
            pairs.append((img_path, mat_path))
    
    return pairs


# ==================================================
# ✅ FIXED: Test Functions
# ==================================================

def test_method_on_dataset(method_name: str, method_func, pairs: List[Tuple[str, str]], 
                           dataset_name: str, is_mat_gt: bool = False, 
                           max_images: int = None, save_images: bool = True) -> Dict:
    """Test a fuzzy method on a dataset with distance tolerance"""
    
    print(f"\n{'='*70}")
    print(f"Testing {method_name} on {dataset_name}")
    print(f"{'='*70}")
    
    if max_images:
        pairs = pairs[:max_images]
        print(f"Limited to {max_images} images for testing")
    
    print(f"Total images: {len(pairs)}")
    
    if save_images:
        output_dir = f"results/{method_name.replace(' ', '_')}/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    all_basic_metrics = []
    all_ois = []
    all_aps = []
    all_predictions_soft = []
    all_ground_truths = []
    
    successful = 0
    failed = 0
    
    for idx, (img_path, gt_path) in enumerate(tqdm(pairs, desc=f"{method_name} | {dataset_name}", unit="img"), 1):
        print(f"\n[{idx}/{len(pairs)}] Processing: {os.path.basename(img_path)}")
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"  ❌ Failed to load image")
                failed += 1
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"  Image size: {image.shape}")
            
            # Load ground truth
            if is_mat_gt:
                gt = read_mat_ground_truth(gt_path)
                if gt is None:
                    print(f"  ❌ Failed to load ground truth")
                    failed += 1
                    continue
                gt = (normalize(gt) * 255).astype(np.uint8)
            else:
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    print(f"  ❌ Failed to load ground truth")
                    failed += 1
                    continue
                gt = gt.astype(np.uint8)

                  # MIEDT için özel preprocessing
            if dataset_name == 'MIEDT':
                gt = preprocess_miedt_ground_truth(gt)
            else:
                gt = gt.astype(np.uint8)
            
            # Get binary and soft edge maps
            pred_binary = method_func(image, return_soft=False)
            pred_soft = method_func(image, return_soft=True)
            
            if pred_binary.dtype != np.uint8:
                pred_binary = (normalize(pred_binary) * 255).astype(np.uint8)
            if pred_soft.dtype != np.uint8:
                pred_soft = (normalize(pred_soft) * 255).astype(np.uint8)
            
            # Resize GT if needed
            if pred_binary.shape != gt.shape:
                gt_resized = cv2.resize(gt, (pred_binary.shape[1], pred_binary.shape[0]))
            else:
                gt_resized = gt.copy()
            
            # Save images
            if save_images:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_binary.png"), pred_binary)
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_soft.png"), pred_soft)
                
                comparison = np.hstack([
                    cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image,
                    gt_resized,
                    pred_binary,
                    pred_soft
                ])
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_comparison.png"), comparison)
                
                print(f"  💾 Saved: {base_name}_*.png")
            
            # Basic metrics on binary edges
            basic = calculate_basic_metrics(pred_binary, gt_resized)
            all_basic_metrics.append(basic)
            
            # ✅ OIS + AP with distance tolerance
            ois, _, ap, _ = calculate_ois_with_tolerance(pred_soft, gt_resized, max_dist=0.0075)
            all_ois.append(ois)
            all_aps.append(ap)
            
            # Store for ODS calculation
            all_predictions_soft.append(pred_soft)
            all_ground_truths.append(gt_resized)
            
            successful += 1
            
            print(f"  ✓ MSE: {basic['mse']:.2f}, PSNR: {basic['psnr']:.2f}, SSIM: {basic['ssim']:.4f}")
            print(f"  ✓ OIS: {ois:.4f}, AP: {ap:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    if successful == 0:
        return {
            'method': method_name,
            'dataset': dataset_name,
            'status': 'failed',
            'successful': 0,
            'failed': failed
        }
    
    # Calculate averages
    avg_mse = np.mean([m['mse'] for m in all_basic_metrics])
    avg_psnr = np.mean([m['psnr'] for m in all_basic_metrics if m['psnr'] != float('inf')])
    avg_ssim = np.mean([m['ssim'] for m in all_basic_metrics])
    avg_ois = np.mean(all_ois)
    avg_ap = np.mean(all_aps)
    
    # ✅ ODS with distance tolerance
    ods, ods_thresh = calculate_ods_with_tolerance(
        all_predictions_soft, all_ground_truths, max_dist=0.0075
    )
    
    print(f"\n  ✓ {method_name} on {dataset_name}: ODS={ods:.4f}, OIS={avg_ois:.4f}, AP={avg_ap:.4f}")
    
    return {
        'method': method_name,
        'dataset': dataset_name,
        'status': 'success',
        'mse': float(avg_mse),
        'psnr': float(avg_psnr),
        'ssim': float(avg_ssim),
        'ods': float(ods),
        'ods_threshold': float(ods_thresh),
        'ois': float(avg_ois),
        'ap': float(avg_ap),
        'successful': successful,
        'failed': failed
    }


def run_all_fuzzy_tests(stare_root=None, biped_root=None, bsds_root=None, 
                        max_images_per_dataset=None, skip_type1=False,
                        save_images=True,miedt_root: str = None):
    """Run all fuzzy methods on all datasets"""
    
    detector = FuzzyEdgeDetectors()
    
    methods = {}
    
    if not skip_type1:
        methods['Type-1 Fuzzy'] = detector.detect_type1_fast
    
    methods['Type-2 Fuzzy'] = detector.detect_type2
    
    if not skip_type1:
        methods['Hybrid-1 Fuzzy'] = detector.detect_hybrid1
        methods['Hybrid-2 Fuzzy'] = detector.detect_hybrid2
    
    datasets = {}
    
    if stare_root and os.path.exists(stare_root):
        pairs = get_stare_pairs(stare_root)
        if pairs:
            datasets['STARE'] = (pairs, False)
            print(f"✓ STARE: {len(pairs)} pairs found")
    
    if biped_root and os.path.exists(biped_root):
        pairs = get_biped_pairs(biped_root)
        if pairs:
            datasets['BIPED'] = (pairs, False)
            print(f"✓ BIPED: {len(pairs)} pairs found")
    
    if bsds_root and os.path.exists(bsds_root):
        pairs = get_bsds500_pairs(bsds_root, 'test')
        if pairs:
            datasets['BSDS500'] = (pairs, True)
            print(f"✓ BSDS500: {len(pairs)} pairs found")

      # MIEDT eklendi
    if miedt_root and os.path.exists(miedt_root):
        print("Loading MIEDT dataset...")
        miedt_pairs = get_miedt_pairs(miedt_root)
        print(f"MIEDT dataset: {len(miedt_pairs)} pairs found")
        if miedt_pairs:
            datasets['MIEDT'] = (miedt_pairs, False)
            print(f"MIEDT dataset: {len(miedt_pairs)} pairs found")
    
    
    if not datasets:
        print("\n❌ ERROR: No datasets found!")
        return []
    
    print(f"\n{'='*70}")
    print(f"CONFIGURATION")
    print(f"{'='*70}")
    print(f"Methods to test: {list(methods.keys())}")
    print(f"Datasets: {list(datasets.keys())}")
    if max_images_per_dataset:
        print(f"Max images per dataset: {max_images_per_dataset}")
    if skip_type1:
        print("⚠️  Type-1 methods SKIPPED (use --include-type1 to enable)")
    print(f"Distance Tolerance: 0.0075 (0.75% of diagonal)")
    print(f"{'='*70}")
    
    all_results = []
    
    for method_name, method_func in methods.items():
        for dataset_name, (pairs, is_mat) in datasets.items():
            result = test_method_on_dataset(
                method_name, method_func, pairs, dataset_name, is_mat,
                max_images=max_images_per_dataset,
                save_images=save_images
            )
            all_results.append(result)
    
    return all_results


def save_results(results: List[Dict], output_file: str = "fuzzy_results_fixed.txt"):
    """Save results to file"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FUZZY EDGE DETECTION - EVALUATION RESULTS (FIXED)\n")
        f.write("="*80 + "\n\n")
        f.write("Methods: Type-1 Fuzzy, Type-2 Fuzzy, Hybrid-1, Hybrid-2\n")
        f.write("Datasets: STARE, BIPED, BSDS500\n")
        f.write("Metrics: MSE, PSNR, SSIM, ODS, OIS, AP\n")
        f.write("Distance Tolerance: 0.0075 (0.75% of diagonal)\n\n")
        f.write("="*80 + "\n\n")
        
        methods = {}
        for result in results:
            method = result['method']
            if method not in methods:
                methods[method] = []
            methods[method].append(result)
        
        for method_name, method_results in methods.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"{method_name}\n")
            f.write(f"{'='*80}\n\n")
            
            for result in method_results:
                f.write(f"Dataset: {result['dataset']}\n")
                f.write(f"{'-'*40}\n")
                
                if result['status'] == 'failed':
                    f.write("Status: Failed\n")
                    f.write(f"Failed: {result['failed']}\n")
                else:
                    f.write(f"Successful: {result['successful']}, Failed: {result['failed']}\n")
                    f.write(f"MSE   : {result['mse']:.6f}\n")
                    f.write(f"PSNR  : {result['psnr']:.4f} dB\n")
                    f.write(f"SSIM  : {result['ssim']:.6f}\n")
                    f.write(f"ODS   : {result['ods']:.6f} @ threshold={result['ods_threshold']:.3f}\n")
                    f.write(f"OIS   : {result['ois']:.6f}\n")
                    f.write(f"AP    : {result['ap']:.6f}\n")
                f.write("\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("SUMMARY TABLE\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Method':<20} {'Dataset':<10} {'MSE':<12} {'PSNR':<10} {'SSIM':<10} {'ODS':<10} {'OIS':<10} {'AP':<10}\n")
        f.write("-"*100 + "\n")
        
        for result in results:
            if result['status'] == 'success':
                f.write(f"{result['method']:<20} {result['dataset']:<10} "
                       f"{result['mse']:<12.6f} {result['psnr']:<10.4f} "
                       f"{result['ssim']:<10.6f} {result['ods']:<10.6f} "
                       f"{result['ois']:<10.6f} {result['ap']:<10.6f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fuzzy Edge Detection Evaluation (FIXED)')
    parser.add_argument('--stare', type=str, help='Path to STARE dataset')
    parser.add_argument('--biped', type=str, help='Path to BIPED dataset')
    parser.add_argument('--bsds', type=str, help='Path to BSDS500 dataset')
    parser.add_argument('--max-images', type=int, help='Max images per dataset')
    parser.add_argument('--include-type1', action='store_true', help='Include Type-1 methods (VERY SLOW!)')
    parser.add_argument('--output', type=str, default='fuzzy_results_fixed.txt', help='Output file')
    
    args = parser.parse_args()
    
    STARE_ROOT = args.stare or r"stare"
    BIPED_ROOT = args.biped or r"biped_dataset/BIPED/BIPED/edges"
    BSDS_ROOT = args.bsds or r"bsds500"
    MIEDT_ROOT = r"MIEDT"
    
    print("\n" + "="*80)
    print("FUZZY EDGE DETECTION EVALUATION (FIXED WITH DISTANCE TOLERANCE)")
    print("="*80)
    
    results = run_all_fuzzy_tests(
        stare_root=STARE_ROOT,
        biped_root=BIPED_ROOT,
        bsds_root=BSDS_ROOT,
        miedt_root=MIEDT_ROOT,
        max_images_per_dataset=args.max_images,
        skip_type1=not args.include_type1
    )
    
    if results:
        save_results(results, args.output)
        
        json_file = args.output.replace('.txt', '.json')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"✓ JSON saved to: {json_file}")
        
        print("\n" + "="*80)
        print("✓ EVALUATION COMPLETED!")
        print("="*80)
    else:
        print("\n❌ No results generated!")