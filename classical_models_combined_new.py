"""
Classical Edge Detection Methods Evaluation - FIXED VERSION
Distance tolerance + proper ODS/OIS calculation + MIEDT support
"""

import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm
import json
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import scipy.io as sio
from typing import Tuple, Dict, List
from sklearn.metrics import average_precision_score

# ==================================================
# CLASSICAL EDGE DETECTORS
# ==================================================

class ClassicalEdgeDetectors:
    """Classical edge detection methods"""
    
    @staticmethod
    def sobel_edge_detection(image: np.ndarray, return_soft: bool = False) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        gray = gray.astype(np.float64)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
        
        Gx = ndimage.convolve(gray, sobel_x)
        Gy = ndimage.convolve(gray, sobel_y)
        G = np.sqrt(Gx**2 + Gy**2)
        
        if return_soft:
            G_norm = (G / (G.max() + 1e-8) * 255).astype(np.uint8)
            return G_norm
        else:
            threshold = np.mean(G) + 2 * np.std(G)
            binary = (G > threshold).astype(np.uint8) * 255
            return binary
    
    @staticmethod
    def prewitt_edge_detection(image: np.ndarray, return_soft: bool = False) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float64)
        
        Gx = ndimage.convolve(gray.astype(np.float64), prewitt_x)
        Gy = ndimage.convolve(gray.astype(np.float64), prewitt_y)
        G = np.sqrt(Gx**2 + Gy**2)
        
        if return_soft:
            G_norm = (G / (G.max() + 1e-8) * 255).astype(np.uint8)
            return G_norm
        else:
            G_norm = (G / (G.max() + 1e-8) * 255).astype(np.uint8)
            _, binary = cv2.threshold(G_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
    
    @staticmethod
    def roberts_edge_detection(image: np.ndarray, return_soft: bool = False) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
        roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)
        
        Gx = ndimage.convolve(gray.astype(np.float64), roberts_x)
        Gy = ndimage.convolve(gray.astype(np.float64), roberts_y)
        G = np.sqrt(Gx**2 + Gy**2)
        
        if return_soft:
            G_norm = (G / (G.max() + 1e-8) * 255).astype(np.uint8)
            return G_norm
        else:
            G_norm = (G / (G.max() + 1e-8) * 255).astype(np.uint8)
            _, binary = cv2.threshold(G_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
    
    @staticmethod
    def canny_edge_detection(image: np.ndarray, return_soft: bool = False) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        if return_soft:
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            magnitude_norm = (magnitude / (magnitude.max() + 1e-8) * 255).astype(np.uint8)
            return magnitude_norm
        else:
            edges = cv2.Canny(gray, 50, 150)
            return edges


# ==================================================
# DISTANCE TOLERANCE FUNCTIONS
# ==================================================

def normalize(img):
    """Normalize to 0-1 range"""
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def compute_f_score(precision, recall):
    """F-score hesapla"""
    with np.errstate(divide='ignore', invalid='ignore'):
        f_score = 2 * (precision * recall) / (precision + recall)
        f_score = np.nan_to_num(f_score)
    return f_score


def correspondence_with_tolerance(pred_binary, gt_binary, max_dist=0.0075):
    """
    Distance tolerance ile TP/FP/FN hesapla
    
    Args:
        pred_binary: Binary prediction (0 or 1)
        gt_binary: Binary ground truth (0 or 1)
        max_dist: Maximum distance tolerance (default 0.0075)
    
    Returns:
        tp, fp, fn: True Positives, False Positives, False Negatives
    """
    # Diagonal hesapla
    diag = np.sqrt(pred_binary.shape[0]**2 + pred_binary.shape[1]**2)
    max_dist_px = max_dist * diag
    
    # Distance transform
    if np.sum(gt_binary) > 0:
        dist_gt = distance_transform_edt(1 - gt_binary)
    else:
        dist_gt = np.ones_like(gt_binary) * np.inf
        
    if np.sum(pred_binary) > 0:
        dist_pred = distance_transform_edt(1 - pred_binary)
    else:
        dist_pred = np.ones_like(pred_binary) * np.inf
    
    # True Positives: predicted edges close to GT edges
    tp = np.sum((pred_binary == 1) & (dist_gt <= max_dist_px))
    
    # False Positives: predicted edges far from GT edges
    fp = np.sum((pred_binary == 1) & (dist_gt > max_dist_px))
    
    # False Negatives: GT edges not matched by prediction
    fn = np.sum((gt_binary == 1) & (dist_pred > max_dist_px))
    
    return tp, fp, fn


def calculate_ois_with_tolerance(pred, gt, thresholds=99, max_dist=0.0075):
    """
    OIS hesapla - distance tolerance ile
    
    Args:
        pred: Continuous prediction [0, 1]
        gt: Ground truth [0, 1]
        thresholds: Number of thresholds to test
        max_dist: Distance tolerance
    
    Returns:
        best_f, best_thresh, ap
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
    ODS hesapla - distance tolerance ile
    
    Args:
        all_predictions: List of continuous predictions
        all_ground_truths: List of ground truths
        thresholds: Number of thresholds to test
        max_dist: Distance tolerance
    
    Returns:
        best_f, best_thresh
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
# BASIC METRICS (MSE, PSNR, SSIM)
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
        max_pixel = 255.0
        psnr_val = 10 * np.log10((max_pixel ** 2) / mse_val)
    
    ssim_val = ssim(gt, pred, data_range=255)
    
    return {
        'mse': float(mse_val),
        'psnr': float(psnr_val),
        'ssim': float(ssim_val)
    }


# ==================================================
# DATASET LOADERS
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
    """Get STARE dataset pairs"""
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
        base = gt_name.replace(".vk", "")
        base = os.path.splitext(base)[0]
        
        if base in raw_dict:
            pairs.append((raw_dict[base], gt_path))
    
    return pairs


def get_biped_pairs(biped_root: str) -> List[Tuple[str, str]]:
    """Get BIPED dataset pairs"""
    test_list_file = os.path.join(biped_root, 'test_rgb.lst')
    
    if not os.path.exists(test_list_file):
        return []
    
    pairs = []
    with open(test_list_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
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
    """Get BSDS500 dataset pairs"""
    images_dir = os.path.join(bsds_root, "images", split)
    gt_dir = os.path.join(bsds_root, "ground_truth", split)
    
    if not os.path.exists(images_dir) or not os.path.exists(gt_dir):
        return []
    
    image_files = glob(os.path.join(images_dir, "*.jpg"))
    if not image_files:
        image_files = glob(os.path.join(images_dir, "*.png"))
    
    pairs = []
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        mat_path = os.path.join(gt_dir, f"{base_name}.mat")
        
        if os.path.exists(mat_path):
            pairs.append((img_path, mat_path))
    
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


# ==================================================
# TESTING FUNCTION
# ==================================================

def test_method_on_dataset(method_name: str, method_func, pairs: List[Tuple[str, str]], 
                           dataset_name: str, is_mat_gt: bool = False) -> Dict:
    """Test a single method on a dataset with distance tolerance"""
    
    print(f"\nTesting {method_name} on {dataset_name}...")
    
    all_basic_metrics = []
    all_ois = []
    all_aps = []
    all_predictions_soft = []
    all_ground_truths = []
    
    successful = 0
    failed = 0
    
    for img_path, gt_path in tqdm(pairs, desc=f"{method_name} on {dataset_name}"):
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                failed += 1
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load ground truth
            if is_mat_gt:
                gt = read_mat_ground_truth(gt_path)
                if gt is None:
                    failed += 1
                    continue
                gt = (normalize(gt) * 255).astype(np.uint8)
            else:
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    failed += 1
                    continue
                
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
            
            # Basic metrics on binary edges
            basic = calculate_basic_metrics(pred_binary, gt_resized)
            all_basic_metrics.append(basic)
            
            # OIS + AP with distance tolerance
            ois, _, ap, _ = calculate_ois_with_tolerance(pred_soft, gt_resized, max_dist=0.0075)
            all_ois.append(ois)
            all_aps.append(ap)
            
            # Store for ODS calculation
            all_predictions_soft.append(pred_soft)
            all_ground_truths.append(gt_resized)
            
            successful += 1
            
        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {e}")
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
    
    # ODS with distance tolerance
    ods, ods_thresh = calculate_ods_with_tolerance(
        all_predictions_soft, all_ground_truths, max_dist=0.0075
    )
    
    print(f"  ✓ {method_name} on {dataset_name}: ODS={ods:.4f}, OIS={avg_ois:.4f}, AP={avg_ap:.4f}")
    
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


# ==================================================
# RUN ALL TESTS
# ==================================================

def run_all_tests(stare_root: str = None, biped_root: str = None, 
                  bsds_root: str = None, miedt_root: str = None) -> Dict:
    """Run all methods on all available datasets"""
    
    detector = ClassicalEdgeDetectors()
    
    methods = {
        'Sobel': detector.sobel_edge_detection,
        'Prewitt': detector.prewitt_edge_detection,
        'Roberts': detector.roberts_edge_detection,
        'Canny': detector.canny_edge_detection
    }
    
    datasets = {}
    
    if stare_root and os.path.exists(stare_root):
        stare_pairs = get_stare_pairs(stare_root)
        if stare_pairs:
            datasets['STARE'] = (stare_pairs, False)
            print(f"STARE dataset: {len(stare_pairs)} pairs found")
   
    if biped_root and os.path.exists(biped_root):
        biped_pairs = get_biped_pairs(biped_root)
        if biped_pairs:
            datasets['BIPED'] = (biped_pairs, False)
            print(f"BIPED dataset: {len(biped_pairs)} pairs found")
    
    if bsds_root and os.path.exists(bsds_root):
        bsds_pairs = get_bsds500_pairs(bsds_root, 'test')
        if bsds_pairs:
            datasets['BSDS500'] = (bsds_pairs, True)
            print(f"BSDS500 dataset: {len(bsds_pairs)} pairs found")
    
    # MIEDT eklendi
    if miedt_root and os.path.exists(miedt_root):
        print("Loading MIEDT dataset...")
        miedt_pairs = get_miedt_pairs(miedt_root)
        print(f"MIEDT dataset: {len(miedt_pairs)} pairs found")
        if miedt_pairs:
            datasets['MIEDT'] = (miedt_pairs, False)
            print(f"MIEDT dataset: {len(miedt_pairs)} pairs found")
    
    if not datasets:
        print("ERROR: No datasets found!")
        return {}
    
    all_results = []
    
    for method_name, method_func in methods.items():
        print(f"\n{'='*60}")
        print(f"Testing {method_name}")
        print(f"{'='*60}")
        
        for dataset_name, (pairs, is_mat) in datasets.items():
            result = test_method_on_dataset(method_name, method_func, pairs, dataset_name, is_mat)
            all_results.append(result)
            print(all_results)
    
    return all_results


def save_results(results: List[Dict], output_file: str = "results_fixed.txt"):
    """Save results to text file"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CLASSICAL EDGE DETECTION - EVALUATION RESULTS (WITH MIEDT)\n")
        f.write("="*80 + "\n\n")
        f.write("Methods: Sobel, Prewitt, Roberts, Canny\n")
        f.write("Datasets: STARE, BIPED, BSDS500, MIEDT\n")
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
            f.write(f"{method_name} EDGE DETECTION\n")
            f.write(f"{'='*80}\n\n")
            
            for result in method_results:
                f.write(f"Dataset: {result['dataset']}\n")
                f.write(f"{'-'*40}\n")
                
                if result['status'] == 'failed':
                    f.write("Status: Failed\n")
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
        f.write(f"{'Method':<12} {'Dataset':<15} {'MSE':<12} {'PSNR':<10} {'SSIM':<10} {'ODS':<10} {'OIS':<10} {'AP':<10}\n")
        f.write("-"*95 + "\n")
        
        for result in results:
            if result['status'] == 'failed':
                f.write(f"{result['method']:<12} {result['dataset']:<15} {'-':<12} {'-':<10} {'-':<10} {'-':<10} {'-':<10} {'-':<10}\n")
            else:
                f.write(f"{result['method']:<12} {result['dataset']:<15} {result['mse']:<12.6f} {result['psnr']:<10.4f} {result['ssim']:<10.6f} {result['ods']:<10.6f} {result['ois']:<10.6f} {result['ap']:<10.6f}\n")
        
        f.write("\n" + "="*95 + "\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    STARE_ROOT = r"stare"
    BIPED_ROOT = r"biped_dataset/BIPED/BIPED/edges"
    BSDS_ROOT = r"bsds500"
    MIEDT_ROOT = r"MIEDT"  # Bu doğru - MIEDT klasörünün içinde Original/ ve Ground Truth/ olmalı
    
    print("="*80)
    print("CLASSICAL EDGE DETECTION EVALUATION (WITH MIEDT)")
    print("="*80)
    
    results = run_all_tests(
        stare_root=STARE_ROOT,
        biped_root=BIPED_ROOT,
        bsds_root=BSDS_ROOT,
        miedt_root=MIEDT_ROOT
    )
