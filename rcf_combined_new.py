#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RCF Test Script - UNIFIED VERSION
Tests on BSDS500, STARE, BIPED, and MIEDT datasets
All with distance tolerance and proper metric calculations
"""

import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import scipy.io as sio
import cv2
import argparse
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import average_precision_score
from scipy.ndimage import distance_transform_edt
from models import RCF
import re


# ============================================================================
# MODEL LOADING AND INFERENCE
# ============================================================================

def load_model(model_path, device='cuda'):
    """RCF modelini yükle"""
    model = RCF().to(device)
    if os.path.isfile(model_path):
        print(f"=> Loading checkpoint from '{model_path}'")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("=> Checkpoint loaded successfully")
    else:
        print(f"=> ERROR: No checkpoint found at '{model_path}'")
        return None
    model.eval()
    return model


def preprocess_image(img_path, device='cuda'):
    """Görüntüyü RCF için hazırla"""
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array.transpose((2, 0, 1))  # HWC -> CHW
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).to(device)
    return img_tensor, img.size


def infer_edges(model, img_tensor, normalize=False):
    """RCF ile kenar tespiti"""
    with torch.no_grad():
        results = model(img_tensor)
        fuse_res = torch.squeeze(results[-1]).cpu().numpy()
    
    # MIEDT için normalization gerekiyor
    if normalize:
        fuse_min = fuse_res.min()
        fuse_max = fuse_res.max()
        
        if fuse_max - fuse_min > 1e-8:
            fuse_res = (fuse_res - fuse_min) / (fuse_max - fuse_min)
        else:
            fuse_res = np.zeros_like(fuse_res)
    
    return fuse_res


# ============================================================================
# DATASET-SPECIFIC FUNCTIONS
# ============================================================================

def get_bsds500_pairs(bsds_root, split='test'):
    """BSDS500 test çiftlerini bul"""
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


def read_mat_ground_truth(mat_path):
    """BSDS500 .mat dosyasından ground truth oku"""
    try:
        mat = sio.loadmat(mat_path)
        
        if 'groundTruth' in mat:
            gt_cell = mat['groundTruth']
            
            if gt_cell.size > 0:
                gt_struct = gt_cell[0, 0]
                
                if 'Boundaries' in gt_struct.dtype.names:
                    boundaries = gt_struct['Boundaries'][0, 0]
                    boundaries = np.array(boundaries, dtype=np.float32)
                    return boundaries
                
                elif 'Segmentation' in gt_struct.dtype.names:
                    seg = gt_struct['Segmentation'][0, 0]
                    seg = np.array(seg, dtype=np.float32)
                    
                    seg_uint8 = (seg / seg.max() * 255).astype(np.uint8) if seg.max() > 0 else seg.astype(np.uint8)
                    boundaries = cv2.Canny(seg_uint8, 50, 150)
                    boundaries = boundaries.astype(np.float32) / 255.0
                    return boundaries
        
        return None
        
    except Exception as e:
        print(f"Error reading {mat_path}: {e}")
        return None


def read_all_ground_truths(mat_path):
    """BSDS500 .mat dosyasından TÜM ground truth annotasyonlarını oku"""
    try:
        mat = sio.loadmat(mat_path)
        boundaries_list = []
        
        if 'groundTruth' in mat:
            gt_cell = mat['groundTruth']
            
            for i in range(gt_cell.shape[1]):
                gt_struct = gt_cell[0, i]
                
                if 'Boundaries' in gt_struct.dtype.names:
                    boundaries = gt_struct['Boundaries'][0, 0]
                    boundaries = np.array(boundaries, dtype=np.float32)
                    boundaries_list.append(boundaries)
                elif 'Segmentation' in gt_struct.dtype.names:
                    seg = gt_struct['Segmentation'][0, 0]
                    seg = np.array(seg, dtype=np.float32)
                    
                    seg_uint8 = (seg / seg.max() * 255).astype(np.uint8) if seg.max() > 0 else seg.astype(np.uint8)
                    boundaries = cv2.Canny(seg_uint8, 50, 150)
                    boundaries = boundaries.astype(np.float32) / 255.0
                    boundaries_list.append(boundaries)
        
        return boundaries_list if boundaries_list else None
        
    except Exception as e:
        return None


def get_stare_pairs(stare_root):
    """STARE test çiftlerini bul"""
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


def get_biped_pairs(biped_root):
    """BIPED test çiftlerini bul"""
    test_list_file = os.path.join(biped_root, 'test_rgb.lst')
    
    if not os.path.exists(test_list_file):
        return []
    
    pairs = []
    
    with open(test_list_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
            
        img_rel, gt_rel = parts[0], parts[1]
        
        img_path = os.path.join(biped_root, 'imgs', 'test', img_rel)
        gt_path = os.path.join(biped_root, 'edge_maps', 'test', gt_rel)
        
        if os.path.exists(img_path) and os.path.exists(gt_path):
            pairs.append((img_path, gt_path))
    
    return pairs


def get_miedt_pairs(miedt_root):
    """MIEDT test çiftlerini bul"""
    images_dir = os.path.join(miedt_root, "ct_brain_original")
    gt_dir = os.path.join(miedt_root, "ct_brain_ground_truth")
    
    if not os.path.exists(images_dir) or not os.path.exists(gt_dir):
        print(f"Images or GT directory not found in {miedt_root}")
        return []
    
    image_extensions = ['*.jpg', '*.png', '*.tif', '*.tiff', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(images_dir, ext)))
    
    pairs = []
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        gt_base_name = re.sub('IMG', 'GT', base_name)
        
        gt_found = False
        for ext in ['.png', '.jpg', '.tif', '.tiff', '.bmp']:
            gt_path = os.path.join(gt_dir, gt_base_name + ext)
            
            if os.path.exists(gt_path):
                pairs.append((img_path, gt_path))
                gt_found = True
                break
        
        if not gt_found:
            print(f"Warning: No GT found for {img_name}")
    
    return pairs


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_to_01(img):
    """Normalize array to [0, 1] range"""
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()
    
    if img_max - img_min > 0:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
    
    return img


def compute_f_score(precision, recall):
    """F-score hesapla - SAFE VERSION"""
    if precision == 0 and recall == 0:
        return 0.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        f_score = 2 * (precision * recall) / (precision + recall)
        if np.isnan(f_score) or np.isinf(f_score):
            f_score = 0.0
    
    return float(f_score)


# ============================================================================
# METRIC COMPUTATION FUNCTIONS
# ============================================================================

def calculate_basic_metrics(pred, gt):
    """
    Calculate MSE, PSNR, SSIM on binary images (0/255)
    Classical models ile aynı yaklaşım
    """
    # Boyutları eşitle
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    
    # uint8'e dönüştür (0-255 aralığında)
    if pred.dtype != np.uint8:
        pred = (normalize_to_01(pred) * 255).astype(np.uint8)
    if gt.dtype != np.uint8:
        gt = (normalize_to_01(gt) * 255).astype(np.uint8)
    
    # MSE hesapla
    mse_val = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    
    # PSNR hesapla
    if mse_val == 0:
        psnr_val = float('inf')
    else:
        max_pixel = 255.0
        psnr_val = 10 * np.log10((max_pixel ** 2) / mse_val)
    
    # SSIM hesapla (data_range=255)
    ssim_val = ssim(gt, pred, data_range=255)
    
    return {
        'mse': float(mse_val),
        'psnr': float(psnr_val),
        'ssim': float(ssim_val)
    }


def correspondence_with_tolerance(pred_binary, gt_binary, max_dist=0.0075):
    """Distance tolerance ile TP/FP/FN hesapla"""
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
    
    return int(tp), int(fp), int(fn)


def compute_ois_with_tolerance(pred, gt_list, thresholds=99, max_dist=0.0075):
    """OIS hesapla - distance tolerance ile"""
    pred = np.clip(pred, 0, 1)
    
    # gt_list tek elemanlı liste olabilir
    if not isinstance(gt_list, list):
        gt_list = [gt_list]
    
    best_f_overall = 0.0
    best_thresh_overall = 0.0
    
    for gt in gt_list:
        gt = normalize_to_01(gt)
        gt_binary = (gt > 0.5).astype(np.uint8)
        
        threshold_values = np.linspace(0, 1, thresholds)
        f_scores = []
        
        for thresh in threshold_values:
            pred_binary = (pred >= thresh).astype(np.uint8)
            
            tp, fp, fn = correspondence_with_tolerance(pred_binary, gt_binary, max_dist)
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f = compute_f_score(precision, recall)
            
            f_scores.append(f)
        
        f_scores = np.array(f_scores)
        best_idx = np.argmax(f_scores)
        best_f = f_scores[best_idx]
        
        if best_f > best_f_overall:
            best_f_overall = best_f
            best_thresh_overall = threshold_values[best_idx]
    
    return float(best_f_overall), float(best_thresh_overall)


def compute_ods_with_tolerance(all_predictions, all_ground_truths_list, thresholds=99, max_dist=0.0075):
    """ODS hesapla - distance tolerance ile"""
    threshold_values = np.linspace(0, 1, thresholds)
    dataset_f_scores = []
    
    for thresh in threshold_values:
        total_tp, total_fp, total_fn = 0, 0, 0
        
        for pred, gt_list in zip(all_predictions, all_ground_truths_list):
            pred = np.clip(pred, 0, 1)
            
            # gt_list tek elemanlı liste olabilir
            if not isinstance(gt_list, list):
                gt_list = [gt_list]
            
            best_tp, best_fp, best_fn = 0, 0, 0
            best_f = 0.0
            
            for gt in gt_list:
                gt = normalize_to_01(gt)
                pred_binary = (pred >= thresh).astype(np.uint8)
                gt_binary = (gt > 0.5).astype(np.uint8)
                
                tp, fp, fn = correspondence_with_tolerance(pred_binary, gt_binary, max_dist)
                
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f = compute_f_score(precision, recall)
                
                if f > best_f:
                    best_f = f
                    best_tp, best_fp, best_fn = tp, fp, fn
            
            total_tp += best_tp
            total_fp += best_fp
            total_fn += best_fn
        
        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f = compute_f_score(precision, recall)
        dataset_f_scores.append(f)
    
    dataset_f_scores = np.array(dataset_f_scores)
    
    if len(dataset_f_scores) > 0:
        best_idx = np.argmax(dataset_f_scores)
        return float(dataset_f_scores[best_idx]), float(threshold_values[best_idx])
    else:
        return 0.0, 0.0


def compute_ap(pred, gt):
    """AP hesapla"""
    pred = np.clip(pred, 0, 1)
    gt = normalize_to_01(gt)
    
    pred_flat = pred.flatten()
    gt_flat = (gt > 0.5).astype(int).flatten()
    
    if np.sum(gt_flat) == 0:
        return 0.0
    
    try:
        ap = average_precision_score(gt_flat, pred_flat)
        return float(ap)
    except:
        return 0.0


# ============================================================================
# DATASET TEST FUNCTIONS
# ============================================================================

def test_bsds500(model, bsds_root, split='test', device='cuda'):
    """BSDS500 test"""
    pairs = get_bsds500_pairs(bsds_root, split)
    
    if len(pairs) == 0:
        print(f"No BSDS500 test pairs found in {bsds_root}")
        return None
    
    print(f"Found {len(pairs)} BSDS500 test pairs")
    
    mse_values, psnr_values, ssim_values = [], [], []
    ois_values, ap_values = [], []
    all_predictions = []
    all_ground_truths_list = []
    
    successful_count = 0
    failed_count = 0
    
    for img_path, gt_path in tqdm(pairs, desc=f"BSDS500 {split}"):
        try:
            img_tensor, original_size = preprocess_image(img_path, device=device)
            pred = infer_edges(model, img_tensor)
            pred = np.clip(pred, 0, 1)
            
            gt = read_mat_ground_truth(gt_path)
            gt_list = read_all_ground_truths(gt_path)
            
            if gt is None or gt_list is None:
                failed_count += 1
                continue
            
            gt = normalize_to_01(gt)
            
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
            
            gt_list_resized = []
            for gt_single in gt_list:
                gt_single = normalize_to_01(gt_single)
                if gt_single.shape != pred.shape:
                    gt_single = cv2.resize(gt_single, (pred.shape[1], pred.shape[0]))
                gt_list_resized.append(gt_single)
            
            # Binary edge maps oluştur (0-255)
            pred_binary = (pred > 0.5).astype(np.uint8) * 255
            gt_binary = (gt > 0.5).astype(np.uint8) * 255
            
            # Basic metrics hesapla
            basic = calculate_basic_metrics(pred_binary, gt_binary)
            mse_values.append(basic['mse'])
            psnr_values.append(basic['psnr'])
            ssim_values.append(basic['ssim'])
            
            ois_f, _ = compute_ois_with_tolerance(pred, gt_list_resized, max_dist=0.0075)
            ois_values.append(ois_f)
            
            ap = compute_ap(pred, gt_list_resized[0])
            ap_values.append(ap)
            
            all_predictions.append(pred)
            all_ground_truths_list.append(gt_list_resized)
            
            successful_count += 1
            
        except Exception as e:
            print(f"\nError processing {os.path.basename(img_path)}: {e}")
            failed_count += 1
            continue
    
    if len(mse_values) > 0:
        # PSNR için inf değerleri filtrele
        psnr_filtered = [p for p in psnr_values if p != float('inf')]
        avg_psnr = float(np.mean(psnr_filtered)) if psnr_filtered else 100.0
        
        results = {
            "dataset": "BSDS500",
            "split": split,
            "total_images": len(mse_values),
            "successful": successful_count,
            "failed": failed_count,
            "mse": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values)),
            "psnr": avg_psnr,
            "psnr_std": float(np.std(psnr_filtered)) if psnr_filtered else 0.0,
            "ssim": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values)),
        }
        
        # if len(ois_values) > 0:
        #     results["ois"] = float(np.mean(ois_values))
        #     results["ois_std"] = float(np.std(ois_values))
        
        # if len(ap_values) > 0:
        #     results["ap"] = float(np.mean(ap_values))
        #     results["ap_std"] = float(np.std(ap_values))
        
        # if len(all_predictions) > 0:
        #     print("\nCalculating ODS (this may take a while)...")
        #     ods_f, ods_threshold = compute_ods_with_tolerance(
        #         all_predictions, all_ground_truths_list, max_dist=0.0075
        #     )
        #     results["ods"] = float(ods_f)
        #     results["ods_threshold"] = float(ods_threshold)
        
        return results
    
    return None


def test_stare(model, stare_root, device='cuda'):
    """STARE test"""
    pairs = get_stare_pairs(stare_root)
    
    if len(pairs) == 0:
        print(f"No STARE test pairs found in {stare_root}")
        return None
    
    print(f"Found {len(pairs)} STARE test pairs")
    
    mse_values, psnr_values, ssim_values = [], [], []
    ois_values, ap_values = [], []
    all_predictions, all_ground_truths = [], []
    
    successful_count = 0
    failed_count = 0
    
    for img_path, gt_path in tqdm(pairs, desc="STARE"):
        try:
            img_tensor, original_size = preprocess_image(img_path, device=device)
            pred = infer_edges(model, img_tensor)
            pred = np.clip(pred, 0, 1)
            
            gt = Image.open(gt_path).convert('L')
            gt_array = np.array(gt).astype(np.float32) / 255.0
            
            if pred.shape != gt_array.shape:
                pred = cv2.resize(pred, (gt_array.shape[1], gt_array.shape[0]))
            
            # Binary edge maps (0-255)
            pred_binary = (pred > 0.5).astype(np.uint8) * 255
            gt_binary = (gt_array > 0.5).astype(np.uint8) * 255
            
            # Basic metrics
            basic = calculate_basic_metrics(pred_binary, gt_binary)
            mse_values.append(basic['mse'])
            psnr_values.append(basic['psnr'])
            ssim_values.append(basic['ssim'])
            
            ois_f, _ = compute_ois_with_tolerance(pred, gt_array, max_dist=0.0075)
            ois_values.append(ois_f)
            
            ap = compute_ap(pred, gt_array)
            ap_values.append(ap)
            
            all_predictions.append(pred)
            all_ground_truths.append(gt_array)
            
            successful_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            failed_count += 1
            continue
    
    if len(mse_values) > 0:
        psnr_filtered = [p for p in psnr_values if p != float('inf')]
        avg_psnr = float(np.mean(psnr_filtered)) if psnr_filtered else 100.0
        
        results = {
            "dataset": "STARE",
            "total_images": len(mse_values),
            "successful": successful_count,
            "failed": failed_count,
            "mse": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values)),
            "psnr": avg_psnr,
            "psnr_std": float(np.std(psnr_filtered)) if psnr_filtered else 0.0,
            "ssim": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values)),
        }
        
        if len(ois_values) > 0:
            results["ois"] = float(np.mean(ois_values))
            results["ois_std"] = float(np.std(ois_values))
        
        if len(ap_values) > 0:
            results["ap"] = float(np.mean(ap_values))
            results["ap_std"] = float(np.std(ap_values))
        
        # if len(all_predictions) > 0:
        #     ods_f, ods_threshold = compute_ods_with_tolerance(
        #         all_predictions, all_ground_truths, max_dist=0.0075
        #     )
        #     results["ods"] = float(ods_f)
        #     results["ods_threshold"] = float(ods_threshold)
        
        return results
    
    return None


def test_biped(model, biped_root, device='cuda'):
    """BIPED test"""
    pairs = get_biped_pairs(biped_root)
    
    if len(pairs) == 0:
        print(f"No BIPED test pairs found in {biped_root}")
        return None
    
    print(f"Found {len(pairs)} BIPED test pairs")
    
    mse_values, psnr_values, ssim_values = [], [], []
    ois_values, ap_values = [], []
    all_predictions, all_ground_truths = [], []
    
    successful_count = 0
    failed_count = 0
    
    for img_path, gt_path in tqdm(pairs, desc="BIPED"):
        try:
            img_tensor, original_size = preprocess_image(img_path, device=device)
            pred = infer_edges(model, img_tensor)
            pred = np.clip(pred, 0, 1)
            
            gt = Image.open(gt_path).convert('L')
            gt_array = np.array(gt).astype(np.float32) / 255.0
            
            if pred.shape != gt_array.shape:
                pred = cv2.resize(pred, (gt_array.shape[1], gt_array.shape[0]))
            
            # Binary edge maps (0-255)
            pred_binary = (pred > 0.5).astype(np.uint8) * 255
            gt_binary = (gt_array > 0.5).astype(np.uint8) * 255
            
            # Basic metrics
            basic = calculate_basic_metrics(pred_binary, gt_binary)
            mse_values.append(basic['mse'])
            psnr_values.append(basic['psnr'])
            ssim_values.append(basic['ssim'])
            
            ois_f, _ = compute_ois_with_tolerance(pred, gt_array, max_dist=0.0075)
            ois_values.append(ois_f)
            
            ap = compute_ap(pred, gt_array)
            ap_values.append(ap)
            
            all_predictions.append(pred)
            all_ground_truths.append(gt_array)
            
            successful_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            failed_count += 1
            continue
    
    if len(mse_values) > 0:
        psnr_filtered = [p for p in psnr_values if p != float('inf')]
        avg_psnr = float(np.mean(psnr_filtered)) if psnr_filtered else 100.0
        
        results = {
            "dataset": "BIPED",
            "total_images": len(mse_values),
            "successful": successful_count,
            "failed": failed_count,
            "mse": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values)),
            "psnr": avg_psnr,
            "psnr_std": float(np.std(psnr_filtered)) if psnr_filtered else 0.0,
            "ssim": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values)),
        }
        
        if len(ois_values) > 0:
            results["ois"] = float(np.mean(ois_values))
            results["ois_std"] = float(np.std(ois_values))
        
        if len(ap_values) > 0:
            results["ap"] = float(np.mean(ap_values))
            results["ap_std"] = float(np.std(ap_values))
        
        if len(all_predictions) > 0:
            ods_f, ods_threshold = compute_ods_with_tolerance(
                all_predictions, all_ground_truths, max_dist=0.0075
            )
            results["ods"] = float(ods_f)
            results["ods_threshold"] = float(ods_threshold)
        
        return results
    
    return None


def test_miedt(model, miedt_root, device='cuda'):
    """MIEDT test"""
    pairs = get_miedt_pairs(miedt_root)
    
    if len(pairs) == 0:
        print(f"No MIEDT test pairs found in {miedt_root}")
        return None
    
    print(f"Found {len(pairs)} MIEDT test pairs")
    
    mse_values, psnr_values, ssim_values = [], [], []
    ois_values, ap_values = [], []
    all_predictions, all_ground_truths = [], []
    
    successful_count = 0
    failed_count = 0
    
    for img_path, gt_path in tqdm(pairs, desc="MIEDT"):
        try:
            img_tensor, _ = preprocess_image(img_path, device=device)
            pred = infer_edges(model, img_tensor, normalize=True)
            pred = np.clip(pred, 0, 1)
            
            gt = Image.open(gt_path).convert('L')
            gt_array = np.array(gt).astype(np.float32) / 255.0
            gt_binary_01 = (gt_array > 0.5).astype(np.float32)
            
            if pred.shape != gt_binary_01.shape:
                pred = cv2.resize(pred, (gt_binary_01.shape[1], gt_binary_01.shape[0]))
            
            # Binary edge maps (0-255)
            pred_binary = (pred > 0.5).astype(np.uint8) * 255
            gt_binary = (gt_binary_01 > 0.5).astype(np.uint8) * 255
            
            # Basic metrics
            basic = calculate_basic_metrics(pred_binary, gt_binary)
            mse_values.append(basic['mse'])
            psnr_values.append(basic['psnr'])
            ssim_values.append(basic['ssim'])
            
            ois_f, _ = compute_ois_with_tolerance(pred, gt_binary_01, max_dist=0.0075)
            ois_values.append(ois_f)
            
            ap = compute_ap(pred, gt_binary_01)
            ap_values.append(ap)
            
            all_predictions.append(pred)
            all_ground_truths.append(gt_binary_01)
            
            successful_count += 1
            
        except Exception as e:
            print(f"\nError processing {os.path.basename(img_path)}: {e}")
            failed_count += 1
            continue
    
    if len(mse_values) > 0:
        psnr_filtered = [p for p in psnr_values if p != float('inf')]
        avg_psnr = float(np.mean(psnr_filtered)) if psnr_filtered else 100.0
        
        results = {
            "dataset": "MIEDT",
            "total_images": len(mse_values),
            "successful": successful_count,
            "failed": failed_count,
            "mse": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values)),
            "psnr": avg_psnr,
            "psnr_std": float(np.std(psnr_filtered)) if psnr_filtered else 0.0,
            "ssim": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values)),
        }
        
        if len(ois_values) > 0:
            results["ois"] = float(np.mean(ois_values))
            results["ois_std"] = float(np.std(ois_values))
        
        if len(ap_values) > 0:
            results["ap"] = float(np.mean(ap_values))
            results["ap_std"] = float(np.std(ap_values))
        
        if len(all_predictions) > 0:
            ods_f, ods_threshold = compute_ods_with_tolerance(
                all_predictions, all_ground_truths, max_dist=0.0075
            )
            results["ods"] = float(ods_f)
            results["ods_threshold"] = float(ods_threshold)
        
        return results
    
    return None


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(all_results, output_file):
    """Tüm sonuçları kaydet"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RCF MODEL - UNIFIED TEST RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Distance Tolerance: 0.0075 (0.75% of diagonal)\n")
        f.write("="*80 + "\n\n")
        
        for results in all_results:
            if results is None:
                continue
                
            f.write("="*80 + "\n")
            f.write(f"Dataset: {results['dataset']}")
            if 'split' in results:
                f.write(f" ({results['split']})")
            f.write("\n")
            f.write("="*80 + "\n")
            f.write(f"Total Images    : {results['total_images']}\n")
            f.write(f"Successful      : {results['successful']}\n")
            f.write(f"Failed          : {results['failed']}\n\n")
            
            f.write("--- Basic Metrics ---\n")
            f.write(f"MSE             : {results['mse']:.6f} ± {results['mse_std']:.6f}\n")
            f.write(f"PSNR            : {results['psnr']:.4f} dB ± {results['psnr_std']:.4f}\n")
            f.write(f"SSIM            : {results['ssim']:.6f} ± {results['ssim_std']:.6f}\n\n")
            
            if 'ois' in results or 'ods' in results or 'ap' in results:
                f.write("--- Edge Detection Metrics ---\n")
                if 'ois' in results:
                    f.write(f"OIS (F-score)   : {results['ois']:.6f} ± {results['ois_std']:.6f}\n")
                if 'ods' in results:
                    f.write(f"ODS (F-score)   : {results['ods']:.6f} @ threshold={results['ods_threshold']:.3f}\n")
                if 'ap' in results:
                    f.write(f"AP              : {results['ap']:.6f} ± {results['ap_std']:.6f}\n")
            
            f.write("\n")
        
        f.write("="*80 + "\n")


def print_summary(all_results):
    """Sonuçları ekrana yazdır"""
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for results in all_results:
        if results is None:
            continue
            
        dataset_name = results['dataset']
        if 'split' in results:
            dataset_name += f" ({results['split']})"
            
        print(f"\n{dataset_name}:")
        print(f"  Processed: {results['successful']}/{results['total_images']} images")
        print(f"  MSE  : {results['mse']:.6f}")
        print(f"  PSNR : {results['psnr']:.4f} dB")
        print(f"  SSIM : {results['ssim']:.6f}")
        if 'ods' in results:
            print(f"  ODS  : {results['ods']:.6f}")
        if 'ois' in results:
            print(f"  OIS  : {results['ois']:.6f}")
        if 'ap' in results:
            print(f"  AP   : {results['ap']:.6f}")
    
    print(f"\n{'='*80}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RCF Unified Test Script')
    parser.add_argument('--model_path', type=str, help='RCF model path',default=r"bsds500_pascal_model.pth")
    parser.add_argument('--bsds_root', type=str,  help='BSDS500 dataset root',default=r"bsds500")
    parser.add_argument('--stare_root', type=str,  help='STARE dataset root',default=r"stare")
    parser.add_argument('--biped_root', type=str,  help='BIPED dataset root',default=r"biped_dataset\BIPED\BIPED\\edges")
    parser.add_argument('--miedt_root', type=str,  help='MIEDT dataset root',default=r"MIEDT")
    parser.add_argument('--bsds_split', type=str, default='test', choices=['test', 'train', 'val'])
    parser.add_argument('--output', type=str, default='rcf_unified_results.txt')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Model yükle
    model = load_model(args.model_path, device=device)
    if model is None:
        exit(1)
    
    # Test edilecek datasetler
    all_results = []
    
    # BSDS500
    if args.bsds_root:
        print(f"\n{'='*80}")
        print("Testing on BSDS500...")
        print(f"{'='*80}")
        results = test_bsds500(model, args.bsds_root, args.bsds_split, device=device)
        all_results.append(results)
    
    # STARE
    if args.stare_root:
        print(f"\n{'='*80}")
        print("Testing on STARE...")
        print(f"{'='*80}")
        results = test_stare(model, args.stare_root, device=device)
        all_results.append(results)
    
    # BIPED
    if args.biped_root:
        print(f"\n{'='*80}")
        print("Testing on BIPED...")
        print(f"{'='*80}")
        results = test_biped(model, args.biped_root, device=device)
        all_results.append(results)
    
    # MIEDT
    if args.miedt_root:
        print(f"\n{'='*80}")
        print("Testing on MIEDT...")
        print(f"{'='*80}")
        results = test_miedt(model, args.miedt_root, device=device)
        all_results.append(results)
    
    # Sonuçları kaydet ve göster
    if all_results:
        save_results(all_results, args.output)
        print_summary(all_results)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\nNo datasets were tested. Please specify at least one dataset root.")