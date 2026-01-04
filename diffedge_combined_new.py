#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DiffusionEdge Combined Test Script - BSDS500 + STARE + BIPED + MIEDT (FIXED VERSION)
Her dört dataset üzerinde DiffusionEdge modelini test eder ve sonuçları kaydeder
Metrics: MSE, PSNR, SSIM, ODS, OIS, AP

DÜZELTMELER:
1. Distance tolerance eklendi (0.0075)
2. ODS/OIS hesaplaması düzeltildi
3. Threshold search düzgün çalışıyor
4. DiffusionEdge model kullanımı
5. MIEDT dataset desteği eklendi
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import yaml
from glob import glob
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import scipy.io as sio
from sklearn.metrics import average_precision_score
from scipy.ndimage import distance_transform_edt
import argparse
from datetime import datetime
import cv2
from pathlib import Path
from fvcore.common.config import CfgNode

# DiffusionEdge imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DiffusionEdge'))

from DiffusionEdge.denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
from DiffusionEdge.denoising_diffusion_pytorch.mask_cond_unet import Unet
from DiffusionEdge.denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion

# ----------------------------
# Config Yükleme
# ----------------------------

def load_conf(config_file):
    """YAML config dosyasını yükle"""
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
    return exp_conf

# ----------------------------
# Model Yükleme
# ----------------------------

def load_diffusion_edge_model(config_path, model_path, first_stage_path=None, device='cuda'):
    """DiffusionEdge modelini config ve checkpoint'ten yükle"""
    print(f"Config yükleniyor: {config_path}")
    cfg_dict = load_conf(config_path)
    cfg = CfgNode(cfg_dict)
    
    print(f"Model yükleniyor: {model_path}")
    
    # First stage model (AutoEncoder)
    model_cfg = cfg.model
    first_stage_cfg = model_cfg.first_stage
    
    # First stage checkpoint yolunu override et
    if first_stage_path and os.path.exists(first_stage_path):
        first_stage_cfg.ckpt_path = first_stage_path
        print(f"✓ First stage kullanılıyor: {first_stage_path}")
    elif first_stage_path:
        print(f"⚠️ First stage bulunamadı: {first_stage_path}")
        print("⚠️ First stage olmadan devam ediliyor...")
        first_stage_cfg.ckpt_path = None
    else:
        print("⚠️ First stage checkpoint belirtilmedi")
        if hasattr(first_stage_cfg, 'ckpt_path') and first_stage_cfg.ckpt_path:
            if not os.path.exists(first_stage_cfg.ckpt_path):
                first_stage_cfg.ckpt_path = None
    
    first_stage_model = AutoencoderKL(
        ddconfig=first_stage_cfg.ddconfig,
        lossconfig=first_stage_cfg.lossconfig,
        embed_dim=first_stage_cfg.embed_dim,
        ckpt_path=first_stage_cfg.get('ckpt_path', None),
    )
    
    # U-Net model
    unet_cfg = model_cfg.unet
    unet = Unet(
        dim=unet_cfg.dim,
        channels=unet_cfg.channels,
        dim_mults=unet_cfg.dim_mults,
        learned_variance=unet_cfg.get('learned_variance', False),
        out_mul=unet_cfg.out_mul,
        cond_in_dim=unet_cfg.cond_in_dim,
        cond_dim=unet_cfg.cond_dim,
        cond_dim_mults=unet_cfg.cond_dim_mults,
        window_sizes1=unet_cfg.window_sizes1,
        window_sizes2=unet_cfg.window_sizes2,
        fourier_scale=unet_cfg.fourier_scale,
        cfg=unet_cfg,
    )
    
    # Latent Diffusion Model
    ldm = LatentDiffusion(
        model=unet,
        auto_encoder=first_stage_model,
        train_sample=model_cfg.train_sample,
        image_size=model_cfg.image_size,
        timesteps=model_cfg.timesteps,
        sampling_timesteps=5,
        loss_type=model_cfg.loss_type,
        objective=model_cfg.objective,
        scale_factor=model_cfg.scale_factor,
        scale_by_std=model_cfg.scale_by_std,
        scale_by_softsign=model_cfg.scale_by_softsign,
        default_scale=model_cfg.get('default_scale', False),
        input_keys=model_cfg.input_keys,
        ckpt_path=None,
        ignore_keys=model_cfg.ignore_keys,
        only_model=model_cfg.only_model,
        start_dist=model_cfg.start_dist,
        perceptual_weight=model_cfg.perceptual_weight,
        use_l1=model_cfg.get('use_l1', True),
        cfg=model_cfg,
    )
    
    # Checkpoint yükle
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'ema' in checkpoint:
        sd = checkpoint['ema']
        new_sd = {}
        for k in sd.keys():
            if k.startswith("ema_model."):
                new_k = k[10:]
                new_sd[new_k] = sd[k]
            else:
                new_sd[k] = sd[k]
        ldm.load_state_dict(new_sd, strict=False)
        print("✓ EMA weights yüklendi")
    elif 'model' in checkpoint:
        ldm.load_state_dict(checkpoint['model'], strict=False)
        print("✓ Model weights yüklendi")
    else:
        ldm.load_state_dict(checkpoint, strict=False)
        print("✓ Checkpoint yüklendi")
    
    if 'model' in checkpoint and 'scale_factor' in checkpoint['model']:
        ldm.scale_factor = checkpoint['model']['scale_factor']
    
    ldm.to(device)
    ldm.eval()
    
    print("✓ Model hazır")
    return ldm, cfg

# ----------------------------
# Preprocessing & Inference
# ----------------------------

def preprocess_image(img_path, target_size=(320, 320)):
    """Görüntüyü DiffusionEdge için uygun formata çevir"""
    img = Image.open(img_path).convert('RGB')
    original_size = img.size
    
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, original_size

def infer_edges_diffusion(model, img_tensor, device='cuda'):
    """DiffusionEdge ile kenar tespiti"""
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        pred_tensor = model.sample(batch_size=1, cond=img_tensor, mask=None)
        pred = pred_tensor.squeeze().cpu().numpy()
        pred = np.clip(pred, 0, 1)
    
    return pred

def normalize(img):
    """0-1 aralığına normalize et"""
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img

# ----------------------------
# BSDS500 Functions
# ----------------------------
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
        pred = (normalize(pred) * 255).astype(np.uint8)
    if gt.dtype != np.uint8:
        gt = (normalize(gt) * 255).astype(np.uint8)
    
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
        
        for key in ['bw', 'boundary', 'edge']:
            if key in mat:
                data = mat[key]
                return np.array(data, dtype=np.float32)
        
        return None
        
    except Exception as e:
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

def get_test_pairs_bsds500(bsds_root, split='test'):
    """BSDS500 dataset için görüntü-ground truth eşleştirmesi"""
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

# ----------------------------
# STARE Functions
# ----------------------------

def get_test_pairs_stare(stare_root):
    """STARE test görüntü ve GT çiftlerini bul"""
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

# ----------------------------
# BIPED Functions
# ----------------------------

def get_test_pairs_biped(biped_root):
    """BIPED test görüntü-GT çiftlerini lst dosyasına göre bul"""
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

# ----------------------------
# MIEDT Functions
# ----------------------------

def get_test_pairs_miedt(miedt_root):
    """
    MIEDT test görüntü-GT çiftlerini bul
    
    Expected structure:
        miedt_root/
            ct_brain_original/
                IMG-001.png, IMG-002.png, ...
            ct_brain_ground_truth/
                GT-001.png, GT-002.png, ...
    """
    img_dir = os.path.join(miedt_root, 'ct_brain_original')
    gt_dir = os.path.join(miedt_root, 'ct_brain_ground_truth')
    
    if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
        return []
    
    # Get image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']:
        found = glob(os.path.join(img_dir, ext))
        if found:
            image_files.extend(found)
    
    if not image_files:
        return []
    
    pairs = []
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        
        # Extract number from IMG-XXX.ext
        if img_name.startswith('IMG-'):
            parts = img_name[4:]  # Remove 'IMG-'
            number_part = os.path.splitext(parts)[0]
            img_ext = os.path.splitext(parts)[1]
            
            # Construct GT filename: GT-XXX.png
            gt_name = f"GT-{number_part}.png"
            gt_path = os.path.join(gt_dir, gt_name)
            
            # Also try with same extension as image
            if not os.path.exists(gt_path):
                gt_name = f"GT-{number_part}{img_ext}"
                gt_path = os.path.join(gt_dir, gt_name)
            
            if os.path.exists(gt_path):
                pairs.append((img_path, gt_path))
    
    return pairs

def preprocess_miedt_ground_truth(gt_image):
    """
    MIEDT ground truth preprocessing
    - Convert segmentation mask to edge map if needed
    - Normalize to 0-1 range
    """
    if len(gt_image.shape) == 3:
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
    
    # Normalize to 0-1
    gt_normalized = normalize(gt_image)
    
    # If it's a binary mask, extract edges
    unique_values = np.unique(gt_normalized)
    if len(unique_values) <= 2:  # Binary mask
        gt_uint8 = (gt_normalized * 255).astype(np.uint8)
        edges = cv2.Canny(gt_uint8, 50, 150)
        return edges.astype(np.float32) / 255.0
    else:
        # Already an edge map
        return gt_normalized

# ----------------------------
# ✅ FIXED: ODS, OIS, AP Metrics with Distance Tolerance
# ----------------------------

def compute_f_score(precision, recall):
    """F-score hesapla"""
    with np.errstate(divide='ignore', invalid='ignore'):
        f_score = 2 * (precision * recall) / (precision + recall)
        f_score = np.nan_to_num(f_score)
    return f_score

def correspondence_with_tolerance(pred_binary, gt_binary, max_dist=0.0075):
    """
    ✅ FIXED: Distance tolerance ile TP/FP/FN hesapla
    
    Args:
        pred_binary: Binary prediction (0 or 1) - ALREADY BINARIZED!
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

def compute_ois_with_tolerance(pred, gt_list, thresholds=99, max_dist=0.0075):
    """
    ✅ FIXED: OIS hesapla - distance tolerance ile
    
    Args:
        pred: Continuous prediction [0, 1]
        gt_list: List of ground truth arrays
        thresholds: Number of thresholds to test
        max_dist: Distance tolerance (default 0.0075)
    
    Returns:
        best_f: Best F-score
        best_thresh: Best threshold
    """
    # GT'leri birleştir (maksimum)
    gt_combined = np.zeros_like(gt_list[0])
    for gt in gt_list:
        gt_combined = np.maximum(gt_combined, gt)
    
    gt_binary = (gt_combined > 0.5).astype(np.uint8)
    
    threshold_values = np.linspace(0, 1, thresholds)
    f_scores = []
    
    for thresh in threshold_values:
        # ✅ THRESHOLD İLE BİNARİZE ET
        pred_binary = (pred >= thresh).astype(np.uint8)
        
        # ✅ DISTANCE TOLERANCE İLE TP/FP/FN HESAPLA
        tp, fp, fn = correspondence_with_tolerance(pred_binary, gt_binary, max_dist)
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f = compute_f_score(precision, recall)
        f_scores.append(f)
    
    f_scores = np.array(f_scores)
    best_idx = np.argmax(f_scores)
    
    return f_scores[best_idx], threshold_values[best_idx]

def compute_ap(pred, gt):
    """AP (Average Precision) hesapla - sklearn ile"""
    pred_flat = pred.flatten()
    gt_flat = (gt.flatten() > 0.5).astype(int)
    
    try:
        ap = average_precision_score(gt_flat, pred_flat)
        return ap
    except:
        return 0.0

def compute_ods_with_tolerance(all_predictions, all_ground_truths, thresholds=99, max_dist=0.0075):
    """
    ✅ FIXED: ODS hesapla - distance tolerance ile
    
    Args:
        all_predictions: List of continuous predictions
        all_ground_truths: List of ground truths
        thresholds: Number of thresholds to test
        max_dist: Distance tolerance
    
    Returns:
        best_f: Best F-score
        best_thresh: Best threshold
    """
    threshold_values = np.linspace(0, 1, thresholds)
    dataset_f_scores = []
    
    for thresh in threshold_values:
        total_tp, total_fp, total_fn = 0, 0, 0
        
        for pred, gt in zip(all_predictions, all_ground_truths):
            # ✅ THRESHOLD İLE BİNARİZE ET
            pred_binary = (pred >= thresh).astype(np.uint8)
            gt_binary = (gt > 0.5).astype(np.uint8)
            
            # ✅ DISTANCE TOLERANCE İLE TP/FP/FN HESAPLA
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
    
    return dataset_f_scores[best_idx], threshold_values[best_idx]

# ----------------------------
# Metric Calculation
# ----------------------------
def calculate_metrics(pred, gt, gt_list=None):
    """MSE, PSNR, SSIM, OIS, AP hesapla"""
    pred = normalize(pred)
    gt = normalize(gt)
    
    # Boyutları eşitle
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    
    # Binary edge maps oluştur (0-255)
    pred_binary = (pred > 0.5).astype(np.uint8) * 255
    gt_binary = (gt > 0.5).astype(np.uint8) * 255
    
    # Basic metrics hesapla (0-255 üzerinde)
    basic = calculate_basic_metrics(pred_binary, gt_binary)
    
    metrics = {
        'mse': basic['mse'],
        'psnr': basic['psnr'],
        'ssim': basic['ssim']
    }
    
    # ✅ FIXED: OIS ve AP hesapla
    if gt_list is not None and len(gt_list) > 0:
        gt_list_resized = []
        for gt_single in gt_list:
            if gt_single.shape != pred.shape:
                gt_single = cv2.resize(gt_single, (pred.shape[1], pred.shape[0]))
            gt_list_resized.append(gt_single)
        
        # ✅ Distance tolerance ile OIS
        ois_f, ois_thresh = compute_ois_with_tolerance(pred, gt_list_resized, max_dist=0.0075)
        metrics['ois_f'] = float(ois_f)
        metrics['ois_threshold'] = float(ois_thresh)
        
        # AP
        ap = compute_ap(pred, gt_list_resized[0])
        metrics['ap'] = float(ap)
    
    return metrics, pred

# ----------------------------
# Test Functions
# ----------------------------

def test_bsds500(model, bsds_root, split, device, target_size):
    """BSDS500 test"""
    pairs = get_test_pairs_bsds500(bsds_root, split)
    
    if len(pairs) == 0:
        return None
    
    mse_values, psnr_values, ssim_values = [], [], []
    ois_values, ap_values = [], []
    all_predictions, all_ground_truths = [], []
    
    successful_count = 0
    failed_count = 0
    
    for img_path, gt_path in tqdm(pairs, desc=f"BSDS500 {split}"):
        try:
            # Görüntü işle
            img_tensor, original_size = preprocess_image(img_path, target_size=target_size)
            
            # Kenar tespiti
            pred = infer_edges_diffusion(model, img_tensor, device=device)
            pred = np.clip(pred, 0, 1)
            
            # GT oku
            gt = read_mat_ground_truth(gt_path)
            gt_list = read_all_ground_truths(gt_path)
            
            if gt is None:
                failed_count += 1
                continue
            
            # Metrikleri hesapla
            metrics, pred = calculate_metrics(pred, gt, gt_list)
            
            mse_values.append(metrics['mse'])
            psnr_values.append(metrics['psnr'])
            ssim_values.append(metrics['ssim'])
            
            if 'ois_f' in metrics:
                ois_values.append(metrics['ois_f'])
            if 'ap' in metrics:
                ap_values.append(metrics['ap'])
            
            all_predictions.append(pred)
            all_ground_truths.append(gt)
            
            successful_count += 1
            
        except Exception as e:
            failed_count += 1
            continue
    
    if len(mse_values) > 0:
        # PSNR için inf değerleri filtrele
        psnr_filtered = [p for p in psnr_values if p != float('inf')]
        avg_psnr = float(np.mean(psnr_filtered)) if psnr_filtered else 100.0

        avg_results = {
            "dataset": "BSDS500",
            "split": split,
            "mse": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values)),
            "psnr": avg_psnr,
            "psnr_std": float(np.std(psnr_filtered)) if psnr_filtered else 0.0,
            "ssim": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values)),
            "total_images": len(mse_values),
            "successful": successful_count,
            "failed": failed_count
        }
        
        if len(ois_values) > 0:
            avg_results["ois"] = float(np.mean(ois_values))
            avg_results["ois_std"] = float(np.std(ois_values))
        
        if len(ap_values) > 0:
            avg_results["ap"] = float(np.mean(ap_values))
            avg_results["ap_std"] = float(np.std(ap_values))
        
        # ✅ FIXED: Distance tolerance ile ODS
        if len(all_predictions) > 0:
            ods_f, ods_threshold = compute_ods_with_tolerance(
                all_predictions, all_ground_truths, max_dist=0.0075
            )
            avg_results["ods"] = float(ods_f)
            avg_results["ods_threshold"] = float(ods_threshold)
        
        return avg_results
    
    return None

def test_stare(model, stare_root, device, target_size):
    """STARE test"""
    pairs = get_test_pairs_stare(stare_root)
    
    if len(pairs) == 0:
        return None
    
    mse_values, psnr_values, ssim_values = [], [], []
    ois_values, ap_values = [], []
    all_predictions, all_ground_truths = [], []
    
    successful_count = 0
    failed_count = 0
    
    for img_path, gt_path in tqdm(pairs, desc="STARE"):
        try:
            # Görüntü işle
            img_tensor, original_size = preprocess_image(img_path, target_size=target_size)
            
            # Kenar tespiti
            pred = infer_edges_diffusion(model, img_tensor, device=device)
            pred = np.clip(pred, 0, 1)
            
            # GT oku
            gt = Image.open(gt_path).convert('L')
            gt_array = np.array(gt).astype(np.float32) / 255.0
            
            # STARE için GT'yi liste olarak hazırla (OIS için)
            gt_list = [gt_array]
            
            # Metrikleri hesapla
            metrics, pred = calculate_metrics(pred, gt_array, gt_list)
            
            mse_values.append(metrics['mse'])
            psnr_values.append(metrics['psnr'])
            ssim_values.append(metrics['ssim'])
            
            if 'ois_f' in metrics:
                ois_values.append(metrics['ois_f'])
            if 'ap' in metrics:
                ap_values.append(metrics['ap'])
            
            all_predictions.append(pred)
            all_ground_truths.append(gt_array)
            
            successful_count += 1
            
        except Exception as e:
            failed_count += 1
            continue
    
    if len(mse_values) > 0:
        
        psnr_filtered = [p for p in psnr_values if p != float('inf')]
        avg_psnr = float(np.mean(psnr_filtered)) if psnr_filtered else 100.0

        avg_results = {
            "dataset": "STARE",
            "mse": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values)),
            "psnr": avg_psnr,
            "psnr_std": float(np.std(psnr_filtered)) if psnr_filtered else 0.0,
            "ssim": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values)),
            "total_images": len(mse_values),
            "successful": successful_count,
            "failed": failed_count
        }
        
        if len(ois_values) > 0:
            avg_results["ois"] = float(np.mean(ois_values))
            avg_results["ois_std"] = float(np.std(ois_values))
        
        if len(ap_values) > 0:
            avg_results["ap"] = float(np.mean(ap_values))
            avg_results["ap_std"] = float(np.std(ap_values))
        
        # ✅ FIXED: Distance tolerance ile ODS
        if len(all_predictions) > 0:
            ods_f, ods_threshold = compute_ods_with_tolerance(
                all_predictions, all_ground_truths, max_dist=0.0075
            )
            avg_results["ods"] = float(ods_f)
            avg_results["ods_threshold"] = float(ods_threshold)
        
        return avg_results
    
    return None

def test_biped(model, biped_root, device, target_size):
    """BIPED test"""
    pairs = get_test_pairs_biped(biped_root)
    
    if len(pairs) == 0:
        return None
    
    mse_values, psnr_values, ssim_values = [], [], []
    ois_values, ap_values = [], []
    all_predictions, all_ground_truths = [], []
    
    successful_count = 0
    failed_count = 0
    
    for img_path, gt_path in tqdm(pairs, desc="BIPED"):
        try:
            # Görüntü işle
            img_tensor, original_size = preprocess_image(img_path, target_size=target_size)
            
            # Kenar tespiti
            pred = infer_edges_diffusion(model, img_tensor, device=device)
            pred = np.clip(pred, 0, 1)
            
            # GT oku
            gt = Image.open(gt_path).convert('L')
            
            # GT'yi liste olarak hazırla (OIS için)
            gt_array = np.array(gt).astype(np.float32) / 255.0
            gt_list = [gt_array]
            
            # Metrikleri hesapla
            metrics, pred = calculate_metrics(pred, gt_array, gt_list)
            
            mse_values.append(metrics['mse'])
            psnr_values.append(metrics['psnr'])
            ssim_values.append(metrics['ssim'])
            
            if 'ois_f' in metrics:
                ois_values.append(metrics['ois_f'])
            if 'ap' in metrics:
                ap_values.append(metrics['ap'])
            
            all_predictions.append(pred)
            all_ground_truths.append(gt_array)
            
            successful_count += 1
            
        except Exception as e:
            failed_count += 1
            continue
    
    if len(mse_values) > 0:
        psnr_filtered = [p for p in psnr_values if p != float('inf')]
        avg_psnr = float(np.mean(psnr_filtered)) if psnr_filtered else 100.0

        avg_results = {
            "dataset": "BIPED",
            "mse": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values)),
            "psnr": avg_psnr,
            "psnr_std": float(np.std(psnr_filtered)) if psnr_filtered else 0.0,
            "ssim": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values)),
            "total_images": len(mse_values),
            "successful": successful_count,
            "failed": failed_count
        }
        
        if len(ois_values) > 0:
            avg_results["ois"] = float(np.mean(ois_values))
            avg_results["ois_std"] = float(np.std(ois_values))
        
        if len(ap_values) > 0:
            avg_results["ap"] = float(np.mean(ap_values))
            avg_results["ap_std"] = float(np.std(ap_values))
        
        # ✅ FIXED: Distance tolerance ile ODS
        if len(all_predictions) > 0:
            ods_f, ods_threshold = compute_ods_with_tolerance(
                all_predictions, all_ground_truths, max_dist=0.0075
            )
            avg_results["ods"] = float(ods_f)
            avg_results["ods_threshold"] = float(ods_threshold)
        
        return avg_results
    
    return None

def test_miedt(model, miedt_root, device, target_size):
    """MIEDT test"""
    pairs = get_test_pairs_miedt(miedt_root)
    
    if len(pairs) == 0:
        return None
    
    mse_values, psnr_values, ssim_values = [], [], []
    ois_values, ap_values = [], []
    all_predictions, all_ground_truths = [], []
    
    successful_count = 0
    failed_count = 0
    
    for img_path, gt_path in tqdm(pairs, desc="MIEDT"):
        try:
            # Görüntü işle
            img_tensor, original_size = preprocess_image(img_path, target_size=target_size)
            
            # Kenar tespiti
            pred = infer_edges_diffusion(model, img_tensor, device=device)
            pred = np.clip(pred, 0, 1)
            
            # GT oku ve preprocess
            gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_image is None:
                failed_count += 1
                continue
            
            # MIEDT için özel preprocessing
            gt_array = preprocess_miedt_ground_truth(gt_image)
            
            # GT'yi liste olarak hazırla (OIS için)
            gt_list = [gt_array]
            
            # Metrikleri hesapla
            metrics, pred = calculate_metrics(pred, gt_array, gt_list)
            
            mse_values.append(metrics['mse'])
            psnr_values.append(metrics['psnr'])
            ssim_values.append(metrics['ssim'])
            
            if 'ois_f' in metrics:
                ois_values.append(metrics['ois_f'])
            if 'ap' in metrics:
                ap_values.append(metrics['ap'])
            
            all_predictions.append(pred)
            all_ground_truths.append(gt_array)
            
            successful_count += 1
            
        except Exception as e:
            failed_count += 1
            continue
    
    if len(mse_values) > 0:
        psnr_filtered = [p for p in psnr_values if p != float('inf')]
        avg_psnr = float(np.mean(psnr_filtered)) if psnr_filtered else 100.0
        avg_results = {
            "dataset": "MIEDT",
            "mse": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values)),
            "psnr": avg_psnr,
            "psnr_std": float(np.std(psnr_filtered)) if psnr_filtered else 0.0,
            "ssim": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values)),
            "total_images": len(mse_values),
            "successful": successful_count,
            "failed": failed_count
        }
        
        if len(ois_values) > 0:
            avg_results["ois"] = float(np.mean(ois_values))
            avg_results["ois_std"] = float(np.std(ois_values))
        
        if len(ap_values) > 0:
            avg_results["ap"] = float(np.mean(ap_values))
            avg_results["ap_std"] = float(np.std(ap_values))
        
        # ✅ FIXED: Distance tolerance ile ODS
        if len(all_predictions) > 0:
            ods_f, ods_threshold = compute_ods_with_tolerance(
                all_predictions, all_ground_truths, max_dist=0.0075
            )
            avg_results["ods"] = float(ods_f)
            avg_results["ods_threshold"] = float(ods_threshold)
        
        return avg_results
    
    return None

# ----------------------------
# Save Results to TXT
# ----------------------------

def save_results_to_txt(bsds_results, stare_results, biped_results, miedt_results, output_file):
    """Sonuçları TXT dosyasına kaydet"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DIFFUSIONEDGE MODEL - COMBINED TEST RESULTS (FIXED VERSION)\n")
        f.write("="*80 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Distance Tolerance: 0.0075 (0.75% of diagonal)\n")
        f.write("="*80 + "\n\n")
        
        # BSDS500 Results
        if bsds_results:
            f.write("=" * 80 + "\n")
            f.write(f"BSDS500 DATASET - {bsds_results['split'].upper()} SPLIT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Images    : {bsds_results['total_images']}\n")
            f.write(f"Successful      : {bsds_results['successful']}\n")
            f.write(f"Failed          : {bsds_results['failed']}\n")
            f.write("\n--- Basic Metrics ---\n")
            f.write(f"MSE             : {bsds_results['mse']:.6f} ± {bsds_results['mse_std']:.6f}\n")
            f.write(f"PSNR            : {bsds_results['psnr']:.4f} dB ± {bsds_results['psnr_std']:.4f}\n")
            f.write(f"SSIM            : {bsds_results['ssim']:.6f} ± {bsds_results['ssim_std']:.6f}\n")
            
            if 'ois' in bsds_results or 'ods' in bsds_results or 'ap' in bsds_results:
                f.write("\n--- Edge Detection Metrics ---\n")
                if 'ois' in bsds_results:
                    f.write(f"OIS (F-score)   : {bsds_results['ois']:.6f} ± {bsds_results['ois_std']:.6f}\n")
                if 'ods' in bsds_results:
                    f.write(f"ODS (F-score)   : {bsds_results['ods']:.6f} @ threshold={bsds_results['ods_threshold']:.3f}\n")
                if 'ap' in bsds_results:
                    f.write(f"AP              : {bsds_results['ap']:.6f} ± {bsds_results['ap_std']:.6f}\n")
            
            f.write("\n")
        else:
            f.write("\nBSDS500: Test edilemedi\n\n")
        
        # STARE Results
        if stare_results:
            f.write("=" * 80 + "\n")
            f.write("STARE DATASET\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Images    : {stare_results['total_images']}\n")
            f.write(f"Successful      : {stare_results['successful']}\n")
            f.write(f"Failed          : {stare_results['failed']}\n")
            f.write("\n--- Basic Metrics ---\n")
            f.write(f"MSE             : {stare_results['mse']:.6f} ± {stare_results['mse_std']:.6f}\n")
            f.write(f"PSNR            : {stare_results['psnr']:.4f} dB ± {stare_results['psnr_std']:.4f}\n")
            f.write(f"SSIM            : {stare_results['ssim']:.6f} ± {stare_results['ssim_std']:.6f}\n")
            
            if 'ois' in stare_results or 'ods' in stare_results or 'ap' in stare_results:
                f.write("\n--- Edge Detection Metrics ---\n")
                if 'ois' in stare_results:
                    f.write(f"OIS (F-score)   : {stare_results['ois']:.6f} ± {stare_results['ois_std']:.6f}\n")
                if 'ods' in stare_results:
                    f.write(f"ODS (F-score)   : {stare_results['ods']:.6f} @ threshold={stare_results['ods_threshold']:.3f}\n")
                if 'ap' in stare_results:
                    f.write(f"AP              : {stare_results['ap']:.6f} ± {stare_results['ap_std']:.6f}\n")
            
            f.write("\n")
        else:
            f.write("\nSTARE: Test edilemedi\n\n")
        
        # BIPED Results
        if biped_results:
            f.write("=" * 80 + "\n")
            f.write("BIPED DATASET\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Images    : {biped_results['total_images']}\n")
            f.write(f"Successful      : {biped_results['successful']}\n")
            f.write(f"Failed          : {biped_results['failed']}\n")
            f.write("\n--- Basic Metrics ---\n")
            f.write(f"MSE             : {biped_results['mse']:.6f} ± {biped_results['mse_std']:.6f}\n")
            f.write(f"PSNR            : {biped_results['psnr']:.4f} dB ± {biped_results['psnr_std']:.4f}\n")
            f.write(f"SSIM            : {biped_results['ssim']:.6f} ± {biped_results['ssim_std']:.6f}\n")
            
            if 'ois' in biped_results or 'ods' in biped_results or 'ap' in biped_results:
                f.write("\n--- Edge Detection Metrics ---\n")
                if 'ois' in biped_results:
                    f.write(f"OIS (F-score)   : {biped_results['ois']:.6f} ± {biped_results['ois_std']:.6f}\n")
                if 'ods' in biped_results:
                    f.write(f"ODS (F-score)   : {biped_results['ods']:.6f} @ threshold={biped_results['ods_threshold']:.3f}\n")
                if 'ap' in biped_results:
                    f.write(f"AP              : {biped_results['ap']:.6f} ± {biped_results['ap_std']:.6f}\n")
            
            f.write("\n")
        else:
            f.write("\nBIPED: Test edilemedi\n\n")
        
        # MIEDT Results
        if miedt_results:
            f.write("=" * 80 + "\n")
            f.write("MIEDT DATASET\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Images    : {miedt_results['total_images']}\n")
            f.write(f"Successful      : {miedt_results['successful']}\n")
            f.write(f"Failed          : {miedt_results['failed']}\n")
            f.write("\n--- Basic Metrics ---\n")
            f.write(f"MSE             : {miedt_results['mse']:.6f} ± {miedt_results['mse_std']:.6f}\n")
            f.write(f"PSNR            : {miedt_results['psnr']:.4f} dB ± {miedt_results['psnr_std']:.4f}\n")
            f.write(f"SSIM            : {miedt_results['ssim']:.6f} ± {miedt_results['ssim_std']:.6f}\n")
            
            if 'ois' in miedt_results or 'ods' in miedt_results or 'ap' in miedt_results:
                f.write("\n--- Edge Detection Metrics ---\n")
                if 'ois' in miedt_results:
                    f.write(f"OIS (F-score)   : {miedt_results['ois']:.6f} ± {miedt_results['ois_std']:.6f}\n")
                if 'ods' in miedt_results:
                    f.write(f"ODS (F-score)   : {miedt_results['ods']:.6f} @ threshold={miedt_results['ods_threshold']:.3f}\n")
                if 'ap' in miedt_results:
                    f.write(f"AP              : {miedt_results['ap']:.6f} ± {miedt_results['ap_std']:.6f}\n")
            
            f.write("\n")
        else:
            f.write("\nMIEDT: Test edilemedi\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DiffusionEdge Combined Test (BSDS500 + STARE + BIPED + MIEDT) - FIXED VERSION')
    parser.add_argument('--bsds_root', type=str, 
                        default=r"bsds500",
                        help='BSDS500 dataset root')
    parser.add_argument('--stare_root', type=str,
                        default=r"stare",
                        help='STARE dataset root')
    parser.add_argument('--biped_root', type=str,
                        default='biped_dataset/BIPED/BIPED/edges',
                        help='BIPED dataset root')
    parser.add_argument('--miedt_root', type=str,
                        default=r"MIEDT",
                        help='MIEDT dataset root')
    parser.add_argument('--bsds_split', type=str,
                        default='test',
                        choices=['test', 'train', 'val'],
                        help='BSDS500 split')
    parser.add_argument('--config', type=str,
                        default='DiffusionEdge/configs/BSDS_sample.yaml',
                        help='Config YAML file')
    parser.add_argument('--model_path', type=str,
                        default='bsds_diffedge.pt',
                        help='DiffusionEdge model file (.pt)')
    parser.add_argument('--first_stage', type=str,
                        default=None,
                        help='First stage checkpoint (optional)')
    parser.add_argument('--output_txt', type=str,
                        default='diffusionedge_combined_results_fixed.txt',
                        help='Output TXT file name')
    parser.add_argument('--target_size', type=int, nargs=2,
                        default=[320, 320],
                        help='Model input size (height width)')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        help='Device: cuda or cpu')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*80}")
    print("DiffusionEdge Model Combined Test - BSDS500 + STARE + BIPED + MIEDT (FIXED VERSION)")
    print(f"{'='*80}")
    print(f"Device          : {device}")
    print(f"BSDS500 Root    : {args.bsds_root}")
    print(f"STARE Root      : {args.stare_root}")
    print(f"BIPED Root      : {args.biped_root}")
    print(f"MIEDT Root      : {args.miedt_root}")
    print(f"BSDS500 Split   : {args.bsds_split}")
    print(f"Config          : {args.config}")
    print(f"Model Path      : {args.model_path}")
    if args.first_stage:
        print(f"First Stage     : {args.first_stage}")
    print(f"Output TXT      : {args.output_txt}")
    print(f"Distance Tol.   : 0.0075 (0.75% of diagonal)")
    print(f"{'='*80}\n")
    
    # Model yükle
    try:
        model, cfg = load_diffusion_edge_model(
            args.config, 
            args.model_path, 
            first_stage_path=args.first_stage,
            device=device
        )
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test sonuçları
    bsds_results = None
    stare_results = None
    biped_results = None
    miedt_results = None
    
    # BSDS500 Test
    if os.path.exists(args.bsds_root):
        print("\n" + "="*80)
        print("BSDS500 TEST BAŞLIYOR...")
        print("="*80)
        bsds_results = test_bsds500(
            model=model,
            bsds_root=args.bsds_root,
            split=args.bsds_split,
            device=device,
            target_size=tuple(args.target_size)
        )
        
        if bsds_results:
            print("\n✓ BSDS500 test tamamlandı")
            print(f"  MSE  : {bsds_results['mse']:.6f}")
            print(f"  PSNR : {bsds_results['psnr']:.4f} dB")
            print(f"  SSIM : {bsds_results['ssim']:.6f}")
            if 'ods' in bsds_results:
                print(f"  ODS  : {bsds_results['ods']:.6f}")
            if 'ois' in bsds_results:
                print(f"  OIS  : {bsds_results['ois']:.6f}")
    else:
        print(f"\n⚠️ BSDS500 dataset bulunamadı: {args.bsds_root}")
    
    # STARE Test
    if os.path.exists(args.stare_root):
        print("\n" + "="*80)
        print("STARE TEST BAŞLIYOR...")
        print("="*80)
        stare_results = test_stare(
            model=model,
            stare_root=args.stare_root,
            device=device,
            target_size=tuple(args.target_size)
        )
        
        if stare_results:
            print("\n✓ STARE test tamamlandı")
            print(f"  MSE  : {stare_results['mse']:.6f}")
            print(f"  PSNR : {stare_results['psnr']:.4f} dB")
            print(f"  SSIM : {stare_results['ssim']:.6f}")
            if 'ods' in stare_results:
                print(f"  ODS  : {stare_results['ods']:.6f}")
            if 'ois' in stare_results:
                print(f"  OIS  : {stare_results['ois']:.6f}")
    else:
        print(f"\n⚠️ STARE dataset bulunamadı: {args.stare_root}")
    
    # BIPED Test
    if os.path.exists(args.biped_root):
        print("\n" + "="*80)
        print("BIPED TEST BAŞLIYOR...")
        print("="*80)
        biped_results = test_biped(
            model=model,
            biped_root=args.biped_root,
            device=device,
            target_size=tuple(args.target_size)
        )
        
        if biped_results:
            print("\n✓ BIPED test tamamlandı")
            print(f"  MSE  : {biped_results['mse']:.6f}")
            print(f"  PSNR : {biped_results['psnr']:.4f} dB")
            print(f"  SSIM : {biped_results['ssim']:.6f}")
            if 'ods' in biped_results:
                print(f"  ODS  : {biped_results['ods']:.6f}")
            if 'ois' in biped_results:
                print(f"  OIS  : {biped_results['ois']:.6f}")
    else:
        print(f"\n⚠️ BIPED dataset bulunamadı: {args.biped_root}")
    
    # MIEDT Test
    if os.path.exists(args.miedt_root):
        print("\n" + "="*80)
        print("MIEDT TEST BAŞLIYOR...")
        print("="*80)
        miedt_results = test_miedt(
            model=model,
            miedt_root=args.miedt_root,
            device=device,
            target_size=tuple(args.target_size)
        )
        
        if miedt_results:
            print("\n✓ MIEDT test tamamlandı")
            print(f"  MSE  : {miedt_results['mse']:.6f}")
            print(f"  PSNR : {miedt_results['psnr']:.4f} dB")
            print(f"  SSIM : {miedt_results['ssim']:.6f}")
            if 'ods' in miedt_results:
                print(f"  ODS  : {miedt_results['ods']:.6f}")
            if 'ois' in miedt_results:
                print(f"  OIS  : {miedt_results['ois']:.6f}")
    else:
        print(f"\n⚠️ MIEDT dataset bulunamadı: {args.miedt_root}")
    
    # Sonuçları TXT'ye kaydet
    save_results_to_txt(bsds_results, stare_results, biped_results, miedt_results, args.output_txt)
    
    print("\n" + "="*80)
    print("TÜM TESTLER TAMAMLANDI!")
    print("="*80)
    print(f"✓ Sonuçlar kaydedildi: {args.output_txt}")
    print("="*80 + "\n")