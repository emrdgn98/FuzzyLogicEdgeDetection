#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RCF + BIPED Test - PyTorch
Basit inference scripti
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

from results.biped_rcf_results.models import RCF  # senin RCF modelin burada olmalı

# ----------------------------
# Helper Functions
# ----------------------------

def load_model(model_path='bsds500_pascal_model.pth', device='cuda'):
    """RCF modelini yükle"""
    model = RCF()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(img_path, device='cuda'):
    """Görüntüyü tensor formatına çevir"""
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, img.size

def infer_edges(model, img_tensor):
    """RCF ile kenar tespiti"""
    with torch.no_grad():
        outputs = model(img_tensor)
        # outputs RCF listesi, son eleman final fused score
        out = outputs[-1].squeeze().cpu().numpy()
    return out

def normalize(img):
    """0-1 aralığına normalize et"""
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img

# ----------------------------
# Dataset Handling
# ----------------------------

def get_test_pairs(biped_root):
    """Test görüntü ve GT çiftlerini lst dosyasına göre bul"""
    test_list_file = os.path.join(biped_root, 'test_rgb.lst')
    pairs = []

    with open(test_list_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        img_rel, gt_rel = line.strip().split()
        img_path = os.path.join(biped_root, 'imgs', 'test', img_rel)
        gt_path  = os.path.join(biped_root, 'edge_maps', 'test', gt_rel)
        if os.path.exists(img_path):
            pairs.append((img_path, gt_path if os.path.exists(gt_path) else None))
        else:
            print("Bulunamadı:", img_path)
    return pairs

# ----------------------------
# Metrics
# ----------------------------

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(pred, gt):
    """MSE, PSNR, SSIM hesapla"""
    pred = normalize(pred)
    gt = normalize(np.array(gt))
    mse_val = np.mean((pred - gt) ** 2)
    psnr_val = psnr(gt, pred, data_range=1.0)
    ssim_val = ssim(gt, pred, data_range=1.0)
    return {'mse': float(mse_val), 'psnr': float(psnr_val), 'ssim': float(ssim_val)}

# ----------------------------
# Main Test Loop
# ----------------------------

def test_biped(model, biped_root, output_dir='biped_rcf_results'):
    os.makedirs(output_dir, exist_ok=True)
    pairs = get_test_pairs(biped_root)
    results = []
    
    mse_values, psnr_values, ssim_values = [], [], []
    
    for img_path, gt_path in tqdm(pairs):
        img_tensor, (W,H) = preprocess_image(img_path)
        pred = infer_edges(model, img_tensor)
        pred = np.clip(pred, 0, 1)
        
        metrics = {}
        metrics['image'] = os.path.basename(img_path)
        
        if gt_path:
            gt_img = Image.open(gt_path).convert('L')
            m = calculate_metrics(pred, gt_img)
            metrics.update(m)
            if m['mse'] is not None:
                mse_values.append(m['mse'])
                psnr_values.append(m['psnr'])
                ssim_values.append(m['ssim'])
        else:
            metrics.update({'mse': None, 'psnr': None, 'ssim': None})
        
        # Sonucu kaydet
        pred_img = (pred * 255).astype(np.uint8)
        pred_pil = Image.fromarray(pred_img)
        pred_pil.save(os.path.join(output_dir, metrics['image'].replace('.jpg', '_pred.png')))
        
        results.append(metrics)
    
    # JSON olarak kaydet
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # --------------------------
    # Terminalde final metricleri yazdır
    # --------------------------
    if len(mse_values) > 0:
        print("\n" + "="*50)
        print("TEST SONUÇLARI (ORTALAMA)")
        print("="*50)
        print(f"Test Edilen Görüntü Sayısı: {len(mse_values)}")
        print(f"MSE  : {np.mean(mse_values):.6f} ± {np.std(mse_values):.6f}")
        print(f"PSNR : {np.mean(psnr_values):.4f} dB ± {np.std(psnr_values):.4f}")
        print(f"SSIM : {np.mean(ssim_values):.6f} ± {np.std(ssim_values):.6f}")
        print("="*50)
    else:
        print("GT bulunamadığı için metric hesaplanamadı!")
    
    print(f"\n✓ Test tamamlandı, sonuçlar kaydedildi: {output_dir}")
    return results
# ----------------------------
# Run
# ----------------------------

if __name__ == "__main__":
    BIPED_ROOT = "biped_dataset\\BIPED\\BIPED\\edges"  # burayı kendi dataset yoluna göre değiştir
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = load_model('bsds500_pascal_model.pth', device=DEVICE)
    results = test_biped(model, BIPED_ROOT)
