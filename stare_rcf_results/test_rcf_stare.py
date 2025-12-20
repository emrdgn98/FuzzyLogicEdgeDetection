#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RCF + STARE Test Script
STARE dataset üzerinde RCF modelini test eder ve metrikleri hesaplar
Metrics: MSE, PSNR, SSIM
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
from glob import glob
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from results.biped_rcf_results.models import RCF  # RCF modelinizi buradan import edin

# ----------------------------
# Model Yükleme
# ----------------------------

def load_model(model_path='bsds500_pascal_model.pth', device='cuda'):
    """RCF modelini yükle"""
    model = RCF()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ----------------------------
# Görüntü İşleme
# ----------------------------

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
# STARE Dataset İşleme
# ----------------------------

def get_test_pairs_stare(stare_root):
    """STARE test görüntü ve GT çiftlerini bul"""
    raw_dir = os.path.join(stare_root, "stare_raw")
    gt_dir = os.path.join(stare_root, "stare_vk")
    
    if not os.path.exists(raw_dir):
        print(f"HATA: Raw dizin bulunamadı: {raw_dir}")
        return []
    
    if not os.path.exists(gt_dir):
        print(f"HATA: GT dizin bulunamadı: {gt_dir}")
        return []
    
    raw_images = glob(os.path.join(raw_dir, "*.ppm"))
    gt_images = glob(os.path.join(gt_dir, "*.ppm"))
    
    # Raw görüntüleri dict'e dönüştür
    raw_dict = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in raw_images}
    
    pairs = []
    for gt_path in gt_images:
        gt_name = os.path.basename(gt_path).lower()
        # .vk uzantısını kaldır
        base = gt_name.replace(".vk", "")
        base = os.path.splitext(base)[0]
        
        if base in raw_dict:
            pairs.append((raw_dict[base], gt_path))
        else:
            print(f"Uyarı: Eşleşmeyen GT dosyası: {gt_name}")
    
    print(f"\n✓ {len(pairs)} görüntü çifti bulundu")
    return pairs

# ----------------------------
# Metrik Hesaplama
# ----------------------------

def calculate_metrics(pred, gt):
    """MSE, PSNR, SSIM hesapla"""
    pred = normalize(pred)
    gt = normalize(np.array(gt))
    
    mse_val = np.mean((pred - gt) ** 2)
    psnr_val = psnr(gt, pred, data_range=1.0)
    ssim_val = ssim(gt, pred, data_range=1.0)
    
    return {
        'mse': float(mse_val),
        'psnr': float(psnr_val),
        'ssim': float(ssim_val)
    }

# ----------------------------
# Ana Test Fonksiyonu
# ----------------------------

def test_stare_rcf(model, stare_root, output_dir='stare_rcf_results', device='cuda'):
    """STARE dataset üzerinde RCF testini çalıştır"""
    os.makedirs(output_dir, exist_ok=True)
    
    pairs = get_test_pairs_stare(stare_root)
    
    if len(pairs) == 0:
        print("HATA: Test edilecek görüntü bulunamadı!")
        return None
    
    results_per_image = []
    mse_values, psnr_values, ssim_values = [], [], []
    
    print("\n" + "="*50)
    print("RCF ile STARE testi başlatılıyor...")
    print("="*50)
    
    for img_path, gt_path in tqdm(pairs, desc="Test ediliyor"):
        try:
            # Görüntüyü işle
            img_tensor, (W, H) = preprocess_image(img_path, device=device)
            
            # Kenar tespiti yap
            pred = infer_edges(model, img_tensor)
            pred = np.clip(pred, 0, 1)
            
            # GT'yi yükle
            gt_img = Image.open(gt_path).convert('L')
            
            # Metrikleri hesapla
            metrics = calculate_metrics(pred, gt_img)
            metrics['image'] = os.path.basename(img_path)
            
            mse_values.append(metrics['mse'])
            psnr_values.append(metrics['psnr'])
            ssim_values.append(metrics['ssim'])
            
            results_per_image.append(metrics)
            
            # Sonucu kaydet
            pred_img = (pred * 255).astype(np.uint8)
            pred_pil = Image.fromarray(pred_img)
            output_name = os.path.basename(img_path).replace('.ppm', '_rcf_pred.png')
            pred_pil.save(os.path.join(output_dir, output_name))
            
        except Exception as e:
            print(f"\nHATA [{os.path.basename(img_path)}]: {str(e)}")
            continue
    
    # Ortalama metrikleri hesapla
    avg_results = {
        "mse": float(np.mean(mse_values)),
        "mse_std": float(np.std(mse_values)),
        "psnr": float(np.mean(psnr_values)),
        "psnr_std": float(np.std(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "ssim_std": float(np.std(ssim_values)),
        "total_images": len(mse_values)
    }
    
    # JSON olarak kaydet
    results = {
        "average": avg_results,
        "per_image": results_per_image
    }
    
    json_path = os.path.join(output_dir, 'stare_rcf_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Terminal çıktısı
    print("\n" + "="*50)
    print("STARE RCF TEST SONUÇLARI (ORTALAMA)")
    print("="*50)
    print(f"Test Edilen Görüntü Sayısı: {avg_results['total_images']}")
    print(f"MSE  : {avg_results['mse']:.6f} ± {avg_results['mse_std']:.6f}")
    print(f"PSNR : {avg_results['psnr']:.4f} dB ± {avg_results['psnr_std']:.4f}")
    print(f"SSIM : {avg_results['ssim']:.6f} ± {avg_results['ssim_std']:.6f}")
    print("="*50)
    print(f"\n✓ Test tamamlandı!")
    print(f"✓ Sonuçlar kaydedildi: {output_dir}")
    print(f"✓ JSON raporu: {json_path}")
    
    return results

# ----------------------------
# Ana Program
# ----------------------------

if __name__ == "__main__":
    # Yapılandırma
    STARE_ROOT = r"C:\Users\emredogan\Desktop\itu_master\fuzzlogicproject\stare"
    MODEL_PATH = r"C:\Users\emredogan\Desktop\itu_master\fuzzlogicproject\bsds500_pascal_model.pth"
    OUTPUT_DIR = "stare_rcf_results"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nCihaz: {DEVICE}")
    print(f"STARE Root: {STARE_ROOT}")
    print(f"Model Path: {MODEL_PATH}")
    
    # Dataset kontrolü
    if not os.path.exists(STARE_ROOT):
        print(f"\nHATA: STARE dataset bulunamadı: {STARE_ROOT}")
        exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"\nHATA: Model dosyası bulunamadı: {MODEL_PATH}")
        exit(1)
    
    # Model yükle
    print("\nModel yükleniyor...")
    model = load_model(MODEL_PATH, device=DEVICE)
    print("✓ Model yüklendi")
    
    # Test çalıştır
    results = test_stare_rcf(
        model=model,
        stare_root=STARE_ROOT,
        output_dir=OUTPUT_DIR,
        device=DEVICE
    )