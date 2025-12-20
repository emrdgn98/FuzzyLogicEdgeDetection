#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HED + BIPED Test - OpenCV
Hızlı kullanım için basitleştirilmiş versiyon
"""

import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt


def load_hed_model(model_path='hed_pretrained_bsds.caffemodel', 
                   prototxt_path='deploy.prototxt'):
    """HED modelini yükle"""
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net


def detect_edges(net, image):
    """Kenar tespiti yap"""
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                                 mean=(104.00698793, 116.66876762, 122.67891434),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    return hed


def normalize(img):
    """0-1 aralığına normalize et"""
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def calculate_metrics(pred, gt):
    """MSI, PSNR, SSIM hesapla"""
    msi = np.mean((pred - gt) ** 2)
    psnr_val = psnr(gt, pred, data_range=1.0)
    ssim_val = ssim(gt, pred, data_range=1.0)
    return {'msi': float(msi), 'psnr': float(psnr_val), 'ssim': float(ssim_val)}
def get_test_pairs(biped_root):
    """Test görüntü-GT çiftlerini lst dosyasına göre bul"""

    test_list_file = os.path.join(biped_root, 'test_rgb.lst')

    print(f"Test listesi: {test_list_file}")

    pairs = []

    with open(test_list_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        img_rel, gt_rel = line.strip().split()

        img_path = os.path.join(biped_root, 'imgs', 'test', img_rel)
        gt_path = os.path.join(biped_root, 'edge_maps', 'test', gt_rel)

        if os.path.exists(img_path) and os.path.exists(gt_path):
            pairs.append((img_path, gt_path))
        else:
            print("Bulunamadı:")
            print(" IMG:", img_path)
            print(" GT :", gt_path)

    return pairs


def test_biped(biped_root, output_dir='biped_results', save_images=True, max_images=None):
    """BIPED dataset'i test et"""
    
    # Model yükle
    print("Model yükleniyor...")
    net = load_hed_model()
    
    # Test çiftlerini al
    print("Test görüntüleri bulunuyor...")
    pairs = get_test_pairs(biped_root)
    
    if max_images:
        pairs = pairs[:max_images]
    
    print(f"Toplam {len(pairs)} görüntü test edilecek")
    
    # Sonuç dizini
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrik listeleri
    all_results = []
    msi_values = []
    psnr_values = []
    ssim_values = []
    
    # Test
    print("\nTest başlıyor...")
    for img_path, gt_path in tqdm(pairs):
        # Görüntü oku
        image = cv2.imread(img_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or gt is None:
            continue
        
        # Kenar tespiti
        pred = detect_edges(net, image)
        
        # GT'yi resize et
        gt_resized = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
        
        # Normalize et
        pred_norm = normalize(pred)
        gt_norm = normalize(gt_resized)
        
        # Metrikleri hesapla
        metrics = calculate_metrics(pred_norm, gt_norm)
        metrics['image'] = os.path.basename(img_path)
        
        all_results.append(metrics)
        msi_values.append(metrics['msi'])
        psnr_values.append(metrics['psnr'])
        ssim_values.append(metrics['ssim'])
        
        # Görüntüleri kaydet
        if save_images:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Tahmin
            pred_path = os.path.join(output_dir, f'{base_name}_pred.png')
            cv2.imwrite(pred_path, (pred_norm * 255).astype(np.uint8))
            
            # Karşılaştırma
            h, w = pred.shape
            original_resized = cv2.resize(image, (w, h))
            pred_color = cv2.cvtColor((pred_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            gt_color = cv2.cvtColor((gt_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            comparison = np.hstack([original_resized, pred_color, gt_color])
            
            comp_path = os.path.join(output_dir, f'{base_name}_comp.png')
            cv2.imwrite(comp_path, comparison)
    
    # Ortalama metrikler
    avg_metrics = {
        'avg_msi': float(np.mean(msi_values)),
        'avg_psnr': float(np.mean(psnr_values)),
        'avg_ssim': float(np.mean(ssim_values)),
        'std_msi': float(np.std(msi_values)),
        'std_psnr': float(np.std(psnr_values)),
        'std_ssim': float(np.std(ssim_values)),
        'total_images': len(all_results)
    }
    
    # JSON kaydet
    results = {
        'average_metrics': avg_metrics,
        'per_image_results': all_results
    }
    
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Sonuçları yazdır
    print("\n" + "="*60)
    print("SONUÇLAR")
    print("="*60)
    print(f"Test Edilen Görüntü: {avg_metrics['total_images']}")
    print(f"MSI  : {avg_metrics['avg_msi']:.6f} ± {avg_metrics['std_msi']:.6f}")
    print(f"PSNR : {avg_metrics['avg_psnr']:.4f} dB ± {avg_metrics['std_psnr']:.4f}")
    print(f"SSIM : {avg_metrics['avg_ssim']:.6f} ± {avg_metrics['std_ssim']:.6f}")
    print("="*60)
    
    # Grafik oluştur
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(msi_values, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(msi_values), color='red', linestyle='--')
    axes[0].set_xlabel('MSI')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('MSI Distribution')
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(psnr_values, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1].axvline(np.mean(psnr_values), color='red', linestyle='--')
    axes[1].set_xlabel('PSNR (dB)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('PSNR Distribution')
    axes[1].grid(alpha=0.3)
    
    axes[2].hist(ssim_values, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[2].axvline(np.mean(ssim_values), color='red', linestyle='--')
    axes[2].set_xlabel('SSIM')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('SSIM Distribution')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'metrics.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Sonuçlar kaydedildi: {output_dir}/")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Komut satırından veya doğrudan kod içinden kullanım
    if len(sys.argv) > 1:
        BIPED_ROOT = sys.argv[1]
    else:
        # BURAYA KENDİ YOLUNUZU YAZIN
        # Örnek: BIPED_ROOT = "D:/datasets/biped_dataset"
        BIPED_ROOT = "biped_dataset\BIPED\BIPED\edges"
    
    if not os.path.exists(BIPED_ROOT):
        print(f"Hata: Dataset bulunamadı: {BIPED_ROOT}")
        print("\nKullanım:")
        print("  python biped_hed_test.py /path/to/biped_dataset")
        print("\nVeya kodu düzenleyip BIPED_ROOT değişkenini güncelleyin")
        sys.exit(1)
    
    # Test et
    results = test_biped(
        biped_root=BIPED_ROOT,
        output_dir='biped_results',
        save_images=True,
        max_images=None  # Hepsini test etmek için None, test için sayı (örn: 10)
    )