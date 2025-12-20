#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DiffusionEdge + BIPED Test Script
BIPED dataset üzerinde DiffusionEdge modelini test eder ve metrikleri hesaplar
Metrics: MSE, PSNR, SSIM
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
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import argparse

# DiffusionEdge imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DiffusionEdge'))

from DiffusionEdge.denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
from DiffusionEdge.denoising_diffusion_pytorch.mask_cond_unet import Unet
from DiffusionEdge.denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion
from fvcore.common.config import CfgNode

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
    """
    DiffusionEdge modelini config ve checkpoint'ten yükle
    
    Args:
        config_path: Config YAML dosyası yolu
        model_path: Model checkpoint (.pt) dosyası yolu
        first_stage_path: First stage checkpoint yolu (opsiyonel)
        device: 'cuda' veya 'cpu'
    
    Returns:
        model: Yüklenmiş LatentDiffusion modeli
    """
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
        print("⚠️ First stage checkpoint belirtilmedi, config'teki yol kullanılacak")
        if hasattr(first_stage_cfg, 'ckpt_path') and first_stage_cfg.ckpt_path:
            if not os.path.exists(first_stage_cfg.ckpt_path):
                print(f"⚠️ Config'teki first stage bulunamadı: {first_stage_cfg.ckpt_path}")
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
        sampling_timesteps=5,  # Hızlı inference
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
    
    # EMA kullanarak yükle
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
    
    # Scale factor varsa yükle
    if 'model' in checkpoint and 'scale_factor' in checkpoint['model']:
        ldm.scale_factor = checkpoint['model']['scale_factor']
    
    ldm.to(device)
    ldm.eval()
    
    print("✓ Model hazır")
    return ldm, cfg

def preprocess_image(img_path, target_size=(320, 320)):
    """Görüntüyü DiffusionEdge için uygun formata çevir"""
    img = Image.open(img_path).convert('RGB')
    original_size = img.size
    
    # DiffusionEdge preprocessing
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, original_size

def infer_edges_diffusion(model, img_tensor, device='cuda'):
    """
    DiffusionEdge ile kenar tespiti
    
    Args:
        model: LatentDiffusion modeli
        img_tensor: Input tensor (1, 3, H, W)
        device: cuda veya cpu
    
    Returns:
        pred: Kenar haritası (H, W) numpy array
    """
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        # Model inference - sample metodu ile
        pred_tensor = model.sample(batch_size=1, cond=img_tensor, mask=None)
        
        # Tensor'ı numpy'a çevir
        pred = pred_tensor.squeeze().cpu().numpy()
        
        # [0, 1] aralığına normalize et
        pred = np.clip(pred, 0, 1)
    
    return pred

def normalize(img):
    """0-1 aralığına normalize et"""
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img

# ----------------------------
# BIPED Dataset İşleme
# ----------------------------

def get_test_pairs_biped(biped_root):
    """
    BIPED test görüntü ve GT çiftlerini lst dosyasına göre bul
    
    Args:
        biped_root: BIPED dataset root dizini
    
    Returns:
        pairs: [(img_path, gt_path), ...] listesi
    """
    test_list_file = os.path.join(biped_root, 'test_rgb.lst')
    
    if not os.path.exists(test_list_file):
        print(f"HATA: test_rgb.lst dosyası bulunamadı: {test_list_file}")
        return []
    
    pairs = []
    
    with open(test_list_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            img_rel, gt_rel = parts[0], parts[1]
            img_path = os.path.join(biped_root, 'imgs', 'test', img_rel)
            gt_path = os.path.join(biped_root, 'edge_maps', 'test', gt_rel)
            
            if os.path.exists(img_path):
                pairs.append((img_path, gt_path if os.path.exists(gt_path) else None))
            else:
                print(f"Uyarı: Görüntü bulunamadı: {img_path}")
    
    print(f"\n✓ {len(pairs)} görüntü çifti bulundu")
    return pairs

# ----------------------------
# Metrik Hesaplama
# ----------------------------

def calculate_metrics(pred, gt):
    """MSE, PSNR, SSIM hesapla"""
    pred = normalize(pred)
    gt = normalize(np.array(gt))
    
    # Boyutları eşitle
    if pred.shape != gt.shape:
        pred_pil = Image.fromarray((pred * 255).astype(np.uint8))
        pred_pil = pred_pil.resize((gt.shape[1], gt.shape[0]), Image.BILINEAR)
        pred = np.array(pred_pil) / 255.0
    
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

def test_biped_diffusion_edge(model, biped_root, output_dir='biped_diffusion_results', 
                               device='cuda', target_size=(320, 320)):
    """BIPED dataset üzerinde DiffusionEdge testini çalıştır"""
    os.makedirs(output_dir, exist_ok=True)
    
    pairs = get_test_pairs_biped(biped_root)
    
    if len(pairs) == 0:
        print("HATA: Test edilecek görüntü bulunamadı!")
        return None
    
    results_per_image = []
    mse_values, psnr_values, ssim_values = [], [], []
    
    print("\n" + "="*50)
    print("DiffusionEdge ile BIPED testi başlatılıyor...")
    print("="*50)
    
    for img_path, gt_path in tqdm(pairs, desc="Test ediliyor"):
        try:
            # Görüntüyü işle
            img_tensor, original_size = preprocess_image(img_path, target_size=target_size)
            
            # Kenar tespiti yap
            pred = infer_edges_diffusion(model, img_tensor, device=device)
            
            # Metrikleri hesapla
            metrics = {}
            metrics['image'] = os.path.basename(img_path)
            
            if gt_path and os.path.exists(gt_path):
                # GT'yi yükle
                gt_img = Image.open(gt_path).convert('L')
                
                # Metrikleri hesapla
                m = calculate_metrics(pred, gt_img)
                metrics.update(m)
                
                if m['mse'] is not None:
                    mse_values.append(m['mse'])
                    psnr_values.append(m['psnr'])
                    ssim_values.append(m['ssim'])
            else:
                metrics.update({'mse': None, 'psnr': None, 'ssim': None})
            
            results_per_image.append(metrics)
            
            # Sonucu kaydet
            pred_resized = Image.fromarray((pred * 255).astype(np.uint8))
            pred_resized = pred_resized.resize(original_size, Image.BILINEAR)
            output_name = os.path.basename(img_path).replace('.jpg', '_diffusion_pred.png')
            output_name = output_name.replace('.png', '_diffusion_pred.png')
            pred_resized.save(os.path.join(output_dir, output_name))
            
        except Exception as e:
            print(f"\nHATA [{os.path.basename(img_path)}]: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # JSON olarak kaydet
    json_path = os.path.join(output_dir, 'biped_diffusion_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_per_image, f, indent=4)
    
    # Ortalama metrikleri hesapla ve göster
    if len(mse_values) > 0:
        avg_results = {
            "mse": float(np.mean(mse_values)),
            "mse_std": float(np.std(mse_values)),
            "psnr": float(np.mean(psnr_values)),
            "psnr_std": float(np.std(psnr_values)),
            "ssim": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values)),
            "total_images": len(mse_values)
        }
        
        # Özet JSON olarak kaydet
        summary_json_path = os.path.join(output_dir, 'summary_results.json')
        with open(summary_json_path, 'w') as f:
            json.dump(avg_results, f, indent=4)
        
        # Terminal çıktısı
        print("\n" + "="*50)
        print("BIPED DIFFUSIONEDGE TEST SONUÇLARI (ORTALAMA)")
        print("="*50)
        print(f"Test Edilen Görüntü Sayısı: {avg_results['total_images']}")
        print(f"MSE  : {avg_results['mse']:.6f} ± {avg_results['mse_std']:.6f}")
        print(f"PSNR : {avg_results['psnr']:.4f} dB ± {avg_results['psnr_std']:.4f}")
        print(f"SSIM : {avg_results['ssim']:.6f} ± {avg_results['ssim_std']:.6f}")
        print("="*50)
        print(f"\n✓ Test tamamlandı!")
        print(f"✓ Sonuçlar kaydedildi: {output_dir}")
        print(f"✓ Detaylı JSON raporu: {json_path}")
        print(f"✓ Özet JSON raporu: {summary_json_path}")
        
        return {
            "average": avg_results,
            "per_image": results_per_image
        }
    else:
        print("\n" + "="*50)
        print("UYARI: GT bulunamadığı için metric hesaplanamadı!")
        print("="*50)
        print(f"\n✓ Test tamamlandı, sonuçlar kaydedildi: {output_dir}")
        print(f"✓ JSON raporu: {json_path}")
        
        return {
            "average": None,
            "per_image": results_per_image
        }

# ----------------------------
# Ana Program
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DiffusionEdge BIPED Test')
    parser.add_argument('--biped_root', type=str, 
                        default=r"biped_dataset\BIPED\BIPED\edges",
                        help='BIPED dataset root dizini')
    parser.add_argument('--config', type=str,
                        default='DiffusionEdge/configs/BSDS_sample.yaml',
                        help='Config YAML dosyası')
    parser.add_argument('--model_path', type=str,
                        default='bsds_diffedge.pt',
                        help='DiffusionEdge model dosyası (.pt)')
    parser.add_argument('--first_stage', type=str,
                        default=None,
                        help='First stage checkpoint (opsiyonel)')
    parser.add_argument('--output_dir', type=str,
                        default='biped_diffusion_results',
                        help='Sonuçların kaydedileceği dizin')
    parser.add_argument('--target_size', type=int, nargs=2,
                        default=[320, 320],
                        help='Model input boyutu (height width)')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        help='Device: cuda veya cpu')
    
    args = parser.parse_args()
    
    # Device kontrolü
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"DiffusionEdge BIPED Test")
    print(f"{'='*60}")
    print(f"Cihaz: {device}")
    print(f"BIPED Root: {args.biped_root}")
    print(f"Config: {args.config}")
    print(f"Model Path: {args.model_path}")
    if args.first_stage:
        print(f"First Stage: {args.first_stage}")
    print(f"{'='*60}\n")
    
    # Dataset kontrolü
    if not os.path.exists(args.biped_root):
        print(f"\nHATA: BIPED dataset bulunamadı: {args.biped_root}")
        exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"\nHATA: Model dosyası bulunamadı: {args.model_path}")
        exit(1)
        
    if not os.path.exists(args.config):
        print(f"\nHATA: Config dosyası bulunamadı: {args.config}")
        print("Lütfen DiffusionEdge/configs/ klasöründeki bir config dosyası belirtin")
        exit(1)
    
    try:
        # Model yükle
        model, cfg = load_diffusion_edge_model(
            args.config, 
            args.model_path, 
            first_stage_path=args.first_stage,
            device=device
        )
        
        # Test çalıştır
        results = test_biped_diffusion_edge(
            model=model,
            biped_root=args.biped_root,
            output_dir=args.output_dir,
            device=device,
            target_size=tuple(args.target_size)
        )
        
    except Exception as e:
        print(f"\nFATAL HATA: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)