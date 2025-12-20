#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HED + STARE (VK) Test
Quantitative evaluation on 20 labeled STARE images
Metrics: Precision, Recall, F1, MSE, PSNR, SSIM
"""

import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# --------------------------------------------------
# MODEL
# --------------------------------------------------

def load_hed_model(model_path='hed_pretrained_bsds.caffemodel',
                   prototxt_path='deploy.prototxt'):
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net


def detect_edges(net, image):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0,
        size=(W, H),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False,
        crop=False
    )
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    return hed


def normalize(img):
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img


# --------------------------------------------------
# STARE PAIRS
# --------------------------------------------------

def get_test_pairs_stare_vk(stare_root):
    raw_dir = os.path.join(stare_root, "stare_raw")
    gt_dir  = os.path.join(stare_root, "stare_vk")

    print("Raw images:", raw_dir)
    print("GT labels :", gt_dir)

    raw_images = glob(os.path.join(raw_dir, "*.ppm"))
    gt_images  = glob(os.path.join(gt_dir, "*.ppm"))

    raw_dict = {
        os.path.splitext(os.path.basename(p))[0].lower(): p
        for p in raw_images
    }

    pairs = []

    for gt_path in gt_images:
        gt_name = os.path.basename(gt_path).lower()
        base = gt_name.replace(".vk", "")
        base = os.path.splitext(base)[0]

        if base in raw_dict:
            pairs.append((raw_dict[base], gt_path))
        else:
            print("Eşleşmeyen GT:", gt_name)

    return pairs


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_prf(pred, gt):
    pred_bin = (pred > 0.3).astype(np.uint8)
    gt_bin   = (gt > 0.5).astype(np.uint8)

    tp = np.sum((pred_bin == 1) & (gt_bin == 1))
    fp = np.sum((pred_bin == 1) & (gt_bin == 0))
    fn = np.sum((pred_bin == 0) & (gt_bin == 1))

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1


def calculate_mse_psnr_ssim(pred, gt):
    mse_val = np.mean((pred - gt) ** 2)
    psnr_val = psnr(gt, pred, data_range=1.0)
    ssim_val = ssim(gt, pred, data_range=1.0)
    return mse_val, psnr_val, ssim_val


# --------------------------------------------------
# TEST
# --------------------------------------------------

def test_stare(stare_root, output_dir="stare_results", save_images=True):
    print("Model yükleniyor...")
    net = load_hed_model()

    print("STARE pairs bulunuyor...")
    pairs = get_test_pairs_stare_vk(stare_root)
    print(f"Toplam {len(pairs)} görüntü test edilecek")

    os.makedirs(output_dir, exist_ok=True)

    precisions, recalls, f1s = [], [], []
    mses, psnrs, ssims = [], [], []
    per_image = []

    for img_path, gt_path in tqdm(pairs):
        image = cv2.imread(img_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if image is None or gt is None:
            continue

        pred = detect_edges(net, image)
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))

        pred = normalize(pred)
        gt   = normalize(gt)

        p, r, f1 = calculate_prf(pred, gt)
        mse_val, psnr_val, ssim_val = calculate_mse_psnr_ssim(pred, gt)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        mses.append(mse_val)
        psnrs.append(psnr_val)
        ssims.append(ssim_val)

        name = os.path.basename(img_path)
        per_image.append({
            "image": name,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "mse": float(mse_val),
            "psnr": float(psnr_val),
            "ssim": float(ssim_val)
        })

        if save_images:
            base = os.path.splitext(name)[0]
            cv2.imwrite(
                os.path.join(output_dir, base + "_hed.png"),
                (pred * 255).astype(np.uint8)
            )

    results = {
        "average": {
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "f1": float(np.mean(f1s)),
            "mse": float(np.mean(mses)),
            "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims)),
            "total_images": len(per_image)
        },
        "per_image": per_image
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 50)
    print("STARE (VK) RESULTS")
    print("=" * 50)
    print(f"Images   : {results['average']['total_images']}")
    print(f"Precision: {results['average']['precision']:.4f}")
    print(f"Recall   : {results['average']['recall']:.4f}")
    print(f"F1-score : {results['average']['f1']:.4f}")
    print(f"MSE      : {results['average']['mse']:.6f}")
    print(f"PSNR     : {results['average']['psnr']:.4f} dB")
    print(f"SSIM     : {results['average']['ssim']:.4f}")
    print("=" * 50)

    return results


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    STARE_ROOT = r"C:\Users\emredogan\Desktop\itu_master\fuzzlogicproject\stare"

    if not os.path.exists(STARE_ROOT):
        print("Hata: STARE dataset bulunamadı:", STARE_ROOT)
        exit(1)

    test_stare(
        stare_root=STARE_ROOT,
        output_dir="stare_results",
        save_images=True
    )
