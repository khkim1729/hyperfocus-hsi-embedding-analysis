import os
import yaml
import numpy as np
import scipy.io as sio
import tifffile as tiff
import matplotlib.pyplot as plt

os.makedirs("images/cross_dataset", exist_ok=True)

# ==========================================
# 1. Dataset Loaders
# ==========================================

def load_indian_pines():
    data = sio.loadmat("data/indian_pines/Indian_pines.mat")
    gt = sio.loadmat("data/indian_pines/Indian_pines_gt.mat")
    img_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    return data[img_key].astype(np.float32), gt[gt_key].astype(np.int32)

def load_botswana():
    data = sio.loadmat("data/botswana/Botswana.mat")
    gt = sio.loadmat("data/botswana/Botswana_gt.mat")
    img_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    return data[img_key].astype(np.float32), gt[gt_key].astype(np.int32)

def load_pavia_university():
    data = sio.loadmat("data/pavia_university/Pavia.mat")
    gt = sio.loadmat("data/pavia_university/Pavia_gt.mat")
    img_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    return data[img_key].astype(np.float32), gt[gt_key].astype(np.int32)

def load_pavia_centre():
    data = sio.loadmat("data/pavia_centre/PaviaCentre.mat")
    gt = sio.loadmat("data/pavia_centre/PaviaCentre_gt.mat")
    img_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    return data[img_key].astype(np.float32), gt[gt_key].astype(np.int32)

def load_hyrank():
    img = tiff.imread("data/hyrank/TrainingSet/Dioni.tif")
    gt = tiff.imread("data/hyrank/TrainingSet/Dioni_GT.tif")
    if img.ndim == 3 and img.shape[0] < img.shape[2]:
        img = img.transpose(1, 2, 0)
    return img.astype(np.float32), gt.astype(np.int32)

def minmax_scale(curves):
    c_min = curves.min(axis=1, keepdims=True)
    c_max = curves.max(axis=1, keepdims=True)
    return (curves - c_min) / (c_max - c_min + 1e-8)

# ==========================================
# 2. Loading Datasets
# ==========================================
print("Loading HSI datasets...")
ip_img, ip_gt = load_indian_pines()
bo_img, bo_gt = load_botswana()
pu_img, pu_gt = load_pavia_university()
pc_img, pc_gt = load_pavia_centre()
hr_img, hr_gt = load_hyrank()

def get_average_spectrum(img, gt, class_ids):
    X = img.reshape(-1, img.shape[2])
    y = gt.reshape(-1)
    mask = np.isin(y, class_ids)
    if np.sum(mask) == 0:
        return None
    pixels = X[mask]
    np.random.seed(42)
    sampled = pixels[np.random.choice(len(pixels), min(500, len(pixels)), replace=False)]
    return minmax_scale(sampled).mean(axis=0)

def interpolate_spectrum(spec, steps=100):
    original_coords = np.linspace(0, 1, len(spec))
    new_coords = np.linspace(0, 1, steps)
    return np.interp(new_coords, original_coords, spec)

# ==========================================
# Plot 1: Semantic Signatures (English Only)
# ==========================================
print("Generating Plot 1: Semantic Signatures...")
bo_water = get_average_spectrum(bo_img, bo_gt, [1])
pc_water = get_average_spectrum(pc_img, pc_gt, [1])
hr_water = get_average_spectrum(hr_img, hr_gt, [13, 14])
water_curves = [interpolate_spectrum(c) for c in [bo_water, pc_water, hr_water] if c is not None]
water_avg = np.mean(water_curves, axis=0)

ip_trees = get_average_spectrum(ip_img, ip_gt, [6, 14])
bo_trees = get_average_spectrum(bo_img, bo_gt, [6, 9])
pu_trees = get_average_spectrum(pu_img, pu_gt, [4])
pc_trees = get_average_spectrum(pc_img, pc_gt, [2])
hr_trees = get_average_spectrum(hr_img, hr_gt, [4, 5, 6, 7, 8])
trees_curves = [interpolate_spectrum(c) for c in [ip_trees, bo_trees, pu_trees, pc_trees, hr_trees] if c is not None]
trees_avg = np.mean(trees_curves, axis=0)

bo_soils = get_average_spectrum(bo_img, bo_gt, [14])
pu_soils = get_average_spectrum(pu_img, pu_gt, [6])
pc_soils = get_average_spectrum(pc_img, pc_gt, [9])
hr_soils = get_average_spectrum(hr_img, hr_gt, [11, 12])
soils_curves = [interpolate_spectrum(c) for c in [bo_soils, pu_soils, pc_soils, hr_soils] if c is not None]
soils_avg = np.mean(soils_curves, axis=0)

ip_urban = get_average_spectrum(ip_img, ip_gt, [16])
pu_urban = get_average_spectrum(pu_img, pu_gt, [1, 3, 5, 7, 8])
pc_urban = get_average_spectrum(pc_img, pc_gt, [3, 4, 5, 6])
hr_urban = get_average_spectrum(hr_img, hr_gt, [1, 2])
urban_curves = [interpolate_spectrum(c) for c in [ip_urban, pu_urban, pc_urban, hr_urban] if c is not None]
urban_avg = np.mean(urban_curves, axis=0)

plt.figure(figsize=(10, 6))
x_steps = np.linspace(400, 1000, 100)
plt.plot(x_steps, water_avg, color='tab:blue', linewidth=2.5, label='Water')
plt.plot(x_steps, trees_avg, color='forestgreen', linewidth=2.5, label='Trees')
plt.plot(x_steps, soils_avg, color='saddlebrown', linewidth=2.5, label='Soils')
plt.plot(x_steps, urban_avg, color='tab:grey', linewidth=2.5, label='Urban')
plt.title("Mean Spectral Reflectance Curves of 4 Common Semantic Groups", fontsize=13, fontweight='bold')
plt.xlabel("Wavelength (pseudo-nm)", fontsize=11)
plt.ylabel("Scaled Reflectance", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.savefig("images/cross_dataset/spectral_signatures_semantic.png", dpi=140)
plt.close()

# ==========================================
# Plot 2: Case Studies (English Only)
# ==========================================
print("Generating Plot 2: Case Studies...")
pu_meadows = get_average_spectrum(pu_img, pu_gt, [2])
pu_metal = get_average_spectrum(pu_img, pu_gt, [5])
pu_soil = get_average_spectrum(pu_img, pu_gt, [6])

pu_asphalt = get_average_spectrum(pu_img, pu_gt, [1])
pc_asphalt = get_average_spectrum(pc_img, pc_gt, [3])

pu_wl = np.linspace(430, 860, len(pu_meadows))
pc_wl = np.linspace(430, 860, len(pc_asphalt))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(pu_wl, pu_meadows, color='forestgreen', linewidth=2, label='Meadows (Vegetation)')
axes[0].plot(pu_wl, pu_metal, color='deepskyblue', linewidth=2, label='Painted Metal Sheets (Metal)')
axes[0].plot(pu_wl, pu_soil, color='saddlebrown', linewidth=2, label='Bare Soil (Soil)')
axes[0].set_title("Case A: Pavia University Highly Separable Classes", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Wavelength (nm)", fontsize=10)
axes[0].set_ylabel("Scaled Reflectance", fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

axes[1].plot(pu_wl, pu_asphalt, color='black', linestyle='-', linewidth=2, label='Pavia University: Asphalt')
axes[1].plot(pc_wl, pc_asphalt, color='grey', linestyle='--', linewidth=2, label='Pavia Centre: Asphalt')
axes[1].plot(pu_wl, pu_trees, color='forestgreen', linestyle='-', linewidth=2, label='Pavia University: Trees')
axes[1].plot(pc_wl, pc_trees, color='limegreen', linestyle='--', linewidth=2, label='Pavia Centre: Trees')
axes[1].set_title("Case B: Pavia U/C Intra-class Spectral Variations", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Wavelength (nm)", fontsize=10)
axes[1].set_ylabel("Scaled Reflectance", fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

plt.tight_layout()
plt.savefig("images/cross_dataset/spectral_signatures_cases.png", dpi=140)
plt.close()

# ==========================================
# Plot 3: Indian Pines (English Only)
# ==========================================
print("Generating Plot 3: Indian Pines...")
ip_corn = get_average_spectrum(ip_img, ip_gt, [2])
ip_grass = get_average_spectrum(ip_img, ip_gt, [5])
ip_soy = get_average_spectrum(ip_img, ip_gt, [10])
ip_woods = get_average_spectrum(ip_img, ip_gt, [14])
ip_stone = get_average_spectrum(ip_img, ip_gt, [16])
ip_alfalfa = get_average_spectrum(ip_img, ip_gt, [1])

ip_bands = np.arange(len(ip_corn))

plt.figure(figsize=(10, 6))
plt.plot(ip_bands, ip_corn, color='orange', label='Corn-notill')
plt.plot(ip_bands, ip_grass, color='limegreen', label='Grass-pasture')
plt.plot(ip_bands, ip_soy, color='forestgreen', label='Soybeans-notill')
plt.plot(ip_bands, ip_woods, color='darkgreen', label='Woods')
plt.plot(ip_bands, ip_stone, color='tab:grey', label='Stone-Steel-Towers')
plt.plot(ip_bands, ip_alfalfa, color='purple', label='Alfalfa')
plt.title("Spectral Signatures of Key Indian Pines Classes", fontsize=13, fontweight='bold')
plt.xlabel("Spectral Band Index", fontsize=11)
plt.ylabel("Scaled Reflectance", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.savefig("images/cross_dataset/spectral_signatures_ip.png", dpi=140)
plt.close()

# ==========================================
# Plot 4: Spectrum Interference (English Only)
# ==========================================
print("Generating Plot 4: Spectrum Interference...")
pu_asphalt_interp = interpolate_spectrum(pu_asphalt, len(ip_soy))
bo_water_interp = interpolate_spectrum(bo_water, len(ip_soy))

plt.figure(figsize=(10, 6))
plt.plot(ip_bands, ip_soy, color='forestgreen', linewidth=2.5, label='Indian Pines: Soybeans (Vegetation)')
plt.plot(ip_bands, pu_asphalt_interp, color='black', linewidth=2.5, label='Pavia University: Asphalt (Urban)')
plt.plot(ip_bands, bo_water_interp, color='tab:blue', linewidth=2.5, label='Botswana: Water (Water)')
plt.title("Spectral Curve Overlaps Causing Manifold Interference in Raw Space", fontsize=13, fontweight='bold')
plt.xlabel("Standardized Band Index", fontsize=11)
plt.ylabel("Scaled Reflectance", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.savefig("images/cross_dataset/spectral_signatures_interference.png", dpi=140)
plt.close()

# ==========================================
# Plot 5: Vegetation vs. Soil Signatures (New / English Only)
# ==========================================
print("Generating Plot 5: Vegetation vs. Soil Signatures...")
bo_riparian = get_average_spectrum(bo_img, bo_gt, [6]) # Vegetation
pu_soil_single = get_average_spectrum(pu_img, pu_gt, [6]) # Soil
bo_soil_single = get_average_spectrum(bo_img, bo_gt, [14]) # Soil

# Interpolate to common scale
veg1 = interpolate_spectrum(trees_avg)
veg2 = interpolate_spectrum(bo_riparian)
soil1 = interpolate_spectrum(pu_soil_single)
soil2 = interpolate_spectrum(bo_soil_single)

plt.figure(figsize=(10, 6))
plt.plot(x_steps, veg1, color='forestgreen', linewidth=2, label='Trees (Vegetation - Common)')
plt.plot(x_steps, veg2, color='limegreen', linewidth=2, label='Riparian (Vegetation - Botswana)')
plt.plot(x_steps, soil1, color='saddlebrown', linewidth=2, label='Bare Soil (Soil - Pavia U)')
plt.plot(x_steps, soil2, color='peru', linewidth=2, label='Exposed Soils (Soil - Botswana)')
plt.title("Spectral Signature Comparison: Vegetation vs. Soil", fontsize=13, fontweight='bold')
plt.xlabel("Wavelength (pseudo-nm)", fontsize=11)
plt.ylabel("Scaled Reflectance", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.savefig("images/cross_dataset/spectral_signatures_vegetation_soil.png", dpi=140)
plt.close()

# ==========================================
# Plot 6: Water vs. Urban Signatures (New / English Only)
# ==========================================
print("Generating Plot 6: Water vs. Urban Signatures...")
bo_water_single = get_average_spectrum(bo_img, bo_gt, [1])
pu_asphalt_single = get_average_spectrum(pu_img, pu_gt, [1])
pu_metal_single = get_average_spectrum(pu_img, pu_gt, [5])

water1 = interpolate_spectrum(bo_water_single)
urban1 = interpolate_spectrum(pu_asphalt_single)
urban2 = interpolate_spectrum(pu_metal_single)

plt.figure(figsize=(10, 6))
plt.plot(x_steps, water1, color='tab:blue', linewidth=2.5, label='Water (Botswana)')
plt.plot(x_steps, urban1, color='tab:grey', linewidth=2.5, label='Asphalt (Urban - Pavia U)')
plt.plot(x_steps, urban2, color='deepskyblue', linewidth=2.5, label='Painted Metal (Urban - Pavia U)')
plt.title("Spectral Signature Comparison: Water vs. Urban", fontsize=13, fontweight='bold')
plt.xlabel("Wavelength (pseudo-nm)", fontsize=11)
plt.ylabel("Scaled Reflectance", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.savefig("images/cross_dataset/spectral_signatures_water_urban.png", dpi=140)
plt.close()

print("All spectral signature plots generated successfully in English!")
