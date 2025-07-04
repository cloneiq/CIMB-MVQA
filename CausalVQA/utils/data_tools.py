import numpy as np
from math import sqrt
import cv2
import os
from PIL import Image

def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0, hard_negative=True, strategy='advanced'):
    lam = np.random.uniform(0, alpha)
    # lam = 0.5
    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    if not hard_negative:
        img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                            h_start:h_start + h_crop,
                                                                                            w_start:w_start + w_crop]
        img2_pha_mix = img2_pha
    elif strategy == 'basic':
        img2_pha_mix = img2_pha.copy()
        mix_ratio = 0.7
        img2_pha_mix[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            (1 - mix_ratio) * img2_pha[h_start:h_start + h_crop, w_start:w_start + w_crop] + \
            mix_ratio * img1_pha[h_start:h_start + h_crop, w_start:w_start + w_crop]
    elif strategy == 'confusing':
        # Special strategy: create hard negative samples, adjust mixing ratio to make model confidence around 50%
        # 1. Retain more phase information from the original image img1 to enhance semantic confusion
        img2_pha_mix = img2_pha.copy()
        
        # Increase phase mixing ratio to 80-90%, retain more phase information from img1
        phase_mix_ratio = 0.85
        img2_pha_mix = (1 - phase_mix_ratio) * img2_pha + phase_mix_ratio * img1_pha
        
        # 2. Partially mix magnitude information, but retain more img2 features in key semantic regions
        # Create radial distance mask
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h/2, w/2
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(h**2 + w**2)/2
        normalized_dist = dist_from_center / max_dist
        
        # Retain more img2 features in the center, more img1 features in the periphery
        for i in range(h):
            for j in range(w):
                # In the center region, keep more img2 magnitude information
                if normalized_dist[i, j] < 0.3:  # center region
                    mix_factor = 0.3  # less mixing, retain more img2 features
                else:  # peripheral region
                    mix_factor = 0.6 + 0.2 * (normalized_dist[i, j] - 0.3)  # more mixing with img1 features
                
                img2_abs[i, j] = mix_factor * img1_abs_[i, j] + (1 - mix_factor) * img2_abs_[i, j]
        
        # 3. Add low-intensity random phase noise to further confuse semantics
        phase_noise = np.random.uniform(-0.1, 0.1, img2_pha.shape)
        img2_pha_mix += 0.1 * phase_noise
    elif strategy == 'frequency_bands':
        # Strategy 1: Frequency band hierarchical mixing strategy
        h_mid = h_crop // 3
        w_mid = w_crop // 3
        h_mid_start = h_start + h_mid
        w_mid_start = w_start + w_mid
        
        # Low-frequency region (global structure) slight mixing
        img2_abs[h_start:h_mid_start, w_start:w_mid_start] = \
            0.2 * img1_abs_[h_start:h_mid_start, w_start:w_mid_start] + 0.8 * img2_abs_[h_start:h_mid_start, w_start:w_mid_start]
        
        # Mid-frequency region (main features) strong mixing
        img2_abs[h_mid_start:h_start+2*h_mid, w_mid_start:w_start+2*w_mid] = \
            0.8 * img1_abs_[h_mid_start:h_start+2*h_mid, w_mid_start:w_start+2*w_mid] + 0.2 * img2_abs_[h_mid_start:h_start+2*h_mid, w_mid_start:w_start+2*w_mid]
        
        # Phase mixing
        img2_pha_mix = img2_pha.copy()
        mix_ratio = 0.7
        img2_pha_mix[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            (1 - mix_ratio) * img2_pha[h_start:h_start + h_crop, w_start:w_start + w_crop] + \
            mix_ratio * img1_pha[h_start:h_start + h_crop, w_start:w_start + w_crop]
        
        # Add controlled noise to phase
        phase_noise = np.random.uniform(-0.2, 0.2, img2_pha.shape) 
        img2_pha_mix[h_mid_start:h_start+2*h_mid, w_mid_start:w_start+2*w_mid] += \
            0.1 * phase_noise[h_mid_start:h_start+2*h_mid, w_mid_start:w_start+2*w_mid]
    elif strategy == 'perceptual':
        # Strategy 2: Perceptual importance weighting
        # Convert to grayscale for processing
        if c == 3:
            img1_gray = np.mean(img1, axis=2).astype(np.uint8)
        else:
            img1_gray = img1[..., 0].astype(np.uint8)
            
        # Calculate gradient using Sobel operator
        gradient_x = cv2.Sobel(img1_gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(img1_gray, cv2.CV_64F, 0, 1, ksize=3)
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        # Normalize to [0,1] range
        importance_map = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        # Use importance map to adjust mixing ratio (applied in frequency domain)
        importance_map_resized = cv2.resize(importance_map, (w, h))
        # Apply importance-weighted mixing in the cropped region
        for i in range(h_start, h_start + h_crop):
            for j in range(w_start, w_start + w_crop):
                # Important regions retain more original features, less important regions mix more
                local_mix = 0.3 + 0.6 * importance_map_resized[i, j]  # mixing range from 0.3 to 0.9
                img2_abs[i, j] = local_mix * img1_abs_[i, j] + (1 - local_mix) * img2_abs_[i, j]
        
        # Phase mixing
        img2_pha_mix = img2_pha.copy()
        # Important regions also retain more original phase features
        for i in range(h_start, h_start + h_crop):
            for j in range(w_start, w_start + w_crop):
                local_mix = 0.3 + 0.5 * importance_map_resized[i, j]  # mixing range from 0.3 to 0.8
                img2_pha_mix[i, j] = (1 - local_mix) * img2_pha[i, j] + local_mix * img1_pha[i, j]
    elif strategy == 'radial':
        # Strategy 3: Radial frequency mixing
        # Create radial distance mask from the center
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h/2, w/2
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(h**2 + w**2)/2
        normalized_dist = dist_from_center / max_dist
        
        # Apply radial mixing in the cropped region
        for i in range(h_start, h_start + h_crop):
            for j in range(w_start, w_start + w_crop):
                # The closer to the center, the higher the mixing ratio
                radial_mix = 0.8 * (1 - normalized_dist[i, j]) + 0.2
                img2_abs[i, j] = radial_mix * img1_abs_[i, j] + (1 - radial_mix) * img2_abs_[i, j]
                
        # Phase mixing
        img2_pha_mix = img2_pha.copy()
        for i in range(h_start, h_start + h_crop):
            for j in range(w_start, w_start + w_crop):
                radial_mix = 0.7 * (1 - normalized_dist[i, j]) + 0.2
                img2_pha_mix[i, j] = (1 - radial_mix) * img2_pha[i, j] + radial_mix * img1_pha[i, j]
    elif strategy == 'medical':
        # Hard negative sample generation for medical images
        img2_pha_mix = img2_pha.copy()
        
        # 1. Calculate salient regions of medical images (using simple threshold or Sobel operator)
        if c == 3:
            img2_gray = np.mean(img2, axis=2).astype(np.uint8)
        else:
            img2_gray = img2[..., 0].astype(np.uint8)
        
        # Use adaptive threshold to find possible ROI regions
        _, roi_mask = cv2.threshold(img2_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        roi_mask = cv2.dilate(roi_mask, np.ones((5,5),np.uint8), iterations=1)
        
        # 2. Use different mixing strategies for ROI and non-ROI regions
        # Phase mixing: ROI retains 65% original phase, non-ROI retains 95% original phase
        phase_mix_ratio_roi = 0.65
        phase_mix_ratio_non_roi = 0.95
        
        for i in range(h):
            for j in range(w):
                if roi_mask[i,j] > 0:  # ROI
                    img2_pha_mix[i,j] = (1-phase_mix_ratio_roi) * img2_pha[i,j] + phase_mix_ratio_roi * img1_pha[i,j]
                    
                    img2_abs[i,j] = 0.3 * img1_abs_[i,j] + 0.7 * img2_abs_[i,j]
                else:  
                    img2_pha_mix[i,j] = (1-phase_mix_ratio_non_roi) * img2_pha[i,j] + phase_mix_ratio_non_roi * img1_pha[i,j]
                    img2_abs[i,j] = 0.7 * img1_abs_[i,j] + 0.3 * img2_abs_[i,j]

        phase_noise = np.random.uniform(-0.1, 0.1, img2_pha.shape)
        for i in range(h):
            for j in range(w):
                if roi_mask[i,j] > 0:
                    img2_pha_mix[i,j] += 0.05 * phase_noise[i,j]  
                else:
                    img2_pha_mix[i,j] += 0.15 * phase_noise[i,j]  
    else:  

        h_mid = h_crop // 3
        w_mid = w_crop // 3
        h_mid_start = h_start + h_mid
        w_mid_start = w_start + w_mid
        
        if c == 3:
            img1_gray = np.mean(img1, axis=2).astype(np.uint8)
        else:
            img1_gray = img1[..., 0].astype(np.uint8)
            
        gradient_x = cv2.Sobel(img1_gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(img1_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        importance_map = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        importance_map_resized = cv2.resize(importance_map, (w, h))
        
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h/2, w/2
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(h**2 + w**2)/2
        normalized_dist = dist_from_center / max_dist
        
        for i in range(h_start, h_start + h_crop):
            for j in range(w_start, w_start + w_crop):
            
                combined_factor = 0.5 * importance_map_resized[i, j] + 0.5 * (1 - normalized_dist[i, j])
                if i < h_mid_start and j < w_mid_start:  
                    mix_factor = 0.3 * combined_factor  
                elif h_mid_start <= i < h_start+2*h_mid and w_mid_start <= j < w_start+2*w_mid:  
                    mix_factor = 0.8 * combined_factor  
                else:  
                    mix_factor = 0.5 * combined_factor  
                
                img2_abs[i, j] = mix_factor * img1_abs_[i, j] + (1 - mix_factor) * img2_abs_[i, j]

        img2_pha_mix = img2_pha.copy()
        phase_noise = np.random.uniform(-0.15, 0.15, img2_pha.shape)
        
        for i in range(h_start, h_start + h_crop):
            for j in range(w_start, w_start + w_crop):
                combined_factor = 0.6 * importance_map_resized[i, j] + 0.4 * (1 - normalized_dist[i, j])
                
                if i < h_mid_start and j < w_mid_start:  
                    mix_ratio = 0.2 * combined_factor
                    noise_factor = 0.05
                elif h_mid_start <= i < h_start+2*h_mid and w_mid_start <= j < w_start+2*w_mid:  # 中频区域
                    mix_ratio = 0.7 * combined_factor
                    noise_factor = 0.15
                else:  
                    mix_ratio = 0.4 * combined_factor
                    noise_factor = 0.1
                
                img2_pha_mix[i, j] = (1 - mix_ratio) * img2_pha[i, j] + mix_ratio * img1_pha[i, j]
                img2_pha_mix[i, j] += noise_factor * phase_noise[i, j]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha_mix))  
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12




