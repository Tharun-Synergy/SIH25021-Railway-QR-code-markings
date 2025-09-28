import qrcode
import cv2
import numpy as np
import os
from PIL import Image
import random
import math
import shutil  # For val split

# Create directories
base_path = 'dataset_rail/train'
os.makedirs(f'{base_path}/qr', exist_ok=True)
os.makedirs(f'{base_path}/no_qr', exist_ok=True)
os.makedirs('dataset_rail/val/qr', exist_ok=True)
os.makedirs('dataset_rail/val/no_qr', exist_ok=True)

def generate_rust_texture(h, w):
    """Generate rust-like texture: reddish-brown spots and streaks"""
    # Base metal gray
    metal = np.random.randint(100, 180, (h, w), dtype=np.uint8)
    # Add rust: Random blobs in red hue
    num_blobs = random.randint(5, 20)
    for _ in range(num_blobs):
        cx, cy = random.randint(0, w), random.randint(0, h)
        radius = random.randint(5, 20)
        cv2.circle(metal, (cx, cy), radius, random.randint(50, 120), -1)  # Dark rust spots
    # Streaks (horizontal for rail)
    for _ in range(random.randint(3, 8)):
        y = random.randint(0, h)
        x1, x2 = random.randint(0, w//2), random.randint(w//2, w)
        cv2.line(metal, (x1, y), (x2, y), random.randint(40, 100), random.randint(1, 3))
    return metal

# Generate QR on rusted rail (engraved simulation: low contrast overlay)
for i in range(1500):  # More samples for robustness
    # Create QR
    qr = qrcode.QRCode(version=2, box_size=8, border=4)  # Smaller for engraving sim
    qr.add_data(f'rail_qr_{i}_track_id')  # Rail-specific data
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_array = np.array(qr_img.convert('L'))  # Grayscale
    
    # Resize to railtrack size (e.g., 300x300) - FIXED: Full size for blending
    h, w = 300, 300
    qr_resized = cv2.resize(qr_array, (w, h))  # Full size to match background
    
    # Generate rusted rail background
    rail_bg = generate_rust_texture(h, w)
    
    # Ensure shapes match (debug assertion)
    assert rail_bg.shape == qr_resized.shape, f"Shape mismatch: bg {rail_bg.shape}, qr {qr_resized.shape}"
    
    # Simulate engraving: Overlay inverted QR with low contrast (dark on light metal)
    alpha = 0.4  # Low visibility for engraving
    qr_inverted = 255 - qr_resized  # Dark QR (engraved effect)
    engraved = cv2.addWeighted(rail_bg, 1 - alpha, qr_inverted, alpha, 0)  # Now same size!
    
    # Add wear: Scratches, blur, noise
    # Scratches
    for _ in range(random.randint(2, 5)):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        cv2.line(engraved, (x1, y1), (x2, y2), 0, random.randint(1, 2))  # Black scratches
    # Blur for depth/wear
    engraved = cv2.GaussianBlur(engraved, (3, 3), 0)
    # Salt-pepper noise for dirt
    noise = np.random.random((h, w))
    engraved[noise < 0.01] = 0  # Black noise
    engraved[noise > 0.99] = 255  # White noise
    
    # Perspective distortion (rail angle) - Apply to full image
    pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])  # Full corners
    pts2 = np.float32([[0,0], [w-20,0], [10,h], [w,h+10]])  # Slight skew for perspective
    M = cv2.getPerspectiveTransform(pts1, pts2)
    engraved = cv2.warpPerspective(engraved, M, (w, h))
    
    cv2.imwrite(f'{base_path}/qr/rail_qr_{i}.png', engraved)

# No QR: Pure rusted rail backgrounds
for i in range(1500):
    bg = generate_rust_texture(300, 300)
    # Add more wear
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    cv2.imwrite(f'{base_path}/no_qr/rail_no_{i}.png', bg)

# Val split (20%)
for i in range(300):
    shutil.copy(f'{base_path}/qr/rail_qr_{i}.png', f'dataset_rail/val/qr/')
    shutil.copy(f'{base_path}/no_qr/rail_no_{i}.png', f'dataset_rail/val/no_qr/')

print('Railtrack-specific dataset generated! (3000 images)')