from flask import Flask, request, jsonify, send_from_directory
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from pyzbar.pyzbar import decode
import os

app = Flask(__name__, static_folder='.')  # Serve static files (like index.html) from current directory

# Optional: Load CNN model only if file exists
model_path = 'rail_qr_cnn_model.keras'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    use_cnn = True
    print("CNN model loaded successfully.")
else:
    model = None
    use_cnn = False
    print("Warning: No CNN model found. Using direct decoding.")

def preprocess_image(image_data):
    # Decode base64
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    img_data = base64.b64decode(image_data)
    img = Image.open(BytesIO(img_data))
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Convert to HSV for rust handling (rust is reddish-brown)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    # Mask rust (high saturation red/orange hues: H 0-30, S>50, V>50)
    lower_rust = np.array([0, 50, 50])
    upper_rust = np.array([30, 255, 255])
    rust_mask = cv2.inRange(hsv, lower_rust, upper_rust)
    # Inpaint rust areas (replace with median filter to smooth)
    img_cv[rust_mask > 0] = cv2.medianBlur(img_cv[rust_mask > 0], 5)
    
    # Back to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Denoise (stronger for dirt/scratches on rail)
    denoised = cv2.bilateralFilter(gray, 15, 100, 100)  # Increased params for texture
    
    # Contrast enhancement (CLAHE, but adaptive for uneven metal lighting)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))  # Higher clip for engravings
    enhanced = clahe.apply(denoised)
    
    # Edge sharpening for engravings (Laplacian filter)
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(enhanced - 0.7 * laplacian)
    
    # Adaptive threshold for low-contrast engraved QRs (better than OTSU for varying light)
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations to clean up (close gaps in rusted edges)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # If using CNN: Resize and predict
    if use_cnn:
        resized = cv2.resize(thresh, (224, 224))
        input_array = np.expand_dims(resized, axis=(0, -1)) / 255.0
        prediction = model.predict(input_array, verbose=0)
        print(f"CNN Confidence for QR: {prediction[0][0]:.2f}")  # Debug log
        if prediction[0][0] < 0.3:  # Lowered threshold for degraded QRs
            return None
    
    # DEBUG: Save for inspection (remove after testing)
    cv2.imwrite('debug_rail_qr.png', thresh)
    
    return thresh  # Processed image for decoding

@app.route('/api/scan-qr', methods=['POST'])
def scan_qr():
         try:
             data = request.json
             if not data or 'image' not in data:
                 return jsonify({'error': 'No image provided', 'success': False}), 400
             
             processed_img = preprocess_image(data['image'])
             if processed_img is None:
                 return jsonify({'error': 'QR detection failed (low confidence)', 'success': False})
             
             # Enhanced decoding for railtrack/rusted QRs (multi-scale and rotations)
             if processed_img is not None:
                 # Try multiple scales (for distant rail views)
                 scales = [1.0, 0.8, 1.2]
                 decoded_objects = []
                 for scale in scales:
                     scaled = cv2.resize(processed_img, None, fx=scale, fy=scale)
                     decoded_objects.extend(decode(scaled))
                     if decoded_objects:
                         break
                 
                 # If none, try rotations (0, 90, 180, 270 deg for angled rail)
                 if not decoded_objects:
                     for angle in [0, 90, 180, 270]:
                         if angle == 0:
                             rot_img = processed_img
                         else:
                             rot_img = cv2.rotate(processed_img, 
                                                  cv2.ROTATE_90_COUNTERCLOCKWISE if angle == 90 else 
                                                  cv2.ROTATE_180 if angle == 180 else 
                                                  cv2.ROTATE_90_CLOCKWISE)
                         decoded_objects.extend(decode(rot_img))
                         if decoded_objects:
                             break
                 
                 if decoded_objects:
                     # Use first valid decode (pyzbar handles error correction)
                     code = decoded_objects[0].data.decode('utf-8')
                     return jsonify({'code': code, 'success': True})
                 else:
                     return jsonify({'error': 'No QR decoded. Try closer angle/cleaner view.', 'success': False})
             else:
                 return jsonify({'error': 'Processing failed', 'success': False})
         
         except Exception as e:
             print(f"Error in scan_qr: {str(e)}")  # Log to terminal
             return jsonify({'error': f'Processing error: {str(e)}', 'success': False}), 500

# NEW: Route to serve the frontend (index.html) at root URL
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# NEW: Catch-all route for other static files (e.g., style.css, script.js if added)
@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Allow external access if needed