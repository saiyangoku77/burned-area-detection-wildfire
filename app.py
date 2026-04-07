"""
=============================================================
 Burned Area Detection — Flask Web Application
 app.py
=============================================================

 HOW TO RUN:
     1. Make sure your model is trained and saved:
        model/burned_area_model.h5

     2. Install dependencies (if not done already):
        pip install flask tensorflow pillow numpy opencv-python

     3. From your project root folder, run:
        python app.py

     4. Open browser → http://127.0.0.1:5000

=============================================================
"""

import os
import io
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageDraw, ImageFilter
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import cv2  # OpenCV for drawing border highlights

# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(__name__)

# Maximum allowed upload size: 16 MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Folder where uploaded images are temporarily stored
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed image file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Model settings — must match what was used during training
IMG_SIZE        = (224, 224)   # MobileNetV2 input size
MODEL_PATH      = os.path.join('model', 'burned_area_model.h5')
THRESHOLD       = 0.5          # Decision boundary: below = burned, above = not burned

# Border highlight colors (BGR format for OpenCV)
COLOR_BURNED     = (0, 0, 255)    # Red  → burned area
COLOR_NOT_BURNED = (0, 200, 50)   # Green → not burned

# ============================================================
# LOAD MODEL AT STARTUP
# ============================================================
# Load once when Flask starts — not on every request (too slow)

print("🔄 Loading trained model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\n❌ Model not found at: {MODEL_PATH}\n"
        "   Please run the Jupyter notebook to train and save the model first.\n"
    )

# Load the model — suppress verbose TF output
tf.get_logger().setLevel('ERROR')
model = load_model(MODEL_PATH)
model.predict(np.zeros((1, 224, 224, 3)), verbose=0)  # Warm-up prediction

print(f"✅ Model loaded successfully from: {MODEL_PATH}")
print(f"   Input shape expected: {model.input_shape}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):
    """Check if the uploaded file has a valid image extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(pil_image):
    """
    Preprocess a PIL Image for model prediction.
    
    Steps:
    1. Convert to RGB (handles RGBA, grayscale, etc.)
    2. Resize to 224x224 (model's required input size)
    3. Normalize pixel values to [0, 1]
    4. Add batch dimension: shape (1, 224, 224, 3)
    
    Args:
        pil_image : PIL.Image object
    
    Returns:
        numpy array of shape (1, 224, 224, 3)
    """
    # Convert to RGB — satellite images may be RGBA or other formats
    pil_image = pil_image.convert('RGB')
    
    # Resize to model's expected input size
    pil_image = pil_image.resize(IMG_SIZE, Image.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = keras_image.img_to_array(pil_image)   # Shape: (224, 224, 3)
    img_array = img_array / 255.0                      # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)      # Shape: (1, 224, 224, 3)
    
    return img_array


def predict(img_array):
    """
    Run model prediction and return label + confidence.
    
    The model outputs a single float (sigmoid output):
    - Close to 0 → burned (class index 0)
    - Close to 1 → not_burned (class index 1)
    
    This is because Keras reads folders alphabetically:
    'burned' comes before 'not_burned' → index 0 = burned
    
    Args:
        img_array : preprocessed numpy array (1, 224, 224, 3)
    
    Returns:
        label (str)      : 'BURNED' or 'NOT BURNED'
        confidence (float): percentage confidence in prediction
        raw_score (float) : raw sigmoid output for debugging
    """
    raw_score = float(model.predict(img_array, verbose=0)[0][0])
    
    if raw_score < THRESHOLD:
        # Low score = class 0 = burned
        label      = 'BURNED'
        confidence = round((1.0 - raw_score) * 100, 2)
    else:
        # High score = class 1 = not burned
        label      = 'NOT BURNED'
        confidence = round(raw_score * 100, 2)
    
    return label, confidence, raw_score


def add_border_highlight(pil_image, is_burned):
    """
    Add a colored border around the image.
    - Red border  → burned area detected
    - Green border → area is NOT burned
    
    Args:
        pil_image : Original PIL Image (any size)
        is_burned : bool — True if burned, False if not
    
    Returns:
        PIL Image with colored border, encoded as base64 PNG string
    """
    # Convert PIL image to OpenCV format (RGB → BGR)
    original_rgb = pil_image.convert('RGB')
    img_cv = np.array(original_rgb)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions
    h, w = img_cv.shape[:2]
    
    # Calculate border thickness proportional to image size
    border_thickness = max(8, min(h, w) // 25)
    
    # Choose border color based on prediction
    color = COLOR_BURNED if is_burned else COLOR_NOT_BURNED
    
    # Draw colored rectangle border around the image
    cv2.rectangle(
        img_cv,
        (0, 0),          # Top-left corner
        (w - 1, h - 1),  # Bottom-right corner
        color,           # Border color (BGR)
        border_thickness # Thickness in pixels
    )
    
    # Add a second inner border for visual depth
    inner_offset = border_thickness + 3
    inner_color = (255, 255, 255)  # White inner border for contrast
    cv2.rectangle(
        img_cv,
        (inner_offset, inner_offset),
        (w - 1 - inner_offset, h - 1 - inner_offset),
        inner_color,
        2   # Thin inner white line
    )
    
    # Add label text directly on the image (top-left corner)
    label_text = "BURNED" if is_burned else "NOT BURNED"
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.5, min(h, w) / 600)  # Scale font with image size
    thickness  = 2
    
    # Draw text shadow (black background for readability)
    text_x, text_y = border_thickness + 10, border_thickness + 35
    cv2.putText(img_cv, label_text, (text_x + 2, text_y + 2),
                font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    
    # Draw main label text in color
    cv2.putText(img_cv, label_text, (text_x, text_y),
                font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Convert back to RGB for PIL, then encode as base64
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(img_rgb)
    
    # Encode as base64 PNG string to send directly to HTML
    buffer = io.BytesIO()
    result_pil.save(buffer, format='PNG', quality=95)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return img_base64


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    """
    Main page — serve the HTML upload form.
    Templates are looked up in the 'templates/' folder automatically.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    """
    Prediction endpoint — called when user uploads an image and clicks Detect.
    
    Expects: multipart/form-data with 'image' file field
    Returns: JSON response with prediction results + base64 highlighted image
    
    Response format:
    {
        "success"    : true/false,
        "label"      : "BURNED" | "NOT BURNED",
        "confidence" : 87.4,
        "raw_score"  : 0.126,
        "image_b64"  : "data:image/png;base64,...",
        "error"      : "error message if success=false"
    }
    """
    # --- Validate that a file was uploaded ---
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file found in request.'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected. Please choose an image.'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # --- Read the image from uploaded bytes ---
        # We don't save to disk — read directly into PIL from memory
        file_bytes = file.read()
        pil_image  = Image.open(io.BytesIO(file_bytes))
        
        # Keep original for display (before resizing for model)
        display_image = pil_image.copy()
        
        # --- Preprocess for model ---
        img_array = preprocess_image(pil_image)
        
        # --- Run prediction ---
        label, confidence, raw_score = predict(img_array)
        is_burned = (label == 'BURNED')
        
        # --- Add border highlight to display image ---
        # Resize display image to max 600px width for web display
        max_display_width = 600
        display_w, display_h = display_image.size
        if display_w > max_display_width:
            ratio         = max_display_width / display_w
            display_w     = max_display_width
            display_h     = int(display_h * ratio)
            display_image = display_image.resize((display_w, display_h), Image.LANCZOS)
        
        img_base64 = add_border_highlight(display_image, is_burned)
        
        # --- Return response ---
        return jsonify({
            'success'    : True,
            'label'      : label,
            'confidence' : confidence,
            'raw_score'  : round(raw_score, 4),
            'image_b64'  : f'data:image/png;base64,{img_base64}'
        })
    
    except Exception as e:
        # Return error details for debugging
        import traceback
        return jsonify({
            'success': False,
            'error'  : f'Prediction failed: {str(e)}',
            'trace'  : traceback.format_exc()
        }), 500


# ============================================================
# RUN THE APP
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  🔥 Burned Area Detection — Web App Running")
    print("="*55)
    print("  Open your browser and go to: http://127.0.0.1:5000")
    print("  Press CTRL+C to stop the server\n")
    
    # debug=True → auto-reloads when you change code (dev only)
    # host='0.0.0.0' → accessible on your local network too
    app.run(debug=True, host='0.0.0.0', port=5000)
