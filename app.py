from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import subprocess
import yaml
from flask import send_from_directory



app = Flask(__name__, static_folder='/Frutiripe/YOLOv5_EfficientNetLite_Web/YOLOv5_EfficientNetLite_Web/runs/detect')
app.config['UPLOAD_FOLDER'] = '/Frutiripe/YOLOv5_EfficientNetLite_Web/Flask_ML/Upload'  # specify your upload directory

# Load drug information
cfg_model_path = "D:/Frutiripe/YOLOv5_EfficientNetLite_Web/Flask_ML/runs/train/exp/weights/best.pt"
data_yaml_path = "D:/Frutiripe/YOLOv5_EfficientNetLite_Web/Flask_ML/Frutyripe-4/data.yaml"

# Load class names from YAML
def load_class_names(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
        return data['names']

class_names = load_class_names(data_yaml_path)

# Get the latest directory
def get_latest_directory(base_path):
    exp_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith('exp')]
    if not exp_dirs:
        return None
    return os.path.join(base_path, max(exp_dirs, key=lambda d: int(d[3:]) if d != 'exp' else 0))

# Run detection
def run_detection(image_path):
    if image_path is None:
        raise ValueError("image_path is None")
    output_base_dir = '/Frutiripe/YOLOv5_EfficientNetLite_Web/Flask_ML/runs/detect'
    # Ensure the output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    subprocess.run([
        'python', '/Frutiripe/YOLOv5_EfficientNetLite_Web/Flask_ML/detect.py',
        '--weights', cfg_model_path,
        '--data', data_yaml_path,
        '--img', '512',
        '--conf', '0.4',
        '--source', image_path,
        '--save-txt',
        '--save-conf',
    ])
    return get_latest_directory(output_base_dir)

# Read detected classes
def read_detected_images(detected_image_dir):
    detected_images = []
    for file in os.listdir(detected_image_dir):
        if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
            detected_images.append(os.path.join(detected_image_dir, file))
    return detected_images

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            imgpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            detected_image_dir = run_detection(imgpath)
            detected_images = read_detected_images(detected_image_dir)
            detected_image_paths = [os.path.relpath(img, start=app.static_folder) for img in detected_images]
            return render_template('index.html', detected_images=detected_image_paths)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
