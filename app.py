from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import os
import base64
from io import BytesIO
from PIL import Image
import tempfile
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, 
           static_url_path='/static',
           static_folder='static')
CORS(app)

# Constants
IMG_SIZE = (224, 224)
DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

CLINICAL_THRESHOLDS = {
    'Atelectasis': 0.35, 'Cardiomegaly': 0.3, 'Consolidation': 0.35,
    'Edema': 0.3, 'Effusion': 0.35, 'Emphysema': 0.35, 'Fibrosis': 0.4,
    'Hernia': 0.25, 'Infiltration': 0.45, 'Mass': 0.3, 'Nodule': 0.4,
    'Pleural_Thickening': 0.4, 'Pneumonia': 0.3, 'Pneumothorax': 0.35
}

class ChestXrayModel:
    def __init__(self, model_path='dns_nih_chest_xray_model_finetuned.h5'):
        logger.info(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        logger.info("Model loaded successfully.")
        
        self.last_conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                self.last_conv_layer = layer.name
                break
            elif hasattr(layer, 'output_shape') and len(getattr(layer, 'output_shape')) == 4:
                self.last_conv_layer = layer.name
                break
        
        if not self.last_conv_layer:
            logger.warning("Could not find appropriate convolutional layer automatically.")
            if any('conv5_block16_concat' in layer.name for layer in self.model.layers):
                self.last_conv_layer = 'conv5_block16_concat'
            else:
                for layer in reversed(self.model.layers):
                    if hasattr(layer, '_outbound_nodes') and len(layer.output_shape) == 4:
                        self.last_conv_layer = layer.name
                        break
            logger.info(f"Falling back to {self.last_conv_layer} for GradCAM visualization.")
        else:
            logger.info(f"Using {self.last_conv_layer} for GradCAM visualization.")
    
    def preprocess_dicom(self, dicom_data):
        try:
            # First attempt to read the DICOM file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_file:
                temp_file.write(dicom_data)
                temp_file_path = temp_file.name
            
            dicom = pydicom.dcmread(temp_file_path)
            os.unlink(temp_file_path)
            
            # If the file is valid, proceed with processing
            pixel_array = dicom.pixel_array
            if pixel_array.max() > 0:
                pixel_array = pixel_array / pixel_array.max() * 255
            pixel_array = pixel_array.astype(np.uint8)
            
            if len(pixel_array.shape) == 2:
                img = Image.fromarray(pixel_array)
            elif len(pixel_array.shape) == 3 and pixel_array.shape[2] == 3:
                img = Image.fromarray(pixel_array)
            elif len(pixel_array.shape) == 3 and pixel_array.shape[2] == 1:
                img = Image.fromarray(pixel_array[:, :, 0])
            else:
                raise ValueError(f"Unexpected DICOM image shape: {pixel_array.shape}")
            
            return img
        except Exception as e:
            logger.warning(f"Initial DICOM read failed: {str(e)}. Attempting repair...")
            try:
                # Attempt to repair the DICOM file
                repaired_data = repair_dicom(dicom_data)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_file:
                    temp_file.write(repaired_data)
                    temp_file_path = temp_file.name
                
                dicom = pydicom.dcmread(temp_file_path)
                os.unlink(temp_file_path)
                
                # Process the repaired DICOM file
                pixel_array = dicom.pixel_array
                if pixel_array.max() > 0:
                    pixel_array = pixel_array / pixel_array.max() * 255
                pixel_array = pixel_array.astype(np.uint8)
                
                if len(pixel_array.shape) == 2:
                    img = Image.fromarray(pixel_array)
                elif len(pixel_array.shape) == 3 and pixel_array.shape[2] == 3:
                    img = Image.fromarray(pixel_array)
                elif len(pixel_array.shape) == 3 and pixel_array.shape[2] == 1:
                    img = Image.fromarray(pixel_array[:, :, 0])
                else:
                    raise ValueError(f"Unexpected DICOM image shape: {pixel_array.shape}")
                
                return img
            except Exception as repair_error:
                logger.error(f"Failed to repair DICOM file: {str(repair_error)}")
                raise ValueError("Unable to process DICOM file: Repair failed")
    
    def preprocess_image(self, img):
        img = img.resize(IMG_SIZE)
        if img.mode == 'L':
            img = img.convert('RGB')
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array, img
    
    def predict(self, img, apply_threshold=False):
        img_array, _ = self.preprocess_image(img)
        predictions = self.model.predict(img_array)[0]
        results = {}
        for i, disease in enumerate(DISEASE_LABELS):
            prob = float(predictions[i])
            if apply_threshold:
                threshold = CLINICAL_THRESHOLDS.get(disease, 0.5)
                detected = prob >= threshold
                results[disease] = {'probability': prob, 'threshold': threshold, 'detected': detected}
            else:
                results[disease] = prob
        return results
    
    def get_gradcam(self, img, disease_idx=None, disease_name=None):
        if disease_name and disease_name in DISEASE_LABELS:
            disease_idx = DISEASE_LABELS.index(disease_name)
        elif disease_idx is None:
            raise ValueError("Must provide either disease_idx or disease_name")
        
        img_array, original_img = self.preprocess_image(img)
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.last_conv_layer).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, predictions = grad_model(img_array)
            target_class_output = predictions[:, disease_idx]
        
        grads = tape.gradient(target_class_output, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        heatmap_normalized = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        original_img_np = np.array(original_img.resize(IMG_SIZE)).astype(np.uint8)
        heatmap_resized = cv2.resize(heatmap_colored, (original_img_np.shape[1], original_img_np.shape[0]))
        superimposed_img = cv2.addWeighted(original_img_np, 0.6, heatmap_resized, 0.4, 0)
        
        _, buffer = cv2.imencode('.png', superimposed_img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str

# New function to repair non-standard DICOM files
def repair_dicom(dicom_data):
    try:
        ds = pydicom.dcmread(BytesIO(dicom_data), force=True)
        if not (hasattr(ds, 'file_meta') and 'TransferSyntaxUID' in ds.file_meta):
            ds.file_meta = FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        # Get the pixel array and normalize
        pixel_array = ds.pixel_array
        if pixel_array.max() > 0:
            pixel_array = pixel_array / pixel_array.max() * 255
        pixel_array = pixel_array.astype(np.uint8)
        
        # Create an image from the pixel array
        if len(pixel_array.shape) == 2:
            image = Image.fromarray(pixel_array)
        elif len(pixel_array.shape) == 3:
            if pixel_array.shape[2] == 3:
                image = Image.fromarray(pixel_array)
            elif pixel_array.shape[2] == 1:
                image = Image.fromarray(pixel_array[:, :, 0])
            else:
                raise ValueError(f"Unexpected DICOM image shape: {pixel_array.shape}")
        else:
            raise ValueError(f"Unexpected DICOM image shape: {pixel_array.shape}")
        
        # Save the image to a bytes buffer (PNG format)
        with BytesIO() as output:
            image.save(output, format="PNG")
            converted_data = output.getvalue()
        
        logger.info("DICOM converted to image successfully")
        return converted_data
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise ValueError(f"DICOM conversion failed: {str(e)}")
    
# Initialize the model
model = None

@app.route('/repair_dicom', methods=['POST'])
def repair_dicom_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        file_content = file.read()
        # repaired_data now contains PNG data from the repair_dicom function
        repaired_png_data = repair_dicom(file_content)
        return repaired_png_data, 200, {
            'Content-Type': 'image/png',
            'Content-Disposition': 'attachment; filename=repaired.png'
        }
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        if model is None:
            model = ChestXrayModel()
        
        file_content = file.read()
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        is_dicom = request.form.get('is_dicom', 'false').lower() == 'true'
        
        if file_ext == '.dcm' or is_dicom:
            logger.info("Processing DICOM file")
            img = model.preprocess_dicom(file_content)
        else:
            img = Image.open(BytesIO(file_content))
        
        predictions = model.predict(img, apply_threshold=True)
        response = {
            'predictions': predictions,
            'summary': {
                'positive_findings': [
                    {'disease': disease, 'probability': data['probability']}
                    for disease, data in predictions.items()
                    if isinstance(data, dict) and data['detected']
                ]
            }
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({'error': f"DICOM processing failed: {str(e)}"}), 500

@app.route('/gradcam', methods=['POST'])
def gradcam():
    global model
    if 'file' not in request.files or 'disease' not in request.form:
        return jsonify({'error': 'Missing file or disease'}), 400
    
    file = request.files['file']
    disease = request.form['disease']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        if model is None:
            model = ChestXrayModel()
        
        file_content = file.read()
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext == '.dcm':
            img = model.preprocess_dicom(file_content)
        else:
            img = Image.open(BytesIO(file_content))
        
        gradcam_img = model.get_gradcam(img, disease_name=disease)
        return jsonify({'gradcam_image': f'data:image/png;base64,{gradcam_img}'})
    except Exception as e:
        logger.error(f"Error in GradCAM: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/report', methods=['POST'])
def report():
    global model
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        if model is None:
            model = ChestXrayModel()
        
        file_content = file.read()
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext == '.dcm':
            img = model.preprocess_dicom(file_content)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_file:
                temp_file.write(file_content)
                dicom = pydicom.dcmread(temp_file.name)
                os.unlink(temp_file.name)
            metadata = {
                'PatientID': str(dicom.get('PatientID', '')),
                'PatientName': str(dicom.get('PatientName', '')),
                'PatientAge': str(dicom.get('PatientAge', '')),
                'PatientSex': str(dicom.get('PatientSex', '')),
                'StudyDate': str(dicom.get('StudyDate', '')),
                'StudyDescription': str(dicom.get('StudyDescription', '')),
                'Modality': str(dicom.get('Modality', ''))
            }
        else:
            img = Image.open(BytesIO(file_content))
            metadata = {}
        
        predictions = model.predict(img, apply_threshold=True)
        positive_findings = [
            {'disease': disease, 'probability': data['probability']}
            for disease, data in predictions.items()
            if isinstance(data, dict) and data['detected']
        ]
        low_confidence_findings = [
            {'disease': disease, 'probability': data['probability']}
            for disease, data in predictions.items()
            if isinstance(data, dict) and not data['detected'] and data['probability'] > 0.2
        ]
        
        report_data = {
            'filename': file.filename,
            'positive_findings': positive_findings,
            'low_confidence_findings': low_confidence_findings,
            'dicom_metadata': metadata
        }
        return jsonify(report_data)
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9090)