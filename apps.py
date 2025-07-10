from flask import Flask, render_template, request, jsonify
from keras.models import load_model

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions
from keras.applications.mobilenet_v2 import MobileNetV2

import tensorflow as tf
from tensorflow import keras
from skimage import transform, io
import numpy as np
import os
import requests
from PIL import Image
from datetime import datetime
from keras.preprocessing import image
from flask_cors import CORS

app = Flask(__name__)

# Load model Xception
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1-VGqY-wKfAT2ax4Qm_-KjYxfnB-PlIfq'
MODEL_PATH = 'Xception.h5'

if not os.path.exists(MODEL_PATH):
    print("Model belum ditemukan. Mengunduh dari Google Drive...")
    r = requests.get(MODEL_URL, allow_redirects=True)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)
    print("Model berhasil diunduh.")

modelxception = load_model(MODEL_PATH)

# Konfigurasi folder upload
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tipe file yang diperbolehkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def main():
    return render_template("cnn.html")

@app.route("/classification")
def classification():
    return render_template("classifications.html")

@app.route('/submit', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'message': 'No image in the request'}), 400

    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'message': 'Invalid file type'}), 400

    # Simpan file dengan timestamp unik
    timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
    filename = f"{timestamp}.png"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    img = Image.open(file).convert('RGB')
    img.save(filepath)

    # Preprocessing untuk model
    img = image.load_img(filepath, target_size=(128, 128))
    x = image.img_to_array(img) / 127.5 - 1
    x = np.expand_dims(x, axis=0)

    # Prediksi
    prediction = modelxception.predict(x)
    class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = '{:.0f}%'.format(100 * np.max(prediction))

    # Informasi klasifikasi
    info = {
    'Brown spot': {
        'description': 'Brown spot adalah penyakit penting pada padi yang disebabkan oleh jamur <i>Bipolaris oryzae</i>.',
        'symptoms': '<ul><li>Bercak berwarna cokelat kehitaman di daun.</li><li>Daun mengering dan tanaman tampak lemah.</li></ul>',
        'causes': '<ul><li>Jamur Bipolaris oryzae berkembang pada kelembapan tinggi.</li><li>Tanaman kekurangan kalium atau kelebihan nitrogen lebih rentan.</li></ul>',
        'prevention': '<ul><li>Gunakan varietas tahan penyakit.</li><li>Lakukan rotasi tanaman dan sanitasi lahan.</li><li>Gunakan fungisida berbahan aktif seperti mancozeb.</li></ul>',
        'reference': '''
            <ul>
                <li><a href="https://plantix.net/id/library/plant-diseases/100064/brown-spot-of-rice/" target="_blank">Plantix - Brown Spot of Rice</a></li>
                <li><a href="https://en.wikipedia.org/wiki/Cochliobolus_miyabeanus" target="_blank">Wikipedia - Cochliobolus miyabeanus</a></li>
            </ul>
        '''
    },
    'Bacterial leaf blight': {
        'description': 'Bacterial leaf blight disebabkan oleh bakteri <i>Xanthomonas oryzae</i> dan menyebar cepat saat musim hujan.',
        'symptoms': '<ul><li>Ujung daun menguning lalu menyebar ke seluruh daun.</li><li>Daun tampak kering seperti terbakar.</li></ul>',
        'causes': '<ul><li>Bakteri masuk melalui luka daun atau alat pertanian kotor.</li><li>Kelembaban tinggi dan angin mempercepat penyebaran.</li></ul>',
        'prevention': '<ul><li>Gunakan benih bersertifikat tahan BLB.</li><li>Sanitasi lahan dan alat pertanian.</li><li>Rotasi tanaman dengan non-padi.</li></ul>',
        'reference': '''
            <ul>
                <li><a href="http://www.knowledgebank.irri.org/decision-tools/rice-doctor/rice-doctor-fact-sheets/item/bacterial-blight" target="_blank">IRRI - Bacterial Blight</a></li>
            </ul>
        '''
    },
    'Leaf smut': {
        'description': 'Leaf smut disebabkan oleh jamur <i>Entyloma oryzae</i> yang menyebabkan bercak hitam pada daun.',
        'symptoms': '<ul><li>Bercak kecil lonjong berwarna gelap pada daun.</li><li>Bercak bisa menyatu dan menyebabkan daun mati.</li></ul>',
        'causes': '<ul><li>Jamur menyebar lewat angin dan cipratan air hujan.</li><li>Sering terjadi pada kondisi lembap dan suhu tinggi.</li></ul>',
        'prevention': '<ul><li>Gunakan varietas tahan penyakit.</li><li>Buang dan bakar daun terinfeksi.</li><li>Semprot fungisida sistemik jika diperlukan.</li></ul>',
        'reference': '''
            <ul>
                <li><a href="https://www.gardeningknowhow.com/edible/grains/rice/how-to-treat-leaf-smut-of-rice.htm" target="_blank">Gardening Know How - Leaf smut</a></li>
            </ul>
        '''
    }
}


    selected_info = info[predicted_class]

    return render_template("classifications.html",
        img_path=filepath,
        predictionxception=predicted_class,
        confidenceexception=confidence,
        descriptionxception=selected_info['description'],
        symptomsxception=selected_info['symptoms'],
        causesxception=selected_info['causes'],
        preventionxception=selected_info['prevention'],
        referencesxception=selected_info['reference']
    )

if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
