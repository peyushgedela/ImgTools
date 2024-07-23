from flask import Flask, request, send_file
from rembg import remove
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import cv2
import io
#testing
app = Flask(__name__)

@app.route('/remove-background', methods=['POST'])
def remove_background():
    file = request.files['image']
    input_image = Image.open(file.stream)
    output_image = remove(input_image)
    byte_io = io.BytesIO()
    output_image.save(byte_io, 'PNG')
    byte_io.seek(0)
    return send_file(byte_io, mimetype='image/png')

@app.route('/compress-image', methods=['POST'])
def compress_image():
    file = request.files['image']
    n_colors = int(request.form['n_colors'])
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    new_colors = kmeans.cluster_centers_.astype(int)
    new_image = new_colors[kmeans.labels_].reshape(image.shape)
    _, buffer = cv2.imencode('.jpg', new_image)
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

@app.route('/upscale-image', methods=['POST'])
def upscale_image():
    return "Upscaling not implemented"

if __name__ == '__main__':
    app.run(debug=True)
