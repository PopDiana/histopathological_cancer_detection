from flask import Flask, render_template, request
from predict import *
import os

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('upload.html')


@app.route('/example/', methods=['POST'])
def use_example():
    # a positive image example
    result = predict_value('static/img/example.tif')
    return result


@app.route('/predict/', methods=['POST'])
def predict():
    uploaded_image = request.files.get('file')
    image_path = 'static/temp/' + uploaded_image.filename
    # save image in the temp folder
    uploaded_image.save(image_path)
    # convert it to tiff
    to_tiff(image_path)
    result = predict_value(TEMP)
    # finally remove the temporary image
    if os.path.exists(image_path):
        os.remove(image_path)
    return result
