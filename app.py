from flask import Flask, render_template, request
from flask.helpers import url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np


app = Flask(__name__)
app = Flask(__name__, static_folder='images')


#defining the model and other necessary variables
img_height = 180
img_width = 180
model = tf.keras.models.load_model('model\model.h5')  

@app.route('/', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join('images', filename))
            test_image_path = os.path.join('images', filename)
            
            img = tf.keras.utils.load_img(test_image_path, target_size=(img_height, img_width))
            x = tf.keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            predictions = model.predict(x)
            score = tf.nn.softmax(predictions[0])
            predicted_label = ["DOOM", "ANIMAL CROSSING"]
            
            context = {'filePathName': test_image_path, 'predictedLabel': predicted_label[np.argmax(score)], 'percentage': 100 * np.max(score)}
            return render_template('index.html', context=context)


if __name__=='__main__':
    app.run(port=3000,debug=True)