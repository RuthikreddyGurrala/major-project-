
import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

app = Flask(__name__)

model1 = load_model('trained/level.h5')
model2 = load_model('trained/body.h5')

user_database = {}

@app.route('/')
def index():
    return render_template('about.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/afterreg', methods=['POST'])
def afterreg():
    x = [x for x in request.form.values()]
    user_id = x[1]
    data = {'name': x[0], 'psw': x[2]}
    if user_id in user_database:
        return render_template('register.html', pred="You are already a member, please login.")
    else:
        user_database[user_id] = data
        return render_template('register.html', pred="Registration Successful. Please login.")

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/afterlogin', methods=['POST'])
def afterlogin():
    user_id = request.form['_id']
    passw = request.form['psw']
    if user_id not in user_database:
        return render_template('login.html', pred="The username is not found.")
    elif user_database[user_id]['psw'] == passw:
        return redirect(url_for('prediction'))
    else:
        return render_template('login.html', pred="Invalid credentials.")

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/result', methods=['POST'])
def res():
    if 'image' not in request.files:
        return render_template('prediction.html', prediction="No file uploaded.")
    f = request.files['image']
    basepath = os.path.dirname(__file__)
    filepath = os.path.join(basepath, 'uploads', f.filename)
    f.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)

    prediction1 = np.argmax(model1.predict(img_data))
    prediction2 = np.argmax(model2.predict(img_data))

    index1 = ['front', 'rear', 'side']
    index2 = ['minor', 'moderate', 'severe']

    result1 = index1[prediction1]
    result2 = index2[prediction2]

    value = ""
    if result1 == "front" and result2 == "minor":
        value = "3000-5000 INR"
    elif result1 == "front" and result2 == "moderate":
        value = "6000-8000 INR"
    elif result1 == "front" and result2 == "severe":
        value = "9000-11000 INR"
    elif result1 == "rear" and result2 == "minor":
        value = "4000-6000 INR"
    elif result1 == "rear" and result2 == "moderate":
        value = "7000-9000 INR"
    elif result1 == "rear" and result2 == "severe":
        value = "11000-13000 INR"
    elif result1 == "side" and result2 == "minor":
        value = "6000-8000 INR"
    elif result1 == "side" and result2 == "moderate":
        value = "9000-11000 INR"
    elif result1 == "side" and result2 == "severe":
        value = "12000-15000 INR"
    else:
        value = "16000-50000 INR"

    return render_template('prediction.html', prediction=value)

@app.route('/logout')
def logout():
    return render_template('logout.html')

if __name__ == "__main__":
    app.run(debug=True)
