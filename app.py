# Importing required libs
from flask import Flask, render_template, request, redirect, url_for,Markup
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from quality_model import preprocess_img_quality, predict_result_quality
from insect_model import preprocess_img_insect, predict_result_insect
from diesease_model import preprocess_img_dd, predict_result_dd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import cv2
import torch
import io
from PIL import Image
import os
from werkzeug.utils import send_from_directory
from ultralytics import YOLO
import joblib
from datetime import datetime
import pandas as pd

# Instantiating flask app
app = Flask(__name__)
model = pickle.load(open('modelRF.pkl', 'rb'))  # loading the model
model2 = pickle.load(open('modelForTestsvm-fin.pkl', 'rb'))  # loading the model
model3 = pickle.load(open('Quality_Model.pkl', 'rb'))  # loading the model
model6 = joblib.load(open('modelFordp.sav', 'rb')) # loading the model
model7 = pickle.load(open('ArimaHTP.pkl', 'rb'))  # loading the model

# Home route
@app.route("/")
def main():
    return render_template("packages.html")

@app.route("/quality")
def quality_prediction():
    return render_template("quality_main2.html")

@app.route("/insects")
def infestation_prediction():
    return render_template("insect_main2.html")

@app.route("/disease")
def disease_prediction():
    return render_template("diesease_main2.html")

@app.route("/harvest")
def harvest_prediction():
    return render_template("harvest_main2.html")

@app.route("/harvest_index")
def harvest_index():
    return render_template("harvest_index.html")

@app.route("/harvest_index2")
def harvest_index2():
    return render_template("harvest_index2.html")

@app.route("/quality_index2")
def quality_index2():
    return render_template("quality_index2.html")

@app.route("/disease_index2")
def disease_index2():
    return render_template("disease_index2.html")

@app.route("/quality_index")
def quality_index():
    return render_template("quality_index.html")

@app.route("/insect_index")
def insect_index():
    return render_template("insect_index.html")

@app.route("/insect_index2")
def insect_index2():
    return render_template("insect_index2.html")

@app.route("/dd_index")
def dd_index():
    return render_template("dd_index.html")











# Prediction route for quality
@app.route('/prediction_quality', methods=['POST'])
@cross_origin()
def predict_image_file_quality():
    try:
        if request.method == 'POST':
            print(request.files['file'].stream)
            img = preprocess_img_quality(request.files['file'].stream)
            print(img)
            pred = predict_result_quality(img)
            return jsonify({'status': True, 'prediction': str(pred)})
    except Exception as e:
        error = "Error: " + str(e)
        return jsonify({'status': False, 'error': error})


# Prediction route for quality ML
@app.route('/MushroomQuality_predict', methods=['POST'])
@cross_origin()
def MushroomQuality_predict():
    try:
        # Get input values from the form
        Temperature = float(request.form["Temperature"])
        Humidity = float(request.form["Humidity"])
        WHC = float(request.form["WHC"])
        pH_level = float(request.form["pH_level"])

        # Make a prediction using the model
        input_data = np.array([[pH_level, WHC, Temperature,Humidity]])
        prediction = model6.predict(input_data)

        # Assuming your model predicts 1 for "Good quality" and 0 for "Not good quality"
        if prediction[0] == 1:
            quality = "Good Quality Casing"
        else:
            quality = "Not Good Quality Casing"
        return jsonify({'status': True, 'prediction': str(quality)})
    except Exception as e:
        error = "Error: " + str(e)
        return jsonify({'status': False, 'error': error})


# Prediction route for insects
@app.route('/prediction_insects', methods=['POST'])
@cross_origin()
def predict_image_file_insect():
    try:
        if request.method == 'POST':
            img = preprocess_img_insect(request.files['file'].stream)
            pred = predict_result_insect(img)
            return jsonify({'status': True, 'prediction': str(pred)})
    except Exception as e:
        error = "Error: " + str(e)
        return jsonify({'status': False, 'error': error})


# Prediction route for dieseases
@app.route('/prediction_dieseases', methods=['POST'])
@cross_origin()
def predict_image_file_dd():
    try:
        if request.method == 'POST':
            img = preprocess_img_dd(request.files['file'].stream)
            pred = predict_result_dd(img)

            return jsonify({'status': True, 'prediction': str(pred)})
            # return render_template("disease_result.html", predictions=str(pred))
    except Exception as e:
        error = "Error: " + str(e)
        return jsonify({'status': False, 'error': error})



# Create a label encoder for mapping range_levels to numerical values

label_encoder = LabelEncoder()

range_levels = ['high','medium', 'low']

label_encoder.fit(range_levels)

# Prediction route for diesease ML
@app.route('/prediction_earlyD',methods=['POST'])
@cross_origin()
def predict_earlyD():

    """Grabs the input values and uses them to make prediction"""

    temperature = float(request.form["temperature"])
    humidity = int(request.form["humidity"])

    # Use the label encoder to convert the selected ventilation level to a numerical value

    ventilation = request.form["ventilation"]
    ventilation = label_encoder.transform([ventilation])[0]

    # Use the label encoder to convert the selected light intensity level to a numerical value
    light_intensity = request.form["light"]
    light_intensity = label_encoder.transform([light_intensity])[0]
    ph = float(request.form["ph"])
    prediction = model2.predict([[temperature, humidity, ventilation, light_intensity, ph]])
 
    if prediction == 0:
        ptext = 'High'
    elif prediction == 1:
        ptext = 'Moderate'
    else:
        ptext = 'Low'

    def find_diseaseType(value, rh):
        diseaseType = []
        if 18 <= value <= 25:
            diseaseType.append("Bacterial Blotch")            
        if 20 <= value <= 26:
            diseaseType.append("Cobweb")            
        if 19 <= value <= 20:
            diseaseType.append("Black Mould")            
        if 20 <= value <= 35:
            diseaseType.append("Green Mould")
        if 25 <= value <= 35:
            diseaseType.append("Mite attack")
        if  value > 35:
            diseaseType.append("Cobweb")
        if  value < 18 and rh > 70:
            diseaseType.append("Black Mould")
        if value < 18:
            diseaseType.append("White Mold, Wet Bubble")
        

        return diseaseType 
   
    result = find_diseaseType(temperature, humidity)

   
    #Disease_type = f"Disease types in the range are:  {', '.join(result)}"
    #prediction_text = f'Disease growth possibility level is: {ptext}'
     
    Disease_type = f"{', '.join(result)}"
    prediction_text = f'{ptext}'
  

    #return jsonify({'status': True, 'prediction': str(prediction_text)})
    return jsonify({'status': True,'prediction': str(prediction_text),'disease_typ': str(Disease_type)})


#Create label encoders for variables

#Label Encoder for Growing room
label_encoder2 = LabelEncoder()
range_levels2 = ['B1', 'A3', 'A2', 'A1', 'B3', 'B2']
label_encoder2.fit(range_levels2)

#Label Encoder for Type of paddy straw
label_encoder3 = LabelEncoder()
range_levels3 = ['50% dura,50% AKP', '100% dura']
label_encoder3.fit(range_levels3)

#Label Encoder for Type of spawn
label_encoder4 = LabelEncoder()
range_levels4 = ['50% syl, 50% myc', '100% sylvarn', '97% syl,3%local (MN)',
       '100% sylvarn new lot', '70% syl,30% myc', '23% syl,77%local (KU)',
       '17% syl,83% local(KU)']
label_encoder4.fit(range_levels4)

#Label Encoder for Casing
label_encoder5 = LabelEncoder()
range_levels5 = ['LC-678  92%  TT-52  8%', 'LC100%', 'TOPTTERA  100%',
       'LC-374BGS-53%  TT-329 47%', 'LC-689  96% TT-22  3%',
       'LC-53%  TT 47%']
label_encoder5.fit(range_levels5)

# Prediction route for harvest yield
@app.route('/prediction_yield', methods=['POST'])
@cross_origin()
def predict_yield():

    Groowing_room = request.form["Groowing_room"]
    Groowing_room = label_encoder2.transform([Groowing_room])[0]

    No_of_bags = int(request.form["No_of_bags"])

    Type_of_paddy_straw = request.form["Type_of_paddy_straw"]
    Type_of_paddy_straw = label_encoder3.transform([Type_of_paddy_straw])[0]

    Type_of_spawn = request.form["Type_of_spawn"]
    Type_of_spawn = label_encoder4.transform([Type_of_spawn])[0]

    Casing = request.form["Casing"]
    Casing = label_encoder5.transform([Casing])[0]

    prediction = model.predict([[Groowing_room, No_of_bags, Type_of_paddy_straw,Type_of_spawn, Casing]])

    output = round(prediction[0], 1)
    return jsonify({'status': True, 'prediction': str({output} )})

@app.route('/prediction_yieldT', methods=['POST'])
@cross_origin()
def predict_yieldT():

    Date = datetime.strptime(
                     request.form['date'],
                     '%Y-%m-%d').date()

    #date1 = datetime.strptime('2023-10-25', '%Y-%m-%d').date()
    #date2 = datetime.strptime('2023-4-19', '%Y-%m-%d').date()
    #date1 = datetime.date(2023, 4, 19)
    #date2 = datetime.date(2023, 4, 25)
    #delta = Date - date1
    #print(delta.days)
    #d1 = delta.days + 383
    #print(d1)

    df1=pd.read_csv('Sarima_HP1_6year_predic.csv', index_col=0) 
    d1= str(Date)
    prediction = df1.loc[d1]

    #prediction=model7.predict(start=d1,end=d1)
    prediction = np.array(prediction)

    output = round(prediction[0], 1)
    return jsonify({'status': True, 'prediction': str(output)})

# Prediction route for harvest time
@app.route('/prediction_time',methods=['GET','POST'])
@cross_origin()
def predict_time():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            print("upload folder is ",filepath)
            f.save(filepath)
            global imgpath
            predict_time.imgpath = f.filename
            print("printing predict image...",predict_time)

            file_extension = f.filename.rsplit('.',1)[1].lower()
            print("printing predict image...",file_extension)

            if file_extension == 'jpg':
                img = cv2.imread(filepath)
                frame = cv2.imencode('.jpg', cv2.UMat(img))[1].tobytes()
                image = Image.open(io.BytesIO(frame))
                model4 = YOLO("htpv14yolov8l.pt")
                results = model4(image,stream=True)

                for r in results:
                    im_array = r.plot()  # plot a BGR numpy array of predictions
                    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                    #im.show()  # show image
                    im.save('static/images/results.jpg')  # save image

            elif file_extension == 'png':
                img = cv2.imread(filepath)
                frame = cv2.imencode('.png', cv2.UMat(img))[1].tobytes()
                image = Image.open(io.BytesIO(frame))
                model4 = YOLO("htpv14yolov8l.pt")
                results = model4(image,stream=True)

                for r in results:
                    im_array = r.plot()  # plot a BGR numpy array of predictions
                    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                    #im.show()  # show image
                    im.save('static/images/results.jpg')  # save image
                    
                    
    return jsonify({'status': True, 'prediction': 'http://127.0.0.1:5000/static/images/results.jpg'})

# Prediction route for risk
@app.route('/risk_level',methods=['GET','POST'])
@cross_origin()
def risk_level():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            print("upload folder is ",filepath)
            f.save(filepath)
            global imgpath
            risk_level.imgpath = f.filename
            print("printing predict image...",risk_level)

            file_extension = f.filename.rsplit('.',1)[1].lower()
            print("printing predict image...",file_extension)

            if file_extension == 'jpg':
                img = cv2.imread(filepath)
                frame = cv2.imencode('.jpg', cv2.UMat(img))[1].tobytes()
                image = Image.open(io.BytesIO(frame))
                model4 = YOLO("best-risklvl.pt")
                results = model4(image,stream=True)

                for r in results:
                    im_array = r.plot()  # plot a BGR numpy array of predictions
                    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                    #im.show()  # show image
                    im.save('static/images/resultsRL.jpg')  # save image

            elif file_extension == 'png':
                img = cv2.imread(filepath)
                frame = cv2.imencode('.png', cv2.UMat(img))[1].tobytes()
                image = Image.open(io.BytesIO(frame))
                model4 = YOLO("best-risklvl.pt")
                results = model4(image,stream=True)

                for r in results:
                    im_array = r.plot()  # plot a BGR numpy array of predictions
                    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                    #im.show()  # show image
                    im.save('static/images/resultsRL.jpg')  # save image

    return jsonify({'status': True, 'prediction': 'http://127.0.0.1:5000/static/images/resultsRL.jpg'})

# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)