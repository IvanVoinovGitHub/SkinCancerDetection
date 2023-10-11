#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#

from secrets import token_bytes
from flask import Flask, render_template, request, session
import logging
from logging import Formatter, FileHandler
from forms import *
from werkzeug.utils import secure_filename

import os
# -----------------------------------------KERAS MODEL IMPLEMENTATION----------------------------------------------------- #

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image as im

# -----------------------------------------KERAS MODEL IMPLEMENTATION----------------------------------------------------- #


import json

#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#

app = Flask(__name__)
app.config.from_object('config')

app.config['IMAGE UPLOADS'] = '/Users/Riyan/Downloads/CAC_Skin_Cancer/static/image'
# in order to use flask sessions, we need secret key :)
random_string = os.urandom(12).hex()
print("Secret key is: ", random_string)
app.secret_key = random_string


#----------------------------------------------------------------------------#
# RF model variable older
#----------------------------------------------------------------------------#

#   1.  HighBP                  v1          P1
# 	2.  HighChol                v2          P1
# 	3.  CholCheck               v16         P4
# 	4.  BMI	                    v3, v4      P1
#   5.  Smoker                  v7          P2
# 	6.  Stroke                  v5          P1
# 	7.  HeartDiseaseorAttack    v6          P1
# 	8.  PhysActivity            v8          P2
# 	9.  Fruits                  v9          P2
# 	10. Veggies                 v10         P2
# 	11. HvyAlcoholConsump       v11         P2
# 	12. AnyHealthcare           v17         P4
# 	13. NoDocbcCost             v18         P4
# 	14. GenHlth                 v12         P3     
# 	15. MentHlth                v13         P3
# 	16. PhysHlth                v14         P3
# 	17. DiffWalk                v15         P3
# 	18. Sex                     v20         P5
# 	19. Age                     v19         P5
# 	20. Education               v21         P5
# 	21. Income                  v22         P5

#----------------------------------------------------------------------------#
# Controllers.
#----------------------------------------------------------------------------#


@app.route('/')
def home():
    return render_template("pages/placeholder.home.html")
    


@app.route('/about')
def about():
    return render_template('pages/placeholder.about.html')


@app.route("/q1", methods=['GET', 'POST'])
def q1():
    if request.method == 'POST':
        return render_template("forms/q1.html")
    elif request.method == 'GET':
        return render_template("forms/q1.html")

@app.route("/q2", methods=['GET', 'POST'])
def q2():
    if request.method == 'POST':
        # Blood Pressure, convert it to binary
        #session['HighBP'] = request.form['v1']
        # Cholestrol, convert it to binary
        #session['HighChol'] = request.form['v2']
        # Weight
        # Check whether or not the value of weight and height are floats or integers - if so, continue, and if not, return text and q1.html
        #weight = float(request.form['v3'])
        # Height
        #height = float(request.form['v4'])
        #session['BMI'] = (weight / (height * height)) * 703.07
        # convert to binary
        #session['Stroke'] = request.form['v5']
        # convert to binary
        #session['HeartDiseaseorAttack']  = request.form['v6']
        
        image = request.files['file']
        filename = secure_filename(image.filename)
        basedir = os.path.abspath(os.path.dirname(__file__))
        local_filename = os.path.join(basedir, app.config['IMAGE UPLOADS'], filename)
        image.save(local_filename)

        model_path = 'SkinCancerClassificationModel.keras'

        # dimensions of our images
        img_width, img_height = 28, 28

        # load the model we saved
        model = load_model(model_path)

        # load image
        img = image.load_img(local_filename, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])

        # predict image
        prediction_metrics = ((model.predict(images, batch_size=10)).tolist())[0]

        # round prediction metrics
        for i in range(len(prediction_metrics)):
            prediction_metrics[i] = (round(prediction_metrics[i] * 1000)) / 10.0

        # sort metrics in asc order
        sorted_metrics = np.argsort(np.array(prediction_metrics))

        # retrieve diagnosis
        diagnosis_class = np.argmax(classes)
        
        print(prediction_metrics)
        # -----------------------------------------KERAS MODEL IMPLEMENTATION-----------------------------------------------------
        
        #FOR NOW - none is <0.1%, low is 0.1-1%, medium is 1-10%, and high is 10-100%

        diagnosisNo = "Congratulations! Your test results show little to no risk of any skin condition."

        diagnosisLow = ["low risk of Actinic Keratoses and Intraepithelial Carcinoma (also known as Bowen's Disease).", 
                        "low risk of Basal Cell Carcinoma.",
                        "low risk of Benign Keratosis-Like Lesions.",
                        "low risk of Dermatofibroma.",
                        "low risk of Melanoma.",
                        "low risk of Melanocytic Nevi.",
                        "low risk of Vascular Lesions."]
        
        diagnosisMedium = ["medium risk of Actinic Keratoses and Intraepithelial Carcinoma (also known as Bowen's Disease).", 
                            "medium risk of Basal Cell Carcinoma.",
                            "medium risk of Benign Keratosis-Like Lesions.",
                            "medium risk of Dermatofibroma.",
                            "medium risk of Melanoma.",
                            "medium risk of Melanocytic Nevi.",
                            "medium risk of Vascular Lesions."]
        
        diagnosisHigh = ["high risk of Actinic Keratoses and Intraepithelial Carcinoma (also known as Bowen's Disease).", 
                        "high risk of Basal Cell Carcinoma.",
                        "high risk of Benign Keratosis-Like Lesions.",
                        "high risk of Dermatofibroma.",
                        "high risk of Melanoma.",
                        "high risk of Melanocytic Nevi.",
                        "high risk of Vascular Lesions."]

        

        #Percentages
        akiec = "_"
        bcc = "_"
        bkl = "_"
        df = "_"
        mel = "_"
        nv = "_"
        vasc = "_"

        description = ["akiec desc", "bcc desc", "bkl desc", "df desc", "mel desc", "nv desc", "vasc desc"]

        return render_template("forms/q2.html", filename=filename, diagnosis=diagnosis, akiec=akiec, bcc=bcc, bkl=bkl, df=df, mel=mel, nv=nv, vasc=vasc, description=description[1])

    elif request.method == 'GET':
        return render_template("forms/q2.html")

@app.route('/q2/<filename>')
def displayQ2():
    return redirect(url_for('static', filename = '/image/' + filename), code=301)


@app.route('/statGender')
def statGender():
    return render_template('pages/statGender.html')

@app.route('/statAge')
def statAge():
    return render_template('pages/statAge.html')

@app.route('/statEducation')
def statEducation():
    return render_template('pages/statEducation.html')

@app.route('/statIncome')
def statIncome():
    return render_template('pages/statIncome.html')

@app.route('/statPhysActivity')
def statPhysActivity():
    return render_template('pages/statPhysActivity.html')

@app.route('/statSmoking')
def statSmoking():
    return render_template('pages/statSmoking.html')

@app.route('/register')
def register():
    form = RegisterForm(request.form)
    return render_template('forms/register.html', form=form)


@app.route('/forgot')
def forgot():
    form = ForgotForm(request.form)
    return render_template('forms/forgot.html', form=form)


# Error handlers.

@app.errorhandler(500)
def internal_error(error):
    #db_session.rollback()
    return render_template('errors/500.html'), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

if not app.debug:
    file_handler = FileHandler('error.log')
    file_handler.setFormatter(
        Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    )
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info('errors')

#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

# Default port:
'''
if __name__ == '__main__':
    app.run()
'''
# Or specify port manually:

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 2500))
    app.run(host='0.0.0.0', port=port)
