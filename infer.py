from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from joblib import load
import webbrowser

app = Flask(__name__)

# Load pre-trained model
clf = load("./saved_model/random_forest.joblib")

@app.route('/')
def home():
    symptoms = {'itching': 0, 'skin_rash': 0, 'nodal_skin_eruptions': 0, 'continuous_sneezing': 0,
                'shivering': 0, 'chills': 0, 'joint_pain': 0, 'stomach_pain': 0, 'acidity': 0, 'ulcers_on_tongue': 0,
                'muscle_wasting': 0, 'vomiting': 0, 'burning_micturition': 0, 'spotting_ urination': 0, 'fatigue': 0,
                'weight_gain': 0, 'anxiety': 0, 'cold_hands_and_feets': 0, 'mood_swings': 0, 'weight_loss': 0,
                'restlessness': 0, 'lethargy': 0, 'patches_in_throat': 0, 'irregular_sugar_level': 0, 'cough': 0,
                'high_fever': 0, 'sunken_eyes': 0, 'breathlessness': 0, 'sweating': 0, 'dehydration': 0,
                'indigestion': 0, 'headache': 0, 'yellowish_skin': 0, 'dark_urine': 0, 'nausea': 0, 'loss_of_appetite': 0,
                'pain_behind_the_eyes': 0, 'back_pain': 0, 'constipation': 0, 'abdominal_pain': 0, 'diarrhoea': 0, 'mild_fever': 0,
                'yellow_urine': 0, 'yellowing_of_eyes': 0, 'acute_liver_failure': 0, 'fluid_overload': 0, 'swelling_of_stomach': 0,
                'swelled_lymph_nodes': 0, 'malaise': 0, 'blurred_and_distorted_vision': 0, 'phlegm': 0, 'throat_irritation': 0,
                'redness_of_eyes': 0, 'sinus_pressure': 0, 'runny_nose': 0, 'congestion': 0, 'chest_pain': 0, 'weakness_in_limbs': 0,
                'fast_heart_rate': 0, 'pain_during_bowel_movements': 0, 'pain_in_anal_region': 0, 'bloody_stool': 0,
                'irritation_in_anus': 0, 'neck_pain': 0, 'dizziness': 0, 'cramps': 0, 'bruising': 0, 'obesity': 0, 'swollen_legs': 0,
                'swollen_blood_vessels': 0, 'puffy_face_and_eyes': 0, 'enlarged_thyroid': 0, 'brittle_nails': 0, 'swollen_extremities': 0,
                'excessive_hunger': 0, 'extra_marital_contacts': 0, 'drying_and_tingling_lips': 0, 'slurred_speech': 0,
                'knee_pain': 0, 'hip_joint_pain': 0, 'muscle_weakness': 0, 'stiff_neck': 0, 'swelling_joints': 0, 'movement_stiffness': 0,
                'spinning_movements': 0, 'loss_of_balance': 0, 'unsteadiness': 0, 'weakness_of_one_body_side': 0, 'loss_of_smell': 0,
                'bladder_discomfort': 0, 'foul_smell_of urine': 0, 'continuous_feel_of_urine': 0, 'passage_of_gases': 0, 'internal_itching': 0,
                'toxic_look_(typhos)': 0, 'depression': 0, 'irritability': 0, 'muscle_pain': 0, 'altered_sensorium': 0,
                'red_spots_over_body': 0, 'belly_pain': 0, 'abnormal_menstruation': 0, 'dischromic _patches': 0, 'watering_from_eyes': 0,
                'increased_appetite': 0, 'polyuria': 0, 'family_history': 0, 'mucoid_sputum': 0, 'rusty_sputum': 0, 'lack_of_concentration': 0,
                'visual_disturbances': 0, 'receiving_blood_transfusion': 0, 'receiving_unsterile_injections': 0, 'coma': 0,
                'stomach_bleeding': 0, 'distention_of_abdomen': 0, 'history_of_alcohol_consumption': 0, 'fluid_overload.1': 0,
                'blood_in_sputum': 0, 'prominent_veins_on_calf': 0, 'palpitations': 0, 'painful_walking': 0, 'pus_filled_pimples': 0,
                'blackheads': 0, 'scurring': 0, 'skin_peeling': 0, 'silver_like_dusting': 0, 'small_dents_in_nails': 0, 'inflammatory_nails': 0,
                'blister': 0, 'red_sore_around_nose': 0, 'yellow_crust_ooze': 0}

    return f'''
        <!doctype html>
        <html>
        <head>
            <title>Symptom Checker</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 0;
                }}
                h1,h2,h4 {{
                    color: #333;
                    text-align: center;
                }}
                form {{
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    width: 90%;
                    margin: 50px auto;
                }}
                label {{
                    display: inline-block;
                    margin-bottom: 10px;
                }}
                input[type="checkbox"] {{
                    margin-right: 5px;
                    display: none; /* Hide the default checkboxes */
                }}
                .ks-cboxtags li {{
                    display: inline-block;
                    margin-right: 10px;
                }}
                .ks-cboxtags li label {{
                    display: inline-block;
                    background-color: rgba(255, 255, 255, .9);
                    border: 2px solid rgba(139, 139, 139, .3);
                    color: #adadad;
                    border-radius: 25px;
                    white-space: nowrap;
                    margin: 3px 0px;
                    cursor: pointer;
                    -webkit-touch-callout: none;
                    -webkit-user-select: none;
                    -moz-user-select: none;
                    -ms-user-select: none;
                    user-select: none;
                    -webkit-tap-highlight-color: transparent;
                    transition: all .2s;
                }}
                .ks-cboxtags li label {{
                    padding: 8px 12px;
                }}
                .ks-cboxtags li label::before {{
                    display: inline-block;
                    font-style: normal;
                    font-variant: normal;
                    text-rendering: auto;
                    -webkit-font-smoothing: antialiased;
                    font-family: "Font Awesome 5 Free";
                    font-weight: 900;
                    font-size: 12px;
                    padding: 2px 6px 2px 2px;
                    content: "\f067";
                    transition: transform .3s ease-in-out;
                }}
                .ks-cboxtags li input[type="checkbox"]:checked + label::before {{
                    content: "\f00c";
                    transform: rotate(-360deg);
                    transition: transform .3s ease-in-out;
                }}
                .ks-cboxtags li input[type="checkbox"]:checked + label {{
                    border: 2px solid #1bdbf8;
                    background-color: #12bbd4;
                    color: #fff;
                    transition: all .2s;
                }}
                input[type="submit"] {{
                    display: flex;
                      align-items: center;
                      font-family: inherit;
                      font-weight: 500;
                      font-size: 16px;
                      padding: 0.7em 1.4em 0.7em 1.1em;
                      color: white;
                      background: #ad5389;
                      background: linear-gradient(0deg, rgba(20,167,62,1) 0%, rgba(102,247,113,1) 100%);
                      border: none;
                      box-shadow: 0 0.7em 1.5em -0.5em #14a73e98;
                      letter-spacing: 0.05em;
                      border-radius: 20em;
                      cursor: pointer;
                      user-select: none;
                      -webkit-user-select: none;
                      touch-action: manipulation;
                }}
                input[type="submit"]:hover {{
                    box-shadow: 0 0.5em 1.5em -0.5em #14a73e98;
                }}
                input[type="submit"]:active {{
                      box-shadow: 0 0.3em 1em -0.5em #14a73e98;
                }}
                footer {{
                        background-color: #333;
                        color: white;
                        padding: 10px;
                        text-align: center;
                    }}
            </style>
        </head>
        <body>
            <h1>Dr.On A Click: AI Based Health Care Solution</h1>
            <h2>Disease Prediction</h2>
            <hr>
            <h4>Select the Symptoms you feel</h6>
            <form id="symptomForm" method="post" action="/predict" onsubmit="return validateForm()">
                <ul class="ks-cboxtags">
                    {"".join(f'<li><input type="checkbox" id="{symptom}" name="{symptom}" value="1"><label for="{symptom}">{symptom}</label></li>' for symptom in symptoms)}
                </ul>
                <br>
                <center><input type="submit" value="Predict" disabled></center>
                
            </form>

            <script>
                function validateForm() {{
                    var checkboxes = document.querySelectorAll('input[type="checkbox"]:checked');
                    var submitButton = document.querySelector('input[type="submit"]');
                    if (checkboxes.length === 0) {{
                        alert("Please select at least one symptom.");
                        return false;
                    }}
                    return true;
                }}

                document.addEventListener('change', function() {{
                    var checkboxes = document.querySelectorAll('input[type="checkbox"]:checked');
                    var submitButton = document.querySelector('input[type="submit"]');
                    submitButton.disabled = checkboxes.length === 0;
                }});
            </script>
        </body>
        <hr>
        <footer>
            <h6>AI Based Disease Prediction completely based on the dataset of Disease-Symptom Knowledge Database, Columbia University.
            <br>This system is Designed by the Department of Information Technology, Shri Ram Murti Smarak, College of Engineering & Technology, Bareilly. @2023</h6>
        </footer>
        </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.to_dict()
    df_test = pd.DataFrame(columns=list(symptoms.keys()))
    df_test.loc[0] = np.array(list(symptoms.values()))

    result = clf.predict(df_test)
    return f'''
        <!doctype html>
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                    min-height: 100vh;
                }}
                h1,h2 {{
                    color: #333;
                    text-align: center;
                }}
                p {{
                    color: #333;
                    text-align: center;
                }}
                a {{
                    color: #4caf50;
                    text-decoration: none;
                    display: block;
                    text-align: center;
                    margin-top: 20px;
                }}
                footer {{
                        background-color: #333;
                        color: white;
                        padding: 10px;
                        text-align: center;
                    }}
            </style>
        </head>
        <body>
            <h1>Dr.On A Click: AI Based Health Care Solution</h1>
            <h2>Disease Prediction</h2>
            <hr>
            <h2>Prediction Result</h2>
            <p>Predicted Disease: {result[0]}</p>
            <a href="/">Go back</a>
        </body>
        <footer>
            <h6>AI Based Disease Prediction completely based on the dataset of Disease-Symptom Knowledge Database, Columbia University.
            <br>This system is Designed by the Department of Information Technology, Shri Ram Murti Smarak, College of Engineering & Technology, Bareilly. @2023</h6>
        </footer>
        </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    webbrowser.open('http://192.168.29.190:5000')
    app.run(debug=True)
