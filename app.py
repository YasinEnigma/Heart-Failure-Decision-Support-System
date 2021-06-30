import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings

from PIL import Image


image = Image.open("background.jpg")

st.set_page_config(page_title="Heart Failure Prediction",
                   page_icon="💉", layout='centered', initial_sidebar_state='auto')  # ❤️
st.image(image)
# Data columns
feature_names_best = ['age', 'sex', 'cpt', 'rbp', 'sch',
                      'fbs', 'res', 'mhr', 'eia', 'opk', 'pes', 'vca', 'tha']

gender_dict = {"Male": 1, "Female": 0}
chest_pain_dict = {"Typical Angina": 0, "Atypical Angina": 1,
                   "Non-Anginal pain": 2, "Asymptomatic": 3	}
ecg_dict = {"Normal": 0, "Wave Abnormality": 1,
            "Left Ventricular Hypertrophy": 2}
slope_dict = {"Unsloping": 0, "Flat": 1,
              "Downsloping": 2}
thal_dict = {"Normal": 0, "Fixed Defect": 1,
             "Reversible Defect": 2}
feature_dict = {"Yes": 1, "No": 0}


def load_image(img):
    im = Image.open(os.path.join(img))
    return im


def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


# title
html_temp = """
<div>
<h1 style="color:crimson;text-align:left;">Early Diagnosis of Heart Failure Prediction</h1>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

if st.checkbox("Information"):
    '''
    One of the most common and complex health problems around the world to date is heart failure (HF) (McRae et al., 2016; Kumar et al., 2016; Bui et al., 2014), which usually occurs when the heart fails to pump sufficient amount of blood to meet the needs of the body. The HF patients often experience some symptoms such as shortness of breath, weakness of the body, and swollen feet (Rahimi et al., 2012; Durairaj & Sivagowry, 2014). The findings from the previous investigations have identified HF and its resulting complications as one of the major causes of reduced quality of life as well as mortality in many technologically advanced countries (Hui & US Jonathan, 2015; Allen et al., 2012). The management of HF disease is quite complex and even worse in developing countries where there are rare possibility of adequate diagnostic tools and limited number of medical experts especially in some rural communities (Sameh et al., 2013). Therefore, accurately predicting the HF risk in patients/individuals would be essential for reducing their associated risks of severe heart problems and increasing their safety and productivity during daily life activities (Al-Shayea, 2011; Kumar, 2012).
    '''
'''
## How does it work ❓ 
Complete all the questions and the machine learning model will predict the patients have heart failure or not? 
'''

if st.checkbox("Exploratory Data Analysis (EDA)"):
    pass

# Logo
st.sidebar.image('real-heart-icon-png-.png', width=120)
st.sidebar.title("Prediction Form📋")

# Age of the patient
age = st.sidebar.number_input("Age (Year)", 1, 100)

# Male or Female
sex = st.sidebar.radio("Sex", tuple(gender_dict.keys()))

# Chest Pain Type
cpt = st.sidebar.radio("Chest Pain Type", tuple(chest_pain_dict.keys()))

# resting blood pressure
rbp = st.sidebar.number_input("Resting Blood Pressure (mmHg)", 1, 7861)

# serum cholesterol
sch = st.sidebar.number_input(
    "Serum Cholesterol (mg/dL)", 1, 7861)

# fasting blood sugar > 120 mg/dL (likely to be diabetic) 1 = true; 0 = false
fbs_num = st.sidebar.number_input("Fasting Blood Sugar (mg/dl)", 1, 500)
if fbs_num > 120:
    fbs = True
else:
    fbs = False

# resting electrocardiogram results
res = st.sidebar.radio(
    "Resting Electrocardiographic Results", tuple(ecg_dict.keys()))

# maximum heart rate achieved
mhr = st.sidebar.number_input("Maximum Heart Rate Achieved", 0.0, 100000.0)

# exercise induced angina
eia = st.sidebar.radio("Exercise Induced Angina", tuple(feature_dict.keys()))

# ST depression induced by exercise relative to rest
opk = st.sidebar.number_input("Old Peak (mm)", 1, 1000)

#  the slope of the peak exercise ST segment
pes = st.sidebar.radio("Peak Exercise Slope", tuple(slope_dict.keys()))

# number of major vessels (0-3) colored by fluoroscopy
vca = st.sidebar.number_input(
    "Number of Major Vessels Colored by fluoroscopy", 1, 1000)

# Thallium testing
tha = st.sidebar.radio("Thallium Scan", tuple(thal_dict.keys()))


feature_list = [age, get_value(sex, gender_dict), get_value(cpt, chest_pain_dict), rbp, sch,
                fbs, get_value(res, ecg_dict), mhr, get_value(eia, feature_dict), opk, get_value(pes, slope_dict), vca, get_value(tha, thal_dict)]
pretty_result = {"Age": age, "Sex": sex, "Chest Pain Type": cpt, "Resting Blood Pressure (mmHg)": rbp, "Serum Cholesterol (mg/dL)": sch, "Fasting Blood Sugar (mg/dl)": fbs, "Resting Electrocardiographic Results": res,
                 "Maximum Heart Rate Achieved": mhr, "Exercise Induced Angina": eia, "Old Peak (mm)": opk, "Peak Exercise Slope": pes, "Number of Major Vessels Colored by fluoroscopy": vca, "Thallium Scan": tha}
'''
## These are the values you entered 🧑‍⚕
'''
st.json(pretty_result)
single_sample = np.array(feature_list).reshape(1, -1)

if st.button("Predict"):
    '''
    ## Results 👁‍🗨

    '''
    loaded_model = load_model('model.pkl')
    prediction = loaded_model.predict(single_sample)
    pred_prob = loaded_model.predict_proba(single_sample)

    if prediction == 1:
        st.error("The person will disease")
    else:
        st.success("The patient will health")

    for health, disease in loaded_model.predict_proba(single_sample):
        health = f"{health*100:.2f}%"
        disease = f"{disease*100:.2f} %"
        st.table(pd.DataFrame({'Health ': health,
                               'Disease': disease}, index=['probability']))
        st.warning(
            "Note: This A.I application is for educational/demo purposes only and cannot be relied upon")

st.sidebar.subheader("Source code")
st.sidebar.info('''

[![Github](https://i.ibb.co/vDLv9z9/iconfinder-mark-github-298822-3.png)](https://github.com/YasinEnigma/Heart-Failure-Decision-Support-System)
**Github**
''')

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
