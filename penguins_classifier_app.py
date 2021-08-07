import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction App

## This app predicts the **Palmer Penguin** species!

Data obtained from the [palmer penguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst. 
""")

# creating the sidebar
st.sidebar.header("User Input Features")
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

upload_file = st.sidebar.file_uploader("Upload your csv file here", type=["csv"])


# sliders part as a function
def user_input_sliders():
    island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    bill_length = st.sidebar.slider('Bill Length (mm)', 32.0, 60.0, 46.5)
    bill_depth = st.sidebar.slider("Bill Depth (mm)", 13.0, 22.0, 14.2)
    flipper_length = st.sidebar.slider("Flipper Length (mm)", 171.0, 232.0, 198.5)
    body_mass = st.sidebar.slider("Body mass (g)", 2600, 6400, 3830)
    data = {
        'island': island,
        'sex': sex,
        'bill_length_mm':bill_length,
        'bill_depth_mm':bill_depth,
        'flipper_length_mm':flipper_length,
        'body_mass_g':body_mass
    }
    features = pd.DataFrame(data,index=[0])
    return features

# condition to get input values


if upload_file is not None:
    input_df = pd.read_csv(upload_file)
else:
    input_df = user_input_sliders()

# showing the user input
st.subheader("User Input Features")
if upload_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)

# importing the penguins file and processing
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1]

# reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl','rb'))

# prediction
prediction = load_clf.predict(df)
prediction_prob = load_clf.predict_proba(df)

# displaying the prediction
st.subheader("Prediction")
penguin_species = np.array((['Adelie','Chinstrap','Gentoo']))
st.write(penguin_species[prediction])

st.subheader("Prediction Probability")
st.write(prediction_prob)


