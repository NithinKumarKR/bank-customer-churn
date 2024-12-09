import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler ,LabelEncoder , OneHotEncoder
from tensorflow.keras.models import load_model
import pandas as pd
import pickle


##load All th model
model=load_model('model.h5')
with open('gender_LabelEncoder.pkl','rb') as f:
    gender_LabelEncoder=pickle.load(f)
with open('geo_encoder.pkl','rb') as f:
    geo_encoder=pickle.load(f)
with open('card_type_encoder.pkl','rb') as f:
    card_type_encoder=pickle.load(f)
with open('scaler.pkl','rb') as f:
    scaler=pickle.load(f)

st.title('Customer Churn Prediction')

geography= st.selectbox('Geography', geo_encoder.categories_[0])
gender= st.selectbox('Gender', gender_LabelEncoder.classes_)
card_type= st.selectbox('Card Type', card_type_encoder.categories_[0])
age= st.slider('Age',18,92)
balance=st.number_input('Balance')
credit=st.slider('Credit Score',0,900)
Salary=st.number_input('Salary')
tenure= st.slider('Tenure',0,50)
products= st.slider('Product',0,10)
HasCrCard=st.selectbox('HasCrCard', [0,1])
IsActiveMember=st.selectbox('IsActiveMember', [0,1])
Complain=st.selectbox('Complain', [0,1])
Satisfaction=st.slider('Satisfaction',0,10)
Point=st.slider('Point Earned',0,1000)


input= {
 'CreditScore': credit,
 'Geography': geography,
 'Gender': gender,
 'Age': age,
 'Tenure': tenure,
 'Balance': balance ,
 'NumOfProducts': products,
 'HasCrCard': HasCrCard,
 'IsActiveMember': IsActiveMember,
 'EstimatedSalary':Salary,
 'Complain': HasCrCard,
 'Satisfaction Score': Satisfaction,
 'Card Type': card_type,
 'Point Earned':Point}

input_df=pd.DataFrame([input])
card_type_encoded=card_type_encoder.transform([[input['Card Type']]]).toarray()

card_type_encoded_df = pd.DataFrame(
    card_type_encoded,
    columns=card_type_encoder.get_feature_names_out(['Card Type']))
card_type_encoded_df = card_type_encoded_df.head(1).to_dict(orient='records')[0]

card_type_encoded=card_type_encoder.transform([[input['Card Type']]]).toarray()
card_type_encoded_df = pd.DataFrame(
    card_type_encoded,
    columns=card_type_encoder.get_feature_names_out(['Card Type']))


input_df['Gender']=gender_LabelEncoder.transform(input_df['Gender'])
input_df=input_df.drop('Card Type', axis=1)


geo_encoded=geo_encoder.transform([[input['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=geo_encoder.get_feature_names_out(['Geography']))

input_df=input_df.drop('Geography', axis=1)

input_df=pd.concat([input_df,geo_encoded_df,card_type_encoded_df],axis=1)


input_df_scaled=scaler.transform(input_df)

st.title('Customer Churn Prediction Output')

st.write("Thank you for giving your inputs")

pred=model.predict(input_df_scaled)

if pred[0][0]>.5:
    st.write("Customer will leave the bank",pred[0][0]) 
else:
    st.write("Customer will not leave the bank",pred[0][0]) 








