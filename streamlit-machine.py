# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:50:45 2024

@author: karthik
"""


import numpy as np
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#Structuring imports

def run_analytics():
  df = pd.read_csv('D:/DS-Nick-brown/Final-Project/campus-placements/Placement_Data_Full_Class.csv')
  df.head(2)
  #Global dataframe used for all the analytical purposes
  

  plt.figure(figsize=(8,6))
  gender_counts = df['gender'].value_counts()
  
  # Plot the pie chart
  fig = px.pie(
    names=gender_counts.index, 
    values=gender_counts.values,
    labels=gender_counts.index,
    hole=0.3,  # Hole size (0-1)
    color_discrete_sequence=['limegreen', 'lightcoral'],  # Color sequence for each category
    width=370,
    height=370
  )
  
  spec_counts = df['specialisation'].value_counts()
  
  fig1 = px.pie(
    names=spec_counts.index, 
    values=spec_counts.values,
    labels=spec_counts.index,
    hole=0.3,  # Hole size (0-1)
    color_discrete_sequence=['teal', 'skyblue'],  # Color sequence for each category
    width=370,
    height=370
  )
  

  # Show the figure using Streamlit
  #st.plotly_chart(fig)
  
  
  # Plot settings
  plt.figure(figsize=(10, 8))
  
  
  
  
  
  
  col1, col_space, col2 = st.columns([3, 0.2, 3])

  # Display the plots
  with col1:
      st.subheader("Gender map")
      st.plotly_chart(fig)
  
  with col2:
     # Analytics part
     ax = sns.countplot(data=df, x='gender', hue='status', palette=['skyblue', 'green'])
     for p in ax.patches:
         ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                     textcoords='offset points')
     
     st.subheader('Placed Vs Not-Placed')
     plt.xlabel('Gender')
     plt.ylabel('Count')
     st.set_option('deprecation.showPyplotGlobalUse', False)
     # Display the plot using Streamlit
     st.pyplot()
      
  col3, col_space, col4 = st.columns([3,0.2,3])
  with col3:
    st.subheader('specialized ratio')
    st.plotly_chart(fig1)
  with col4:
   #--plot for placement specialization--
   
   # Analytics part
   ax = sns.countplot(data=df, x='specialisation', hue='status', palette=['skyblue', 'green'])
   for p in ax.patches:
       ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                   textcoords='offset points')
   
   st.subheader('Popular Major')
   plt.xlabel('Major')
   plt.ylabel('Count')
   st.set_option('deprecation.showPyplotGlobalUse', False)
   st.pyplot()
  

#This is a test comment to check the status of git
def enhance_input(gender,ssc_p,ssc_b,hsc_p,hsc_b,hsc_s,degree_p,degree_t,workex,etest_p,specialisation,mba_p):
  #Handling Gender 
  if gender=="M":
    gender=1
  else:
    gender=0
  
  #Handling ssc_b
  if ssc_b=="Central":
    ssc_b=0
  else:
    ssc_b=1
  #handling hscb
  if hsc_b=="Central":
    hsc_b=0
  else:
    hsc_b=1
  
  #Handling hsc_s
  if hsc_s=="science":
    hsc_s=2
  elif hsc_s=="arts":
    hsc_s=0
  else:
    hsc_s=1
  
  #Handling degree_t
  if degree_t=="tech":
    degree_t=2
  elif degree_t=="mgmt":
    degree_t=0
  else:
    degree_t=1
  
  #Handling workex
  if workex=="Yes":
    workex=1
  else:
    workex=0
  
  #Handling specialization
  if specialisation=="HR":
    specialisation=1
  else:
    specialisation=0
    
  return predict_data((gender,ssc_p,ssc_b,hsc_p,hsc_b,hsc_s,degree_p,degree_t,workex,etest_p,specialisation,mba_p))
  
  
def rerun_model():
  #This is where the model's re-learning phase has to be triggered. 
  
  print('This block of cell is used to retrigger the modelling ipynb file')
  print('Running....')
  
def predict_data(input_data):
  loaded_model = pickle.load(open('D:/DS-Nick-brown/Final-Project/campus-placements/trained_model.sav','rb'))
  
  input_numpy = np.asarray(input_data)
  
  input_numpy = input_numpy.reshape(1,-1)
  
  prediction = loaded_model.predict(input_numpy)
  
  print(prediction)
  
  if prediction[0]==0:
    
    return "You will not be placed"
  else:
    return "You will be placed"
  
#feed = (1,65.00,1,25,1,1,60,2,0,57.00,0,59.00)
#Streamlit code starts (main)

#gender	ssc_p	ssc_b	hsc_p	hsc_b	hsc_s	degree_p	degree_t	
#workex	etest_p	specialisation	mba_p

def main():
    st.title("Campus Placement Prediction Software")
    
    st.subheader("Want to know where you stand?? ")
    gender = st.selectbox("Please choose your gender", ["M", "F"], index=0)
    ssc_p=value = st.slider("Enter your secondary school percentage", min_value=0.0, max_value=100.0, step=0.01, format="%.2f")
    ssc_b=st.selectbox("Choose your secondary school board",["Central","Others"],index=1)
    #HSC details
    hsc_p=st.slider("Enter your high school percentage", min_value=0.0, max_value=100.0, step=0.01, format="%.2f")
    hsc_b=st.selectbox("Choose your high school board",["Central","Others"],index=1)
    hsc_s=st.selectbox("Choose your high school subject",["science","commerce","arts"],index=1)
    #Degree details
    degree_p=st.slider("Enter your degree percentage", min_value=0.0, max_value=100.0, step=0.01, format="%.2f")
    degree_t=st.selectbox("Enter your degree subject",["tech","mgmt","others"],index=1)
    #Work ex
    workex=st.selectbox("Do you have work experience?",["Yes","No"],index=1)
    #entrance test percentage
    etest_p=st.slider("Enter your entrance test percentage", min_value=0.0, max_value=100.0, step=0.01, format="%.2f")
    #Specialization
    specialisation=st.selectbox("What is your specialisation",["HR","finance"],index=1)
    #mba percentage
    mba_p=st.slider("Enter your MBA percentage", min_value=0.0, max_value=100.0, step=0.01, format="%.2f")
    
 
    output=""
    a=""
    if st.button("Rate your chances!"):
       output=enhance_input(gender,ssc_p,ssc_b,hsc_p,hsc_b,hsc_s,degree_p,degree_t,workex,etest_p,specialisation,mba_p)
    st.success(output)
    
    
    st.subheader("Contribute to our software!!")
    
    if(st.button("Run Analytics")):
      run_analytics()
      

if __name__ == '__main__':
    main()

  
  




