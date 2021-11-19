import streamlit as st
import pandas as pd
import pickle
import sklearn
import xgboost
import numpy as np

pipe = pickle.load(open('pipe.pkl','rb'))



teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Delhi Daredevils',
 'Kings XI Punjab',
 'Rajasthan Royals',
 'Chennai Super Kings',
 'Delhi Capitals']

cities = ['Sharjah', 'Mumbai', 'Delhi', 'Bangalore', 'Pune', 'Ranchi',
       'Johannesburg', 'Ahmedabad', 'Chandigarh', 'Cape Town', 'Chennai',
       'Durban', 'Abu Dhabi', 'Jaipur', 'Bengaluru', 'Hyderabad',
       'Kolkata', 'Dubai', 'Port Elizabeth', 'Dharamsala', 'Centurion',
       'Indore', 'East London', 'Kimberley', 'Visakhapatnam', 'Raipur',
       'Cuttack', 'Bloemfontein']

st.title("IPL Score Predictor")

col1,col2 = st.beta_columns(2)

with col1:
    batting_team = st.selectbox("Select batting team",sorted(teams))
with col2:
    bowling_team = st.selectbox("Select bowling team",sorted(teams))

city =  st.selectbox("Select City",sorted(cities))

col3,col4,col5 = st.beta_columns(3)

with col3:
    current_score = st.number_input("Current Score")

with col4:
    overs = st.number_input("Overs done (works for over>5)")

with col5:
    wickets = st.number_input("Wickets Out")

last_five = st.number_input('Runs scored in last 5 overs')
if st.button("Predict Score"):
    balls_left = 120 - (overs*6)
    wickets_left = 10-wickets
    crr = current_score/overs

    input_df = pd.DataFrame(
        {'batting_team':[batting_team],'bowling_team':[bowling_team],'city':city,'current_score':[current_score],
    'balls_left':[balls_left],'wickets_left':[wickets],'crr':[crr],'last_five':[last_five]})


    #st.text(sklearn.__version__)
    result = pipe.predict(input_df)
    st.header("Predicted Score " + str(result[0].round(0)))


