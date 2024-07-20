import streamlit as st
import pandas as pd

st.set_page_config(layout="centered")

st.subheader('Resale, Insurance, and Maintenance Cost Profiles')
cost_profiles_df = pd.read_csv('dataset/cost_profiles.csv')
st.dataframe(cost_profiles_df, hide_index = True)