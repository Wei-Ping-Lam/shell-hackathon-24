import streamlit as st
import pandas as pd

st.set_page_config(layout="centered")
st.header("Here are the maximum limits of your annual CO2 emissions")
st.subheader('You may change your goals here (but keep in mind that these current goals set us to the path to net zero by 2050!)')

if 'carbon_emissions' not in st.session_state:
  carbon_emissions_df = pd.read_csv('./dataset/carbon_emissions.csv')
  st.session_state['carbon_emissions'] = carbon_emissions_df

if 'original_carbon_emissions' not in st.session_state:
  original_carbon_emissions_df = pd.read_csv('./dataset/carbon_emissions.csv')
  st.session_state['original_carbon_emissions'] = original_carbon_emissions_df

carbon_emissions_df = st.data_editor(
  st.session_state['carbon_emissions'], 
  column_config = {"Carbon": st.column_config.NumberColumn("Carbon emission CO2/kg", min_value = 0, step = 1, format = "%d")},
  use_container_width = False, hide_index = True, disabled=("Year"), width = 300)
if not st.session_state['carbon_emissions'].equals(carbon_emissions_df):
    st.session_state['carbon_emissions'] = carbon_emissions_df
    st.rerun()
if not st.session_state['carbon_emissions'].equals(st.session_state['original_carbon_emissions']):
  if st.button('Reset'):
    st.session_state['carbon_emissions'] = st.session_state['original_carbon_emissions']
    st.rerun()