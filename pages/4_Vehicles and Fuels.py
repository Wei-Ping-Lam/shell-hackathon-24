import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

col1, col2, col3 = st.columns([5, 4, 4])

col1.subheader('Vehicles')
vehicles_df = st.session_state['vehicles']
col1.dataframe(vehicles_df, hide_index = True)

col2.subheader('Fuels')
fuels_df = st.session_state['fuels']
col2.dataframe(fuels_df, hide_index = True)

col3.subheader('Vehicle Fuels')
vehicles_fuels_df = st.session_state['vehicles_fuels']
col3.dataframe(vehicles_fuels_df, hide_index = True)
