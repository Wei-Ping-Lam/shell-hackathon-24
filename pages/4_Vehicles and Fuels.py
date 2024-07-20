import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

if 'vehicles_fuels' not in st.session_state:
  st.session_state['vehicles_fuels'] = pd.read_csv('dataset/vehicles_fuels.csv')

if 'fuels' not in st.session_state:
  st.session_state['fuels'] = pd.read_csv('dataset/fuels.csv')

if 'vehicles' not in st.session_state:
  st.session_state['vehicles'] = pd.read_csv('dataset/vehicles.csv')

col1, col2, col3 = st.columns([5, 4, 4])

col1.subheader('Vehicles')
vehicles_df = st.session_state['vehicles']
col1.dataframe(vehicles_df, hide_index = True)
if 'original_vehicles' not in st.session_state:
    st.session_state['original_vehicles'] = pd.read_csv('dataset/vehicles.csv')
if st.session_state['vehicles'].equals(st.session_state['original_vehicles']):
  new_vehicles = col1.file_uploader("Replace Vehicles data", type = '.csv')
  if new_vehicles is not None:
    st.session_state['vehicles'] = pd.read_csv(new_vehicles)
    st.rerun()
if not st.session_state['vehicles'].equals(st.session_state['original_vehicles']):
  if col1.button('Reset to default', key = 'k1'):
    st.session_state['vehicles'] = st.session_state['original_vehicles']
    st.rerun()

col2.subheader('Fuels')
fuels_df = st.session_state['fuels']
col2.dataframe(fuels_df, hide_index = True)
if 'original_fuels' not in st.session_state:
    st.session_state['original_fuels'] = pd.read_csv('dataset/fuels.csv')
if st.session_state['fuels'].equals(st.session_state['original_fuels']):
  new_fuels = col2.file_uploader("Replace Fuels data", type = '.csv')
  if new_fuels is not None:
    st.session_state['fuels'] = pd.read_csv(new_fuels)
    st.rerun()
if not st.session_state['fuels'].equals(st.session_state['original_fuels']):
  if col2.button('Reset to default', key = 'k2'):
    st.session_state['fuels'] = st.session_state['original_fuels']
    st.rerun()

col3.subheader('Vehicle Fuels')
vehicles_fuels_df = st.session_state['vehicles_fuels']
col3.dataframe(vehicles_fuels_df, hide_index = True)
if 'original_vehicles_fuels' not in st.session_state:
    st.session_state['original_vehicles_fuels'] = pd.read_csv('dataset/vehicles_fuels.csv')
if st.session_state['vehicles_fuels'].equals(st.session_state['original_vehicles_fuels']):
  new_vehicles_fuels = col3.file_uploader("Replace Vehicle Fuels data", type = '.csv')
  if new_vehicles_fuels is not None:
    st.session_state['vehicles_fuels'] = pd.read_csv(new_vehicles_fuels)
    st.rerun()
if not st.session_state['vehicles_fuels'].equals(st.session_state['original_vehicles_fuels']):
  if col3.button('Reset to default', key = 'k3'):
    st.session_state['vehicles_fuels'] = st.session_state['original_vehicles_fuels']
    st.rerun()
