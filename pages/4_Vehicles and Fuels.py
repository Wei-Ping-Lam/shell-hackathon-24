import streamlit as st
import pandas as pd
import copy
st.set_page_config(layout="wide")

if 'vehicles_fuels' not in st.session_state:
  st.session_state['vehicles_fuels'] = pd.read_csv('dataset/vehicles_fuels.csv')

if 'fuels' not in st.session_state:
  st.session_state['fuels'] = pd.read_csv('dataset/fuels.csv')

if 'vehicles' not in st.session_state:
  st.session_state['vehicles'] = pd.read_csv('dataset/vehicles.csv')

col1, col2, col3 = st.columns([5, 4.2, 4])

col1.subheader('Vehicles')
col1.markdown('You may change the costs by double clicking in the table or by uploading a new file')

if 'original_vehicles' not in st.session_state:
  st.session_state['original_vehicles'] = pd.read_csv('dataset/vehicles.csv')

vehicles_df = col1.data_editor(st.session_state['vehicles'], column_config = {"Cost": st.column_config.NumberColumn("Cost ($)", min_value = 0, step = 0.01, format = "%.2f")}, hide_index = True, disabled=["ID", "Vehicle", "Size", "Year", "Yearly range (km)", "Distance"])
if not st.session_state['vehicles'].equals(vehicles_df):
    st.session_state['vehicles'] = vehicles_df
    st.rerun()
if not st.session_state['vehicles'].equals(st.session_state['original_vehicles']):
  if col1.button('Reset to default'):
    st.session_state['vehicles'] = st.session_state['original_vehicles']
    del st.session_state['delme']
    st.rerun()

new_vehicles = col1.file_uploader("Replace Vehicles data", type = '.csv', key = 'delme')
col1.download_button("Download Vehicles Template", pd.read_csv('./dataset/vehicles_template.csv').to_csv(index=False), 'vehicles_template.csv')
if new_vehicles is not None:
  st.session_state['vehicles'] = pd.read_csv(new_vehicles)
  del st.session_state['delme']
  new_vehicles = None
  st.rerun()


col2.subheader('Fuels')
col2.markdown('You may change the emissions and costs by double clicking in the table or by uploading a new file')

if 'original_fuels' not in st.session_state:
    st.session_state['original_fuels'] = pd.read_csv('dataset/fuels.csv')

fuels_df = col2.data_editor(st.session_state['fuels'], column_config = {"Emissions": st.column_config.NumberColumn("Emissions (CO2/unit_fuel)", min_value = 0, step = 0.000001, format = "%.6f"), "Cost": st.column_config.NumberColumn("Cost ($/unit_fuel)", min_value = 0, step = 0.000001, format = "%.6f")}, hide_index = True, disabled=["Fuel", "Year"])
if not st.session_state['fuels'].equals(fuels_df):
    st.session_state['fuels'] = fuels_df
    st.rerun()
if not st.session_state['fuels'].equals(st.session_state['original_fuels']):
  if col2.button('Reset to default'):
    st.session_state['fuels'] = st.session_state['original_fuels']
    del st.session_state['delme2']
    st.rerun()

new_fuels = col2.file_uploader("Replace Fuels data", type = '.csv', key = 'delme2')
col2.download_button("Download Fuels Template", pd.read_csv('./dataset/fuels_template.csv').to_csv(index=False), 'fuels_template.csv')
if new_fuels is not None:
  st.session_state['fuels'] = pd.read_csv(new_fuels)
  del st.session_state['delme2']
  new_fuels = None
  st.rerun()


col3.subheader('Vehicle Fuels')
col3.markdown('You may change the consumptions by double clicking in the table or by uploading a new file')

if 'original_vehicles_fuels' not in st.session_state:
    st.session_state['original_vehicles_fuels'] = pd.read_csv('dataset/vehicles_fuels.csv')

vehicles_fuels_df = col3.data_editor(st.session_state['vehicles_fuels'], column_config = {"Consumption": st.column_config.NumberColumn("Consumption (unit_fuel/km)", min_value = 0, step = 0.000001, format = "%.6f")}, hide_index = True, disabled=["ID", "Fuel"])
if not st.session_state['vehicles_fuels'].equals(vehicles_fuels_df):
    st.session_state['vehicles_fuels'] = vehicles_fuels_df
    st.rerun()
if not st.session_state['vehicles_fuels'].equals(st.session_state['original_vehicles_fuels']):
  if col3.button('Reset to default'):
    st.session_state['vehicles_fuels'] = st.session_state['original_vehicles_fuels']
    del st.session_state['delme3']
    st.rerun()

new_vehicles_fuels = col3.file_uploader("Replace Vehicle Fuels data", type = '.csv', key = 'delme3')
col3.download_button("Download Vehicle Fuels Template", pd.read_csv('./dataset/vehicles_fuels_template.csv').to_csv(index=False), 'vehicles_fuels_template.csv')
if new_vehicles_fuels is not None:
  st.session_state['vehicles_fuels'] = pd.read_csv(new_vehicles_fuels)
  del st.session_state['delme3']
  new_vehicles_fuels = None
  st.rerun()

