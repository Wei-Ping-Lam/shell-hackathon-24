import streamlit as st
import pandas as pd

st.set_page_config(layout="centered")
if 'carbon_emissions' not in st.session_state:
  st.session_state['carbon_emissions'] = pd.read_csv('./dataset/carbon_emissions.csv')
if 'cost_profiles' not in st.session_state:
  st.session_state['cost_profiles'] = pd.read_csv('dataset/cost_profiles.csv')
if 'vehicles_fuels' not in st.session_state:
  st.session_state['vehicles_fuels'] = pd.read_csv('dataset/vehicles_fuels.csv')
if 'fuels' not in st.session_state:
  st.session_state['fuels'] = pd.read_csv('dataset/fuels.csv')
if 'vehicles' not in st.session_state:
  st.session_state['vehicles'] = pd.read_csv('dataset/vehicles.csv')
if 'demand' not in st.session_state:
  st.header("Input your demand")
  st.markdown("For each year (2023-2038), there should be 16 inputs. Each 4 vehicle sizes should be included in each 4 distance buckets.")
  demanded = st.file_uploader("Choose a CSV file", type = '.csv')
  st.download_button("Download Demand Template", pd.read_csv('./dataset/demand_template.csv').to_csv(index=False), 'demand_template.csv')
  if st.button("Use default demand data"):
    demand_df = pd.read_csv('./dataset/demand.csv')
    st.session_state['original_demand'] = demand_df
    st.session_state['demand'] = demand_df
    st.rerun()

  if demanded is not None:
    demand_df = pd.read_csv(demanded)
    st.session_state['original_demand'] = demand_df
    st.session_state['demand'] = demand_df
    st.rerun()

if 'demand' in st.session_state:
  st.header("Here is your demand")
  st.subheader('You may change the demand here by double clicking in the table')
  dem_df = st.data_editor(
    st.session_state['demand'], 
    column_config = {"demand": st.column_config.NumberColumn("Demand (km)", min_value = 0, step = 1, format = "%d")},
    use_container_width = False, hide_index = True, disabled=("Year", "Size", "Distance"), width = 400)
  if not st.session_state['demand'].equals(dem_df):
    st.session_state['demand'] = dem_df
    st.rerun()
  
  col1, col2 = st.columns([1, 1.6])
  if col1.button('Remove demand'):
    del st.session_state['demand']
    del st.session_state['original_demand']
    st.rerun()

  if not st.session_state['demand'].equals(st.session_state['original_demand']):
    if col2.button('Reset demand'):
      st.session_state['demand'] = st.session_state['original_demand']
      st.rerun()
  
  if st.button('Go to Fleet Optimizer'):
    st.switch_page("pages/2_Fleet Optimizer.py")