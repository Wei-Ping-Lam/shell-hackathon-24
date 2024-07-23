import streamlit as st
import pandas as pd

st.set_page_config(layout="centered")

st.header('Resale, Insurance, and Maintenance Cost Profiles')
#st.subheader("You may change the cost profiles here by double clicking in the table")

if 'cost_profiles' not in st.session_state:
  st.session_state['cost_profiles'] = pd.read_csv('dataset/cost_profiles.csv')
#if 'original_cost_profiles' not in st.session_state:
#  st.session_state['original_cost_profiles'] = pd.read_csv('dataset/cost_profiles.csv')
st.dataframe(st.session_state['cost_profiles'], hide_index = True)

#cost_profiles_df = st.data_editor(
#  st.session_state['cost_profiles'], 
#  column_config = {"Resale": st.column_config.NumberColumn("Resale Value %", min_value = 0, max_value=100, step = 1, format = "%d"),
#  "Ins": st.column_config.NumberColumn("Insurance Cost %", min_value = 0, step = 1, format = "%d"), 
#  "Man": st.column_config.NumberColumn("Maintenance Cost %", min_value = 0, step = 1, format = "%d")},
#  use_container_width = True, hide_index = True, disabled=("End of Year"))
#if not st.session_state['cost_profiles'].equals(cost_profiles_df):
#    st.session_state['cost_profiles'] = cost_profiles_df
#    st.rerun()
#if not st.session_state['cost_profiles'].equals(st.session_state['original_cost_profiles']):
#  if st.button('Reset to default'):
#    st.session_state['cost_profiles'] = st.session_state['original_cost_profiles']
#    st.rerun()

if 'sell_percent' not in st.session_state:
  st.session_state['sell_percent'] = 20.0

def store_value():
  st.session_state["sell_percent"] = st.session_state["_sell_percent"]

hi = st.number_input(" s", min_value=0.0, max_value=100.0, value = st.session_state["sell_percent"], key = '_sell_percent', label_visibility = 'hidden', on_change=store_value)
st.markdown('Every year at most %g%% of the vehicles in the existing fleet can be sold. (default 20%%)' % hi)