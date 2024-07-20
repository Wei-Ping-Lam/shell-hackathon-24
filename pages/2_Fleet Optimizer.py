import streamlit as st

st.header("Fleet Optimization Solver")

if 'demand' not in st.session_state:
  st.subheader('Import your demand first')
  if st.button('Go to Demand page'):
    st.switch_page("pages/1_Demand.py")

if 'demand' in st.session_state:
  x = 5
