import streamlit as st

st.header("Fleet Optimization Solver")

if 'demand' not in st.session_state:
  st.subheader('Import your demand first')
  if st.button('Go to Demand page'):
    st.switch_page("pages/1_Demand.py")

if 'demand' in st.session_state:
  #option_years = []
  #for i in range(st.session_state['range_start'], st.session_state['range_end']+1):
  #  option_years.append(i)

  #start_year, end_year = st.select_slider(
  #  "Choose range of years to optimize for", 
  #  options=option_years,
  #  value = (st.session_state['year_one'], st.session_state['year_two'])
  #)

  #st.session_state['year_one'] = start_year
  #st.session_state['year_two'] = end_year
