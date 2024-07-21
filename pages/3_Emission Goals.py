import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc, font_manager, cm

st.set_page_config(layout="wide")
c1, c2, c3 = st.columns([1, 4, 1])
c2.header("Here are the maximum limits of your annual CO2 emissions")
c2.subheader('You may change your goals here by double clicking in the table (but keep in mind that these current goals set us to the path to net zero by 2050!)')
st.divider()
col1, col2 = st.columns([5, 6])
if 'carbon_emissions' not in st.session_state:
  st.session_state['carbon_emissions'] = pd.read_csv('./dataset/carbon_emissions.csv')

if 'original_carbon_emissions' not in st.session_state:
  st.session_state['original_carbon_emissions'] = pd.read_csv('./dataset/carbon_emissions.csv')

col4, col5, col6 = col1.columns([1, 2, 1])
carbon_emissions_df = col5.data_editor(
  st.session_state['carbon_emissions'], 
  column_config = {"Carbon": st.column_config.NumberColumn("Carbon emission CO2/kg", min_value = 0, step = 1, format = "%d")},
  use_container_width = False, hide_index = True, disabled=("Year"), width = 300, height = 600)
if not st.session_state['carbon_emissions'].equals(carbon_emissions_df):
    st.session_state['carbon_emissions'] = carbon_emissions_df
    st.rerun()
if not st.session_state['carbon_emissions'].equals(st.session_state['original_carbon_emissions']):
  st.session_state['changedE'] = True
  if col5.button('Reset to default'):
    st.session_state['changedE'] = False
    st.session_state['carbon_emissions'] = st.session_state['original_carbon_emissions']
    st.rerun()

palette = sns.color_palette("colorblind")
palette = sns.color_palette('bright')
sns.set_palette(palette)
ffont = font_manager.FontProperties(family='Arial', style='normal', size=18.0, weight='bold', stretch='normal')
ffontt = font_manager.FontProperties(family='Arial', style='normal', size=22.0, weight='bold', stretch='normal')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(st.session_state['carbon_emissions']['Year'].tolist(), (st.session_state['carbon_emissions']['Carbon emission CO2/kg']/1000000).tolist(), marker = '*', lw = 2, markersize = 9)
ax.set_ylabel('Carbon emission CO$_2$ (kton)', font = ffont)
ax.set_xlabel('Year', font = ffont)
ax.set_title('Annual CO$_2$ Emission Goals', font = ffontt)
ax.spines['top'].set_linewidth(0)
ax.spines['right'].set_linewidth(0)
ax.spines['left'].set_linewidth(1.4)
ax.spines['bottom'].set_linewidth(1.4)
ax.tick_params(axis='x', which='major', direction='out', labelsize=15, width=1.5, length=5)
ax.tick_params(axis='y', which='major', direction='in', labelsize=15, width=1.5, length=5)
for _ in range(0):
  col2.write(' ')
col2.pyplot(fig)