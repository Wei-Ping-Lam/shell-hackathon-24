import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc, font_manager, cm

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

st.set_page_config(layout="centered")
st.title("Fleet Decarbonization")
st.image('./fleet_image.png')

carbon_emissions_df = st.session_state['carbon_emissions']

palette = sns.color_palette("colorblind")
palette = sns.color_palette('bright')
sns.set_palette(palette)

ffont = font_manager.FontProperties(family='Arial', style='normal', size=18.0, weight='bold', stretch='normal')
ffontt = font_manager.FontProperties(family='Arial', style='normal', size=22.0, weight='bold', stretch='normal')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(carbon_emissions_df['Year'].tolist(), (carbon_emissions_df['Carbon emission CO2/kg']/1000000).tolist(), marker = '*', lw = 2, markersize = 9, label = 'Set by Shell.ai')
carbon_year = [2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050]
carbon_emis = [2.404387, 2.140637125, 1.905103375, 1.68424575, 1.4751625, 1.275311875, 1.08251125, 0.89493925, 0.711134125, 0.529993875, 0.3507775, 0.1731035, 0]
ax.plot(carbon_year, carbon_emis, '--', color = 'k', lw = 2, label = 'Path to Net Zero')
ax.set_ylabel('Carbon emission CO$_2$ (kton)', font = ffont)
ax.set_xlabel('Year', font = ffont)
ax.set_title('Reach Net Zero by 2050!', font = ffontt)
ax.spines['top'].set_linewidth(0)
ax.spines['right'].set_linewidth(0)
ax.spines['left'].set_linewidth(1.4)
ax.spines['bottom'].set_linewidth(1.4)
ax.set_xticks([2024, 2026, 2028, 2030, 2032, 2034, 2036, 2038, 2040, 2042, 2044, 2046, 2048, 2050])
ax.tick_params(axis='x', which='major', direction='out', labelsize=15, width=1.5, length=5)
ax.tick_params(axis='y', which='major', direction='in', labelsize=15, width=1.5, length=5)
ax.legend(loc = 'center right', fontsize = 15)

st.pyplot(fig)

st.write(':white_check_mark: Input your demand!')
st.write(':white_check_mark: Confirm your emission goals!')
st.write(':white_check_mark: Solve for your optimal fleet!')
st.write(':white_check_mark: View analytics of your new solution!')

if st.button('Start your journey here!', type='primary'):
  st.switch_page("pages/1_Demand.py")