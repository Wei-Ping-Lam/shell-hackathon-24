import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc, font_manager, cm

st.title("Fleet Decarbonization")
st.image('./fleet_image.png')

carbon_emissions_df = pd.read_csv('./dataset/carbon_emissions.csv')
#st.line_chart(carbon_emissions_df, x='Year', y='Carbon emission CO2/kg', y_label=r'Carbon emission CO2 (kg)')

palette = sns.color_palette("colorblind")
palette = sns.color_palette('bright')
sns.set_palette(palette)

ffont = font_manager.FontProperties(family='Arial', style='normal', size=18.0, weight='bold', stretch='normal')
ffontt = font_manager.FontProperties(family='Arial', style='normal', size=22.0, weight='bold', stretch='normal')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(carbon_emissions_df['Year'].tolist()[:16], (carbon_emissions_df['Carbon emission CO2/kg']/1000000).tolist()[:16], marker = '*', lw = 2, markersize = 9)
ax.plot(carbon_emissions_df['Year'].tolist()[15:], (carbon_emissions_df['Carbon emission CO2/kg']/1000000).tolist()[15:], '--', color = 'k', lw = 2)
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

st.pyplot(fig)

if st.button('Start your journey here!'):
  st.switch_page("pages/1_Demand.py")