import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc, font_manager, cm
import numpy as np
import math
import copy
import random
import time

st.set_page_config(layout="wide")
col1, col2, col3 = st.columns([1, 3, 1])
col2.header("Fleet Optimization Solver")

if 'demand' not in st.session_state:
  col2.subheader('Import your demand first')
  if col2.button('Go to Demand page'):
    st.switch_page("pages/1_Demand.py")

def main_fun():
  with col2.status("Optimizing...", expanded=True) as status:
    st.write("Creating Strong Initial Solution...")

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
    
    st.session_state['sol_demand'] = st.session_state['demand']
    st.session_state['sol_carbon_emissions'] = st.session_state['carbon_emissions']
    st.session_state['sol_cost_profiles'] = st.session_state['cost_profiles']
    st.session_state['sol_vehicles_fuels'] = st.session_state['vehicles_fuels']
    st.session_state['sol_fuels'] = st.session_state['fuels']
    st.session_state['sol_vehicles'] = st.session_state['vehicles']

    demand_df = st.session_state['demand']
    cost_profiles_df = st.session_state['cost_profiles']
    vehicles_fuels_df = st.session_state['vehicles_fuels']
    fuels_df = st.session_state['fuels']
    vehicles_df = st.session_state['vehicles']
    carbon_emissions_df = st.session_state['carbon_emissions']

    class Vehicle:
      def __init__(self, ID, _type, size, year, distance, cost, _range):
        self.ID = ID
        self._type = _type
        self.size = size
        self.year = year
        self.distance = distance
        self.cost = cost
        self._range = _range
        
      def get_vehicle_cost(self):
        return self.cost
      
      def get_insurance_cost(self, year_now):
        time = year_now - self.year
        return cost_profiles_df['Insurance Cost %'][time]/100 * self.cost
      
      def get_maintenance_cost(self, year_now):
        time = year_now - self.year
        return cost_profiles_df['Maintenance Cost %'][time]/100 * self.cost
      
      def get_fuel_cost(self, year_now, fuel_type, distance_driven):
        fuel_cost_per_unit = fuels_df.loc[fuels_df['Fuel'] == fuel_type].loc[fuels_df.loc[fuels_df['Fuel'] == fuel_type]['Year'] == year_now]['Cost ($/unit_fuel)'].tolist()[0]
        fuel_unit_per_km = vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == self.ID].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == self.ID]['Fuel'] == fuel_type]['Consumption (unit_fuel/km)'].tolist()[0]
        return fuel_cost_per_unit*fuel_unit_per_km*distance_driven
      
      def get_emissions(self, year_now, fuel_type, distance_driven):
        emissions_per_unit = fuels_df.loc[fuels_df['Fuel'] == fuel_type].loc[fuels_df.loc[fuels_df['Fuel'] == fuel_type]['Year'] == year_now]['Emissions (CO2/unit_fuel)'].tolist()[0]
        fuel_unit_per_km = vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == self.ID].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == self.ID]['Fuel'] == fuel_type]['Consumption (unit_fuel/km)'].tolist()[0]
        return emissions_per_unit*fuel_unit_per_km*distance_driven
      
      def get_resale(self, year_now):
        time = year_now - self.year
        return cost_profiles_df['Resale Value %'][time]/100 * self.cost
      
    Vehicles = {}
    for i in range(len(vehicles_df)):
      vehicle_ID = vehicles_df['ID'][i]
      vehicle_type = vehicles_df['Vehicle'][i]
      vehicle_size = vehicles_df['Size'][i]
      vehicle_year = vehicles_df['Year'][i]
      vehicle_distance = vehicles_df['Distance'][i]
      vehicle_cost = vehicles_df['Cost ($)'][i]
      vehicle_range = vehicles_df['Yearly range (km)'][i]
      Vehicles[vehicle_ID] = Vehicle(vehicle_ID, vehicle_type, vehicle_size, vehicle_year, vehicle_distance, vehicle_cost, vehicle_range)

    class Demand:
      def __init__(self, year, size, distance, _demand):
        self.year = year
        self.size = size
        self.distance = distance
        self._demand = _demand
        self.met_by = {}
        
      def add_vehicle(self, ID, quant, fuel, kms):
        self.met_by[ID] = [quant, fuel, kms]
        
      def is_met(self):
        satis = 0
        for atrib in self.met_by.values():
          satis += atrib[0]*atrib[2]
        return satis >= self._demand

    demand = []
    for i in range(len(demand_df)):
      demand_year = demand_df['Year'][i]
      demand_size = demand_df['Size'][i]
      demand_distance = demand_df['Distance'][i]
      demand_demand = demand_df['Demand (km)'][i]
      demand.append(Demand(demand_year, demand_size, demand_distance, demand_demand))















    
    st.write("Optimizing Solution and Reducing Costs...")
    time.sleep(1)
    st.write("Reducing Carbon Emissions...")
    time.sleep(1)
    st.session_state['submission'] = pd.read_csv('dataset/sample_submission.csv')
    status.update(label="Solution complete!", state='complete', expanded=True)
    time.sleep(1)


if 'demand' in st.session_state:
  placeholder2 = col2.empty()
  placeholder = col2.empty()
  if not 'submission' in st.session_state:
    if placeholder2.button("Optimize Fleet!", type='primary', use_container_width =True, key='p1'):
      placeholder2.empty()
      main_fun()
    if st.session_state['demand'].equals(st.session_state['original_demand']):
      if placeholder.button("Use a sample solution to default demand", key = 't2'):
        st.session_state['submission'] = pd.read_csv('dataset/sample_submission.csv')
        st.rerun()
  if 'submission' in st.session_state:
    placeholder.empty()
    placeholder2.empty()
    kol1, kol2, kol3 = st.columns([1, 3, 1])

    flag1 = st.session_state['sol_demand'].equals(st.session_state['demand'])
    flag2 = st.session_state['sol_carbon_emissions'].equals(st.session_state['carbon_emissions'])
    flag3 = st.session_state['sol_cost_profiles'].equals(st.session_state['cost_profiles'])
    flag4 = st.session_state['sol_vehicles_fuels'].equals(st.session_state['vehicles_fuels'])
    flag5 = st.session_state['sol_fuels'].equals(st.session_state['fuels'])
    flag6 = st.session_state['sol_vehicles'].equals(st.session_state['vehicles'])
    if not flag1 or not flag2 or not flag3 or not flag4 or not flag5 or not flag6:
      plakeholder2 = kol2.empty()
      plakeholder2.subheader('Your (emissions, vehicle, fuel, vehicle fuel, cost profile) data changed')
      plakeholder = kol2.empty()
      if plakeholder.button("Re-Optimize Fleet!", type='primary', use_container_width =True, key='p6'):
        plakeholder2.empty()
        plakeholder.empty()
        main_fun()

    kol2.header('Here are your solutions costs and analytics!')
    sub = st.session_state['submission']
    c1, c2, c3 = kol2.columns([1, 1, 3])
    c1.download_button("Download solution", sub.to_csv(index=False), 'solution.csv', type='primary')
    if c2.button('Delete solution'):
      del st.session_state['submission']
      st.rerun()

    
    cost_profiles_df = st.session_state['cost_profiles']
    vehicles_fuels_df = st.session_state['vehicles_fuels']
    fuels_df = st.session_state['fuels']
    vehicles_df = st.session_state['vehicles']
    carbon_emissions_df = st.session_state['carbon_emissions']
    if 'original_carbon_emissions' not in st.session_state:
      st.session_state['original_carbon_emissions'] = pd.read_csv('./dataset/carbon_emissions.csv')

    class Vehicle:
      def __init__(self, ID, _type, size, year, distance, cost, _range):
        self.ID = ID
        self._type = _type
        self.size = size
        self.year = year
        self.distance = distance
        self.cost = cost
        self._range = _range
      
      def get_vehicle_cost(self):
        return self.cost
      
      def get_insurance_cost(self, year_now):
        time = year_now - self.year
        return cost_profiles_df['Insurance Cost %'][time]/100 * self.cost
      
      def get_maintenance_cost(self, year_now):
        time = year_now - self.year
        return cost_profiles_df['Maintenance Cost %'][time]/100 * self.cost
      
      def get_fuel_cost(self, year_now, fuel_type, distance_driven):
        fuel_cost_per_unit = fuels_df.loc[fuels_df['Fuel'] == fuel_type].loc[fuels_df.loc[fuels_df['Fuel'] == fuel_type]['Year'] == year_now]['Cost ($/unit_fuel)'].tolist()[0]
        fuel_unit_per_km = vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == self.ID].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == self.ID]['Fuel'] == fuel_type]['Consumption (unit_fuel/km)'].tolist()[0]
        return fuel_cost_per_unit*fuel_unit_per_km*distance_driven
      
      def get_emissions(self, year_now, fuel_type, distance_driven):
        emissions_per_unit = fuels_df.loc[fuels_df['Fuel'] == fuel_type].loc[fuels_df.loc[fuels_df['Fuel'] == fuel_type]['Year'] == year_now]['Emissions (CO2/unit_fuel)'].tolist()[0]
        fuel_unit_per_km = vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == self.ID].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == self.ID]['Fuel'] == fuel_type]['Consumption (unit_fuel/km)'].tolist()[0]
        return emissions_per_unit*fuel_unit_per_km*distance_driven
      
      def get_resale(self, year_now):
        time = year_now - self.year
        return cost_profiles_df['Resale Value %'][time]/100 * self.cost

    Vehicles = {}
    for i in range(len(vehicles_df)):
      vehicle_ID = vehicles_df['ID'][i]
      vehicle_type = vehicles_df['Vehicle'][i]
      vehicle_size = vehicles_df['Size'][i]
      vehicle_year = vehicles_df['Year'][i]
      vehicle_distance = vehicles_df['Distance'][i]
      vehicle_cost = vehicles_df['Cost ($)'][i]
      vehicle_range = vehicles_df['Yearly range (km)'][i]
      Vehicles[vehicle_ID] = Vehicle(vehicle_ID, vehicle_type, vehicle_size, vehicle_year, vehicle_distance, vehicle_cost, vehicle_range)


    fleet = {}

    sub_buy = sub.loc[sub['Type'] == 'Buy']
    sub_use = sub.loc[sub['Type'] == 'Use']
    sub_sell = sub.loc[sub['Type'] =='Sell']

    C_buy = []
    C_ins = []
    C_mnt = []
    C_fuel = []
    CO2_emissions = []
    V_sell = []
    V2_sell = 0
    year_list = []
    total_2028 = 0
    total_2038 = 0
    for i in range(2023, 2039):
      year_list.append(i)
      ids = sub_buy.loc[sub_buy['Year'] == i]['ID'].tolist()
      nums = sub_buy.loc[sub_buy['Year'] == i]['Num_Vehicles'].tolist()
      buy_cost = 0
      for j in range(len(ids)):
        fleet[ids[j]] = nums[j]
        buy_cost += Vehicles[ids[j]].get_vehicle_cost() * nums[j]
      C_buy.append(buy_cost)
      
      ins_cost = 0
      mnt_cost = 0
      for vehicle, stock in fleet.items():
        ins_cost += Vehicles[vehicle].get_insurance_cost(i) * stock
        mnt_cost += Vehicles[vehicle].get_maintenance_cost(i) * stock
      C_ins.append(ins_cost)
      C_mnt.append(mnt_cost)
      
      ids = sub_use.loc[sub_use['Year'] == i]['ID'].tolist()
      nums = sub_use.loc[sub_use['Year'] == i]['Num_Vehicles'].tolist()
      fuels = sub_use.loc[sub_use['Year'] == i]['Fuel'].tolist()
      kms = sub_use.loc[sub_use['Year'] == i]['Distance_per_vehicle(km)'].tolist()
      fuel_cost = 0
      emit = 0
      for j in range(len(ids)):
        fuel_cost += Vehicles[ids[j]].get_fuel_cost(i, fuels[j], kms[j]) * nums[j]
        emit += Vehicles[ids[j]].get_emissions(i, fuels[j], kms[j]) * nums[j]
      C_fuel.append(fuel_cost)
      CO2_emissions.append(emit/1000000)
      
      ids = sub_sell.loc[sub_sell['Year'] == i]['ID'].tolist()
      nums = sub_sell.loc[sub_sell['Year'] == i]['Num_Vehicles'].tolist()
      sell_value = 0
      for j in range(len(ids)):
        fleet[ids[j]] -= nums[j]
        if fleet[ids[j]] == 0:
          del fleet[ids[j]]
        sell_value += Vehicles[ids[j]].get_resale(i) * nums[j]
      V_sell.append(sell_value)
      
      if i == 2028:
        V2_sell = 0
        for vehicle, stock in fleet.items():
          V2_sell += Vehicles[vehicle].get_resale(i) * stock
        total_2028 = sum(C_buy) + sum(C_ins) + sum(C_mnt) + sum(C_fuel) - sum(V_sell) - V2_sell
      
      if i == 2038:
        V2_sell = 0
        for vehicle, stock in fleet.items():
          V2_sell += Vehicles[vehicle].get_resale(i) * stock
        total_2038 = sum(C_buy) + sum(C_ins) + sum(C_mnt) + sum(C_fuel) - sum(V_sell) - V2_sell

      if i == 2038:
        V2_sell = 0
        for vehicle, stock in fleet.items():
          V2_sell += Vehicles[vehicle].get_resale(i) * stock
        total_cost = sum(C_buy) + sum(C_ins) + sum(C_mnt) + sum(C_fuel) - sum(V_sell) - V2_sell
        V_sell[-1] = V2_sell
        #print(100-70*total_cost/65000000)
        #print(100-70*total_cost/172000000)
      
      

    totals = [round(a+b+c+d-e, 2) for a, b, c, d, e in zip(C_buy, C_ins, C_mnt, C_fuel, V_sell)]
    Cc_fuel = [round(x, 2) for x in C_fuel]
    results = pd.DataFrame(list(zip(year_list, C_buy, C_ins, C_mnt, Cc_fuel, V_sell, totals)), columns = ['Year', 'Buy Costs', 'Insurance Costs', 'Maintenance Costs', 'Fuel Costs', 'Selling Profits', 'Total'])
    results.loc["Total"] = results.sum()
    results['Year']['Total'] = 'Total'

    tot = results['Total']['Total']
    kol2.subheader('The total cost of your solution is $' + str(tot))

    tot_2028 = 100-70*total_2028/65000000
    tot_2038 = 100-70*tot/172000000
    st.latex(r'100 - \frac{70*%.2f}{65000000}=%.5f' % (total_2028, tot_2028))
    st.latex(r'100 - \frac{70*%.2f}{172000000}=%.5f' % (tot, tot_2038))

    ffont = font_manager.FontProperties(family='Arial', style='normal', size=18.0, weight='bold', stretch='normal')
    ffontt = font_manager.FontProperties(family='Arial', style='normal', size=22.0, weight='bold', stretch='normal')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(st.session_state['original_carbon_emissions']['Year'].tolist(), (st.session_state['original_carbon_emissions']['Carbon emission CO2/kg']/1000000).tolist(), marker = '*', lw = 2, markersize = 9, label = 'Set by Shell.ai')
    carbon_year = [2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050]
    carbon_emis = [2.404387, 2.140637125, 1.905103375, 1.68424575, 1.4751625, 1.275311875, 1.08251125, 0.89493925, 0.711134125, 0.529993875, 0.3507775, 0.1731035, 0]
    ax.plot(carbon_year, carbon_emis, '--', color = 'k', lw = 2, label = 'Path to Net Zero')
    if not st.session_state['carbon_emissions'].equals(st.session_state['original_carbon_emissions']):
      ax.plot(st.session_state['carbon_emissions']['Year'].tolist(), (st.session_state['carbon_emissions']['Carbon emission CO2/kg']/1000000).tolist(), marker = '*', lw = 2, markersize = 9, label = 'Set by You')
    ax.plot(year_list, CO2_emissions, marker = 'X', lw = 2, markersize = 9, label = 'Your Solution')
    ax.set_ylabel('Carbon emission CO$_2$ (kton)', font = ffont)
    ax.set_xlabel('Year', font = ffont)
    ax.set_title('Annual CO$_2$ Emissions', font = ffontt)
    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['left'].set_linewidth(1.4)
    ax.spines['bottom'].set_linewidth(1.4)
    #ax.set_xticks([2024, 2026, 2028, 2030, 2032, 2034, 2036, 2038, 2040, 2042, 2044, 2046, 2048, 2050])
    ax.tick_params(axis='x', which='major', direction='out', labelsize=15, width=1.5, length=5)
    ax.tick_params(axis='y', which='major', direction='in', labelsize=15, width=1.5, length=5)
    ax.legend(loc = 'center right', fontsize = 15)

    col1, col2, col3 = st.columns([2, 5, 2])
    col2.pyplot(fig, use_container_width = True)

    st.line_chart(results[:16].set_index('Year')/1000000, x_label='Year', y_label='Costs ($1M)', height = 700)

    st.dataframe(results, hide_index = True, use_container_width=True)