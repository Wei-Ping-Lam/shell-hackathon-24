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

def sol_fun():
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
  if 'sell_percent' not in st.session_state:
    st.session_state['sell_percent'] = 20.0
  
  st.session_state['sol_demand'] = st.session_state['demand']
  st.session_state['sol_carbon_emissions'] = st.session_state['carbon_emissions']
  st.session_state['sol_cost_profiles'] = st.session_state['cost_profiles']
  st.session_state['sol_vehicles_fuels'] = st.session_state['vehicles_fuels']
  st.session_state['sol_fuels'] = st.session_state['fuels']
  st.session_state['sol_vehicles'] = st.session_state['vehicles']
  st.session_state['sol_sell_percent'] = st.session_state['sell_percent']
  st.session_state['sol_alpha'] = st.session_state['alpha']
  st.session_state['sol_beta'] = st.session_state['beta']

def main_fun(loops, quicks, alphaa, betaa):
  start_time = time.time()
  with col2.status("Optimizing... (do not leave page)", expanded=True) as status:
    if not quicks:
      progress_text = "Creating Strong Initial Solution..."
    else:
      progress_text = "Creating Initial Solution..."
    my_bar = st.progress(0, text=progress_text)

    sol_fun()

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

    def weiping(y_start, demanz, fleez, grups, grup2, quick, alpha, beta):
      demand = copy.deepcopy(demanz)
      fleet = copy.deepcopy(fleez)
      groups = copy.deepcopy(grups)
      
      if len(grups) == 0:
        group = {"S1 D1": {}, "S1 D2": {}, "S1 D3": {}, "S1 D4": {}, "S2 D1": {}, "S2 D2": {}, "S2 D3": {}, "S2 D4": {}, "S3 D1": {}, "S3 D2": {}, "S3 D3": {}, "S3 D4": {}, "S4 D1": {}, "S4 D2": {}, "S4 D3": {}, "S4 D4": {},}
      else:
        group = copy.deepcopy(grup2)

      maininscost = []
      for i in range(6, 7+beta*10, beta):
        maininscost.append(i)
      sellcost = [90, 80, 70, 60, 50, 40, 30, 30, 30, 30, 30]
      number_of_cars_sold = []
      for y in range(16):
        count = 0
        for i in range(y*16, (y+1)*16):
          size = demand[i].size
          dem = demand[i]._demand
          factor = [102000, 106000, 73000, 118000]
          count += math.ceil(dem/factor[int(size[1])-1])
        number_of_cars_sold.append(math.floor(count*st.session_state['sell_percent']/100))

      for y in range(y_start, 16):
        for i in range(y*16, (y+1)*16):
          year = demand[i].year
          size = demand[i].size
          distance = demand[i].distance
          group_id = size+" "+distance
          dem = demand[i]._demand
          factor = [102000, 106000, 73000, 118000]
          num  = math.ceil(dem/factor[int(size[1])-1])
          dem = dem/num
          num -= sum(group[group_id].values())

          
          potential_vehicles = []
          if int(distance[1]) == 2 and year < 2026:
            potential_vehicles = ['LNG_'+size+'_'+str(year), 'Diesel_'+size+'_'+str(year)]
          elif int(distance[1]) == 3 and year < 2029:
            potential_vehicles = ['LNG_'+size+'_'+str(year), 'Diesel_'+size+'_'+str(year)]
          elif int(distance[1]) == 4 and year < 2032:
            potential_vehicles = ['LNG_'+size+'_'+str(year), 'Diesel_'+size+'_'+str(year)]
          else:
            potential_vehicles = ['BEV_'+size+'_'+str(year), 'LNG_'+size+'_'+str(year), 'Diesel_'+size+'_'+str(year)]
          if num > 0:  
            costs = {}
            for veh in potential_vehicles:
              phys_costs = 0.16*Vehicles[veh].cost

              fuel_consumption = 0
              rate = 0
              if veh[:3] == 'BEV':
                rate = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='Electricity']['Cost ($/unit_fuel)'].tolist()[0]
                fuel_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='Electricity']['Consumption (unit_fuel/km)'].tolist()[0]
              elif veh[:3] == 'LNG':
                rate1 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='LNG']['Cost ($/unit_fuel)'].tolist()[0]
                fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='LNG']['Consumption (unit_fuel/km)'].tolist()[0]
                rate2 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='BioLNG']['Cost ($/unit_fuel)'].tolist()[0]
                fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='BioLNG']['Consumption (unit_fuel/km)'].tolist()[0]
                rates = [rate1, rate2]
                fuel_consumptions = [fuel_consumption1, fuel_consumption2]
                multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                rate = rates[multi.index(min(multi))]
                fuel_consumption = fuel_consumptions[multi.index(min(multi))]
              elif veh[:3] == 'Die':
                rate1 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='B20']['Cost ($/unit_fuel)'].tolist()[0]
                fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='B20']['Consumption (unit_fuel/km)'].tolist()[0]
                rate2 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='HVO']['Cost ($/unit_fuel)'].tolist()[0]
                fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='HVO']['Consumption (unit_fuel/km)'].tolist()[0]
                rates = [rate1, rate2]
                fuel_consumptions = [fuel_consumption1, fuel_consumption2]
                multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                rate = rates[multi.index(min(multi))]
                fuel_consumption = fuel_consumptions[multi.index(min(multi))]

              fuel_costs = dem*rate*fuel_consumption
              costs[phys_costs+fuel_costs] = [veh, num]
            group[group_id][costs[min(costs.keys())][0]] = num
            if costs[min(costs.keys())][0] in fleet:
              fleet[costs[min(costs.keys())][0]] += num
            else:
              fleet[costs[min(costs.keys())][0]] = num

          min_rate = {}
          for veh, quant in group[group_id].items():
            fuel_consumption = 0
            rate = 0
            f = ''
            if veh[:3] == 'BEV':
              rate = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='Electricity']['Cost ($/unit_fuel)'].tolist()[0]
              fuel_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='Electricity']['Consumption (unit_fuel/km)'].tolist()[0]
              f = 'Electricity'
            elif veh[:3] == 'LNG':
              rate1 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='LNG']['Cost ($/unit_fuel)'].tolist()[0]
              fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='LNG']['Consumption (unit_fuel/km)'].tolist()[0]
              rate2 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='BioLNG']['Cost ($/unit_fuel)'].tolist()[0]
              fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='BioLNG']['Consumption (unit_fuel/km)'].tolist()[0]
              rates = [rate1, rate2]
              fuel_consumptions = [fuel_consumption1, fuel_consumption2]
              fs = ['LNG', 'BioLNG']
              multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
              rate = rates[multi.index(min(multi))]
              fuel_consumption = fuel_consumptions[multi.index(min(multi))]
              f = fs[multi.index(min(multi))]
            elif veh[:3] == 'Die':
              rate1 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='B20']['Cost ($/unit_fuel)'].tolist()[0]
              fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='B20']['Consumption (unit_fuel/km)'].tolist()[0]
              rate2 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='HVO']['Cost ($/unit_fuel)'].tolist()[0]
              fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='HVO']['Consumption (unit_fuel/km)'].tolist()[0]
              rates = [rate1, rate2]
              fuel_consumptions = [fuel_consumption1, fuel_consumption2]
              fs = ['B20', 'HVO']
              multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
              rate = rates[multi.index(min(multi))]
              fuel_consumption = fuel_consumptions[multi.index(min(multi))]
              f = fs[multi.index(min(multi))]
            valuee = rate*fuel_consumption+(float(veh[-2:])*0.000000001)
            min_rate[valuee] = [veh, f]
          dem = demand[i]._demand
          while True:
            veh = min_rate[min(min_rate.keys())][0]
            f = min_rate[min(min_rate.keys())][1]
            del min_rate[min(min_rate.keys())]
            if len(min_rate) == 0:
              if group[group_id][veh] > 1 and Vehicles[veh].year == year:
                fuel_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']==f]['Consumption (unit_fuel/km)'].tolist()[0]
                rate = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']==f]['Cost ($/unit_fuel)'].tolist()[0]

                phys_costs = 0.16*Vehicles[veh].cost*(group[group_id][veh])
                fuel_costs = fuel_consumption*rate*dem

                phys_costs_hi = 0.16*Vehicles[veh].cost*(group[group_id][veh]-1)
                dem_hi = (group[group_id][veh]-1)*factor[int(size[1])-1]
                fuel_costs_hi = fuel_consumption*rate*dem_hi
                dem_low = dem - dem_hi
                costs_low = {}
                for veh_low in potential_vehicles:
                  if veh_low != veh:
                    phys_costs_low = 0.16*Vehicles[veh_low].cost

                    rate_low = 0
                    fuel_consumption_low = 0
                    g = ''
                    if veh_low[:3] == 'BEV':
                      rate_low = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='Electricity']['Cost ($/unit_fuel)'].tolist()[0]
                      fuel_consumption_low = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_low].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_low]['Fuel']=='Electricity']['Consumption (unit_fuel/km)'].tolist()[0]
                      g = 'Electricity'
                    elif veh_low[:3] == 'LNG':
                      rate_low1 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='LNG']['Cost ($/unit_fuel)'].tolist()[0]
                      fuel_consumption_low1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_low].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_low]['Fuel']=='LNG']['Consumption (unit_fuel/km)'].tolist()[0]
                      rate_low2 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='BioLNG']['Cost ($/unit_fuel)'].tolist()[0]
                      fuel_consumption_low2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_low].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_low]['Fuel']=='BioLNG']['Consumption (unit_fuel/km)'].tolist()[0]
                      rate_lows = [rate_low1, rate_low2]
                      fuel_consumption_lows = [fuel_consumption_low1, fuel_consumption_low2]
                      gs = ['LNG', 'BioLNG']
                      multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                      rate_low = rate_lows[multi.index(min(multi))]
                      fuel_consumption_low = fuel_consumption_lows[multi.index(min(multi))]
                      g = gs[multi.index(min(multi))]
                    elif veh_low[:3] == 'Die':
                      rate_low1 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='B20']['Cost ($/unit_fuel)'].tolist()[0]
                      fuel_consumption_low1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_low].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_low]['Fuel']=='B20']['Consumption (unit_fuel/km)'].tolist()[0]
                      rate_low2 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='HVO']['Cost ($/unit_fuel)'].tolist()[0]
                      fuel_consumption_low2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_low].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_low]['Fuel']=='HVO']['Consumption (unit_fuel/km)'].tolist()[0]
                      rate_lows = [rate_low1, rate_low2]
                      fuel_consumption_lows = [fuel_consumption_low1, fuel_consumption_low2]
                      gs = ['B20', 'HVO']
                      multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                      rate_low = rate_lows[multi.index(min(multi))]
                      fuel_consumption_low = fuel_consumption_lows[multi.index(min(multi))]
                      g = gs[multi.index(min(multi))]
                    fuel_costs_low = dem_low*rate_low*fuel_consumption_low
                    costs_low[phys_costs_low+fuel_costs_low] = [veh_low, g, dem_low]
                if min(costs_low.keys())+phys_costs_hi+fuel_costs_hi < phys_costs+fuel_costs:
                  demand[i].add_vehicle(veh, group[group_id][veh]-1, f, factor[int(size[1])-1])
                  group[group_id][veh] -= 1
                  fleet[veh] -= 1

                  rep = costs_low[min(costs_low.keys())]
                  demand[i].add_vehicle(rep[0], 1, rep[1], dem_low)

                  group[group_id][rep[0]] = 1
                  if rep[0] in fleet:
                    fleet[rep[0]] += 1
                  else:
                    fleet[rep[0]] = 1

                else: 
                  demand[i].add_vehicle(veh, group[group_id][veh], f, dem/group[group_id][veh]+0.000000001)
                break
              else:
                demand[i].add_vehicle(veh, group[group_id][veh], f, dem/group[group_id][veh]+0.000000001)
                break
            else:
              demand[i].add_vehicle(veh, group[group_id][veh], f,  factor[int(size[1])-1])
              dem -= factor[int(size[1])-1]*group[group_id][veh]

        keep_costs = {}   
        for veh in fleet.keys():
          veh_year = Vehicles[veh].year
          time = y+2023 - veh_year + 1
          ins_mtn_costs = maininscost[time]/100*Vehicles[veh].cost
          sell_costs = (sellcost[time-1]-sellcost[time])/100*Vehicles[veh].cost
          new_veh = veh[:-4]+str(y+2023)
          replace_cost = 0.1*(Vehicles[new_veh].cost-Vehicles[veh].cost)
          coooost = ins_mtn_costs+sell_costs-replace_cost
          keep_costs[coooost] = veh

        sell_num = number_of_cars_sold[y]
        groups.append(copy.deepcopy(group))

        cars_2023 = {}
        cars_2024 = {}
        cars_2025 = {}
        cars_2026 = {}
        cars_2027 = {}
        cars_2028 = {}
        for veh_ID, quant in fleet.items():
          if int(veh_ID[-4:]) == 2023:
            cars_2023[veh_ID] = quant
          if int(veh_ID[-4:]) == 2024:
            cars_2024[veh_ID] = quant
          if int(veh_ID[-4:]) == 2025:
            cars_2025[veh_ID] = quant
          if int(veh_ID[-4:]) == 2026:
            cars_2026[veh_ID] = quant
          if int(veh_ID[-4:]) == 2027:
            cars_2027[veh_ID] = quant
          if int(veh_ID[-4:]) == 2028:
            cars_2028[veh_ID] = quant

        flag23 = True
        flag24 = True
        flag25 = True
        flag26 = True
        flag27 = True
        flag28 = True

        while True:
          if y < 10 and flag23 and sell_num > 0:
            flag23 = False
            if math.ceil(sum(number_of_cars_sold[y+1:10])*alpha) < sum(cars_2023.values()):
              selling = sum(cars_2023.values()) - math.ceil(sum(number_of_cars_sold[y+1:10])*alpha)
              if selling > sell_num:
                continue
              sell_num -= selling
              keep_costs_2023 = {}
              for veh, quant in cars_2023.items():
                veh_year = Vehicles[veh].year
                time = y+2023 - veh_year + 1
                ins_mtn_costs = maininscost[time]/100*Vehicles[veh].cost
                sell_costs = (sellcost[time-1]-sellcost[time])/100*Vehicles[veh].cost
                new_veh = veh[:-4]+str(y+2023)
                replace_cost = 0.1*(Vehicles[new_veh].cost-Vehicles[veh].cost)
                coooost = ins_mtn_costs+sell_costs-replace_cost
                keep_costs_2023[coooost] = veh
              while True:
                highest = keep_costs_2023[max(keep_costs_2023.keys())]
                candids = []
                for grou, fle in group.items():
                  if highest in fle.keys():
                    candids.append(grou)

                if selling > fleet[highest]:
                  selling -= fleet[highest]
                  del fleet[highest]
                  del keep_costs[max(keep_costs_2023.keys())]
                  del keep_costs_2023[max(keep_costs_2023.keys())]
                  for gro in candids:
                    del group[gro][highest]
                  continue
                elif selling == fleet[highest]:
                  del fleet[highest]
                  del keep_costs[max(keep_costs_2023.keys())]
                  for gro in candids:
                    del group[gro][highest]
                  break
                else:
                  fleet[highest] -= selling
                  num_cand = len(candids)
                  disti = []
                  for h in range(len(candids)):
                    disti.append(int(candids[h][-1]))
                  while True:
                    min_dist = min(disti)
                    candid = candids[0][:4] + str(min_dist)
                    if selling > group[candid][highest]:
                      selling -= group[candid][highest]
                      del group[candid][highest]
                      disti.remove(min_dist)
                    elif selling == group[candid][highest]:
                      del group[candid][highest]
                      break
                    else:
                      group[candid][highest] -= selling
                      break
                  break
          if y < 11 and flag24 and sell_num > 0:
            flag24 = False
            if math.ceil(sum(number_of_cars_sold[y+1:11])*alpha) < sum(cars_2024.values()):
              selling = sum(cars_2024.values()) - math.ceil(sum(number_of_cars_sold[y+1:11])*alpha)
              sell_num -= selling
              keep_costs_2024 = {}
              for veh, quant in cars_2024.items():
                veh_year = Vehicles[veh].year
                time = y+2023 - veh_year + 1
                ins_mtn_costs = maininscost[time]/100*Vehicles[veh].cost
                sell_costs = (sellcost[time-1]-sellcost[time])/100*Vehicles[veh].cost
                new_veh = veh[:-4]+str(y+2023)
                replace_cost = 0.1*(Vehicles[new_veh].cost-Vehicles[veh].cost)
                coooost = ins_mtn_costs+sell_costs-replace_cost
                keep_costs_2024[coooost] = veh
              while True:
                highest = keep_costs_2024[max(keep_costs_2024.keys())]
                candids = []
                for grou, fle in group.items():
                  if highest in fle.keys():
                    candids.append(grou)

                if selling > fleet[highest]:
                  selling -= fleet[highest]
                  del fleet[highest]
                  del keep_costs[max(keep_costs_2024.keys())]
                  del keep_costs_2024[max(keep_costs_2024.keys())]
                  for gro in candids:
                    del group[gro][highest]
                  continue
                elif selling == fleet[highest]:
                  del fleet[highest]
                  del keep_costs[max(keep_costs_2024.keys())]
                  for gro in candids:
                    del group[gro][highest]
                  break
                else:
                  fleet[highest] -= selling
                  num_cand = len(candids)
                  disti = []
                  for h in range(len(candids)):
                    disti.append(int(candids[h][-1]))
                  while True:
                    min_dist = min(disti)
                    candid = candids[0][:4] + str(min_dist)
                    if selling > group[candid][highest]:
                      selling -= group[candid][highest]
                      del group[candid][highest]
                      disti.remove(min_dist)
                    elif selling == group[candid][highest]:
                      del group[candid][highest]
                      break
                    else:
                      group[candid][highest] -= selling
                      break
                  break
          if y < 12 and flag25 and sell_num > 0:
            flag25 = False
            if math.ceil(sum(number_of_cars_sold[y+1:12])*alpha) < sum(cars_2025.values()):
              selling = sum(cars_2025.values()) - math.ceil(sum(number_of_cars_sold[y+1:12])*alpha)
              sell_num -= selling
              keep_costs_2025 = {}
              for veh, quant in cars_2025.items():
                veh_year = Vehicles[veh].year
                time = y+2023 - veh_year + 1
                ins_mtn_costs = maininscost[time]/100*Vehicles[veh].cost
                sell_costs = (sellcost[time-1]-sellcost[time])/100*Vehicles[veh].cost
                new_veh = veh[:-4]+str(y+2023)
                replace_cost = 0.1*(Vehicles[new_veh].cost-Vehicles[veh].cost)
                coooost = ins_mtn_costs+sell_costs-replace_cost
                keep_costs_2025[coooost] = veh
              while True:
                highest = keep_costs_2025[max(keep_costs_2025.keys())]
                candids = []
                for grou, fle in group.items():
                  if highest in fle.keys():
                    candids.append(grou)

                if selling > fleet[highest]:
                  selling -= fleet[highest]
                  del fleet[highest]
                  del keep_costs[max(keep_costs_2025.keys())]
                  del keep_costs_2025[max(keep_costs_2025.keys())]
                  for gro in candids:
                    del group[gro][highest]
                  continue
                elif selling == fleet[highest]:
                  del fleet[highest]
                  del keep_costs[max(keep_costs_2025.keys())]
                  for gro in candids:
                    del group[gro][highest]
                  break
                else:
                  fleet[highest] -= selling
                  num_cand = len(candids)
                  disti = []
                  for h in range(len(candids)):
                    disti.append(int(candids[h][-1]))
                  while True:
                    min_dist = min(disti)
                    candid = candids[0][:4] + str(min_dist)
                    if selling > group[candid][highest]:
                      selling -= group[candid][highest]
                      del group[candid][highest]
                      disti.remove(min_dist)
                    elif selling == group[candid][highest]:
                      del group[candid][highest]
                      break
                    else:
                      group[candid][highest] -= selling
                      break
                  break
          if y < 13 and flag26 and sell_num > 0:
            flag26 = False
            if math.ceil(sum(number_of_cars_sold[y+1:13])*alpha) < sum(cars_2026.values()):
              selling = sum(cars_2026.values()) - math.ceil(sum(number_of_cars_sold[y+1:13])*alpha)
              sell_num -= selling
              keep_costs_2026 = {}
              for veh, quant in cars_2026.items():
                veh_year = Vehicles[veh].year
                time = y+2023 - veh_year + 1
                ins_mtn_costs = maininscost[time]/100*Vehicles[veh].cost
                sell_costs = (sellcost[time-1]-sellcost[time])/100*Vehicles[veh].cost
                new_veh = veh[:-4]+str(y+2023)
                replace_cost = 0.1*(Vehicles[new_veh].cost-Vehicles[veh].cost)
                coooost = ins_mtn_costs+sell_costs-replace_cost
                keep_costs_2026[coooost] = veh
              while True:
                highest = keep_costs_2026[max(keep_costs_2026.keys())]
                candids = []
                for grou, fle in group.items():
                  if highest in fle.keys():
                    candids.append(grou)

                if selling > fleet[highest]:
                  selling -= fleet[highest]
                  del fleet[highest]
                  del keep_costs[max(keep_costs_2026.keys())]
                  del keep_costs_2026[max(keep_costs_2026.keys())]
                  for gro in candids:
                    del group[gro][highest]
                  continue
                elif selling == fleet[highest]:
                  del fleet[highest]
                  del keep_costs[max(keep_costs_2026.keys())]
                  for gro in candids:
                    del group[gro][highest]
                  break
                else:
                  fleet[highest] -= selling
                  num_cand = len(candids)
                  disti = []
                  for h in range(len(candids)):
                    disti.append(int(candids[h][-1]))
                  while True:
                    min_dist = min(disti)
                    candid = candids[0][:4] + str(min_dist)
                    if selling > group[candid][highest]:
                      selling -= group[candid][highest]
                      del group[candid][highest]
                      disti.remove(min_dist)
                    elif selling == group[candid][highest]:
                      del group[candid][highest]
                      break
                    else:
                      group[candid][highest] -= selling
                      break
                  break
          if y < 14 and flag27 and sell_num > 0:
            flag27 = False
            if math.ceil(sum(number_of_cars_sold[y+1:14])*alpha) < sum(cars_2027.values()):
              selling = sum(cars_2027.values()) - math.ceil(sum(number_of_cars_sold[y+1:14])*alpha)
              sell_num -= selling
              keep_costs_2027 = {}
              for veh, quant in cars_2027.items():
                veh_year = Vehicles[veh].year
                time = y+2023 - veh_year + 1
                ins_mtn_costs = maininscost[time]/100*Vehicles[veh].cost
                sell_costs = (sellcost[time-1]-sellcost[time])/100*Vehicles[veh].cost
                new_veh = veh[:-4]+str(y+2023)
                replace_cost = 0.1*(Vehicles[new_veh].cost-Vehicles[veh].cost)
                coooost = ins_mtn_costs+sell_costs-replace_cost
                keep_costs_2027[coooost] = veh
              while True:
                highest = keep_costs_2027[max(keep_costs_2027.keys())]
                candids = []
                for grou, fle in group.items():
                  if highest in fle.keys():
                    candids.append(grou)

                if selling > fleet[highest]:
                  selling -= fleet[highest]
                  del fleet[highest]
                  del keep_costs[max(keep_costs_2027.keys())]
                  del keep_costs_2027[max(keep_costs_2027.keys())]
                  for gro in candids:
                    del group[gro][highest]
                  continue
                elif selling == fleet[highest]:
                  del fleet[highest]
                  del keep_costs[max(keep_costs_2027.keys())]
                  for gro in candids:
                    del group[gro][highest]
                  break
                else:
                  fleet[highest] -= selling
                  num_cand = len(candids)
                  disti = []
                  for h in range(len(candids)):
                    disti.append(int(candids[h][-1]))
                  while True:
                    min_dist = min(disti)
                    candid = candids[0][:4] + str(min_dist)
                    if selling > group[candid][highest]:
                      selling -= group[candid][highest]
                      del group[candid][highest]
                      disti.remove(min_dist)
                    elif selling == group[candid][highest]:
                      del group[candid][highest]
                      break
                    else:
                      group[candid][highest] -= selling
                      break
                  break
          if y < 15 and flag28 and sell_num > 0:
            flag28 = False
            if math.ceil(sum(number_of_cars_sold[y+1:15])*alpha) < sum(cars_2028.values()):
              selling = sum(cars_2028.values()) - math.ceil(sum(number_of_cars_sold[y+1:15])*alpha)
              sell_num -= selling

              keep_costs_2028 = {}
              for veh, quant in cars_2028.items():
                veh_year = Vehicles[veh].year
                time = y+2023 - veh_year + 1
                ins_mtn_costs = maininscost[time]/100*Vehicles[veh].cost
                sell_costs = (sellcost[time-1]-sellcost[time])/100*Vehicles[veh].cost
                new_veh = veh[:-4]+str(y+2023)
                replace_cost = 0.1*(Vehicles[new_veh].cost-Vehicles[veh].cost)
                coooost = ins_mtn_costs+sell_costs-replace_cost
                keep_costs_2028[coooost] = veh
              while True:
                highest = keep_costs_2028[max(keep_costs_2028.keys())]
                candids = []
                for grou, fle in group.items():
                  if highest in fle.keys():
                    candids.append(grou)

                if selling > fleet[highest]:
                  selling -= fleet[highest]
                  del fleet[highest]
                  del keep_costs[max(keep_costs_2028.keys())]
                  del keep_costs_2028[max(keep_costs_2028.keys())]
                  for gro in candids:
                    del group[gro][highest]
                  continue
                elif selling == fleet[highest]:
                  del fleet[highest]
                  del keep_costs[max(keep_costs_2028.keys())]
                  for gro in candids:
                    del group[gro][highest]
                  break
                else:
                  fleet[highest] -= selling
                  num_cand = len(candids)
                  disti = []
                  for h in range(len(candids)):
                    disti.append(int(candids[h][-1]))
                  while True:
                    min_dist = min(disti)
                    candid = candids[0][:4] + str(min_dist)
                    if selling > group[candid][highest]:
                      selling -= group[candid][highest]
                      del group[candid][highest]
                      disti.remove(min_dist)
                    elif selling == group[candid][highest]:
                      del group[candid][highest]
                      break
                    else:
                      group[candid][highest] -= selling
                      break
                  break

          if sell_num <= 0:
            break
          highest = keep_costs[max(keep_costs.keys())]
          candids = []
          for grou, fle in group.items():
            if highest in fle.keys():
              candids.append(grou)
          if sell_num > fleet[highest]:
            sell_num -= fleet[highest]
            del fleet[highest]
            del keep_costs[max(keep_costs.keys())]
            for gro in candids:
              del group[gro][highest]
            continue
          elif sell_num == fleet[highest]:
            del fleet[highest]
            for gro in candids:
              del group[gro][highest]
            break
          else:
            fleet[highest] -= sell_num
            num_cand = len(candids)
            disti = []
            for h in range(len(candids)):
              disti.append(int(candids[h][-1]))
            while True:
              min_dist = min(disti)
              candid = candids[0][:4] + str(min_dist)
              if sell_num > group[candid][highest]:
                sell_num -= group[candid][highest]
                del group[candid][highest]
                disti.remove(min_dist)
                continue
              elif sell_num == group[candid][highest]:
                del group[candid][highest]
                break
              else:
                group[candid][highest] -= sell_num
                break
            break


      if quick:
        return demand, fleet, group, groups


      categories = ["S1 D1", "S1 D2", "S1 D3", "S1 D4", "S2 D1", "S2 D2", "S2 D3", "S2 D4", "S4 D1", "S4 D2", "S4 D3", "S4 D4", "S3 D1", "S3 D2", "S3 D3", "S3 D4"]

      new_demand = []
      for i in range(len(demand_df)):
        demand_year = demand_df['Year'][i]
        demand_size = demand_df['Size'][i]
        demand_distance = demand_df['Distance'][i]
        demand_demand = demand_df['Demand (km)'][i]
        new_demand.append(Demand(demand_year, demand_size, demand_distance, demand_demand))

      for c in range(16):
        category = []
        years = 2022
        for gro in groups:
          years += 1
          for veh, quant in gro[categories[c]].items():
            if int(veh[-4:]) == years:
              for _ in range(quant):
                category = category + [{veh:[]}]
        for y in range(16):
          for i in range(y*16, (y+1)*16):
            if i%16 == c:
              vehicles_used = copy.deepcopy(demand[i].met_by)
              for veh, lissst in vehicles_used.items():
                indx = next((category.index(x) for x in category if veh in x.keys()), 'ok')
                for k in range(lissst[0]):
                  category[k+indx][veh].append(lissst[2])
        size = categories[c][:2]
        distance = categories[c][3:]
        new_cat = []
        for V in category:
          for veh, uses in V.items():
            yeard = veh[-4:]
            if int(distance[1]) == 2 and int(yeard) < 2026:
              pot_veh_ids = ['LNG_'+size+'_'+yeard, 'Diesel_'+size+'_'+yeard]
            elif int(distance[1]) == 3 and int(yeard) < 2029:
              pot_veh_ids = ['LNG_'+size+'_'+yeard, 'Diesel_'+size+'_'+yeard]
            elif int(distance[1]) == 4 and int(yeard) < 2032:
              pot_veh_ids = ['LNG_'+size+'_'+yeard, 'Diesel_'+size+'_'+yeard]
            else:
              pot_veh_ids = ['BEV_'+size+'_'+yeard, 'LNG_'+size+'_'+yeard, 'Diesel_'+size+'_'+yeard]

            num_of_years_used = len(uses)
            maininscost = [6, 9, 12, 15, 18, 21, 24, 27, 30, 33]
            sellcost = [90, 80, 70, 60, 50, 40, 30, 30, 30, 30]
            maininsperc = sum(maininscost[:num_of_years_used])/100
            sellperc = sellcost[num_of_years_used-1]/100
            totalcosts = []
            for veh_Id in pot_veh_ids:
              phy_costs = Vehicles[veh_Id].cost*(maininsperc+1-sellperc)
              fuel_costs = 0
              fuel_consumption = 0
              rate = 0
              yearr = int(veh_Id[-4:])
              for km in range(len(uses)):
                range_ = uses[km]
                yearrr = yearr+km
                if veh_Id[:3] == 'BEV':
                  rate = fuels_df.loc[fuels_df['Year'] == yearrr].loc[fuels_df.loc[fuels_df['Year'] == yearrr]['Fuel']=='Electricity']['Cost ($/unit_fuel)'].tolist()[0]
                  fuel_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_Id].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_Id]['Fuel']=='Electricity']['Consumption (unit_fuel/km)'].tolist()[0]
                if veh_Id[:3] == 'LNG':
                  rate1 = fuels_df.loc[fuels_df['Year'] == yearrr].loc[fuels_df.loc[fuels_df['Year'] == yearrr]['Fuel']=='LNG']['Cost ($/unit_fuel)'].tolist()[0]
                  fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_Id].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_Id]['Fuel']=='LNG']['Consumption (unit_fuel/km)'].tolist()[0]
                  rate2 = fuels_df.loc[fuels_df['Year'] == yearrr].loc[fuels_df.loc[fuels_df['Year'] == yearrr]['Fuel']=='BioLNG']['Cost ($/unit_fuel)'].tolist()[0]
                  fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_Id].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_Id]['Fuel']=='BioLNG']['Consumption (unit_fuel/km)'].tolist()[0]
                  rates = [rate1, rate2]
                  fuel_consumptions = [fuel_consumption1, fuel_consumption2]
                  multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                  rate = rates[multi.index(min(multi))]
                  fuel_consumption = fuel_consumptions[multi.index(min(multi))]
                if veh_Id[:3] == 'Die':
                  rate1 = fuels_df.loc[fuels_df['Year'] == yearrr].loc[fuels_df.loc[fuels_df['Year'] == yearrr]['Fuel']=='B20']['Cost ($/unit_fuel)'].tolist()[0]
                  fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_Id].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_Id]['Fuel']=='B20']['Consumption (unit_fuel/km)'].tolist()[0]
                  rate2 = fuels_df.loc[fuels_df['Year'] == yearrr].loc[fuels_df.loc[fuels_df['Year'] == yearrr]['Fuel']=='HVO']['Cost ($/unit_fuel)'].tolist()[0]
                  fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_Id].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh_Id]['Fuel']=='HVO']['Consumption (unit_fuel/km)'].tolist()[0]
                  rates = [rate1, rate2]
                  fuel_consumptions = [fuel_consumption1, fuel_consumption2]
                  multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                  rate = rates[multi.index(min(multi))]
                  fuel_consumption = fuel_consumptions[multi.index(min(multi))]
                  
                fuel_costs += range_*rate*fuel_consumption
              totalcosts.append(fuel_costs+phy_costs)
            idx = totalcosts.index(min(totalcosts))
            if int(yeard)-2023 >= y_start:
              new_cat.append({pot_veh_ids[idx]: uses})
            else:
              new_cat.append({veh: uses})

        for y in range(16):
          for i in range(y*16, (y+1)*16):
            if i%16 == c:
              for V in new_cat:
                for veh, uses in V.items():
                  if y+2023 <= int(veh[-4:])+len(uses)-1 and y+2023 >= int(veh[-4:]):
                    ye = y+2023-int(veh[-4:])
                    if veh in new_demand[i].met_by and uses[ye] == new_demand[i].met_by[veh][2]:
                      new_demand[i].met_by[veh][0] += 1
                    elif veh in new_demand[i].met_by and uses[ye] != new_demand[i].met_by[veh][2]:
                      orig = new_demand[i].met_by[veh][0]
                      distss = orig*new_demand[i].met_by[veh][2] + uses[ye]
                      new_demand[i].met_by[veh][2] = distss/(orig+1)+0.00000000001
                      new_demand[i].met_by[veh][0] += 1
                    else:
                      yaer = y+2023
                      f = 0
                      if veh[:3] == 'BEV':
                        f = 'Electricity'
                      elif veh[:3] == 'LNG':
                        rate1 = fuels_df.loc[fuels_df['Year'] == yaer].loc[fuels_df.loc[fuels_df['Year'] == yaer]['Fuel']=='LNG']['Cost ($/unit_fuel)'].tolist()[0]
                        fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='LNG']['Consumption (unit_fuel/km)'].tolist()[0]
                        rate2 = fuels_df.loc[fuels_df['Year'] == yaer].loc[fuels_df.loc[fuels_df['Year'] == yaer]['Fuel']=='BioLNG']['Cost ($/unit_fuel)'].tolist()[0]
                        fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='BioLNG']['Consumption (unit_fuel/km)'].tolist()[0]
                        multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                        fs = ['LNG', 'BioLNG']
                        f = fs[multi.index(min(multi))]
                      elif veh[:3] == 'Die':
                        rate1 = fuels_df.loc[fuels_df['Year'] == yaer].loc[fuels_df.loc[fuels_df['Year'] == yaer]['Fuel']=='B20']['Cost ($/unit_fuel)'].tolist()[0]
                        fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='B20']['Consumption (unit_fuel/km)'].tolist()[0]
                        rate2 = fuels_df.loc[fuels_df['Year'] == yaer].loc[fuels_df.loc[fuels_df['Year'] == yaer]['Fuel']=='HVO']['Cost ($/unit_fuel)'].tolist()[0]
                        fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='HVO']['Consumption (unit_fuel/km)'].tolist()[0]
                        multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                        fs = ['B20', 'HVO']
                        f = fs[multi.index(min(multi))]
                      new_demand[i].add_vehicle(veh, 1, copy.deepcopy(f), uses[ye])
      
      new_fleet = {}
      new_group2 = {"S1 D1": {}, "S1 D2": {}, "S1 D3": {}, "S1 D4": {}, "S2 D1": {}, "S2 D2": {}, "S2 D3": {}, "S2 D4": {}, "S3 D1": {}, "S3 D2": {}, "S3 D3": {}, "S3 D4": {}, "S4 D1": {}, "S4 D2": {}, "S4 D3": {}, "S4 D4": {}}
      for y in range(y_start+1, y_start+2):
        if y == 16:
          break
        for i in range(y*16, (y+1)*16):
          siz = new_demand[i].size
          distancc = new_demand[i].distance
          grouu = siz + ' ' + distancc
          for veh, liss in new_demand[i].met_by.items():
            if int(veh[-4:]) != y+2023:
              new_group2[grouu][veh] = liss[0]
              if veh in new_fleet:
                new_fleet[veh] += liss[0]
              else:
                new_fleet[veh] = liss[0]
                      
      new_group = {"S1 D1": {}, "S1 D2": {}, "S1 D3": {}, "S1 D4": {}, "S2 D1": {}, "S2 D2": {}, "S2 D3": {}, "S2 D4": {}, "S3 D1": {}, "S3 D2": {}, "S3 D3": {}, "S3 D4": {}, "S4 D1": {}, "S4 D2": {}, "S4 D3": {}, "S4 D4": {}}
      for y in range(y_start, y_start+1):
        for i in range(y*16, (y+1)*16):
          siz = new_demand[i].size
          distancc = new_demand[i].distance
          grouu = siz + ' ' + distancc
          for veh, liss in new_demand[i].met_by.items():
            new_group[grouu][veh] = liss[0]
      
      for y in range(y_start+1, 16):
        for i in range(y*16, (y+1)*16):
          new_demand[i].met_by = {}
      
      return new_demand, new_fleet, new_group, new_group2
    
    demand = []
    for i in range(len(demand_df)):
      demand_year = demand_df['Year'][i]
      demand_size = demand_df['Size'][i]
      demand_distance = demand_df['Distance'][i]
      demand_demand = demand_df['Demand (km)'][i]
      demand.append(Demand(demand_year, demand_size, demand_distance, demand_demand))
    fleet = {}
    groups = []
    group2 = {}

    if not quicks:
      for x in range(16):##############################################################################################################
        #print(x)
        demand, fleet, group, group2 = copy.deepcopy(weiping(x, demand, fleet, groups, group2, False, alphaa, betaa))
        my_bar.progress((x+1)/16, text=progress_text)
        groups.append(group)
        #for i in range(x*16, (x+1)*16):
          #print(demand[i].met_by)
    else:
      demand, fleet, group, groups = copy.deepcopy(weiping(0, demand, fleet, groups, group2, True, alphaa, betaa))
      my_bar.progress(1.0, text=progress_text)

    if loops != 0:
      progress_text_2  = "Optimizing Solution and Reducing Costs..."
      my_bar_2 = st.progress(0, text=progress_text_2)
    

    categories = ["S1 D1", "S1 D2", "S1 D3", "S1 D4", "S2 D1", "S2 D2", "S2 D3", "S2 D4", "S4 D1", "S4 D2", "S4 D3", "S4 D4", "S3 D1", "S3 D2", "S3 D3", "S3 D4"]
    def calc_group_costs10(catt, process):
      categories = ["S1 D1", "S1 D2", "S1 D3", "S1 D4", "S2 D1", "S2 D2", "S2 D3", "S2 D4", "S4 D1", "S4 D2", "S4 D3", "S4 D4", "S3 D1", "S3 D2", "S3 D3", "S3 D4"]
      inx = categories.index(catt)
      factor = [102000, 106000, 73000, 118000]
      factory = factor[int(catt[1])-1]
      total_costs = 0
      for p in range(len(process)):
        year = p+2023
        demand_index = p*16+inx
        amount = demand[demand_index]._demand
        min_rate = {}
        for veh, quant in process[p].items():
          if int(veh[-4:]) == year:
            total_costs += Vehicles[veh].cost*quant # buy costs
          total_costs += Vehicles[veh].get_insurance_cost(year)*quant # insurance costs
          total_costs += Vehicles[veh].get_maintenance_cost(year)*quant # maintenance costs
          
          fuel_consumption = 0
          rate = 0
          f = ''
          if veh[:3] == 'BEV':
            rate = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='Electricity']['Cost ($/unit_fuel)'].tolist()[0]
            fuel_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='Electricity']['Consumption (unit_fuel/km)'].tolist()[0]
            f = 'Electricity'
          elif veh[:3] == 'LNG':
            rate1 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='LNG']['Cost ($/unit_fuel)'].tolist()[0]
            fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='LNG']['Consumption (unit_fuel/km)'].tolist()[0]
            rate2 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='BioLNG']['Cost ($/unit_fuel)'].tolist()[0]
            fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='BioLNG']['Consumption (unit_fuel/km)'].tolist()[0]
            rates = [rate1, rate2]
            fuel_consumptions = [fuel_consumption1, fuel_consumption2]
            fs = ['LNG', 'BioLNG']
            multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
            rate = rates[multi.index(min(multi))]
            fuel_consumption = fuel_consumptions[multi.index(min(multi))]
            f = fs[multi.index(min(multi))]
          elif veh[:3] == 'Die':
            rate1 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='B20']['Cost ($/unit_fuel)'].tolist()[0]
            fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='B20']['Consumption (unit_fuel/km)'].tolist()[0]
            rate2 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='HVO']['Cost ($/unit_fuel)'].tolist()[0]
            fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='HVO']['Consumption (unit_fuel/km)'].tolist()[0]
            rates = [rate1, rate2]
            fuel_consumptions = [fuel_consumption1, fuel_consumption2]
            fs = ['B20', 'HVO']
            multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
            rate = rates[multi.index(min(multi))]
            fuel_consumption = fuel_consumptions[multi.index(min(multi))]
            f = fs[multi.index(min(multi))]
          valuee = rate*fuel_consumption+(float(veh[-2:])*0.000000001)
          min_rate[valuee] = [veh, f, rate*fuel_consumption, quant]
            
        while True:
          veh = min_rate[min(min_rate.keys())][0]
          f = min_rate[min(min_rate.keys())][1]
          rater = min_rate[min(min_rate.keys())][2]
          quan = min_rate[min(min_rate.keys())][3]
          del min_rate[min(min_rate.keys())]
          if len(min_rate) == 0:
            total_costs += rater*amount # fuel costs
            break
          else:
            total_costs += rater*factory*quan # fuel costs
            amount -= factory*quan
            
        if year == 2038:
          for veh, quant in process[p].items():
              total_costs -= Vehicles[veh].get_resale(year)*quant # sell costs
        else:
          for veh, quant in process[p].items():
            if veh in process[p+1].keys():
              if process[p+1][veh] != quant:
                diff = np.abs(process[p+1][veh] - quant)
                total_costs -= Vehicles[veh].get_resale(year)*diff # sell costs
            else:
              total_costs -= Vehicles[veh].get_resale(year)*quant # sell costs
      return total_costs


    def selling_gain(catt, y_ear): #get an extra sell in a year, what is max decrease in costs
      categories = ["S1 D1", "S1 D2", "S1 D3", "S1 D4", "S2 D1", "S2 D2", "S2 D3", "S2 D4", "S4 D1", "S4 D2", "S4 D3", "S4 D4", "S3 D1", "S3 D2", "S3 D3", "S3 D4"]
      c = categories.index(catt)
      factor = [102000, 106000, 73000, 118000]
      factory = factor[int(catt[1])-1]
      process = []
      for g in range(len(groups)):
        process.append(groups[g][catt])
      orig_cos = calc_group_costs10(catt, process)
      category = []
      years = 2022
      for gro in groups:
        years += 1
        for veh, quant in gro[categories[c]].items():
          if int(veh[-4:]) == years:
            for _ in range(quant):
              category = category + [{veh:[]}]
      for y in range(16):
        for i in range(y*16, (y+1)*16):
          if i%16 == c:
            vehicles_used = copy.deepcopy(demand[i].met_by)
            for veh, lissst in vehicles_used.items():
              indx = next((category.index(x) for x in category if veh in x.keys()), 'ok')
              for k in range(lissst[0]):
                category[k+indx][veh].append(lissst[2])
      size = categories[c][:2]
      distance = categories[c][3:]
      max_dec = {}
      for V in category:
        for veh, uses in V.items():
          if int(veh[-4:]) <= y_ear and int(veh[-4:])+len(uses)-1 > y_ear:
            cat_to_comp = copy.deepcopy(category)
            cat_to_comp.remove(V)
            split_ind = y_ear - int(veh[-4:]) + 1
            year1 = int(veh[-4:])
            uses1 = uses[:split_ind]
            year2 = year1+split_ind
            uses2 = uses[split_ind:]
            por_phys = [0.16, 0.35, 0.57, 0.82, 1.1, 1.41, 1.75, 2.02, 2.32, 2.65]
            
            potential_vehicles1 = []
            if int(distance[1]) == 2 and year1 < 2026:
              potential_vehicles1 = ['LNG_'+size+'_'+str(year1), 'Diesel_'+size+'_'+str(year1)]
            elif int(distance[1]) == 3 and year1 < 2029:
              potential_vehicles1 = ['LNG_'+size+'_'+str(year1), 'Diesel_'+size+'_'+str(year1)]
            elif int(distance[1]) == 4 and year1 < 2032:
              potential_vehicles1 = ['LNG_'+size+'_'+str(year1), 'Diesel_'+size+'_'+str(year1)]
            else:
              potential_vehicles1 = ['BEV_'+size+'_'+str(year1), 'LNG_'+size+'_'+str(year1), 'Diesel_'+size+'_'+str(year1)]
            costs1 = {}
            for veh in potential_vehicles1:
              phys_costs = por_phys[len(uses1)-1]*Vehicles[veh].cost
              fuel_consumption = 0
              
              fuel_costs = 0
              for u in range(len(uses1)):
                yearu = year1+u
                rate = 0
                f = ''
                if veh[:3] == 'BEV':
                  rate = fuels_df.loc[fuels_df['Year'] == yearu].loc[fuels_df.loc[fuels_df['Year'] == yearu]['Fuel']=='Electricity']['Cost ($/unit_fuel)'].tolist()[0]
                  fuel_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='Electricity']['Consumption (unit_fuel/km)'].tolist()[0]
                  f = 'Electricity'
                elif veh[:3] == 'LNG':
                  rate1 = fuels_df.loc[fuels_df['Year'] == yearu].loc[fuels_df.loc[fuels_df['Year'] == yearu]['Fuel']=='LNG']['Cost ($/unit_fuel)'].tolist()[0]
                  fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='LNG']['Consumption (unit_fuel/km)'].tolist()[0]
                  rate2 = fuels_df.loc[fuels_df['Year'] == yearu].loc[fuels_df.loc[fuels_df['Year'] == yearu]['Fuel']=='BioLNG']['Cost ($/unit_fuel)'].tolist()[0]
                  fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='BioLNG']['Consumption (unit_fuel/km)'].tolist()[0]
                  rates = [rate1, rate2]
                  fuel_consumptions = [fuel_consumption1, fuel_consumption2]
                  fs = ['LNG', 'BioLNG']
                  multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                  rate = rates[multi.index(min(multi))]
                  fuel_consumption = fuel_consumptions[multi.index(min(multi))]
                  f = fs[multi.index(min(multi))]
                elif veh[:3] == 'Die':
                  rate1 = fuels_df.loc[fuels_df['Year'] == yearu].loc[fuels_df.loc[fuels_df['Year'] == yearu]['Fuel']=='B20']['Cost ($/unit_fuel)'].tolist()[0]
                  fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='B20']['Consumption (unit_fuel/km)'].tolist()[0]
                  rate2 = fuels_df.loc[fuels_df['Year'] == yearu].loc[fuels_df.loc[fuels_df['Year'] == yearu]['Fuel']=='HVO']['Cost ($/unit_fuel)'].tolist()[0]
                  fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='HVO']['Consumption (unit_fuel/km)'].tolist()[0]
                  rates = [rate1, rate2]
                  fuel_consumptions = [fuel_consumption1, fuel_consumption2]
                  fs = ['B20', 'HVO']
                  multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                  rate = rates[multi.index(min(multi))]
                  fuel_consumption = fuel_consumptions[multi.index(min(multi))]
                  f = fs[multi.index(min(multi))]
                fuel_costs += rate*fuel_consumption*uses1[u]
              costs1[phys_costs+fuel_costs] = veh
            cat_to_comp.append({costs1[min(costs1.keys())]:uses1})
            
            potential_vehicles2 = []
            if int(distance[1]) == 2 and year2 < 2026:
              potential_vehicles2 = ['LNG_'+size+'_'+str(year2), 'Diesel_'+size+'_'+str(year2)]
            elif int(distance[1]) == 3 and year2 < 2029:
              potential_vehicles2 = ['LNG_'+size+'_'+str(year2), 'Diesel_'+size+'_'+str(year2)]
            elif int(distance[1]) == 4 and year2 < 2032:
              potential_vehicles2 = ['LNG_'+size+'_'+str(year2), 'Diesel_'+size+'_'+str(year2)]
            else:
              potential_vehicles2 = ['BEV_'+size+'_'+str(year2), 'LNG_'+size+'_'+str(year2), 'Diesel_'+size+'_'+str(year2)]
            costs2 = {}
            for veh in potential_vehicles2:
              phys_costs = por_phys[len(uses2)-1]*Vehicles[veh].cost
              fuel_consumption = 0
              fuel_costs = 0
              for u in range(len(uses2)):
                yearu = year2+u
                rate = 0
                f = ''
                if veh[:3] == 'BEV':
                  rate = fuels_df.loc[fuels_df['Year'] == yearu].loc[fuels_df.loc[fuels_df['Year'] == yearu]['Fuel']=='Electricity']['Cost ($/unit_fuel)'].tolist()[0]
                  fuel_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='Electricity']['Consumption (unit_fuel/km)'].tolist()[0]
                  f = 'Electricity'
                elif veh[:3] == 'LNG':
                  rate1 = fuels_df.loc[fuels_df['Year'] == yearu].loc[fuels_df.loc[fuels_df['Year'] == yearu]['Fuel']=='LNG']['Cost ($/unit_fuel)'].tolist()[0]
                  fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='LNG']['Consumption (unit_fuel/km)'].tolist()[0]
                  rate2 = fuels_df.loc[fuels_df['Year'] == yearu].loc[fuels_df.loc[fuels_df['Year'] == yearu]['Fuel']=='BioLNG']['Cost ($/unit_fuel)'].tolist()[0]
                  fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='BioLNG']['Consumption (unit_fuel/km)'].tolist()[0]
                  rates = [rate1, rate2]
                  fuel_consumptions = [fuel_consumption1, fuel_consumption2]
                  fs = ['LNG', 'BioLNG']
                  multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                  rate = rates[multi.index(min(multi))]
                  fuel_consumption = fuel_consumptions[multi.index(min(multi))]
                  f = fs[multi.index(min(multi))]
                elif veh[:3] == 'Die':
                  rate1 = fuels_df.loc[fuels_df['Year'] == yearu].loc[fuels_df.loc[fuels_df['Year'] == yearu]['Fuel']=='B20']['Cost ($/unit_fuel)'].tolist()[0]
                  fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='B20']['Consumption (unit_fuel/km)'].tolist()[0]
                  rate2 = fuels_df.loc[fuels_df['Year'] == yearu].loc[fuels_df.loc[fuels_df['Year'] == yearu]['Fuel']=='HVO']['Cost ($/unit_fuel)'].tolist()[0]
                  fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='HVO']['Consumption (unit_fuel/km)'].tolist()[0]
                  rates = [rate1, rate2]
                  fuel_consumptions = [fuel_consumption1, fuel_consumption2]
                  fs = ['B20', 'HVO']
                  multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                  rate = rates[multi.index(min(multi))]
                  fuel_consumption = fuel_consumptions[multi.index(min(multi))]
                  f = fs[multi.index(min(multi))]
                fuel_costs += rate*fuel_consumption*uses2[u]
              costs2[phys_costs+fuel_costs] = veh
            cat_to_comp.append({costs2[min(costs2.keys())]:uses2})
            compressed = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
            for y in range(16):
              for B in cat_to_comp:
                for beh, bses in B.items():
                  yeart = int(beh[-4:])-2023
                  if yeart <= y and yeart+len(bses)-1 >= y:
                    if beh in compressed[y].keys():
                      compressed[y][beh] += 1
                    else:
                      compressed[y][beh] = 1
            max_dec[orig_cos-calc_group_costs10(catt, compressed)] = compressed
      if len(max_dec) == 0:
        return 0, None
      #print(max_dec[max(max_dec.keys())])
      return max(max_dec.keys()), max_dec[max(max_dec.keys())]

    def selling_loss(catt, y_ear): #have to get rid of a sell, what is min increase in costs
      categories = ["S1 D1", "S1 D2", "S1 D3", "S1 D4", "S2 D1", "S2 D2", "S2 D3", "S2 D4", "S4 D1", "S4 D2", "S4 D3", "S4 D4", "S3 D1", "S3 D2", "S3 D3", "S3 D4"]
      c = categories.index(catt)
      factor = [102000, 106000, 73000, 118000]
      factory = factor[int(catt[1])-1]
      process = []
      for g in range(len(groups)):
        process.append(groups[g][catt])
      orig_cos = calc_group_costs10(catt, process)
      category = []
      years = 2022
      for gro in groups:
        years += 1
        for veh, quant in gro[categories[c]].items():
          if int(veh[-4:]) == years:
            for _ in range(quant):
              category = category + [{veh:[]}]
      for y in range(16):
        for i in range(y*16, (y+1)*16):
          if i%16 == c:
            vehicles_used = copy.deepcopy(demand[i].met_by)
            for veh, lissst in vehicles_used.items():
              indx = next((category.index(x) for x in category if veh in x.keys()), 'ok')
              for k in range(lissst[0]):
                category[k+indx][veh].append(lissst[2])
      size = categories[c][:2]
      distance = categories[c][3:]
      min_inc = {}
      for V in category:
        for veh, uses in V.items():
          if int(veh[-4:])+len(uses)-1 == y_ear:
            for B in category:
              for beh, bses in B.items():
                if int(beh[-4:]) == y_ear+1 and len(bses) <= 10-len(uses):
                  cat_to_comp = copy.deepcopy(category)
                  cat_to_comp.remove(V)
                  cat_to_comp.remove(B)
                  year1 = int(veh[-4:])
                  uses1 = uses + bses
                  por_phys = [0.16, 0.35, 0.57, 0.82, 1.1, 1.41, 1.75, 2.02, 2.32, 2.65]
                  potential_vehicles1 = []
                  if int(distance[1]) == 2 and year1 < 2026:
                    potential_vehicles1 = ['LNG_'+size+'_'+str(year1), 'Diesel_'+size+'_'+str(year1)]
                  elif int(distance[1]) == 3 and year1 < 2029:
                    potential_vehicles1 = ['LNG_'+size+'_'+str(year1), 'Diesel_'+size+'_'+str(year1)]
                  elif int(distance[1]) == 4 and year1 < 2032:
                    potential_vehicles1 = ['LNG_'+size+'_'+str(year1), 'Diesel_'+size+'_'+str(year1)]
                  else:
                    potential_vehicles1 = ['BEV_'+size+'_'+str(year1), 'LNG_'+size+'_'+str(year1), 'Diesel_'+size+'_'+str(year1)]
                  costs1 = {}
                  for veh in potential_vehicles1:
                    phys_costs = por_phys[len(uses1)-1]*Vehicles[veh].cost
                    fuel_costs = 0
                    for u in range(len(uses1)):
                      yearu = year1+u
                      fuel_consumption = 0
                      rate = 0
                      f = ''
                      if veh[:3] == 'BEV':
                        rate = fuels_df.loc[fuels_df['Year'] == yearu].loc[fuels_df.loc[fuels_df['Year'] == yearu]['Fuel']=='Electricity']['Cost ($/unit_fuel)'].tolist()[0]
                        fuel_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='Electricity']['Consumption (unit_fuel/km)'].tolist()[0]
                        f = 'Electricity'
                      elif veh[:3] == 'LNG':
                        rate1 = fuels_df.loc[fuels_df['Year'] == yearu].loc[fuels_df.loc[fuels_df['Year'] == yearu]['Fuel']=='LNG']['Cost ($/unit_fuel)'].tolist()[0]
                        fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='LNG']['Consumption (unit_fuel/km)'].tolist()[0]
                        rate2 = fuels_df.loc[fuels_df['Year'] == yearu].loc[fuels_df.loc[fuels_df['Year'] == yearu]['Fuel']=='BioLNG']['Cost ($/unit_fuel)'].tolist()[0]
                        fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='BioLNG']['Consumption (unit_fuel/km)'].tolist()[0]
                        rates = [rate1, rate2]
                        fuel_consumptions = [fuel_consumption1, fuel_consumption2]
                        fs = ['LNG', 'BioLNG']
                        multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                        rate = rates[multi.index(min(multi))]
                        fuel_consumption = fuel_consumptions[multi.index(min(multi))]
                        f = fs[multi.index(min(multi))]
                      elif veh[:3] == 'Die':
                        rate1 = fuels_df.loc[fuels_df['Year'] == yearu].loc[fuels_df.loc[fuels_df['Year'] == yearu]['Fuel']=='B20']['Cost ($/unit_fuel)'].tolist()[0]
                        fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='B20']['Consumption (unit_fuel/km)'].tolist()[0]
                        rate2 = fuels_df.loc[fuels_df['Year'] == yearu].loc[fuels_df.loc[fuels_df['Year'] == yearu]['Fuel']=='HVO']['Cost ($/unit_fuel)'].tolist()[0]
                        fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='HVO']['Consumption (unit_fuel/km)'].tolist()[0]
                        rates = [rate1, rate2]
                        fuel_consumptions = [fuel_consumption1, fuel_consumption2]
                        fs = ['B20', 'HVO']
                        multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                        rate = rates[multi.index(min(multi))]
                        fuel_consumption = fuel_consumptions[multi.index(min(multi))]
                        f = fs[multi.index(min(multi))]
                      fuel_costs += rate*fuel_consumption*uses1[u]
                    costs1[phys_costs+fuel_costs] = veh
                  cat_to_comp.append({costs1[min(costs1.keys())]:uses1})
                  compressed = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
                  for y in range(16):
                    for P in cat_to_comp:
                      for peh, pses in P.items():
                        yeart = int(peh[-4:])-2023
                        if yeart <= y and yeart+len(pses)-1 >= y:
                          if peh in compressed[y].keys():
                            compressed[y][peh] += 1
                          else:
                            compressed[y][peh] = 1
                  min_inc[calc_group_costs10(catt, compressed)-orig_cos] = compressed
      if len(min_inc) == 0:
        return 999999999999999, None
      return min(min_inc.keys()), min_inc[min(min_inc.keys())]

    categories = ["S1 D1", "S1 D2", "S1 D3", "S1 D4", "S2 D1", "S2 D2", "S2 D3", "S2 D4", "S4 D1", "S4 D2", "S4 D3", "S4 D4", "S3 D1", "S3 D2", "S3 D3", "S3 D4"]
    
    for mm in range(loops):
      for y in range(0, 15):
        yeary = y+2023
        increases = {}
        decreases = {}
        for c in range(len(categories)):
          amount, compr = selling_loss(categories[c], yeary)
          if amount != 999999999999999:
            increases[amount+random.random()*0.000000001] = [copy.deepcopy(compr), c]
          amount, compr = selling_gain(categories[c], yeary)
          if amount != 0:
            decreases[amount+random.random()*0.000000001] = [copy.deepcopy(compr), c]
        #print(y)
        while True:
          flag = False
          temp_key = None
          temp_val = None
          if max(decreases.keys()) > min(increases.keys()) and decreases[max(decreases.keys())][1] == increases[min(increases.keys())][1]:
            flag = True
            #temp_key = copy.deepcopy(min(increases.keys()))
            #temp_val = copy.deepcopy(increases[min(increases.keys())])
            del increases[min(increases.keys())]
          if max(decreases.keys()) > min(increases.keys()):
            for m in range(16):
              year = m+2023
              groups[m][categories[decreases[max(decreases.keys())][1]]] = decreases[max(decreases.keys())][0][m]
              groups[m][categories[increases[min(increases.keys())][1]]] = increases[min(increases.keys())][0][m]
              for j in range(m*16, (m+1)*16):
                if j%16 == decreases[max(decreases.keys())][1]:
                  size = categories[decreases[max(decreases.keys())][1]][:2]
                  distance = categories[decreases[max(decreases.keys())][1]][3:]
                  factor = [102000, 106000, 73000, 118000]
                  factory = factor[int(size[1])-1]
                  min_rate = {}
                  for veh, quant in decreases[max(decreases.keys())][0][m].items():
                    fuel_consumption = 0
                    rate = 0
                    f = ''
                    if veh[:3] == 'BEV':
                      rate = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='Electricity']['Cost ($/unit_fuel)'].tolist()[0]
                      fuel_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='Electricity']['Consumption (unit_fuel/km)'].tolist()[0]
                      f = 'Electricity'
                    elif veh[:3] == 'LNG':
                      rate1 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='LNG']['Cost ($/unit_fuel)'].tolist()[0]
                      fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='LNG']['Consumption (unit_fuel/km)'].tolist()[0]
                      rate2 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='BioLNG']['Cost ($/unit_fuel)'].tolist()[0]
                      fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='BioLNG']['Consumption (unit_fuel/km)'].tolist()[0]
                      rates = [rate1, rate2]
                      fuel_consumptions = [fuel_consumption1, fuel_consumption2]
                      fs = ['LNG', 'BioLNG']
                      multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                      rate = rates[multi.index(min(multi))]
                      fuel_consumption = fuel_consumptions[multi.index(min(multi))]
                      f = fs[multi.index(min(multi))]
                    elif veh[:3] == 'Die':
                      rate1 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='B20']['Cost ($/unit_fuel)'].tolist()[0]
                      fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='B20']['Consumption (unit_fuel/km)'].tolist()[0]
                      rate2 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='HVO']['Cost ($/unit_fuel)'].tolist()[0]
                      fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='HVO']['Consumption (unit_fuel/km)'].tolist()[0]
                      rates = [rate1, rate2]
                      fuel_consumptions = [fuel_consumption1, fuel_consumption2]
                      fs = ['B20', 'HVO']
                      multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                      rate = rates[multi.index(min(multi))]
                      fuel_consumption = fuel_consumptions[multi.index(min(multi))]
                      f = fs[multi.index(min(multi))]
                    valuee = rate*fuel_consumption+(float(veh[-2:])*0.000000001)
                    min_rate[valuee] = [veh, f, quant]
                  dem = demand[j]._demand
                  demand[j].met_by = {}
                  while True:
                    veh = min_rate[min(min_rate.keys())][0]
                    f = min_rate[min(min_rate.keys())][1]
                    quan = min_rate[min(min_rate.keys())][2]
                    del min_rate[min(min_rate.keys())]
                    if len(min_rate) == 0:
                      demand[j].add_vehicle(veh, quan, f, dem/quan+0.000000001)
                      break
                    else:
                      demand[j].add_vehicle(veh, quan, f, factory)
                      dem -= factory*quan
                if j%16 == increases[min(increases.keys())][1]:
                  size = categories[increases[min(increases.keys())][1]][:2]
                  distance = categories[increases[min(increases.keys())][1]][3:]
                  factor = [102000, 106000, 73000, 118000]
                  factory = factor[int(size[1])-1]
                  min_rate = {}
                  for veh, quant in increases[min(increases.keys())][0][m].items():
                    fuel_consumption = 0
                    rate = 0
                    f = ''
                    if veh[:3] == 'BEV':
                      rate = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='Electricity']['Cost ($/unit_fuel)'].tolist()[0]
                      fuel_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='Electricity']['Consumption (unit_fuel/km)'].tolist()[0]
                      f = 'Electricity'
                    elif veh[:3] == 'LNG':
                      rate1 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='LNG']['Cost ($/unit_fuel)'].tolist()[0]
                      fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='LNG']['Consumption (unit_fuel/km)'].tolist()[0]
                      rate2 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='BioLNG']['Cost ($/unit_fuel)'].tolist()[0]
                      fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='BioLNG']['Consumption (unit_fuel/km)'].tolist()[0]
                      rates = [rate1, rate2]
                      fuel_consumptions = [fuel_consumption1, fuel_consumption2]
                      fs = ['LNG', 'BioLNG']
                      multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                      rate = rates[multi.index(min(multi))]
                      fuel_consumption = fuel_consumptions[multi.index(min(multi))]
                      f = fs[multi.index(min(multi))]
                    elif veh[:3] == 'Die':
                      rate1 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='B20']['Cost ($/unit_fuel)'].tolist()[0]
                      fuel_consumption1 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='B20']['Consumption (unit_fuel/km)'].tolist()[0]
                      rate2 = fuels_df.loc[fuels_df['Year'] == year].loc[fuels_df.loc[fuels_df['Year'] == year]['Fuel']=='HVO']['Cost ($/unit_fuel)'].tolist()[0]
                      fuel_consumption2 = vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID']==veh]['Fuel']=='HVO']['Consumption (unit_fuel/km)'].tolist()[0]
                      rates = [rate1, rate2]
                      fuel_consumptions = [fuel_consumption1, fuel_consumption2]
                      fs = ['B20', 'HVO']
                      multi = [rate1*fuel_consumption1, rate2*fuel_consumption2]
                      rate = rates[multi.index(min(multi))]
                      fuel_consumption = fuel_consumptions[multi.index(min(multi))]
                      f = fs[multi.index(min(multi))]
                    valuee = rate*fuel_consumption+(float(veh[-2:])*0.000000001)
                    min_rate[valuee] = [veh, f, quant]
                  dem = demand[j]._demand
                  demand[j].met_by = {}
                  while True:
                    veh = min_rate[min(min_rate.keys())][0]
                    f = min_rate[min(min_rate.keys())][1]
                    quan = min_rate[min(min_rate.keys())][2]
                    del min_rate[min(min_rate.keys())]
                    if len(min_rate) == 0:
                      demand[j].add_vehicle(veh, quan, f, dem/quan+0.000000001)
                      break
                    else:
                      demand[j].add_vehicle(veh, quan, f, factory)
                      dem -= factory*quan
            cd = decreases[max(decreases.keys())][1]
            ci = increases[min(increases.keys())][1]
            #print(categories[ci], categories[cd], min(increases.keys()) - max(decreases.keys()))
            del decreases[max(decreases.keys())]
            del increases[min(increases.keys())]
            amount, compr = selling_loss(categories[ci], yeary)
            if amount != 999999999999999:
              increases[amount+random.random()*0.000000001] = [copy.deepcopy(compr), ci]
            
            holder = -12341234
            for amound, freak in decreases.items():
              if freak[1] == ci:
                holder = amound
            if holder != -12341234:
              del decreases[holder]
            amount, compr = selling_gain(categories[ci], yeary)
            if amount != 0:
              decreases[amount+random.random()*0.000000001] = [copy.deepcopy(compr), ci]
            
            amount, compr = selling_gain(categories[cd], yeary)
            if amount != 0:
              decreases[amount+random.random()*0.000000001] = [copy.deepcopy(compr), cd]
            
            holder = -123412345
            for amound, freak in increases.items():
              if freak[1] == cd:
                holder = amound
            if holder != -123412345:
              del increases[holder]
            amount, compr = selling_loss(categories[cd], yeary)
            if amount != 999999999999999:
              increases[amount+random.random()*0.000000001] = [copy.deepcopy(compr), cd]
            #if flag:
            #  increases[temp_key] = temp_val
              
          else:
            break
        my_bar_2.progress((mm*16+y+1)/(loops*16-1), text=progress_text_2)

    progress_text_3 = "Reducing Carbon Emissions..."
    my_bar_3 = st.progress(0, text=progress_text_3)

    new_demand = copy.deepcopy(demand)
    solution = pd.DataFrame(columns = ['Year', 'ID', 'Num_Vehicles', 'Type', 'Fuel', 'Distance_bucket', 'Distance_per_vehicle(km)'])
    sleet = {}
    for i in range(len(new_demand)):
      year = math.floor(i/16)+2023
      for veh, lit in new_demand[i].met_by.items():
        if int(veh[-4:]) == year:
          if veh in sleet:
            sleet[veh] += lit[0]
            solution.loc[solution['ID'] == veh, ['Num_Vehicles']] += lit[0]
          else:
            sleet[veh] = lit[0]
            solution.loc[len(solution.index)] = [year, veh, lit[0], 'Buy', np.nan, np.nan, '0']

    sheet = {}
    jay = []
    for y in range(16):
      cou = 0
      sleet_prior = copy.deepcopy(sheet)
      sheet = {}
      for i in range(y*16, (y+1)*16):
        for veh, lit in new_demand[i].met_by.items():
          if veh in sheet:
            sheet[veh] += lit[0]
          else:
            sheet[veh] = lit[0]
      if y >= 1:
        for veh_ID, quant in sleet_prior.items():
          if veh_ID in sheet.keys():
            diff = sleet_prior[veh_ID] - sheet[veh_ID]
            if diff > 0:
              solution.loc[len(solution.index)] = [y+2023-1, veh_ID, diff, 'Sell', np.nan, np.nan, '0']
              cou += diff
          else:
            solution.loc[len(solution.index)] = [y+2023-1, veh_ID, sleet_prior[veh_ID], 'Sell', np.nan, np.nan, '0']
            cou += sleet_prior[veh_ID]
      jay.append(cou)
        
        
    for i in range(len(new_demand)):
      year = new_demand[i].year
      for veh, lits in new_demand[i].met_by.items():
        solution.loc[len(solution.index)] = [year, veh, lits[0], 'Use', lits[1], new_demand[i].distance, lits[2]]

    sub = solution
    sub_use = sub.loc[sub['Type'] == 'Use']
    progr = 0
    for i in range(2023, 2039):
      ids = sub_use.loc[sub_use['Year'] == i]['ID'].tolist()
      nums = sub_use.loc[sub_use['Year'] == i]['Num_Vehicles'].tolist()
      fuels = sub_use.loc[sub_use['Year'] == i]['Fuel'].tolist()
      buckets = sub_use.loc[sub_use['Year'] == i]['Distance_bucket'].tolist()
      kms = sub_use.loc[sub_use['Year'] == i]['Distance_per_vehicle(km)'].tolist()
      emit = 0
      for j in range(len(ids)):
        emit += Vehicles[ids[j]].get_emissions(i, fuels[j], kms[j]) * nums[j]
      
      if emit <= carbon_emissions_df["Carbon emission CO2/kg"].tolist()[i-2023]:
        progr += 1
        my_bar_3.progress(progr/16, text=progress_text_3)
        continue
      else:
        reduction = emit - carbon_emissions_df["Carbon emission CO2/kg"].tolist()[i-2023]
        dollar_per_CO2 = {}
        for j in range(len(ids)):
          if ids[j][:3] == 'LNG' and fuels[j] == 'LNG':
            LNG_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == ids[j]].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == ids[j]]['Fuel'] == 'LNG']['Consumption (unit_fuel/km)'].tolist()[0]
            BioLNG_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == ids[j]].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == ids[j]]['Fuel'] == 'BioLNG']['Consumption (unit_fuel/km)'].tolist()[0]
            LNG_emission = fuels_df.loc[fuels_df['Year'] == i].loc[fuels_df.loc[fuels_df['Year'] == i]['Fuel'] == 'LNG']['Emissions (CO2/unit_fuel)'].tolist()[0]
            BioLNG_emission = fuels_df.loc[fuels_df['Year'] == i].loc[fuels_df.loc[fuels_df['Year'] == i]['Fuel'] == 'BioLNG']['Emissions (CO2/unit_fuel)'].tolist()[0]
            LNG_cost = fuels_df.loc[fuels_df['Year'] == i].loc[fuels_df.loc[fuels_df['Year'] == i]['Fuel'] == 'LNG']['Cost ($/unit_fuel)'].tolist()[0]
            BioLNG_cost = fuels_df.loc[fuels_df['Year'] == i].loc[fuels_df.loc[fuels_df['Year'] == i]['Fuel'] == 'BioLNG']['Cost ($/unit_fuel)'].tolist()[0]
            
            redpv = kms[j]*(LNG_consumption*LNG_emission - BioLNG_consumption*BioLNG_emission)
            redpkm = LNG_consumption*LNG_emission - BioLNG_consumption*BioLNG_emission
            if redpkm <= 0:
              continue
            costpkm = BioLNG_consumption*BioLNG_cost - LNG_consumption*LNG_cost
            rateio = (BioLNG_consumption*BioLNG_cost - LNG_consumption*LNG_cost)/(LNG_consumption*LNG_emission - BioLNG_consumption*BioLNG_emission)
            dollar_per_CO2[rateio + random.random()*0.00000001] = [ids[j], nums[j], 'BioLNG', buckets[j], kms[j], redpv, redpkm, costpkm]
          if ids[j][:3] == 'LNG' and fuels[j] == 'BioLNG':
            LNG_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == ids[j]].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == ids[j]]['Fuel'] == 'LNG']['Consumption (unit_fuel/km)'].tolist()[0]
            BioLNG_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == ids[j]].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == ids[j]]['Fuel'] == 'BioLNG']['Consumption (unit_fuel/km)'].tolist()[0]
            LNG_emission = fuels_df.loc[fuels_df['Year'] == i].loc[fuels_df.loc[fuels_df['Year'] == i]['Fuel'] == 'LNG']['Emissions (CO2/unit_fuel)'].tolist()[0]
            BioLNG_emission = fuels_df.loc[fuels_df['Year'] == i].loc[fuels_df.loc[fuels_df['Year'] == i]['Fuel'] == 'BioLNG']['Emissions (CO2/unit_fuel)'].tolist()[0]
            LNG_cost = fuels_df.loc[fuels_df['Year'] == i].loc[fuels_df.loc[fuels_df['Year'] == i]['Fuel'] == 'LNG']['Cost ($/unit_fuel)'].tolist()[0]
            BioLNG_cost = fuels_df.loc[fuels_df['Year'] == i].loc[fuels_df.loc[fuels_df['Year'] == i]['Fuel'] == 'BioLNG']['Cost ($/unit_fuel)'].tolist()[0]
            
            redpv = kms[j]*(BioLNG_consumption*BioLNG_emission - LNG_consumption*LNG_emission)
            redpkm = BioLNG_consumption*BioLNG_emission - LNG_consumption*LNG_emission
            if redpkm <= 0:
              continue
            costpkm = LNG_consumption*LNG_cost - BioLNG_consumption*BioLNG_cost
            rateio = (LNG_consumption*LNG_cost - BioLNG_consumption*BioLNG_cost)/(BioLNG_consumption*BioLNG_emission - LNG_consumption*LNG_emission)
            dollar_per_CO2[rateio + random.random()*0.00000001] = [ids[j], nums[j], 'BioLNG', buckets[j], kms[j], redpv, redpkm, costpkm]
          if ids[j][:3] == 'Die' and fuels[j] == 'B20':
            B20_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == ids[j]].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == ids[j]]['Fuel'] == 'B20']['Consumption (unit_fuel/km)'].tolist()[0]
            HVO_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == ids[j]].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == ids[j]]['Fuel'] == 'HVO']['Consumption (unit_fuel/km)'].tolist()[0]
            B20_emission = fuels_df.loc[fuels_df['Year'] == i].loc[fuels_df.loc[fuels_df['Year'] == i]['Fuel'] == 'B20']['Emissions (CO2/unit_fuel)'].tolist()[0]
            HVO_emission = fuels_df.loc[fuels_df['Year'] == i].loc[fuels_df.loc[fuels_df['Year'] == i]['Fuel'] == 'HVO']['Emissions (CO2/unit_fuel)'].tolist()[0]
            B20_cost = fuels_df.loc[fuels_df['Year'] == i].loc[fuels_df.loc[fuels_df['Year'] == i]['Fuel'] == 'B20']['Cost ($/unit_fuel)'].tolist()[0]
            HVO_cost = fuels_df.loc[fuels_df['Year'] == i].loc[fuels_df.loc[fuels_df['Year'] == i]['Fuel'] == 'HVO']['Cost ($/unit_fuel)'].tolist()[0]
            
            redpv = kms[j]*(B20_consumption*B20_emission - HVO_consumption*HVO_emission)
            redpkm = B20_consumption*B20_emission - HVO_consumption*HVO_emission
            if redpkm <= 0:
              continue
            costpkm = HVO_consumption*HVO_cost - B20_consumption*B20_cost
            rateio = (HVO_consumption*HVO_cost - B20_consumption*B20_cost)/(B20_consumption*B20_emission - HVO_consumption*HVO_emission)
            dollar_per_CO2[rateio + random.random()*0.00000001] = [ids[j], nums[j], 'HVO', buckets[j], kms[j], redpv, redpkm, costpkm]
          if ids[j][:3] == 'Die' and fuels[j] == 'HVO':
            B20_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == ids[j]].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == ids[j]]['Fuel'] == 'B20']['Consumption (unit_fuel/km)'].tolist()[0]
            HVO_consumption = vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == ids[j]].loc[vehicles_fuels_df.loc[vehicles_fuels_df['ID'] == ids[j]]['Fuel'] == 'HVO']['Consumption (unit_fuel/km)'].tolist()[0]
            B20_emission = fuels_df.loc[fuels_df['Year'] == i].loc[fuels_df.loc[fuels_df['Year'] == i]['Fuel'] == 'B20']['Emissions (CO2/unit_fuel)'].tolist()[0]
            HVO_emission = fuels_df.loc[fuels_df['Year'] == i].loc[fuels_df.loc[fuels_df['Year'] == i]['Fuel'] == 'HVO']['Emissions (CO2/unit_fuel)'].tolist()[0]
            B20_cost = fuels_df.loc[fuels_df['Year'] == i].loc[fuels_df.loc[fuels_df['Year'] == i]['Fuel'] == 'B20']['Cost ($/unit_fuel)'].tolist()[0]
            HVO_cost = fuels_df.loc[fuels_df['Year'] == i].loc[fuels_df.loc[fuels_df['Year'] == i]['Fuel'] == 'HVO']['Cost ($/unit_fuel)'].tolist()[0]
            
            redpv = kms[j]*(HVO_consumption*HVO_emission - B20_consumption*B20_emission)
            redpkm = HVO_consumption*HVO_emission - B20_consumption*B20_emission
            if redpkm <= 0:
              continue
            costpkm = B20_consumption*B20_cost - HVO_consumption*HVO_cost
            rateio = (B20_consumption*B20_cost - HVO_consumption*HVO_cost)/(HVO_consumption*HVO_emission - B20_consumption*B20_emission)
            dollar_per_CO2[rateio + random.random()*0.00000001] = [ids[j], nums[j], 'HVO', buckets[j], kms[j], redpv, redpkm, costpkm]
        while len(dollar_per_CO2) > 0:
          veh_id = dollar_per_CO2[min(dollar_per_CO2.keys())][0]
          quant = dollar_per_CO2[min(dollar_per_CO2.keys())][1]
          fue = dollar_per_CO2[min(dollar_per_CO2.keys())][2]
          bucke = dollar_per_CO2[min(dollar_per_CO2.keys())][3]
          distan = dollar_per_CO2[min(dollar_per_CO2.keys())][4]
          reduction_per_veh = dollar_per_CO2[min(dollar_per_CO2.keys())][5]
          reduction_per_km = dollar_per_CO2[min(dollar_per_CO2.keys())][6]
          cost_per_km = dollar_per_CO2[min(dollar_per_CO2.keys())][7]
          num_needed = math.ceil(reduction/reduction_per_veh)
          if num_needed >= quant:
            intex = sub.loc[sub['Year']==i].loc[sub.loc[sub['Year']==i]['ID']==veh_id].loc[sub.loc[sub['Year']==i].loc[sub.loc[sub['Year']==i]['ID']==veh_id]['Distance_bucket']==bucke].index.tolist()[0]
            sub.at[intex, 'Fuel'] = fue
            reduction -= reduction_per_veh*quant
            if reduction <= 0:
              break
            del dollar_per_CO2[min(dollar_per_CO2.keys())]
            continue
          else:
            kilometers = reduction/reduction_per_km
            factor = [102000, 106000, 73000, 118000]
            factory = factor[int(veh_id[5])-1]
            demanddd = distan*quant
            num = 0
            while True:
              num += 1
              min_de = demanddd-(quant-num)*factory
              max_de = num*factory
              if kilometers <= min_de or kilometers <= max_de:
                break
            num_old = quant - num
            demand_old = demanddd-kilometers
            if demand_old > num_old*factory:
              kilometers += demand_old-num_old*factory
              demand_old = num_old*factory

            intex = sub.loc[sub['Year']==i].loc[sub.loc[sub['Year']==i]['ID']==veh_id].loc[sub.loc[sub['Year']==i].loc[sub.loc[sub['Year']==i]['ID']==veh_id]['Distance_bucket']==bucke].index.tolist()[0]
            sub.at[intex, 'Num_Vehicles'] = quant - num
            demand_amount1 = demand_old/num_old
            if demand_amount1 < factory:
              demand_amount1 += 0.0000001
            sub.at[intex, 'Distance_per_vehicle(km)'] = demand_amount1
            demand_amount2 = kilometers/num
            if demand_amount2 < factory:
              demand_amount2 += 0.0000001
            sub.loc[len(sub)] = [i, veh_id, num, 'Use', fue, bucke, demand_amount2]

            break
      progr += 1
      my_bar_3.progress(progr/16, text=progress_text_3)

    st.session_state['submission'] = sub.sort_values(['Year', 'Type', 'Distance_bucket'], kind='quicksort')

    status.update(label="Solution complete!", state='complete', expanded=True)
    time.sleep(1)
  end_time = time.time()
  elapsed_time = end_time - start_time
  minutes = elapsed_time//60
  seconds = elapsed_time%60
  col2.markdown('Elapsed Time to Solution: %d minutes %.1f seconds' % (minutes, seconds))

if 'demand' in st.session_state:
  st.session_state['alpha'] = col2.slider("Alpha: parameter affecting the timeliness of selling 2023 vehicles", min_value=0.2, max_value=1.0, value = 0.2, step = 0.01, format='%.2f')
  st.session_state['beta'] = col2.slider("Beta: parameter describing weight of insurance/maintenance costs on selling scheme", min_value=3, max_value=20, value = 11, step = 1, format='%.1f')
  l1, l2, l3 = col2.columns([1, 1, 1])
  placeholder2 = l1.empty()
  placeholder9 = l2.empty()
  placeholder7 = l3.empty()
  placeholder8 = col2.empty()
  placeholder = col2.empty()
  if not 'submission' in st.session_state:
    if placeholder2.button("Optimize Fleet! (~2.5 hours)", type='primary', use_container_width =True, key='p1'):
      placeholder2.empty()
      placeholder9.empty()
      placeholder7.empty()
      if placeholder8.button('Stop'):
        st.stop()
        st.rerun()
      main_fun(6, False, st.session_state['alpha'], st.session_state['beta'])
    if placeholder9.button("Find Quick Solution! (~5 minutes)", type='primary', use_container_width =True, key='p9'):
      placeholder2.empty()
      placeholder9.empty()
      placeholder7.empty()
      if placeholder8.button('Stop'):
        st.stop()
        st.rerun()
      main_fun(0, False, st.session_state['alpha'], st.session_state['beta'])
    if placeholder7.button("Find Quicker Solution! (~5 seconds)", type='primary', use_container_width =True, key='u7'):
      placeholder2.empty()
      placeholder9.empty()
      placeholder7.empty()
      if placeholder8.button('Stop'):
        st.stop()
        st.rerun()
      main_fun(0, True, st.session_state['alpha'], st.session_state['beta'])
    if st.session_state['demand'].equals(st.session_state['original_demand']):
      if placeholder.button("Use a sample solution to default demand", key = 't2'):
        st.session_state['submission'] = pd.read_csv('dataset/sample_submission.csv')
        sol_fun()
        st.rerun()
  if 'submission' in st.session_state:

    placeholder.empty()
    placeholder2.empty()
    placeholder8.empty()
    placeholder7.empty()
    placeholder9.empty()
    kol1, kol2, kol3 = st.columns([1, 3, 1])

    flag1 = st.session_state['sol_demand'].equals(st.session_state['demand'])
    flag2 = st.session_state['sol_carbon_emissions'].equals(st.session_state['carbon_emissions'])
    #flag3 = st.session_state['sol_cost_profiles'].equals(st.session_state['cost_profiles'])
    flag4 = st.session_state['sol_vehicles_fuels'].equals(st.session_state['vehicles_fuels'])
    flag5 = st.session_state['sol_fuels'].equals(st.session_state['fuels'])
    flag6 = st.session_state['sol_vehicles'].equals(st.session_state['vehicles'])
    flag7 = st.session_state['sol_sell_percent'] == st.session_state['sell_percent']
    flag8 = st.session_state['sol_alpha'] == st.session_state['alpha']
    flag9 = st.session_state['sol_beta'] == st.session_state['beta']
    if not flag1 or not flag2 or not flag4 or not flag5 or not flag6 or not flag7 or not flag8 or not flag9:
      plakeholder2 = kol2.empty()
      plakeholder2.subheader('Your (demand, emissions, vehicle, fuel, vehicle fuel, annual vehicle sell precentage, alpha, or beta) data changed')
      plakeholder = kol2.empty()
      if plakeholder.button("Re-Optimize Fleet!", type='primary', use_container_width =True, key='p6'):
        plakeholder2.empty()
        plakeholder.empty()
        del st.session_state['submission']
        st.rerun()

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
    st.latex(r'Public \space Leaderboard \space Score \Longrightarrow 100 - \frac{70*%.2f}{65000000}=%.5f' % (total_2028, tot_2028))
    st.latex(r'Private \space Leaderboard \space Score \Longrightarrow 100 - \frac{70*%.2f}{172000000}=%.5f' % (tot, tot_2038))

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