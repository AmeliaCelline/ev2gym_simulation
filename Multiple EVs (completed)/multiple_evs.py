import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ev2gym.models.ev2gym_env import EV2Gym
from collections import defaultdict

# initialization
config_file = "multiple_evs.yaml"
env = EV2Gym(config_file=config_file,
              save_replay=False,
              save_plots=False)


# - SIMULATION DESCRIPTION ------------------------------------

# goal of this simulation is carbon neutrality/ power balance
# unlike before, we don't have traditional power generator anymore. Because in real grid, traditional generator is controlled based on the demand.
# now the goal is to balance the power from solar power, load, and EV charging station
# solar power + power from EV discharging == load +  power from EV charging

# this simulation assume only 1 transformer and 1 port in charging station (so only 1 car can connect to charging station) so we can just use the total power of the transformer to determine the power balance

# this simulation allows multiple evs to spawn in the simulation

# the transformer here act as the grid basically

# note: there is also no "actual algorithm" to forecast the load and solar power for the current timestep, so instead, I will use the value from the previous timestep as forecast.


# ------------------------------------------------------------

# Initialize variables

# store SOC of EVs

# the key of this dictionary is the id of the charging station and the time of arrival
# it is important to note that the id of EV is unique per charging station, so to store the SOC of the EV, we should use the id of the charging station and not the id of the EV
# I use time of arrival as well because another EV can use the same charging station at a later time and has the same CS id, so we need to differentiate them
soc_evs = defaultdict(list)


class MySolution:
    def __init__(self, env):
        self.env = env

    def get_action(self, env) -> np.ndarray:

        # forecast
        if env.current_step > 0:
            pv_forecast = env.tr_solar_power[0, env.current_step-1] * -1 # because pv is negative from the original value. So I basically make it positive here

            load_forecast = env.tr_inflexible_loads[0, env.current_step-1]
        else:
            pv_forecast = 0
            load_forecast = 0

        remaining_power =  pv_forecast - load_forecast

        # if remaining power is positive, we need to charge
        # each EV will be charged with the average power
        
        action_list = np.zeros(env.number_of_ports)

        if remaining_power > 0:
            
            # however we want to calculate the maximum amount of power we can charge (because it can be limited by the charger)

            counter = 0

            # get the number of EVs that will be charged in this timestep

            charge_EVs = []
            
            for cs in env.charging_stations:
                if cs.evs_connected[0] is not None: # 0 because we assume only 1 port in charging station

                    # store EV SOC for plotting
                    soc_evs[(cs.id,  cs.evs_connected[0].time_of_arrival)].append(cs.evs_connected[0].get_soc() * 100)

                    # ev battery not yet full
                    if cs.evs_connected[0].get_soc() < 1:
                        charge_EVs.append(cs)
                        counter+=1
           
            if counter != 0:
                average_power = remaining_power / counter
                # set the action
                for cs in charge_EVs:

                    # note if average power > cs.get_max_power, it will actually charge with the max power of the charger
                    action_list[cs.id] = average_power / cs.get_max_power()


        # if remaining power is negative, we need to discharge
        else:
            counter = 0
            discharge_EVs = []
            # determine the number of EV that has higher battery than their desired capactiy
            for cs in env.charging_stations:
                if cs.evs_connected[0] is not None:
                    # store EV SOC for plotting
                    soc_evs[(cs.id,  cs.evs_connected[0].time_of_arrival)].append(cs.evs_connected[0].get_soc() * 100)

                    if cs.evs_connected[0].current_capacity > cs.evs_connected[0].desired_capacity:
                        discharge_EVs.append(cs)
                        counter += 1

            
            if counter != 0:
                # the average power EV need to discharge
                average_power = remaining_power / counter

                for cs in discharge_EVs:

                    # calculate the available power EV can discharge
                    available_power = cs.evs_connected[0].desired_capacity - cs.evs_connected[0].current_capacity

                    #if ev doesn't have enough power to discharge using the average power, it will discharge the remaining available power it has
                    if available_power * -1 < average_power * -1: # I multiply -1 here to first make them positive, and then compare. It is easier to read this way.
                        action_list[cs.id] = available_power / cs.get_min_power() * (-1) # multiply -1 to signify discharging
                    else:
                        action_list[cs.id] = average_power / cs.get_min_power() * (-1)
                        
        return action_list
    
# test the solution
agent = MySolution(env)

for t in range(env.simulation_length):
    # get action
    actions = agent.get_action(env)
    env.step(actions)

# all the codes below for plotting.

df_EV_SOC = pd.DataFrame()
df = pd.DataFrame(0, index= range(env.simulation_length), columns=['total_cs_power'])

# plotting

fig, axs = plt.subplots(4, 1)  # 4 rows, 1 column

# plotting the SOC of EVs

for ev in env.EVs:
    soc = soc_evs[(ev.location, ev.time_of_arrival)]

    if ev.time_of_departure <= env.simulation_length:
        stay_time = range(ev.time_of_arrival, ev.time_of_departure + 1)
    else:
        stay_time = range(ev.time_of_arrival, env.simulation_length)

    axs[3].plot(stay_time, soc)

# plotting the arrival and departure time
for i, ev in enumerate(env.EVs):
    if i == 0:
        for i in range(3):
            axs[i].axvline(x= ev.time_of_arrival, color='green', label='arrival time', linestyle='--')
            axs[i].axvline(x = ev.time_of_departure, color='red', label='departure time', linestyle='--')
    else:
        for i in range(3):
            axs[i].axvline(x= ev.time_of_arrival, color='green', linestyle='--')
            axs[i].axvline(x = ev.time_of_departure, color='red', linestyle='--')


# plotting the CS power
for cs in env.charging_stations:
    df['total_cs_power'] += env.cs_power[cs.id, :]

df['solar_power'] = env.tr_solar_power[0, :] * -1
df ['inflexible_load'] = env.tr_inflexible_loads[0, :]
df["power_balance"] = df['solar_power'] - df ['inflexible_load'] - df['total_cs_power']

axs[0].plot(df.index, df['solar_power'], label='Solar Power', color='orange', linewidth = 3)
axs[0].plot(df.index, df['inflexible_load'], label='Load', color='blue', linewidth = 3)

axs[1].plot(df.index, df['total_cs_power'], label='CS power', color='purple', linewidth = 3)

axs[2].plot(df.index, df['power_balance'], label='Power Balance', color='black', linewidth = 2, linestyle='--')

axs[3].axhline(y= env.EVs[0].desired_capacity/env.EVs[0].battery_capacity * 100, label='desired SOC', color='red', linestyle='--')
axs[3].set_ylabel('SOC')


for i in range(4):
    axs[i].set_xlim(0,env.simulation_length-1)
    axs[i].legend()
    axs[i].grid(True)
    axs[i].set_xlabel('simulation timestep')

for i in range(3):
    axs[i].set_ylabel('Power')

plt.show()


