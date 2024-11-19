import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ev2gym.models.ev2gym_env import EV2Gym

# initialization
config_file = "multiple_evs.yaml"

env = EV2Gym(config_file=config_file,
              save_replay=False,
              save_plots=False)

#create my own solution

# IN THIS SIMULATION, procured/setpoint power is the generated power from the grid, so we want to make sure:
# power balance = setpoint power + power generated from PV - power consumed by load - power consumed by EV
# Goal: power balance = 0

# this simulation assume only 1 transformer (acting as the grid)
# multiple evs and multiple charging stations (1 CS still only has one port, meaning 1 CS can only connect to 1 EV)

# note: there is also no "actual algorithm" to forecast the load and solar power for the current timestep, so instead, I will use the value from the previous timestep as forecast.

class MySolution:
    def __init__(self, env):
        self.env = env

    def get_action(self, env) -> np.ndarray:
        transformer_power = env.power_setpoints[env.current_step]

        # forecast
        if env.current_step > 0:
            pv_forecast = env.tr_solar_power[0, env.current_step-1] * -1 # because pv is negative from the original value. So I basically make it positive here

            load_forecast = env.tr_inflexible_loads[0, env.current_step-1]
        else:
            pv_forecast = 0
            load_forecast = 0

        remaining_power = transformer_power + pv_forecast - load_forecast

        # if remaining power is positive, we need to charge
        # each EV will be charged with the average power
        # however if the average power > the maximum power of the charger, then we will charge with the max power of the charger.
        # note: I did not mention the max power of EV battery, because in ev.py, they will adjust it if the power of cs > power of EV
        

        action_list = np.zeros(env.number_of_ports)

        if remaining_power > 0:
            
            # however we want to calculate the maximum amount of power we can charge (because it can be limited by the charger)

            counter = 0

            # get the number of EVs that will be charged in this timestep

            charge_EVs = []
            for cs in env.charging_stations:
                if cs.evs_connected[0] is not None: # 0 because we assume only 1 port in charging station
                    # ev battery not yet full
                    if cs.evs_connected[0].get_soc() < 1:
                        charge_EVs.append((counter, cs))
                        counter += 1

            if counter != 0:
                average_power = remaining_power / counter


                # set the action
                for i, cs in charge_EVs:
                        max_power_of_cs = cs.get_max_power()

                        # we want to charge with the minimum of the two
                        max_power_to_charge = min(max_power_of_cs, average_power)
                        action_list[i] = max_power_to_charge / max_power_of_cs


        # if remaining power is negative, we need to discharge
        else:
            counter = 0
            discharge_EVs = []
            # determine the number of EV that has higher battery than their desired capactiy
            for cs in env.charging_stations:
                if cs.evs_connected[0] is not None:
                    if cs.evs_connected[0].current_capacity > cs.evs_connected[0].desired_capacity:
                        discharge_EVs.append((counter, cs))
                        counter += 1

            

            if counter != 0:
                # the average power EV need to discharge
                average_power = remaining_power / counter

                for i, cs in discharge_EVs:
                    if cs.evs_connected[0].current_capacity - average_power > cs.evs_connected[0].desired_capacity:
                        action_list[i] = average_power / cs.get_max_power()
                    else:
                        action_list[i]=0
        return action_list
    
# test the solution
agent = MySolution(env)

for t in range(env.simulation_length):
    # get action
    actions = agent.get_action(env)
    env.step(actions)

#for plotting

fig, axs = plt.subplots(4, 1)  # 4 rows, 1 column


df_EV_SOC = pd.DataFrame()
df = pd.DataFrame(0, index= range(env.simulation_length), columns=["total_cs_power"])


for cs in env.charging_stations:
    df["total_cs_power"] += env.cs_power[cs.id, :]
    for port in range(cs.n_ports):
        df_EV_SOC[f'cs_{cs.id}'] = (env.port_energy_level[port, cs.id, :])

df['solar_power'] = env.tr_solar_power[0, :] * -1
df ['inflexible_load'] = env.tr_inflexible_loads[0, :]
df["power_setpoint"] = env.power_setpoints[:]
df["power_balance"] = df['solar_power'] + df['power_setpoint'] - df ['inflexible_load'] - df['total_cs_power']

axs[0].plot(df.index, df['solar_power'], label='Solar Power', color='orange', linewidth = 3)
axs[0].plot(df.index, df['power_setpoint'], label='Power generated by grid', color='green', linewidth = 3)
axs[0].plot(df.index, df['inflexible_load'], label='Load', color='blue', linewidth = 3)

axs[1].plot(df.index, df['total_cs_power'], label='CS power', color='purple', linewidth = 3)

axs[2].plot(df.index, df['power_balance'], label='Power Balance', color='black', linewidth = 2, linestyle='--')

for cs in env.charging_stations:
    for port in range(cs.n_ports):
        axs[3].plot(df_EV_SOC[f'cs_{cs.id}'])


counter_ev = 0
# get the arrival and departure time
for ev in env.EVs:
    if counter_ev == 0:
        for i in range(3):
            axs[i].axvline(x= ev.time_of_arrival, color='red', label='arrival time', linestyle='--')
            axs[i].axvline(x = ev.time_of_departure, color='red', label='departure time', linestyle='--')
    else:
        for i in range(3):
            axs[i].axvline(x= ev.time_of_arrival, color='red', linestyle='--')
            axs[i].axvline(x = ev.time_of_departure, color='red', linestyle='--')

# get the desired SOC
axs[3].axhline(y= env.EVs[0].desired_capacity/env.EVs[0].battery_capacity, label='desired SOC', color='red', linestyle='--')

axs[3].set_ylabel('SOC')

for i in range(4):
    axs[3].set_xlabel('simulation timestep')
    axs[i].legend()
    axs[i].grid(True)

for i in range(3):
    axs[i].set_xlabel('simulation timestep')
    axs[i].set_ylabel('Power')

plt.show()


