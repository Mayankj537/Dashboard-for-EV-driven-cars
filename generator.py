import pandas as pd
import numpy as np
import random
import threading
import time
import os
from datetime import datetime
 
battery_csv = '../public/battery_dynamic.csv'
 
# Parameters
Q = 50  # Nominal battery capacity in Ah
delta_t = 1/60  # 1-minute time step
n_samples = 1
 
driver_profiles = ['aggressive', 'moderate', 'conservative']
charging_habits = ['slow', 'fast']
batches = ['batch_A', 'batch_B', 'batch_C']
 
def generate_battery_data():
    temperature = np.random.uniform(10, 45, n_samples)
    voltage = np.random.uniform(3.0, 4.2, n_samples)
    current = np.random.uniform(0, 50, n_samples)
    charge_cycles = np.random.randint(0, 500, n_samples)
    humidity = np.random.uniform(10, 90, n_samples)
    charging_habit = [random.choice(charging_habits) for _ in range(n_samples)]
    batch = [random.choice(batches) for _ in range(n_samples)]
    driver_profile = [random.choice(driver_profiles) for _ in range(n_samples)]
 
    soc = [random.uniform(0.8, 1.0)]
    for i in range(1, n_samples):
        soc_next = soc[-1] + current[i] * delta_t / Q
        soc_next = min(max(soc_next, 0.0), 1.0)
        soc.append(soc_next)
 
    data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # ADD THIS LINE
        'Temperature': temperature,
        'Voltage': voltage,
        'Current': current,
        'Charge_Cycles': charge_cycles,
        'SoC': soc,
        'Humidity': humidity,
        'Charging_Habit': charging_habit,
        'Batch': batch,
        'Driver_Profile': driver_profile,
    }
 
    return pd.DataFrame(data)
 
def battery_health_condition(row):
    score = (
        (row['Temperature'] < 35) * 0.2 +
        (row['Voltage'] > 3.5) * 0.2 +
        (abs(row['Current']) < 30) * 0.2 +
        (row['Charge_Cycles'] < 200) * 0.2 +
        (row['SoC'] > 0.5) * 0.2 +
        (row['Humidity'] < 70) * 0.1
    )
    if row['Charging_Habit'] == 'slow':
        score += 0.1
    if row['Driver_Profile'] == 'conservative':
        score += 0.1
 
    if score > 0.8:
        return 2  # good
    elif score > 0.5:
        return 1  # fair
    else:
        return 0  # poor
 
def append_battery_data():
    while True:
        df = generate_battery_data()
        df['Battery_Health'] = df.apply(battery_health_condition, axis=1)
 
        file_exists = os.path.exists(battery_csv)
        df.to_csv(battery_csv, mode='a', header=not file_exists, index=False)
 
        print(f"{datetime.now()} - Appended {len(df)} rows to {battery_csv}")
        time.sleep(10)
 
# Start background data generation thread
battery_thread = threading.Thread(target=append_battery_data, daemon=True)
battery_thread.start()
 
# Keep main thread alive
try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    print("Stopped.")