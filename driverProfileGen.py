import pandas as pd
import numpy as np
import random
import threading
import time
import os
from datetime import datetime
 
driver_csv = 'driver_profile_dynamic.csv'  # Correct CSV name for driver data
 
def append_driver_data():
    """Appends driver profile data to driver-specific CSV every second."""
    while True:
        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'average_speed': np.random.normal(65, 15),
            'max_acceleration': np.abs(np.random.normal(2.5, 1.0)),  # Acceleration can't be negative
            'hard_brakes': np.random.poisson(2),
            'smooth_brakes': np.random.poisson(3),
            'energy_consumption': np.abs(np.random.normal(18, 4)),  # Consumption can't be negative
            'idle_time': np.abs(np.random.normal(45, 15)),
            'driver_profile': random.choice([0, 1, 2])  # 0: Aggressive, 1: Moderate, 2: Conservative
        }
 
        # Use proper file existence check and header handling
        header = not os.path.exists(driver_csv)
        pd.DataFrame([row]).to_csv(driver_csv, mode='a', header=header, index=False)
        print(row)
        time.sleep(10)
 
# Start thread
driver_thread = threading.Thread(target=append_driver_data, daemon=True)
driver_thread.start()
 
# Keep main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped. Driver data saved to:", driver_csv)