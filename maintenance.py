import pandas as pd
import json
import os
from datetime import datetime
import time
 
def generate_maintenance_suggestion_json(
    battery_csv='../public/battery_dynamic.csv',
    output_json='../src/assets/json/maintenance.json'
):
    if not os.path.exists(battery_csv):
        print(f"{battery_csv} not found. Waiting for data...")
        return
 
    try:
        df = pd.read_csv(battery_csv)
    except Exception as e:
        print(f"Error reading {battery_csv}: {e}")
        return
 
    if df.empty:
        print("Battery dynamic CSV is empty.")
        return
 
    # Use the latest 100 rows
    df = df.tail(100)
 
    required_cols = ['timestamp', 'Charge_Cycles', 'Battery_Health']
    for col in required_cols:
        if col not in df.columns:
            print(f"Column {col} missing in data. Cannot generate maintenance suggestion.")
            return
 
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
 
    latest = df.iloc[-1]
 
    # Map Battery_Health class to percentage if needed
    health_map = {2: 95, 1: 80, 0: 60}
    battery_health_percent = health_map.get(latest['Battery_Health'], 60)
 
    # Maintenance suggestion logic
    if latest['Charge_Cycles'] >= 400 or battery_health_percent < 75:
        message = (
            f"Maintenance due: Battery health at {battery_health_percent}%, "
            f"Charge cycles: {latest['Charge_Cycles']}. Please schedule service soon."
        )
    else:
        message = "Battery is healthy. No maintenance needed at this time."

    print(message)
 
    suggestion_json = {
        "message": message
    }
 
    with open(output_json, 'w') as f:
        json.dump(suggestion_json, f, indent=2)
 
# To run every minute, use:
if __name__ == '__main__':
    while True:
        generate_maintenance_suggestion_json()
        time.sleep(10)