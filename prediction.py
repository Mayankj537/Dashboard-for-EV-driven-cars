import pandas as pd
import joblib
import json
import time
import os
import random
import numpy as np
 
# --- Load the trained model and encoders ---
model = joblib.load('gbc_battery_health_model.pkl')
le_charging = joblib.load('le_charging.pkl')
le_batch = joblib.load('le_batch.pkl')
le_driver = joblib.load('le_driver.pkl')
 
battery_csv = '../public/battery_dynamic.csv'
output_csv = '../public/battery_health_predictions.csv'
 
# Chart.js configuration for colors and filenames
chartjs_config = {
    'parameters': {
        'Temperature': {
            'filename': '../src/assets/json/temperature_chartjs.json',
            'borderColor': 'rgba(255, 99, 132, 1)',
            'backgroundColor': 'rgba(255, 99, 132, 0.2)'
        },
        'Voltage': {
            'filename': '../src/assets/json/voltage_chartjs.json',
            'borderColor': 'rgba(54, 162, 235, 1)',
            'backgroundColor': 'rgba(54, 162, 235, 0.2)'
        },
        'Current': {
            'filename': '../src/assets/json/current_chartjs.json',
            'borderColor': 'rgba(255, 206, 86, 1)',
            'backgroundColor': 'rgba(255, 206, 86, 0.2)'
        },
        'Humidity': {
            'filename': '../src/assets/json/humidity_chartjs.json',
            'borderColor': 'rgba(75, 192, 192, 1)',
            'backgroundColor': 'rgba(75, 192, 192, 0.2)'
        },
        'SoC': {
            'filename': '../src/assets/json/soc_chartjs.json',
            'borderColor': 'rgba(153, 102, 255, 1)',
            'backgroundColor': 'rgba(153, 102, 255, 0.2)'
        }
    }
}
 
 
last_processed = 0
 
# --- Health class to percentage mapping ---
def health_class_to_soh_percent(health_class):
    if health_class == 2:    # Good
        return round(random.uniform(85,99), 2)
    elif health_class == 1:  # Fair
        return round(random.uniform(70,84), 2)
    else:                    # Poor
        return round(random.uniform(58,69), 2)
 
# --- Chart.js JSON generation ---
def generate_chartjs_json_with_timestamps():
    # Read the latest 20 rows
    if not os.path.exists(battery_csv):
        print(f"{battery_csv} not found.")
        return
    df = pd.read_csv(battery_csv).tail(20)
    if 'timestamp' not in df.columns:
        print("No 'timestamp' column in data.")
        return
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    labels = df['timestamp'].dt.strftime('%H:%M:%S').tolist()
 
    for param, config in chartjs_config['parameters'].items():
        if param in df.columns:
            values = df[param].tolist()
            chart_data = {
                'labels': labels,
                'datasets': [{
                    'label': param,
                    'data': values,
                    'borderColor': config['borderColor'],
                    'backgroundColor': config['backgroundColor'],
                    'fill': True
                }]
            }
            # Save to JSON file
            with open(config['filename'], 'w') as f:
                json.dump(chart_data, f, indent=2)
            print(f"Updated {config['filename']}")
        else:
            print(f"Column {param} not found in data.")
 
# --- EV metrics calculation ---
def calculate_metrics(df_dynamic):
    ICE_CO2_PER_KM = 0.192
    EV_CO2_PER_KWH = 0.5
    EV_CONSUMPTION_KWH_PER_KM = 0.15
    ICE_COST_PER_KM = 10 / 15
    EV_COST_PER_KWH = 8
    NOMINAL_BATTERY_CAPACITY_KWH = 60
    IDEAL_CONSUMPTION_KWH_PER_100KM = 15
    EPA_ADJUSTMENT_FACTOR = 0.7
 
    def soh_percent_to_fraction(soh_percent):
        return soh_percent / 100.0
 
    if not all(col in df_dynamic.columns for col in ['timestamp', 'Predicted_Battery_Health', 'SoC']):
        print("Required columns missing for metrics calculation.")
        return None
 
    latest = df_dynamic.iloc[-1]
    soh_frac = soh_percent_to_fraction(latest['Predicted_Battery_Health'])
    soc = latest['SoC']
 
    usable_energy = soh_frac * NOMINAL_BATTERY_CAPACITY_KWH * soc
    ideal_range = usable_energy * 100 / IDEAL_CONSUMPTION_KWH_PER_100KM
    epa_range = ideal_range * EPA_ADJUSTMENT_FACTOR
 
    ev_co2 = epa_range * EV_CONSUMPTION_KWH_PER_KM * EV_CO2_PER_KWH
    ice_co2 = epa_range * ICE_CO2_PER_KM
    co2_saved = ice_co2 - ev_co2
 
    ev_cost_per_km = EV_CONSUMPTION_KWH_PER_KM * EV_COST_PER_KWH
    ice_cost_per_km = ICE_COST_PER_KM
    cost_saved = (ice_cost_per_km - ev_cost_per_km) * epa_range
 
    degradation_rate = 0.018
    years = 3
    future_soh = soh_frac * ((1 - degradation_rate) ** years)
    future_usable_energy = future_soh * NOMINAL_BATTERY_CAPACITY_KWH * soc
    future_ideal_range = future_usable_energy * 100 / IDEAL_CONSUMPTION_KWH_PER_100KM
    future_epa_range = future_ideal_range * EPA_ADJUSTMENT_FACTOR
 
    return {
        'timestamp': latest['timestamp'],
        'CO2_Saved_kg': round(co2_saved, 2),
        'Cost_Saved_INR': abs(round(cost_saved, 2)),
        'Expected_Range_km': round(ideal_range, 2),
        'Real_Time_Range_km': round(epa_range, 2),
        'Projected_3yr_Range_km': round(future_epa_range, 2)
    }
 
# --- Metrics CSV update ---
def update_metrics_csv(metrics, filename='../public/ev_metrics_dynamic.csv'):
    import csv
    file_exists = os.path.exists(filename)
    fieldnames = [
        'timestamp', 'CO2_Saved_kg', 'Cost_Saved_INR',
        'Expected_Range_km', 'Real_Time_Range_km', 'Projected_3yr_Range_km'
    ]
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
 
# --- MAIN LOOP ---
while True:
    if os.path.exists(battery_csv):
        df = pd.read_csv(battery_csv)
        new_rows = df.iloc[last_processed:]
       
        if not new_rows.empty:
            # --- ML prediction ---
            new_rows['Charging_Habit'] = le_charging.transform(new_rows['Charging_Habit'])
            new_rows['Batch'] = le_batch.transform(new_rows['Batch'])
            new_rows['Driver_Profile'] = le_driver.transform(new_rows['Driver_Profile'])
 
            features = new_rows.drop(columns=['timestamp', 'Battery_Health'], errors='ignore')
            predictions = model.predict(features)
            new_rows['Predicted_Battery_Health'] = [health_class_to_soh_percent(p) for p in predictions]
 
            new_rows.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
            print(f"Processed {len(new_rows)} new rows. Predictions saved to {output_csv}.")
 
            last_processed += len(new_rows)
           
            # --- Chart.js JSON generation ---
            generate_chartjs_json_with_timestamps()
 
            # --- EV Metrics calculation and CSV update ---
            df_pred = pd.read_csv(output_csv)
            metrics = calculate_metrics(df_pred)
            if metrics is not None:
                update_metrics_csv(metrics)
                print("EV metrics updated:", metrics)
           
        else:
            print("No new data to process.")
    else:
        print("Waiting for dynamic CSV data...")
 
    time.sleep(30)  # Wait 30 seconds before next check