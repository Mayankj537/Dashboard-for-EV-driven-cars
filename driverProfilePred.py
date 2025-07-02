import pandas as pd
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
 
# 1. Load and preprocess static dataset (run this once at startup)
static_df = pd.read_csv('driver_profile_dataset.csv')
y = static_df['driver_profile']
X = static_df.drop('driver_profile', axis=1)
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
clf = RandomForestClassifier(n_estimators=1000, random_state=42)
clf.fit(X_train, y_train)
 
# Suggestion generator with multiple messages per profile
def generate_suggestion(profile):
    suggestions = {
        'Aggressive': [
            "Hello Jay",
            "⚠️ Aggressive driving detected! Slow down and drive more smoothly to save battery and improve safety.",
            "Consider keeping a safe distance and following speed limits to prevent risky behavior."
        ],
        'Moderate': [
            "Hello Jay",
            "👍 Good driving! Keep maintaining steady speeds for optimal efficiency.",
            "Your balanced approach not only ensures efficiency but also maintains road safety."
        ],
        'Conservative': [
            "Hello Jay",
            "✅ Excellent driving behavior! You're maximizing battery life and safety.",
            "Continue your cautious approach — it minimizes risk and promotes long-term vehicle health."
        ]
    }
    return suggestions.get(profile, ["Drive safely!", "Always remain alert and abide by road rules."])
 
def save_driver_profile_curve_and_suggestion_json():
    dynamic_csv = 'driver_profile_dynamic.csv'
    profile_labels = ['Aggressive', 'Moderate', 'Conservative']
    profile_scores = {'Aggressive': 0, 'Moderate': 1, 'Conservative': 2}
    output_curve_json = '../src/assets/json/driver_profile_curve.json'
    output_suggestion_json = '../src/assets/json/suggestions.json'
 
    # Load latest 20 rows for Chart.js format
    try:
        df_dynamic = pd.read_csv(dynamic_csv).tail(20).copy()
    except FileNotFoundError:
        print("Dynamic CSV not found. Please generate data first.")
        return
 
    # Predict if not already present
    if 'Predicted_Profile' not in df_dynamic.columns:
        features = df_dynamic.drop(columns=['timestamp', 'driver_profile'], errors='ignore')
        predictions = clf.predict(features)
        df_dynamic['Predicted_Profile'] = [profile_labels[p] for p in predictions]
 
    # Map profile to score for plotting
    df_dynamic['Profile_Score'] = df_dynamic['Predicted_Profile'].map(profile_scores)
    df_dynamic['timestamp'] = pd.to_datetime(df_dynamic['timestamp'])
 
    # Smooth the curve using a moving average (window size 3 for smaller dataset)
    df_dynamic['Smoothed_Score'] = df_dynamic['Profile_Score'].rolling(window=3, min_periods=1).mean()
 
    # Format timestamps for labels (show time only for readability)
    labels = df_dynamic['timestamp'].dt.strftime('%H:%M:%S').tolist()
    data = df_dynamic['Smoothed_Score'].tolist()
 
    # ---- 1. Chart.js driver profile curve JSON (unchanged) ----
    chart_data = {
        "labels": labels,
        "datasets": [{
            "label": "Driver Profile Score",
            "data": data,
            "borderColor": "rgba(255, 159, 64, 1)",
            "backgroundColor": "rgba(255, 159, 64, 0.2)",
            "fill": True,
            "tension": 0.4
        }]
    }
    with open(output_curve_json, 'w') as f:
        json.dump(chart_data, f, indent=2)
    print(f"[{pd.Timestamp.now()}] Driver profile curve saved to {output_curve_json}")
 
    # ---- 2. Suggestion JSON (multiple messages in a single object) ----
    latest_profile = df_dynamic['Predicted_Profile'].iloc[-1]
    suggestion_list = generate_suggestion(latest_profile)
    # The JSON will contain one object with the key "message" associated with an array of suggestion strings
    suggestion_json = [{"message": suggestion_list}]
    with open(output_suggestion_json, 'w') as f:
        json.dump(suggestion_json, f, indent=2)
    print(f"[{pd.Timestamp.now()}] Suggestions saved to {output_suggestion_json}: {suggestion_list}")
 
# --- Run the update every 20 seconds ---
if __name__ == "__main__":
    while True:
        save_driver_profile_curve_and_suggestion_json()
        time.sleep(20)
