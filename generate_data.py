import pandas as pd
import numpy as np

# Configuration
num_points = 500
utilization_base = 50
power_base = 200
noise_level = 5
power_ratio = 3.5  # Watts per % utilization

# Generate normal data
timestamps = pd.to_datetime(pd.date_range(start='2025-07-20', periods=num_points, freq='T'))
utilization = utilization_base + np.random.randn(num_points) * noise_level
# Power should correlate with utilization
power = power_base + (utilization * power_ratio) + np.random.randn(num_points) * (noise_level * 2)

# Create DataFrame
df = pd.DataFrame({'timestamp': timestamps, 'utilization_percent': utilization, 'power_watts': power})

# --- Inject Anomalies ---

# Anomaly 1: High power, low utilization (PSU/Firmware issue)
df.loc[100:105, 'power_watts'] += 150
df.loc[100:105, 'utilization_percent'] -= 20

# Anomaly 2: Power spike (Failing PSU)
df.loc[250:252, 'power_watts'] *= 1.5

# Anomaly 3: System crash (Utilization and power drop)
df.loc[400:402, 'utilization_percent'] = 0
df.loc[400:402, 'power_watts'] = 50 # Standby power

# Ensure utilization is within bounds [0, 100]
df['utilization_percent'] = df['utilization_percent'].clip(0, 100)

# Save to CSV
df.to_csv('hardware_data.csv', index=False)
print("Dummy data file 'hardware_data.csv' created successfully.")