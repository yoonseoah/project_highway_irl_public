import gym
import numpy as np
import pandas as pd
from irl_graphics_v2 import EnvViewer2
from shared.transform_obs import transform_obs_v2 as transform_obs
import highway_irl_v2  # Your custom driving environment
import os
import time

os.environ["SDL_VIDEODRIVER"] = "cocoa"

# Configurations
config_file = 'config_35_quick120.npy'
config = np.load('env_configure/' + config_file, allow_pickle=True).tolist()
max_speed = config['max_speed']
discretize = False

# Load CSV data
file_path = "highway_task/data/345/01_episode0.csv"  # Update this path if needed
df = pd.read_csv(file_path)

# Load overtaking indices
overtaking_indices = np.load("overtaking_indices.npy")

print(f"✅ 렌더링할 총 프레임 개수: {len(overtaking_indices)}")

# Initialize environment
env = gym.make('IRL-v2')
env.config = config

if env.viewer is None:
    env.viewer = EnvViewer2(env)

# Reset environment
env.reset()
lane_speed = [20, 20, 20]  # Default lane speeds
obs = env.observation_type.observe()
state, lane_speed = transform_obs(obs, lane_speed, discretize, max_speed)
score = 0

# Disable AI for other cars (NPCs)
for vehicle in env.road.vehicles:
    vehicle.enable_lane_change = False  # Prevent AI-controlled lane changes
    vehicle.speed = 0  # Stop AI from accelerating

# Replay loop for only overtaking moments
for t in overtaking_indices:
    if t < 0 or t >= len(df) - 10:
        continue  # Skip invalid indices

    env.render()

    # Extract the 10 rows corresponding to this timestep
    rows = df.iloc[t:t+10]

    if len(rows) < 10:
        print(f"Skipping step {t} - Not enough vehicle data")
        continue

    # Get time difference from the dataset
    if t == 0:
        dt = 0.2  # Default for first step
    else:
        dt = rows.iloc[0]['time'] - df.iloc[t-10]['time']  # Compute actual time difference

    # Get action from the CSV (assume 'control' column stores the ego action)
    action = int(rows.iloc[0]['control'])  # Ego vehicle action

    # Step the environment (to keep the simulation in sync)
    env.step(action)

    # Ensure only the correct 10 vehicles exist in the scene
    while len(env.road.vehicles) > 10:
        env.road.vehicles.pop()

    # Manually update the 10 vehicles with logged positions
    for i, vehicle in enumerate(env.road.vehicles[:10]):  # Only update first 10 vehicles
        vehicle.position = np.array([rows.iloc[i]['x'], rows.iloc[i]['y']])
        vx, vy = rows.iloc[i]['vx'], rows.iloc[i]['vy']
        vehicle.speed = np.sqrt(vx**2 + vy**2)
        vehicle.position += np.array([vx * dt, vy * dt])

    # Force Ego Vehicle’s Position Again to Prevent Drifting
    ego_vehicle = env.road.vehicles[0]  # Ego vehicle is always first
    ego_vehicle.position = np.array([rows.iloc[0]['x'], rows.iloc[0]['y']])  # Strictly enforce position

    # Transform observation into state
    next_obs = np.array([
        [rows.iloc[i]['x'], rows.iloc[i]['y'], rows.iloc[i]['control'], rows.iloc[i]['vx'], rows.iloc[i]['vy']]
        for i in range(10)
    ])
    next_state, lane_speed = transform_obs(next_obs, lane_speed, discretize, max_speed)

    # Update state and score
    state = next_state
    score += rows.iloc[0]['rewards']

    # Debug output (optional)
    print(f"Step {t//10}: Time={rows.iloc[0]['time']}, dt={dt:.3f}, Action={action}, Reward={rows.iloc[0]['rewards']}")

    # Add delay to match real replay time
    time.sleep(dt)

# Close the environment after replay
env.close()
print("✅ 추월 장면 렌더링 완료.")