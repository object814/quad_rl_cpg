import pybullet as p
import pybullet_data
import numpy as np
import time
from envs.gym_env import UnitreeA1Env  # Assuming the environment is in unitree_a1_env.py

def test_unitree_a1():
    # Connect to PyBullet with GUI to visualize the robot
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Create environment instance
    env = UnitreeA1Env(client_id=client, dt=0.01, debug=True)

    # Reset the environment
    obs = env.reset()

    # Number of steps to test
    num_steps = 1000

    # Action to increase amplitude for all legs
    increase_amplitude_action = np.array([1, 1, 1, 1])  # Increase amplitude for all legs

    for step in range(num_steps):
        # Apply the increase action and stop once the target amplitude is reached
        obs, reward, done, info = env.step(increase_amplitude_action)

        # Sleep to match the real-time visualization
        time.sleep(env.dt)

        if done:
            break

    # Close the environment
    env.close()

if __name__ == "__main__":
    test_unitree_a1()
