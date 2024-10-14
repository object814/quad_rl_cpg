import pybullet as p
import pybullet_data
import numpy as np
import time
from envs.gym_env import UnitreeA1Env

def test_unitree_a1():
<<<<<<< Updated upstream
    steps = 1000
    dt = 0.01

    # Initialize Environment
    env = UnitreeA1Env(dt=dt, debug=False)
    env.reset()

    # Action space: [FR_amplitude, FR_frequency, FR_phase, FL_amplitude, FL_frequency, FL_phase, RR_amplitude, RR_frequency, RR_phase, RL_amplitude, RL_frequency, RL_phase]
    # Test by increasing amplitude of all legs
    action = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    for i in range(steps):
        observation, reward, done, info = env.step(action)
        time.sleep(dt)
        # input("Press Enter to continue...")
    
=======
    # Create environment instance
    env = UnitreeA1Env(dt=0.01, debug=False)

    # Reset the environment
    obs = env.reset()

    # Number of steps to test
    num_steps = 1000

    # Action to increase amplitude for all legs
    increase_amplitude_action = np.array([1, 1, 1, 1])  # Increase amplitude for all legs

    for step in range(num_steps):
        # Apply the increase action and stop once the target amplitude is reached
        obs, reward, done, info = env.step(increase_amplitude_action)
        print(f"Step {step+1}/{num_steps}, Amplitude: {env.amplitude}, Reward: {reward}")

        # Sleep to match the real-time visualization
        time.sleep(env.dt)

        if done:
            break

    # Close the environment
>>>>>>> Stashed changes
    env.close()

if __name__ == "__main__":
    test_unitree_a1()