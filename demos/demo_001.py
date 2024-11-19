import pybullet as p
import pybullet_data
import numpy as np
import time
from envs.gym_env import UnitreeA1Env
import envs.config as cfg

def demo_001():
    steps = 5000
    dt = cfg.dt

    # Initialize Environment
    env = UnitreeA1Env(dt=dt, debug=False, animate_cpg=False, env_add_weight=True)
    env.reset()

    # Action space: [FR_amplitude, FR_frequency, FR_phase, FL_amplitude, FL_frequency, FL_phase, RR_amplitude, RR_frequency, RR_phase, RL_amplitude, RL_frequency, RL_phase]
    # Test by increasing amplitude of all legs
    for i in range(steps):
        action = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) if i % 2 else np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        observation, reward, done, info = env.step(action)
        time.sleep(dt)
    
    env.close()

if __name__ == "__main__":
    demo_001()