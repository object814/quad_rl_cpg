import pybullet as p
import pybullet_data
import numpy as np
import time
from envs.gym_env import UnitreeA1Env
import envs.config as cfg

def test_unitree_a1():
    steps = 1000
    dt = cfg.dt

    # Initialize Environment
    env = UnitreeA1Env(dt=dt, debug=False, animate_cpg=True, fixed_base=True)
    env.reset()

    # Action space: [FR_amplitude, FR_frequency, FR_phase, FL_amplitude, FL_frequency, FL_phase, RR_amplitude, RR_frequency, RR_phase, RL_amplitude, RL_frequency, RL_phase]
    # Test by increasing amplitude of all legs
    action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    #action = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    for i in range(steps):
        observation, reward, done, info = env.step(action)
        reward = (cfg.discount_factor ** i) * reward
        time.sleep(dt)
        # input("Press Enter to continue...")
    
    env.close()

if __name__ == "__main__":
    test_unitree_a1()