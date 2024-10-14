import pybullet as p
import pybullet_data
import numpy as np
import time
from envs.gym_env import UnitreeA1Env  # Assuming the environment is in unitree_a1_env.py
import matplotlib.pyplot as plt

forward_progress_rewards = []
roll_pitch_stability_penalties = []
payload_drop_penalties = []
foot_slip_penalties = []
rewards = []

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line1, = ax.plot(forward_progress_rewards, label='Forward Progress Reward')
line2, = ax.plot(roll_pitch_stability_penalties, label='Roll/Pitch Stability Penalty')
line3, = ax.plot(payload_drop_penalties, label='Payload Drop Penalty')
line4, = ax.plot(foot_slip_penalties, label='Foot Slip Penalty')
line5, = ax.plot(rewards, label='Total Reward')
ax.legend()
ax.set_xlabel('Step')
ax.set_ylabel('Value')
ax.set_title('Rewards and Penalties Over Steps')
ax.grid(True)

def test_unitree_a1():
    steps = 1000
    dt = 0.01

    # Initialize Environment
    env = UnitreeA1Env(dt=dt, debug=False, animate_cpg=False)
    env.reset()

    # Action space: [FR_amplitude, FR_frequency, FR_phase, FL_amplitude, FL_frequency, FL_phase, RR_amplitude, RR_frequency, RR_phase, RL_amplitude, RL_frequency, RL_phase]
    # Test by increasing amplitude of all legs
    action = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])

    for step in range(steps):
        obs, reward, forward_progress_reward, roll_pitch_stability_penalty, payload_drop_penalty, foot_slip_penalty, done, info = env.step_for_reward_testing(action)
        print(f"Step {step+1}/{steps}, Forward Progress Reward: {forward_progress_reward}, Roll_Pitch_Stability_Penalty: {roll_pitch_stability_penalty}, Payload Drop Penalty: {payload_drop_penalty}, Foot Slip Penalty: {foot_slip_penalty}, Reward: {reward}, Done: {done}")
        
        forward_progress_rewards.append(forward_progress_reward)
        roll_pitch_stability_penalties.append(roll_pitch_stability_penalty)
        payload_drop_penalties.append(payload_drop_penalty)
        foot_slip_penalties.append(foot_slip_penalty)
        rewards.append(reward)

        line1.set_ydata(forward_progress_rewards)
        line1.set_xdata(range(len(forward_progress_rewards)))
        line2.set_ydata(roll_pitch_stability_penalties)
        line2.set_xdata(range(len(roll_pitch_stability_penalties)))
        line3.set_ydata(payload_drop_penalties)
        line3.set_xdata(range(len(payload_drop_penalties)))
        line4.set_ydata(foot_slip_penalties)
        line4.set_xdata(range(len(foot_slip_penalties)))
        line5.set_ydata(rewards)
        line5.set_xdata(range(len(rewards)))
        ax.relim()  # Recalculate limits
        ax.autoscale_view()  # Autoscale
        plt.draw()
        plt.pause(0.001)  # Pause to update the plot

        # Sleep to match the real-time visualization
        time.sleep(dt)

        if done:
            break

    # Close the environment
    env.close()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    test_unitree_a1()
