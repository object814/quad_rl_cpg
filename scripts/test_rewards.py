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
        obs, reward, forward_progress_reward, roll_pitch_stability_penalty, payload_drop_penalty, foot_slip_penalty, done, info = env.step_for_reward_testing(increase_amplitude_action)
        print(f"Step {step+1}/{num_steps}, Forward Progress Reward: {forward_progress_reward}, Roll_Pitch_Stability_Penalty: {roll_pitch_stability_penalty}, Payload Drop Penalty: {payload_drop_penalty}, Foot Slip Penalty: {foot_slip_penalty}, Reward: {reward}, Done: {done}")
        
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
        plt.pause(0.01)  # Pause to update the plot

        # Sleep to match the real-time visualization
        time.sleep(env.dt)

        if done:
            break

    # Close the environment
    env.close()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    test_unitree_a1()

