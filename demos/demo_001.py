import pybullet as p
import pybullet_data
import numpy as np
import time
from envs.gym_env import UnitreeA1Env
import envs.config as cfg
import imageio
from pathlib import Path

'''
Demo 001: 
    * The purpose of this script is to demonstrate that a constant CPG
      is enough for walking when no payload is applied to the robot.
    
    * Refer to Demo 002 to see how the robot fails when a payload is 
      applied on the robot under a constant CPG.

    * No payload on the robot. 
    * No learning or policy update (constant CPG).
'''
def demo_001(save_gif = False):
    demo_num = "demo_001"
    steps = 1000
    dt = cfg.dt
    num_episodes = 10

    # Initialize Environment
    env = UnitreeA1Env(dt=dt, debug=False, animate_cpg=False, env_add_weight=False)
    
    for curr_episode in range(num_episodes):
        env.reset()
        # Gif creation: Create path and initialize variables.
        if save_gif:
            frames = []
            dir_path = f"demos/saved_gifs/{demo_num}"
            # Create path if doesn't exist
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            filename = f"{dir_path}/episode_{curr_episode + 1}.gif"
            frame_skip = 10

        for i in range(steps):
            action = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) if i % 2 else np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            observation, reward, done, info = env.step(action)
            # Gif creation: Get and append the frame
            if save_gif and i % frame_skip == 0:
                width, height, rgb_pixels, _, _ = p.getCameraImage(
                            width=640,
                            height=480,
                            renderer=p.ER_TINY_RENDERER,
                        )
                frame = np.reshape(rgb_pixels, (height, width, 4))[:, :, :3]  # Keep only RGB channels
                frame = frame.astype(np.uint8)  # Convert to uint8
                frames.append(frame)
            else:
                time.sleep(dt)
        
        # Save the GIF
        if save_gif:
            try:
                imageio.mimsave(filename, frames, fps=10)
                print(f"[INFO] GIF saved to {filename}")
            except Exception as e:
                print(f"[ERROR] Failed to save GIF: {e}")

    env.close()

if __name__ == "__main__":
    demo_001(save_gif = False)