# test_model.py

import torch
import os
import numpy as np
from networks.ppo_actor_critic import PPOActorCritic
from envs.gym_env import UnitreeA1Env
import imageio
import pybullet as p
from datetime import datetime

class Evaluator:
    def __init__(self, num_episodes=10, render=True, max_steps=1000, save_gifs=True):
        self.num_episodes = num_episodes
        self.render = render
        self.max_steps = max_steps
        self.save_gifs = save_gifs
        self.env = UnitreeA1Env(render=render)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.model = PPOActorCritic(self.observation_dim, self.action_dim)
        self.model_path = "networks/saved_model/ppo_model.pth"
        self.load_model()

        # Generate timestamp for experiment folder
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.exp_folder = f"logs/exp_{self.timestamp}"

        # Ensure directories exist
        if not os.path.exists("logs"):
            os.makedirs("logs")
        if not os.path.exists(self.exp_folder):
            os.makedirs(self.exp_folder)
        if self.save_gifs:
            self.gifs_folder = os.path.join(self.exp_folder, "gifs")
            if not os.path.exists(self.gifs_folder):
                os.makedirs(self.gifs_folder)

        # Metrics
        self.success_count = 0
        self.total_rewards = []
        self.episode_lengths = []
        self.episode_results = []  # Store per-episode results

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"[INFO] Loaded model from {self.model_path}")
        else:
            print(f"[ERROR] Model file not found at {self.model_path}")
            exit(1)

    def evaluate(self):
        '''
        Run the evaluation loop for the saved PPO agent.
        '''
        for ep in range(self.num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            timestep = 0
            frames = []
            step = 0
            frame_skip = 10  # Adjust frame skip for smoother GIFs

            while not done and timestep < self.max_steps:
                # Capture the default camera view
                if self.render and self.save_gifs and step % frame_skip == 0:
                    width, height, rgb_pixels, _, _ = p.getCameraImage(
                        width=640,
                        height=480,
                        renderer=p.ER_TINY_RENDERER,
                    )
                    frame = np.reshape(rgb_pixels, (height, width, 4))[:, :, :3]  # Keep only RGB channels
                    frame = frame.astype(np.uint8)  # Convert to uint8
                    frames.append(frame)

                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

                # Apply random weights to the model's parameters
                with torch.no_grad():
                    self.randomize_weights()
                    action, _, _ = self.model.act(obs_tensor)
                    action_np = action.squeeze(0).numpy()

                obs, reward, done, _ = self.env.step(action_np)
                episode_reward += reward
                timestep += 1
                step += 1

            self.total_rewards.append(episode_reward)
            self.episode_lengths.append(timestep)

            # Check for success
            success = False
            if timestep >= self.max_steps and not done:
                # Agent survived until max_steps without termination
                self.success_count += 1
                success = True

            # Save per-episode result
            self.episode_results.append({
                'episode': ep + 1,
                'reward': episode_reward,
                'length': timestep,
                'success': success
            })

            # Save GIF
            if self.save_gifs:
                gif_path = os.path.join(self.gifs_folder, f"evaluation_episode_{ep + 1}.gif")
                self.save_gif(frames, gif_path)

            print(f"Episode {ep + 1}/{self.num_episodes} - Reward: {episode_reward:.2f}, Steps: {timestep}, Success: {success}")

        self.log_results()

    def randomize_weights(self):
        """Apply random weights to the model."""
        for param in self.model.parameters():
            param.data = torch.randn_like(param.data)

    def save_gif(self, frames, filename):
        """Save the recorded frames as a GIF."""
        if len(frames) == 0:
            print(f"[WARNING] No frames captured for {filename}")
            return
        try:
            imageio.mimsave(filename, frames, fps=10)
            print(f"[INFO] Saved GIF to {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save GIF: {e}")

    def log_results(self):
        """Log the evaluation results."""
        success_rate = self.success_count / self.num_episodes * 100
        average_reward = np.mean(self.total_rewards)
        average_length = np.mean(self.episode_lengths)

        log_content = (
            f"Evaluation Results ({self.timestamp}):\n"
            f"Number of Episodes: {self.num_episodes}\n"
            f"Success Rate: {success_rate:.2f}%\n"
            f"Average Reward: {average_reward:.2f}\n"
            f"Average Episode Length: {average_length:.2f} steps\n"
            "\nDetailed per-episode results:\n"
        )

        for result in self.episode_results:
            log_content += (
                f"Episode {result['episode']}: "
                f"Reward = {result['reward']:.2f}, "
                f"Length = {result['length']}, "
                f"Success = {result['success']}\n"
            )

        print("\n" + log_content)

        # Save to log file
        log_file = os.path.join(self.exp_folder, "evaluation_results.txt")
        with open(log_file, "w") as f:
            f.write(log_content)
        print(f"[INFO] Logged results to {log_file}")

    def close(self):
        self.env.close()

if __name__ == "__main__":
    evaluator = Evaluator(num_episodes=10, render=True, max_steps=1000, save_gifs=True)
    evaluator.evaluate()
    evaluator.close()
