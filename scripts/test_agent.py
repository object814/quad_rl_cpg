import torch
from torch.optim import Adam
from networks.ppo_actor_critic import PPOActorCritic
from envs.gym_env import UnitreeA1Env
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import imageio
import os
import pybullet as p

class LearningAgent:
    def __init__(self):
        # Hyperparameters
        # PPO hyperparameters
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.lam = 0.95
        self.epsilon = 0.2
        self.entropy_coef = 0.01
        # Training hyperparameters
        self.max_timestep = 4000
        self.rollout_steps = 100
        self.epochs = 10 # Number of epochs per update
        self.episode_num = 1000 # Number of episodes to run
        self.batch_size = 32
        self.writer = SummaryWriter("runs/ppo_training")
        self.global_step = 0
        self.payload_drop_count = 0
        # Initialize environment and network
        self.env = UnitreeA1Env(render=False)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.model = PPOActorCritic(self.observation_dim, self.action_dim)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Storage for episode-specific data
        self.episode_buffer = []

        # Initialize run logging folders
        # check if networks/saved_gifs exists
        if not os.path.exists("networks/saved_gifs"):
            os.makedirs("networks/saved_gifs")
        # check if networks/saved_model exists
        if not os.path.exists("networks/saved_model"):
            os.makedirs("networks/saved_model")

    def run(self):
        '''
        Run the training loop for the PPO agent.
        '''
        for ep in range(self.episode_num):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            timestep = 0
            
            while not done and timestep < self.max_timestep:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                # Sample an action from the policy
                with torch.no_grad():
                    action, log_prob, state_value = self.model.act(obs_tensor)
                    action_np = action.squeeze(0).numpy()

                next_obs, reward, done, _ = self.env.step(action_np)
                
                # Estimate the value of the next state
                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    _, _, next_state_value = self.model.act(next_obs_tensor)
                
                # Record data for training
                self.episode_buffer.append({
                    'obs': obs,
                    'action': action.squeeze(0),
                    'reward': reward,
                    'log_prob': log_prob,
                    'state_value': state_value.squeeze(0),
                    'next_state_value': next_state_value.squeeze(0),
                    'done': done
                })

                obs = next_obs
                episode_reward += reward
                timestep += 1
                self.global_step += 1
                if self.env.payload_dropped == True:
                    self.payload_drop_count += 1

                # Train at intervals
                if done or timestep % self.rollout_steps == 0:
                    self.update_network()
                    self.episode_buffer = []  # Clear buffer after update

            # Log episode rewards
            self.writer.add_scalar("Reward/Total", episode_reward, ep)
            
            print(f"Episode {ep + 1}/{self.episode_num} completed with reward: {episode_reward}")

            if (ep + 1) % 10 == 0:
                self.record_gif(f"networks/saved_gifs/training_episode_{ep + 1}.gif")

        # Close environment and writer
        print(f"Payload dropped {self.payload_drop_count}/{self.episode_num} times")
        self.save_model()
        self.env.close()
        self.writer.close()
    
    def record_gif(self, filename="networks/saved_gifs/training_episode.gif"):
        """
        Record an episode and save it as a GIF using PyBullet's camera rendering.

        Args:
            filename (str): Path to save the GIF file.
        """
        frames = []
        obs = self.env.reset()
        done = False
        step = 0
        frame_skip = 10

        while not done:
            # Capture the default camera view
            if step % frame_skip == 0:
                width, height, rgb_pixels, _, _ = p.getCameraImage(
                    width=640,
                    height=480,
                    renderer=p.ER_TINY_RENDERER,
                )
                frame = np.reshape(rgb_pixels, (height, width, 4))[:, :, :3]  # Keep only RGB channels
                frame = frame.astype(np.uint8)  # Convert to uint8
                frames.append(frame)

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = self.model.act(obs_tensor)
            obs, _, done, _ = self.env.step(action.squeeze(0).numpy())
            step += 1

        # Save the GIF
        try:
            imageio.mimsave(filename, frames, fps=10)
            print(f"[INFO] GIF saved to {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save GIF: {e}")

    def save_model(self, filename="networks/saved_model/ppo_model.pth"):
        '''
        Save the model parameters to a file.

        Args:
            - filename (str): Name of the file to save the model to.
        '''
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")

    def update_network(self):
        '''
        Update the network using the collected episode data.
        '''
        MIN_EPISODE_LENGTH = 10  # Set a minimum length for updates
        if len(self.episode_buffer) < MIN_EPISODE_LENGTH:
            print("Episode buffer too small to update network.")
            return  # Skip the update
        
        # Prepare data from episode buffer
        observations = torch.stack([torch.tensor(item['obs'], dtype=torch.float32) for item in self.episode_buffer])
        actions = torch.stack([item['action'] for item in self.episode_buffer])
        rewards = torch.tensor([item['reward'] for item in self.episode_buffer], dtype=torch.float32)
        log_probs_old = torch.tensor([item['log_prob'] for item in self.episode_buffer], dtype=torch.float32)
        state_values = torch.stack([item['state_value'] for item in self.episode_buffer]).squeeze(-1)
        next_state_values = torch.stack([item['next_state_value'] for item in self.episode_buffer]).squeeze(-1)
        dones = torch.tensor([item['done'] for item in self.episode_buffer], dtype=torch.float32)

        # Compute advantages and returns
        advantages = self.compute_advantages(rewards, state_values, next_state_values, dones)
        returns = advantages + state_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple epochs over the data
        dataset_size = observations.size(0)
        indices = np.arange(dataset_size)
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions using current policy
                log_probs, entropy, predicted_state_values = self.model.evaluate_actions(batch_obs, batch_actions)

                # Compute PPO loss
                loss, policy_loss, value_loss = self.ppo_loss(
                    log_probs,
                    batch_log_probs_old,
                    batch_advantages,
                    predicted_state_values.squeeze(-1),
                    batch_returns,
                    entropy
                )

                # Backward pass and optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar("Loss/Total", loss.item(), self.global_step)
                self.writer.add_scalar("Loss/Policy", policy_loss.item(), self.global_step)
                self.writer.add_scalar("Loss/Value", value_loss.item(), self.global_step)

    def compute_advantages(self, rewards, state_values, next_state_values, dones):
        '''
        Advantage computation using Generalized Advantage Estimation (GAE).

        Args:
            - rewards (Tensor): Rewards observed during the episode.
            - state_values (Tensor): Predicted state values.
            - next_state_values (Tensor): Predicted state values for the next state.
            - dones (Tensor): Termination flags for each timestep.

        Returns:
            - advantages (Tensor): Computed advantages.
        '''
        deltas = rewards + self.gamma * next_state_values * (1 - dones) - state_values
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages

    def ppo_loss(self, log_probs, log_probs_old, advantages, values, returns, entropy):
        '''
        PPO loss function with clipped surrogate objective.

        Args:
            - log_probs (Tensor): Log probabilities of actions taken.
            - log_probs_old (Tensor): Log probabilities of actions taken in the previous policy.
            - advantages (Tensor): Computed advantages.
            - values (Tensor): Predicted state values.
            - returns (Tensor): Computed returns.
            - entropy (Tensor): Entropy of the policy.

        Returns:
            - total_loss (Tensor): Total loss for training.
            - policy_loss (Tensor): Policy loss component.
            - value_loss (Tensor): Value function loss component
        '''
        # Compute ratio
        ratio = torch.exp(log_probs - log_probs_old)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value function loss
        value_loss = F.mse_loss(values, returns)
        value_loss = 0.1 * value_loss

        # Entropy bonus
        entropy_bonus = entropy.mean()

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy_bonus

        return total_loss, policy_loss, value_loss

if __name__ == "__main__":
    agent = LearningAgent()
    agent.run()


    