import torch
from torch.optim import Adam
from networks.ppo_actor_critic import PPOActorCritic
from envs.gym_env import UnitreeA1Env
from networks.ppo_actor_critic import compute_advantages  # Assuming this function is implemented in a utils module
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class LearningAgent:
    def __init__(self):
        # Hyperparameters
        self.learning_rate = 1e-3
        self.gamma = 0.99
        self.lam = 0.95
        self.epsilon = 0.2
        self.max_timestep = 1000
        self.rollout_steps = 100
        self.num_rollouts = 10
        self.epochs = 10
        self.batch_size = 32
        self.writer = SummaryWriter("runs/ppo_training")
        self.global_step = 0
        
        # Initialize environment and network
        self.env = UnitreeA1Env()
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.model = PPOActorCritic(self.observation_dim, self.action_dim)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Storage for episode-specific data
        self.episode_buffer = []

    def run(self):
        for epoch in range(self.epochs):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            timestep = 0
            
            while not done and timestep < self.max_timestep:
                action, log_prob, state_value = self.model.act(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                next_obs, reward, done, _ = self.env.step(action.squeeze().numpy())
                
                # Record data for training
                self.episode_buffer.append((obs, action, reward, log_prob, state_value))
                obs = next_obs
                episode_reward += reward
                timestep += 1
                self.global_step += 1

                # Train at intervals
                if done or timestep % self.rollout_steps == 0:
                    self.update_network()

            # Log episode rewards
            self.writer.add_scalar("Reward/Total", episode_reward, epoch)
            
            print(f"Episode {epoch + 1}/{self.epochs} completed with reward: {episode_reward}")

        # Close environment and writer
        self.env.close()
        self.writer.close()

    def update_network(self):
        # Unpack and structure the episode buffer as in the test script
        observations, actions, rewards, log_probs, state_values = zip(*self.episode_buffer)

        # Convert lists to tensors in a structured way
        observations = torch.tensor(np.array(observations, dtype=np.float32), dtype=torch.float32)
        actions = [np.array(action, dtype=np.float32).flatten() for action in actions]

        # Now convert to a tensor
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        old_log_probs = torch.tensor(np.array([log_prob.detach().numpy() for log_prob in log_probs], dtype=np.float32), dtype=torch.float32)
        state_values = torch.tensor(np.array([state_value.detach().numpy() for state_value in state_values], dtype=np.float32), dtype=torch.float32)
        
        # Calculate advantages and returns
        advantages = compute_advantages(rewards, state_values, gamma=self.gamma)
        returns = advantages + state_values.numpy()  # PPO typically uses returns = advantages + values

        # Convert advantages and returns to tensors
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Ensure dimensions are consistent for loss calculations
        advantages = advantages.unsqueeze(-1).expand_as(actions)  # Shape advantages to match actions if necessary
        returns = returns.view(-1, 1)  # Match `state_values` shape

        # Train using PPO loss
        log_probs, entropy, predicted_state_values = self.model.evaluate_actions(observations, actions)
        loss, policy_loss, value_loss = self.ppo_loss(log_probs, old_log_probs, advantages, predicted_state_values, returns, entropy)
        
        # Backward pass and optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar("Loss/Total", loss.item(), self.global_step)
        self.writer.add_scalar("Loss/Policy", policy_loss.item(), self.global_step)
        self.writer.add_scalar("Loss/Value", value_loss.item(), self.global_step)


        # Clear the episode buffer after training
        self.episode_buffer = []



    def calculate_returns_and_advantages(self, rewards, state_values):
        # Calculate advantages and returns using Generalized Advantage Estimation (GAE)
        returns = []
        advantages = compute_advantages(rewards, state_values, self.gamma, self.lam)
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns, advantages

    def ppo_loss(self, log_probs, old_log_probs, advantages, values, returns, entropy):
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        value_loss = F.mse_loss(values.squeeze(-1), returns)
        entropy_bonus = -0.01 * entropy.mean()
        total_loss = policy_loss + 0.5 * value_loss + entropy_bonus
        return total_loss, policy_loss, value_loss

if __name__ == "__main__":
    agent = LearningAgent()
    agent.run()
