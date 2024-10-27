import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PPOActorCritic(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_size=256):
        """
        Initialize the PPO Actor-Critic network.

        Args:
            observation_dim (int): Dimension of the input observation vector.
            action_dim (int): Action dimension.
            hidden_size (int): Hidden layer size.

        For now, we have a simple network structure served as a baseline to start with, network structure:
        - Shared base layers:
            - Linear layer 1
            - ReLU activation
            - Linear layer 2
            - ReLU activation
        - Actor head:
            - Linear layer
            - ReLU activation
            - Output layer
            - Sigmoid activation
        - Critic head:
            - Linear layer
            - ReLU activation
            - Output layer
        """
        super(PPOActorCritic, self).__init__()
        
        # Shared fully connected layers
        self.shared_fc1 = nn.Linear(observation_dim, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head
        self.actor_fc = nn.Linear(hidden_size, hidden_size)
        self.actor_out = nn.Linear(hidden_size, action_dim)
        
        # Critic head
        self.critic_fc = nn.Linear(hidden_size, hidden_size)
        self.critic_out = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.
        """
        for layer in [self.shared_fc1, self.shared_fc2, 
                      self.actor_fc, self.actor_out, 
                      self.critic_fc, self.critic_out]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor (env observation).

        Returns:
            actor_logits (torch.Tensor): Probabilities for each action dimension.
            state_value (torch.Tensor): Estimated value of the current state.
        """
        # Shared layers
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        
        # Actor forward pass
        actor = F.relu(self.actor_fc(x))
        actor_logits = torch.sigmoid(self.actor_out(actor))  # Probabilities for each action dimension, 1 represents increase, 0 represents decrease
        
        # Critic forward pass
        critic = F.relu(self.critic_fc(x))
        state_value = self.critic_out(critic)  # State value (reward prediction)
        
        return actor_logits, state_value
    
    def act(self, observation):
        """
        Given an observation, sample an action and compute log probabilities.

        Args:
            observation (torch.Tensor): Input observation tensor.

        Returns:
            actions (torch.Tensor): Sampled actions mapped to {-1, +1}.
            log_probs (torch.Tensor): Log probabilities of the sampled actions.
            state_value (torch.Tensor): Estimated value of the current state.
        """
        actor_logits, state_value = self.forward(observation)
        # Sample actions: for each action dimension, decide -1 or +1 based on probability
        probs = actor_logits
        dist = torch.distributions.Bernoulli(probs)
        action_sample = dist.sample()
        log_probs = dist.log_prob(action_sample)
        
        # Convert sampled actions from {0,1} to {-1, +1}
        actions = 2 * action_sample - 1
        
        return actions, log_probs, state_value
    
    def evaluate_actions(self, observations, actions):
        """
        Evaluate actions for given observations. Returns log probabilities and state values.

        Args:
            observations (torch.Tensor): Batch of observation tensors.
            actions (torch.Tensor): Batch of actions.

        Returns:
            log_probs (torch.Tensor): Log probabilities of the actions.
            entropy (torch.Tensor): Entropy of the action distributions.
            state_values (torch.Tensor): Estimated values of the states.
        """
        actor_logits, state_values = self.forward(observations) # get actor logits and state values using current network
        probs = actor_logits # probabilities for each action dimension
        dist = torch.distributions.Bernoulli(probs) # bernoulli distribution for each action dimension based on current probabilities
        # Convert actions from {-1, +1} to {0, 1} for probability computation
        actions_binary = (actions + 1) / 2
        log_probs = dist.log_prob(actions_binary)
        entropy = dist.entropy()
        return log_probs, entropy, state_values


def compute_advantages(rewards, state_values, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = state_values[t]
        else:
            next_value = state_values[t + 1]
        delta = rewards[t] + gamma * next_value - state_values[t]
        advantages[t] = last_gae_lam = delta + gamma * lam * last_gae_lam
    return advantages

def ppo_loss(log_probs, old_log_probs, advantages, values, returns, entropy, epsilon=0.2):

    # Calculate ratio
    ratio = torch.exp(log_probs - old_log_probs)
    
    
    # Clipped objective
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    
    # Value loss
    value_loss = F.mse_loss(values, returns)

    # Total loss
    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()  # 0.5 is a scaling factor for value loss
    return total_loss