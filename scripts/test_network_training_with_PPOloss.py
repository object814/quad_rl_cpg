import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from networks.ppo_actor_critic import PPOActorCritic, ppo_loss, compute_advantages
from envs.gym_env import UnitreeA1Env


def generate_rollout_data(env, model, rollout_steps=100, num_rollouts=10, gamma=0.99):
    print("\nStarting rollout data generation...\n")
    observations, actions, state_values, log_probs, rewards, dones = [], [], [], [], [], []

    for rollout in range(num_rollouts):
        print(f"Generating rollout {rollout + 1}/{num_rollouts}...")
        obs = env.reset()
        episode_rewards = []

        for step in range(rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, log_prob, state_value = model.act(obs_tensor)

            observations.append(obs)
            actions.append(action.squeeze().numpy())
            state_values.append(state_value.item())
            log_probs.append(log_prob.squeeze().detach().numpy())

            random_action = np.random.choice([-1, 1], size=(env.action_space.shape[0],))

            next_obs, reward, done, _ = env.step(random_action)
            episode_rewards.append(reward)
            obs = next_obs
            rewards.append(reward)
            dones.append(done)

            if done:
                print("\n[INFO] Rollout ended early due to termination condition.\n")
                break
        print(f"Completed rollout {rollout + 1}/{num_rollouts}.\n")

    observations = torch.tensor(np.array(observations, dtype=np.float32), dtype=torch.float32)
    actions = torch.tensor(np.array(actions, dtype=np.float32), dtype=torch.float32)
    state_values = torch.tensor(np.array(state_values, dtype=np.float32), dtype=torch.float32)
    #print(f"State values shape: {state_values.shape}") # For debugging
    log_probs = torch.tensor(np.array(log_probs, dtype=np.float32), dtype=torch.float32)
    rewards = torch.tensor(np.array(rewards, dtype=np.float32), dtype=torch.float32)
    
    # Compute advantages and returns
    advantages = compute_advantages(rewards, state_values, gamma=gamma)
    #print(f"Advantages shape: {advantages.shape}") # For debugging
    returns = advantages + state_values.numpy()
    #print(f"Returns shape: {returns.shape}") # For debugging

    # Convert advantages and returns to tensors
    advantages = torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1).expand_as(actions)
    returns = torch.tensor(returns, dtype=torch.float32)

    #print("[INFO] Rollout data generation complete.\n")
    return observations, actions, log_probs, state_values, advantages, returns


def train_ppo_network(model, data_loader, epochs=100, epsilon=0.2):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("\nStarting PPO training...\n")
    
    for epoch in range(epochs):
        total_loss = 0
        batch_idx = 0
        for batch in data_loader:
            observations, actions, old_log_probs, state_values, advantages, returns = batch
            batch_idx += 1

            # Forward pass through the network
            log_probs, entropy, predicted_state_values = model.evaluate_actions(observations, actions)
            #print(f"predicted_state_values shape: {predicted_state_values.shape}") # For debugging


            # Compute PPO loss
            loss = ppo_loss(log_probs, old_log_probs, advantages, predicted_state_values, returns, entropy ,epsilon)

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()

            print(f"Epoch {epoch + 1} | Batch {batch_idx}")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Layer: {name} | Gradient Norm: {param.grad.norm()}")

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    print("\n[INFO] PPO training completed.\n")

def main():
    # Initialize environment and model
    env = UnitreeA1Env(render=False)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = PPOActorCritic(observation_dim, action_dim)

    # Generate rollouts
    observations, actions, old_log_probs, state_values, advantages, returns = generate_rollout_data(env, model)

    # Prepare DataLoader
    dataset = TensorDataset(observations, actions, old_log_probs, state_values, advantages, returns)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train the network with PPO loss
    train_ppo_network(model, data_loader)

    env.close()

if __name__ == "__main__":
    main()
