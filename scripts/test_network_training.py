import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from networks.ppo_actor_critic import PPOActorCritic
from envs.gym_env import UnitreeA1Env

def generate_rollout_data(env, model, rollout_steps=100, num_rollouts=10):
    """
    Generate rollout data by executing random actions in the environment.
    
    Args:
        env (UnitreeA1Env): The gym environment.
        model (PPOActorCritic): The PPO Actor-Critic model.
        rollout_steps (int): Maximum steps per rollout.
        num_rollouts (int): Number of rollouts.

    Returns:
        Tuple[torch.Tensor]: Tensors of observations, actions, state values, and log probabilities.
    """
    print("\nStarting rollout data generation...\n")
    observations, actions, state_values, log_probs = [], [], [], []

    for rollout in range(num_rollouts):
        print(f"Generating rollout {rollout + 1}/{num_rollouts}...")
        obs = env.reset()
        
        for step in range(rollout_steps):
            # Convert observation to tensor and get model outputs
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, log_prob, state_value = model.act(obs_tensor)

            # Store data
            observations.append(obs)
            actions.append(action.squeeze().numpy())
            state_values.append(state_value.item())
            log_probs.append(log_prob.squeeze().detach().numpy())

            # Execute a random action and get the next observation
            random_action = np.random.choice([-1, 1], size=(env.action_space.shape[0],))
            obs, reward, done, _ = env.step(random_action)

            # Log progress within rollout
            print(f"  Step {step + 1}/{rollout_steps}", end="\r")

            if done:
                print("\n[INFO] Rollout ended early due to termination condition.\n")
                break
        print(f"Completed rollout {rollout + 1}/{num_rollouts} with {step + 1} steps.\n")

    observations = torch.tensor(np.array(observations, dtype=np.float32), dtype=torch.float32)
    actions = torch.tensor(np.array(actions, dtype=np.float32), dtype=torch.float32)
    state_values = torch.tensor(np.array(state_values, dtype=np.float32), dtype=torch.float32)
    log_probs = torch.tensor(np.array(log_probs, dtype=np.float32), dtype=torch.float32)

    print("[INFO] Rollout data generation complete.\n")
    return observations, actions, state_values, log_probs

def train_dummy_network(model, data_loader, epochs=100):
    """
    Train the PPO Actor-Critic model with a dummy loss function for demonstration purposes.
    
    Args:
        model (PPOActorCritic): The PPO Actor-Critic model.
        data_loader (DataLoader): DataLoader for batch training.
        epochs (int): Number of training epochs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("\nStarting dummy training...\n")
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            observations, actions, old_log_probs, state_values = batch

            # Forward pass through the network
            log_probs, _, predicted_state_values = model.evaluate_actions(observations, actions)

            # Dummy loss function (for demonstration purposes)
            dummy_loss = torch.mean((predicted_state_values - state_values) ** 2) - torch.mean(log_probs)

            # Backward pass and optimization step
            optimizer.zero_grad()
            dummy_loss.backward()
            optimizer.step()

            total_loss += dummy_loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    print("\n[INFO] Dummy training completed.\n")

def main():
    # Initialize environment and model
    env = UnitreeA1Env(render=False)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = PPOActorCritic(observation_dim, action_dim)

    # Generate rollouts
    observations, actions, state_values, log_probs = generate_rollout_data(env, model)

    # Prepare DataLoader
    dataset = TensorDataset(observations, actions, log_probs, state_values)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train the network with dummy loss
    train_dummy_network(model, data_loader)

    env.close()

if __name__ == "__main__":
    main()
