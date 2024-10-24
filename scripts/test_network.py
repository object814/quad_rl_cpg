import torch
import numpy
from networks.ppo_actor_critic import PPOActorCritic

def test_network():
    """
    Demonstrates a forward and backward pass using the PPOActorCritic network.
    """
    # Define observation and action dimensions based on your environment
    observation_dim = 23
    action_dim = 12
    
    # Initialize the PPO Actor-Critic network
    model = PPOActorCritic(observation_dim, action_dim)
    
    # Create a dummy observation tensor (batch size = 1)
    dummy_observation = torch.randn(1, observation_dim, requires_grad=True)
    
    # Forward pass: sample action, log probabilities, and state value
    actions, log_probs, state_value = model.act(dummy_observation)
    print("\nForward Pass Completed Successfully.")
    print("\nSampled Actions and Their Probabilities:")
    print("{:<5} | {:<15} | {:<10}".format("Dim", "Sampled Action", "Prob"))
    print("-" * 40)
    for dim in range(action_dim):
        log_prob = log_probs[0][dim].detach().numpy()
        action = actions[0][dim].detach().numpy()
        prob = 1 / (1 + numpy.exp(-log_prob))
        action_str = "+1" if action == 1 else "-1"
        print("{:<5} | {:<15} | {:<10.4f}".format(dim+1, action_str, prob))

    print("\nState Value:", state_value.item())
    
    # Backward pass: compute gradients from a dummy loss
    # For example, encourage higher log probabilities and higher state values
    dummy_loss = -torch.mean(log_probs) + torch.mean(state_value)
    print("\nDummy Loss:", dummy_loss.item())
    
    # Backward pass: compute gradients
    dummy_loss.backward()
    print("\nGradients after Backward Pass:")
    print("{:<30} | {:<10}".format("Parameter", "Grad Norm"))
    print("-" * 40)
    for name, param in model.named_parameters():
        if param.grad is not None:
            print("{:<30} | {:<10.4f}".format(name, param.grad.norm().item()))
        else:
            print("{:<30} | {:<10}".format(name, "No Gradient!"))
    
    print("\nBackward pass completed successfully.\n")

if __name__ == "__main__":
    test_network()
