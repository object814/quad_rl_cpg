
# Group  21 - quad_rl_cpg  
ME5418 Project - Group 21: Quadruped Reinforcement Learning with Central Pattern Generators (CPG)

## Environment Setup

1. **Create Conda Environment**  
   Open your Linux terminal and navigate to the project foler "quad_rl_cpg-main", run the following command to create the environment:
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate Conda Environment**  
   Once the environment is created, activate it:
   ```bash
   conda activate group-21-rl-cpg
   ```
---
## Neural Network - Running Tests

### 1. **Single observation: forward pass + backward pass (`scripts/test_network.py`)**
   Samples a single random observation. Performs a forward pass to compute sampled actions, log probabilities, and state values. It then calculates a dummy loss and computes the gradients through a backward pass.
   
   Run the test:
   ```bash
   python3 -m scripts.test_network
   ```

### 2. **Simple Network training (`scripts/test_network_training.py`)**
   Initial test to validate the integration between the agent living in our gym environment and our neural network model. Agent goes through 10 rollouts with 100 timesteps each. It then trains the network with a dummy loss function for demonstration purposes. Graphical rendering is disabled by default, but can be easily enabled by setting `env = UnitreeA1Env(render=True)`.

   Run the test:
   ```bash
   python3 -m scripts.test_network_training
   ```


### 3. **Forward pass/Back propagation test with PPO losses (`scripts/test_network_training_with_PPOloss.py`)**  
   Tests the agent living in our gym environment and neural network training for PPO. Agent goes through 10 rollouts with 100 timestpes each, with a randomized action selection. Then back propagates through the collected data 100 epochs with a batch size of 32 calculating the PPO loses. This test script prints the gradient norm of each neural net layer per batch and total loss per epoch. Graphical rendering is disabled by default, but can be easily enabled by setting `env = UnitreeA1Env(render=True)`.

   
   To run the forward/back propagation test with PPO loss calculation:
   ```bash
   python3 -m scripts.test_network_training_with_PPOloss
   ```


---
## GYM Environment - Running Tests

### 1. **Full Test**  
   This simulates the quadruped robot with a weight block added. The foot positions are controlled by the CPG and visualized for each leg with a scripted trajectory.
   ```bash
   python3 -m scripts.test_all
   ```

### 2. **Full Test with Robot Fixed in Air**  
   Similar to the Full Test, but the robot is fixed in the air for clearer visualization of the leg movements.
   ```bash
   python3 -m scripts.test_fixed
   ```

### 3. **Full Reward Test**  
   Simulates the robot with a weight block added, and each reward/penalty is plotted separately.  

   #### Separate Reward/Penalty Tests:
   1. **Foot Slip Penalty**: Drag the robot on the ground using your cursor. The foot slip penalty will reflect the simulated physics.
   2. **Roll/Pitch Penalty**: Gently rotate the robot in the roll/pitch direction with your cursor. The roll/pitch penalty will reflect the simulated physics.
   3. **Forward Reward**: Drag the robot away from its starting position using your cursor. The forward reward will reflect the simulated physics.
   4. **Payload Drop Penalty**: Flip the robot violently. Once the threshold is reached, the episode will end, and the payload drop penalty will be applied.
   
   To run the Full Reward Test:
   ```bash
   python3 -m scripts.test_rewards
   ```

---

## Configuration Options

The test configurations can be adjusted in the `envs/config.py` file:

1. **Discount Factor**  
   Modify the total reward discount per timestep.
   
2. **Reward/Penalty Weights**  
   Adjust the weights for rewards and penalties.

3. **Penalty Thresholds**  
   Set the thresholds for applying penalties (e.g., foot slip, roll/pitch, payload drop).

4. **Payload Parameters**  
   - `add_weight`: Toggle whether to add weight.  
   - `box_xyz_dim`: Adjust the payload dimensions.  
   - `box_mass`: Modify the payload mass.  
   - `attach_link_id`: Specify the payload position.

---

## Further Documentation

For a more detailed explanation of the code and methods, please refer to the project report.

