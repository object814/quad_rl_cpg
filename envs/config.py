"""
Parameters for the environment gym_env.py
"""

# Environment settings
dt = 0.01                                   # Time step for updating the CPG states
# render = True                               # Whether to render the simulation
# debug = False                               # Whether to enable debug mode
# animate_cpg = True                          # Whether to animate the CPGs
discount_factor = 1.0                      # Discount factor per timestep for total rewards
# fixed_base = False                          # Whether to fix the base of the robot
reset_position = [0, 0, 0.4]                # Initial position of the robot


# Reward / penalty weights
forward_progress_reward_weight = 100.0        # Reward weight for forward progress, distance travelled from the starting position
roll_stability_penalty_weight = 1.0         # Penalty weight for roll stability
pitch_stability_penalty_weight = 1.0        # Penalty weight for pitch stability
payload_drop_penalty_weight = 1.0           # Penalty weight for payload drop
foot_slip_penalty_weight = 0              # Penalty weight for foot slip


# Penalty thresholds
max_roll = 0.7                              # in radians, threashold for simulating weight dropped
min_roll = -0.7                             # in radians, threashold for simulating weight dropped
max_pitch = 0.7                             # in radians, threashold for simulating weight dropped
min_pitch = -0.7                            # in radians, threashold for simulating weight dropped
roll_pitch_range_without_penalty = 0.5235   #  in radians, range for robot base orientation in roll and pitch which won't be penalized
roll_pitch_penalty_threashold = roll_pitch_range_without_penalty ** 2

 # Added box params
add_weight = True
box_x_dim = 0.05
box_y_dim = 0.05
box_z_dim = 0.05
box_mass = 1.0
attach_link_id = 6 #Try 2 -> right forward sholder | 6 -> left forward sholder | 10 -> right back sholder | 14 -> left back sholder