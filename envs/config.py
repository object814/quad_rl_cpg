"""
Parameters for the environment gym_env.py
"""

# Environment settings
dt = 0.01                                   # Time step for updating the CPG states
render = True                               # Whether to render the simulation
debug = False                               # Whether to enable debug mode
animate_cpg = True                          # Whether to animate the CPGs
discount_factor = 0.99                      # Discount factor per timestep for total rewards
fix_base = False                            # Whether to fix the base of the robot
reset_position = [0, 0, 0.5]                # Initial position of the robot


# Reward / penalty weights
forward_progress_reward_weight = 1.0        # Reward weight for forward progress, distance travelled from the starting position
roll_stability_penalty_weight = 1.0         # Penalty weight for roll stability
pitch_stability_penalty_weight = 1.0        # Penalty weight for pitch stability
payload_drop_penalty_weight = 1.0           # Penalty weight for payload drop
foot_slip_penalty_weight = 1.0              # Penalty weight for foot slip


# Penalty thresholds
max_roll = 0.6                              # in radians, threashold for simulating weight dropped
min_roll = -0.6                             # in radians, threashold for simulating weight dropped
max_pitch = 0.6                             # in radians, threashold for simulating weight dropped
min_pitch = -0.6                            # in radians, threashold for simulating weight dropped
roll_pitch_range_without_penalty = 0.1745   #  in radians, range for robot base orientation in roll and pitch which won't be penalized
roll_pitch_penalty_threashold = roll_pitch_range_without_penalty ** 2
