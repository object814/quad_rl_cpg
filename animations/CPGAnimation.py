import matplotlib.pyplot as plt

class CPGAnimation:
    def __init__(self, leg_names):
        """
        Initialize the CPG Animation class.

        Args:
            leg_names (list): List of leg names ["FR", "FL", "RR", "RL"].
        """
        self.leg_names = leg_names

        # Create a figure and subplots for each leg
        self.fig, self.axs = plt.subplots(2, 2, figsize=(8, 8))
        self.axs = self.axs.flatten()

        # Dictionary to store plot data for each leg
        self.foot_positions = {leg: {"x": [], "z": []} for leg in self.leg_names}

        # Initialize plots for each leg
        self.lines = {}
        for i, leg in enumerate(self.leg_names):
            self.axs[i].set_title(f"Leg {leg}")
            self.axs[i].set_xlim(-0.2, 0.2)  # Set x limits (adjust as needed)
            self.axs[i].set_ylim(-0.4, 0.1)  # Set z limits (adjust as needed)
            self.axs[i].set_xlabel("x")
            self.axs[i].set_ylabel("z")

            # Initialize an empty line plot for each leg
            (line,) = self.axs[i].plot([], [], 'bo')  # 'bo' is for blue dots
            self.lines[leg] = line

        # Display the figure
        #plt.ion()  # Turn on interactive mode
        #plt.show()



    def animate(self, target_positions):
        """
        Update the animation with new foot positions for each leg.

        Args:
            target_positions (dict): Dictionary containing the foot positions
                                     for each leg, e.g., {'FR': (x, z), ...}.
        """
        for leg, (x, z) in target_positions.items():
            # Append new foot position to the stored data
            self.foot_positions[leg]["x"].append(x)
            self.foot_positions[leg]["z"].append(z)

            # Update the plot data
            self.lines[leg].set_data(self.foot_positions[leg]["x"], self.foot_positions[leg]["z"])

        # Redraw the figure
        plt.draw()
        #plt.pause(0.001)  # Pause for a short interval to allow for non-blocking updates

        plt.show(block=False)  # Non-blocking
        plt.pause(0.001)  # Allow GUI to process events
        print("HEYYYYY")
