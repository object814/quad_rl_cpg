class RewardNormalizer:
    def __init__(self, alpha=0.99):
        """
        Initialize the RewardNormalizer.

        Args:
            alpha (float): Decay rate for the running mean and variance. Default is 0.99.
        """
        self.mean = 0.0
        self.var = 1.0
        self.alpha = alpha  # Decay rate for moving average

    def normalize(self, reward):
        """
        Normalize the reward.

        Args:
            reward (float): The current reward.

        Returns:
            normalized_reward (float): The normalized reward.
        """
        # Update mean and variance
        self.mean = self.alpha * self.mean + (1 - self.alpha) * reward
        self.var = self.alpha * self.var + (1 - self.alpha) * (reward - self.mean) ** 2
        std = (self.var + 1e-8) ** 0.5  # Avoid division by zero

        # Normalize reward
        return (reward - self.mean) / std
