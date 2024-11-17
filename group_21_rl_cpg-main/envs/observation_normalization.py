import numpy as np

class ObservationNormalizer:
    def __init__(self, obs_dim, alpha=0.99):
        """
        Initialize the Observation Normalizer.

        Args:
            obs_dim (int): Dimension of the observation vector.
            alpha (float): Decay rate for running mean and variance. Default is 0.99.
        """
        self.mean = np.zeros(obs_dim)
        self.var = np.ones(obs_dim)
        self.alpha = alpha  # Decay rate

    def normalize(self, observation):
        """
        Normalize the observation.

        Args:
            observation (np.array): The observation to normalize.

        Returns:
            normalized_observation (np.array): The normalized observation.
        """
        # Update mean and variance
        self.mean = self.alpha * self.mean + (1 - self.alpha) * observation
        self.var = self.alpha * self.var + (1 - self.alpha) * (observation - self.mean) ** 2
        std = np.sqrt(self.var + 1e-8)  # Avoid division by zero

        # Normalize
        return (observation - self.mean) / std
