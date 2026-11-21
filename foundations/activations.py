import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Maps to (0, 1), used for binary classification outputs.
        Vanishing gradients make it a poor choice for hidden layers in deep networks
        - Gradients shrink to nothing and deep layers stop learning 
        """

        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)

        # Overflow fix for large z vals
        z_clipped = np.clip(z, -500, 500)
        return np.round(1 / (1 + np.exp(-z_clipped)), 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Default hidden-layer activation -> gradient is 0 or 1.
        Avoids vanishing gradient problem for +ve inputs
        """
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        return np.maximum(0, z)