import numpy as np
from typing import NamedTuple, List, Deque, Dict


class Experience(NamedTuple):
    """
    An experience contains the data of one Agent transition.
    - Observation
    - Action
    - Reward
    - Done flag
    - Next Observation
    """

    obs: np.ndarray
    action: np.ndarray  # Check on the actual actions required by Unity
    reward: float
    done: bool
    next_obs: np.ndarray


# Pretty cool, these aren't instantiations but like composite typing
# A Trajectory is an ordered sequence of Experiences
Trajectory = List[Experience]  # small enough so I am not afraid of exceeding

# A Buffer is an unordered list of Experiences from multiple Trajectories
Buffer = Deque[Experience]
