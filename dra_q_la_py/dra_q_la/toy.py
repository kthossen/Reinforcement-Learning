
import numpy as np
from .mdp import TabularMDP

def load_toy_mdp():
    """A tiny 3-state, 2-action MDP for smoke testing."""
    S, A = 3, 2
    P = np.zeros((S, A, S), dtype=float)
    # Action 0: move to next state with prob 1.0
    for s in range(S):
        P[s, 0, (s+1)%S] = 1.0
    # Action 1: stay with some randomness
    for s in range(S):
        P[s, 1, s] = 0.7
        P[s, 1, (s+1)%S] = 0.3
    R = np.array([[0, -0.1],
                  [0, 0.0],
                  [1.0, 0.2]], dtype=float)
    gamma = 0.95
    return TabularMDP(P, R, gamma=gamma, initial_state=0)
