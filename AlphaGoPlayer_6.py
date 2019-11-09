import numpy as np
from policyValueNet import PolicyValueNet

class AlphaGoPlayer():
    def __init__(self, init_state, seed, player_color, board_size=13, timesteps=8):
        self.init_state = init_state
        self.seed = seed
        self.player_color = player_color
        self.timesteps = timesteps
        self.policy_value_net = PolicyValueNet(board_size, timesteps)

    # Simulator passes observation as current state
    def get_action(self, cur_state, opponent_action):
        # Do Coolstuff using cur_state
        # Check illegal Move
        print('-------------------------------------')
        print('opponent_action: ' + str(opponent_action))
        print('-------------------------------------')
        action = np.random.randint(0, 169)
        return action

    def get_action(self, cur_state)

    def mcts(self, cur_state, model):
        return policy

    