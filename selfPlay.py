from goSim import GoEnv
from policyValueNet import PolicyValueNet
from policyValueNet import args as nnargs
from MCTS import MCTS
from copy import copy, deepcopy

from config import *
from utils import getNextState, initState, copySimulator
from enums import Colour
import numpy  as np
import timeit
import time

class SelfPlay():

    def __init__(self):
        print(PLAYER_COLOR)
        # self.simulator = GoEnv(player_color=PLAYER_COLOR, observation_type='image3c', illegal_move_mode="raise", board_size=BOARD_SIZE, komi=KOMI_VALUE)
        self.network = PolicyValueNet(nnargs)

    def sampleAction(self, policy):

        action = np.random.choice(NUM_ACTIONS, p=policy)
        assert (action >= 0 and action <= NUM_ACTIONS - 1), "Valid action not selected"

        return action


    def runEpisode(self):
        self.simulator = GoEnv(player_color=PLAYER_COLOR, observation_type='image3c', illegal_move_mode="raise", board_size=BOARD_SIZE, komi=KOMI_VALUE)
        self.simulator.reset()
        # print(self.simulator.state.color)
        # print("Ff~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # sim_copy = 

        # print(self.simulator.state.color)
        
        # sim_copy.state.color = Colour.WHITE.value
        # print(self.simulator.state.color)
        # print(timeit.timeit(my_function, number=1))
        self.mcts = MCTS(self.network, copySimulator(self.simulator))
        states = []
        policies = []
        rewards = []
        players = []
        currState = initState()
        self.currPlayer = Colour.BLACK.value
        i = 0
        while(True):
            states.append(currState)
            players.append(self.currPlayer)
            # print(self.simulator.state.color, self.simulator.player_color)
            start_t = time.time()
            policy = self.mcts.getPolicy(deepcopy(currState))
            end_t = time.time()
            print('Time elapsed for MCTS policy with {} simulations = {}'.format(
                        NUM_SIMULATIONS,
                        end_t - start_t
                    ))            

            # print("Time elapsed for MCTS policy with")
            policies.append(policy)
        
            action = self.sampleAction(policy)

            # print(self.simulator.state.color, self.simulator.player_color)
            self.simulator.set_player_color(self.currPlayer)
            obs_t, action, r_t, done, info, cur_score = self.simulator.step(action)
            print("Action taken  = ", action)
            self.simulator.render()
            # print('################# NEW_STATE ##############################')
            if(done):
                reward = r_t
                winner_colour = self.currPlayer
                break
            # inverted_obs = invertObs(obs_t)
            nextState = getNextState(deepcopy(currState), obs_t)
            # print('#################HERE##############################')
            # print(currState, nextState)
            self.currPlayer = 3 - self.currPlayer
            # print(self.currPlayer)
            # print(nextState[16, :, :])
            assert(self.currPlayer == nextState[16, 0, 0])
            currState = nextState
            i += 1
            
        for plr in players:
            if(winner_colour == plr):
                rewards.append(reward)
            else:
                rewards.append(-reward)
        return states, policies, rewards         

            
selfP = SelfPlay()
all_examples = []
for ep_num in range(NUM_EPISODES):
    print('Episode Count: {}'.format(ep_num))
    states, policies, rewards = selfP.runEpisode()
    all_examples.append((states, policies, rewards))
    # from IPython import embed; embed()
    selfP.network.train((np.array(states), np.array(policies), np.array(rewards)))
    selfP.network.save_checkpoint()
    # selfP.network.load_checkpoint()
