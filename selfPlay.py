from goSim import GoEnv
from policyValueNet import PolicyValueNet
from policyValueNet import args as nnargs
from MCTS import MCTS
from copy import copy, deepcopy

from config import *
from utils import getNextState, initState, copySimulator, getStringState, augmentExamples
from enums import Colour
import numpy  as np
import timeit
import time, random

class SelfPlay():

    def __init__(self):
        # print(PLAYER_COLOR)
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
        action = None
        while(True):
            states.append(currState)
            players.append(self.currPlayer)
            # print(self.simulator.state.color, self.simulator.player_color)
            start_t = time.time()
            policy = self.mcts.getPolicy(deepcopy(currState), prev_action=action)
            end_t = time.time()
            print('Time elapsed for MCTS policy with {} simulations = {}'.format(
                        NUM_SIMULATIONS,
                        end_t - start_t
                    ))            

            # print("Time elapsed for MCTS policy with")
            policies.append(policy)
            
            if(i > 0 and action == PASS_ACTION):
                _, _, win_reward, _, _, _ = self.simulator.decide_winner()

                # print(win_reward)
                # print(black_colour)
                if(win_reward > 0):
                    action = PASS_ACTION
                else:
                    action = self.sampleAction(policy)

            else:
                action = self.sampleAction(policy)

            # print(self.simulator.state.color, self.simulator.player_color)
            print("Action to take  = ", action)
            obs_t, action, r_t, done, info, cur_score = self.simulator.step(action)
            
            self.simulator.render()
            # print('################# NEW_STATE ##############################')
            if(done):
                # from IPython import embed; embed()

                reward = r_t
                curr_colour = self.currPlayer
                break
            # inverted_obs = invertObs(obs_t)
            nextState = getNextState(deepcopy(currState), obs_t)
            # print('#################HERE##############################')
            # print(currState, nextState)
            self.currPlayer = 3 - self.currPlayer
            self.simulator.set_player_color(self.currPlayer)
            self.mcts.updateSimulator(self.simulator)

            # print(self.currPlayer)
            # print(nextState[16, :, :])
            assert(self.currPlayer == nextState[NUM_FEATURES - 1, 0, 0])
            currState = nextState
            i += 1
            if(i == NUM_MOVES):
                _, _, reward, _, _, board_score = self.simulator.decide_winner()
                
                curr_colour = self.simulator.player_color
                print("CURRENT COLOR -----------------------", curr_colour)
                print("REWARD -----------------------", reward)
                print("FINAL SCORE ------------------", board_score)
                break

                
            print("No. of turns = ", i)
            
        for plr in players:
            if(curr_colour == plr):
                rewards.append(reward)
            else:
                rewards.append(-reward)
        # from IPython import embed; embed()
        
        return states, policies, rewards         

            
selfP = SelfPlay()
all_examples = []
for ep_num in range(NUM_EPISODES):
    print('Episode Count: {}'.format(ep_num))
    states, policies, rewards = selfP.runEpisode()

    # print(getStringState(states[8]))
    aug_states, aug_policies, aug_rewards = augmentExamples(states, policies, rewards)
    training_set = list(zip(aug_states, aug_policies, aug_rewards))
    random.shuffle(training_set)
    selfP.network.train(training_set)
    if(ep_num % CHECKPOINT_COUNTER == 0):
        selfP.network.save_checkpoint(tag=str(ep_num))
        # selfP.network = PolicyValueNet(nnargs)
        # selfP.network.load_checkpoint(tag=str(ep_num))
