from goSim import GoEnv
from policyValueNet import PolicyValueNet
from policyValueNet import args as nnargs
from MCTS import MCTS

from config import *
from utils import getNextState
from enums import Colour
import numpy  as np
import random

class SelfPlay():

    def __init__(self):
        print(PLAYER_COLOR)
        self.simulator = GoEnv(player_color=PLAYER_COLOR, observation_type='image3c', illegal_move_mode="raise", board_size=BOARD_SIZE, komi=KOMI_VALUE)
        self.network = PolicyValueNet(nnargs)

    def sampleAction(self, policy):

        action = np.random.choice(NUM_ACTIONS, p=policy)
        assert (action >= 0 and action <= NUM_ACTIONS - 1), "Valid action not selected"

        return action

    def get_dist(self):
    	x = [random.random() for _ in range(NUM_ACTIONS)]
    	y = sum(x)
    	x = [x[i]/y for i in range(len(x))]
    	return x

    def runEpisode(self):
        states = []
        policies = []
        rewards = []
        players = []
        self.simulator.reset()
        currState = np.zeros((NUM_FEATURES, BOARD_SIZE, BOARD_SIZE))
        self.currPlayer = Colour.BLACK.value
        i = 0
        while (True):
        	player_color = (i%2) + 1
        	self.simulator.set_player_color(player_color)
        	states.append(currState)
        	players.append(self.currPlayer)
        	policy = self.get_dist()
        	action = self.sampleAction(policy)
        	policies.append(np.array(policy))
        	obs_t, action, r_t, done, info, cur_score = self.simulator.step(action)
        	if(done):
        		reward = r_t
        		winner_colour = self.currPlayer
        		break
        	nextState = getNextState(currState, obs_t)
        	self.currPlayer = not self.currPlayer
        	assert(self.currPlayer == nextState[16, 0, 0])
        	currState = nextState
        	i += 1

        for plr in players:
            if(winner_colour == plr):
                rewards.append(reward)
            else:
                 rewards.append(-reward)
        return states, policies, rewards

    def train_step(self):
        s,p,r = self.runEpisode()
        self.network.train((s,p,r))

selfP = SelfPlay()
selfP.train_step()

