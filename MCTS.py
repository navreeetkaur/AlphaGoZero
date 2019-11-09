import numpy as np
from policyValueNet import PolicyValueNet
from policyValueNet import args as neuralNetArgs
from enums import Colour
from config import *
from utils import obsToString, stateToObs, getValidMoves, obsToState, invertObs
from copy import copy, deepcopy
import math
import time

class MCTS():

    def __init__(self, nNet, simulator):
        self.nNet  = nNet
        self.Qsa = {}
        self.Psa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Wsa = {}
        self.Rs = {} # game ended -> 0, win -> 1, loss -> -1 (for player 1)
        self.valid_moves = {}
        self.num_simulations = NUM_SIMULATIONS
        self.simulator = simulator
<<<<<<< HEAD
        # print(simulator.player_color)
        # print(simulator.state.color)
=======
        print(simulator.player_color)
        print(simulator.state.color)
>>>>>>> fe170cc3d45ff7dc266c9dacb5186c0bf5ace4ed
        self.currSimulator = None

    # run MCTS with player 1 (BLACK)
    # invert it when the player is WHITE
    def getPolicy(self, state, temp=1):
        # from IPython import embed; embed()
<<<<<<< HEAD
        # print(state[16 ,0, 0])
        if(state[16,0,0] == Colour.WHITE.value):
            state[16, :, :] = Colour.BLACK.value
        for i in range(self.num_simulations):
            
            # print('------------------NEW SIMULATION-------------------')
            # print(state[16 ,0, 0])
=======
        for i in range(self.num_simulations):
            print('------------------NEW SIMULATION-------------------')
>>>>>>> fe170cc3d45ff7dc266c9dacb5186c0bf5ace4ed
            self.search(deepcopy(state), 0)
        curObs = stateToObs(state)
        strObs = obsToString(curObs)
        counts = np.zeros(NUM_ACTIONS)

        for a in range(NUM_ACTIONS):
            if (strObs, a) in self.Nsa:
                counts[a] = self.Nsa[(strObs, a)]

        if(temp == 0):
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs
        
        counts = [x ** (1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs

    def getValidMoves(self, obs, player_colour):
        validActions = np.zeros(NUM_ACTIONS)
        for action in range(NUM_ACTIONS):
            is_legal_action = self.simulator.is_legal_action(obs, action, player_colour)
            if(is_legal_action):
                validActions[action] = 1
        return validActions

    def getNextState(self, action):
        if(self.currSimulator) is None:
            self.currSimulator = copy(self.simulator)
<<<<<<< HEAD
        # print("-------------------getNextState-----------------")
        # print('Initial -> Action: {}, Player: {}, State: {}'.format(
        #     action,
        #     Colour(self.currSimulator.player_color).__str__(),
        #     Colour(self.currSimulator.state.color).__str__()
        #     ))
=======
        print("-------------------getNextState-----------------")
        print('Initial -> Action: {}, Player: {}, State: {}'.format(
            action,
            Colour(self.currSimulator.player_color).__str__(),
            Colour(self.currSimulator.state.color).__str__()
            ))
>>>>>>> fe170cc3d45ff7dc266c9dacb5186c0bf5ace4ed

        obs_t = None
        r_t = None

        if(self.currSimulator.state.color == Colour.WHITE.value):
            self.currSimulator.state.color = Colour.BLACK.value
            self.currSimulator.player_color = Colour.BLACK.value
            obs_t, action, r_t, done, info, cur_score = self.currSimulator.step(action)
            self.currSimulator.state.color = Colour.BLACK.value
<<<<<<< HEAD
            # print(r_t, done)
            # print('Observation Inverted.') # Invert when the state of the simulator is WHITE
            obs_t = invertObs(obs_t)
        else:
            # print('Observation Not Inverted.')
            obs_t, action, r_t, done, info, cur_score = self.currSimulator.step(action)
            # print(r_t, done)
            self.currSimulator.state.color = Colour.WHITE.value
            self.currSimulator.player_color = Colour.WHITE.value
        # print('Final -> Player: {}, State: {}'.format(
        #     Colour(self.currSimulator.player_color).__str__(),
        #     Colour(self.currSimulator.state.color).__str__()
        # ))
        # print(obsToString(obs_t), r_t)
=======
            print('Observation Inverted.') # Invert when the state of the simulator is WHITE
            obs_t = invertObs(obs_t)
        else:
            print('Observation Not Inverted.')
            obs_t, action, r_t, done, info, cur_score = self.currSimulator.step(action)
            self.currSimulator.state.color = Colour.WHITE.value
            self.currSimulator.player_color = Colour.WHITE.value
        print('Final -> Player: {}, State: {}'.format(
            Colour(self.currSimulator.player_color).__str__(),
            Colour(self.currSimulator.state.color).__str__()
        ))
        print(obsToString(obs_t), r_t)
>>>>>>> fe170cc3d45ff7dc266c9dacb5186c0bf5ace4ed
        return obs_t, r_t

    # State is of shape (17 x 13 x 13) -- STATE WITH RESPECT TO BLACK
    def search(self, state, reward):
<<<<<<< HEAD
        # print(state[16, 0, 0])
        # print(Colour.BLACK.value)

        assert(state[16, 0, 0] == Colour.BLACK.value)
=======
        assert(state[16, 0, 0] + 1== Colour.BLACK.value)
>>>>>>> fe170cc3d45ff7dc266c9dacb5186c0bf5ace4ed
        player_colour = Colour.BLACK.value

        # Get Obs from State (Picked Top 2)
        obs = np.zeros((3, state.shape[1], state.shape[2]))
        obs[0, :, :] = state[0, :, :]
        obs[1, :, :] = state[1, :, :]
        obs[2, :, :] = np.logical_not(np.logical_or(obs[0, :, :], obs[1, :, :]))

        # Convert it into String representation
        strState = obsToString(obs)

        if strState not in self.Rs:
            self.Rs[strState] = reward
        else:
            # Correct code for action = 169, 170
            if not (reward == self.Rs[strState]):
                # print(strState)
                print("CORRRECCTTTTT IT IN FUTURE!!!!!!!!!!!!!!!!!!!")
            self.Rs[strState] = reward

        if(self.Rs[strState] != 0):
            # terminal node
            self.currSimulator = None
            return -self.Rs[strState]
        
        if strState not in self.Psa:
            start_t = time.time()
            ps, vs = self.nNet.predict(state)
<<<<<<< HEAD
            end_t = time.time()
            print('Time elapsed for prediction = {}'.format(
                        end_t - start_t
                    ))            

=======
>>>>>>> fe170cc3d45ff7dc266c9dacb5186c0bf5ace4ed
            valids = self.getValidMoves(obs, player_colour)
            ps = ps * valids
            self.Psa[strState] = ps
            if(np.sum(ps) > 0):
                self.Psa[strState] /= np.sum(ps)
            else:
                # Assigning equal probability to all valid moves
                self.Psa[strState] += valids
                self.Psa[strState] /= np.sum(ps)
            self.valid_moves[strState] = valids
            self.Ns[strState] = 0
            self.currSimulator = None
            return -vs

        assert(strState in self.valid_moves)

        valids = self.valid_moves[strState]
        cur_best = -float('inf')
        best_act = -1

        for a in range(NUM_ACTIONS):
            if valids[a]:
                # print(self.Psa[strState].shape)
                # print(self.Ns[strState].shape)

                if (strState,a) in self.Qsa:
                    u = self.Qsa[(strState,a)] + CPUCT*self.Psa[strState][a]*math.sqrt(self.Ns[strState])/(1+self.Nsa[(strState,a)])
                else:
                    u = CPUCT*self.Psa[strState][a]*math.sqrt(self.Ns[strState] + EPSILON)     # Q = 0 ?

                # print(u.shape)
                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
<<<<<<< HEAD
        # if(a == PASS_ACTION):
        #     passes += 1
        next_obs, next_reward = self.getNextState(a)

        # Invert Obs to get Observation with respect to Black
        # print(player_colour)
=======
        next_obs, next_reward = self.getNextState(a)

        # Invert Obs to get Observation with respect to Black
        print(player_colour)
>>>>>>> fe170cc3d45ff7dc266c9dacb5186c0bf5ace4ed
        next_state = obsToState(next_obs, state) # Color is Black

        v = self.search(next_state, next_reward)

        if (strState,a) in self.Qsa:
            self.Wsa[(strState, a)] += v
            self.Nsa[(strState, a)] += 1
            self.Qsa[(strState, a)] = self.Wsa[(strState,a)] / self.Nsa[(strState,a)]
        else:
            self.Wsa[(strState, a)] = v
            self.Qsa[(strState, a)] = v
            self.Nsa[(strState, a)] = 1

        self.Ns[strState] += 1
        self.currSimulator = None

        return -v


