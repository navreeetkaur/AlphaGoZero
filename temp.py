from goSim import GoEnv
from utils import obsToString
from copy import copy
import numpy as np
from enums import Colour

goenv = GoEnv(player_color='black', observation_type='image3c', illegal_move_mode="raise", board_size=13, komi=7.5)
goenv.reset()
goenv_copy = copy(goenv)
goenv_copy.state = copy(goenv.state)


actions = [1,2, 13, 16, 27, 28, 15, 14, 32, 34, 15, 14, 15]
# white_actions = [14, 60, 61, 62]
for i in range(200):
    color = 1
    # print("((((((((((((((", goenv_copy.state.color)
    
    goenv.set_player_color(color)
    # print(goenv.state.color)
    # goenv.state.color = color
    # print("Action to play: {}, {}".format(i, color))
    obs_t, action, r_t, done, info, cur_score = goenv.step(i % 169)
    goenv.state.color = Colour.BLACK.value
    # print(goenv.is_legal_action(obs_t, actions[i + 1], 3 - color))
    goenv.render()
    # print(r_t, done, cur_score)
    # print(info)
    # print()
    # print(goenv.state.color)
    # print("((((((((((((((", goenv_copy.state.color)
    # print()
    # print(r_t)
    # print(done)

    # print(obsToString(obs_t))
    # print(obs_t.shape)
    # print(action)
    # print(r_t)
    # print(done)
    # # print(info)
    # print(cur_score)
# goenv.render()
goenv.close()