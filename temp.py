from goSim import GoEnv
from utils import obsToString
from copy import copy
import numpy as np
from enums import Colour

goenv = GoEnv(player_color='black', observation_type='image3c', illegal_move_mode="raise", board_size=13, komi=7.5)
goenv.reset()
goenv_copy = copy(goenv)
goenv_copy.state = copy(goenv.state)

for i in range(10):
    color = i % 2 + 1
    # print("((((((((((((((", goenv_copy.state.color)
    
    goenv.set_player_color(color)
    print(goenv.state.color)
    goenv.state.color = color
    # obs_t, action, r_t, done, info, cur_score = goenv.step(np.random.randint(0, 168))
    print()
    print(goenv.state.color)
    print("((((((((((((((", goenv_copy.state.color)
    print()
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