exe = './alphazero_c0.25'

import sys
from subprocess import Popen, PIPE, STDOUT
from kaggle_environments.envs.hungry_geese.hungry_geese import Action

parameter_file = "../../rl/parameters/038_01.bin"
cmd = [exe, "-p", parameter_file, "-t", "0.1"]
p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)


def print_environment():
    p = Popen(r"""g++ -### -E - -march=native 2>&1 | sed -r '/cc1/!d;s/(")|(^.* - )|( -mno-[^\ ]+)//g'""", stdout=PIPE, shell=True)
    print(p.stdout.readline().decode().strip())

def write_obs(obs):
    # example:
    # 60.0         # time
    # 6            # step
    # 1 5          # goose (player)
    # 1 0          # goose (opponent)
    # 0            # goose (opponent)
    # 3 7 8 9      # goose (opponent)
    # 40 47        # foods
    
    def write(*args):
        p.stdin.write(f"{' '.join(map(str, args))}\n".encode())
    write(obs["remainingOverageTime"])
    write(obs["step"])
    for goose in obs["geese"]:
        write(len(goose), *goose)
    write(*obs["food"])
    
    p.stdin.flush()

def agent(obs_dict, config_dict):
    if obs_dict["index"] != 0:
        # swap geese
        obs_dict["geese"][0], obs_dict["geese"][obs_dict["index"]] = obs_dict["geese"][obs_dict["index"]], obs_dict["geese"][0]
    write_obs(obs_dict)
    
    #if obs_dict["step"] == 0:
    #    print_environment()

    while True:
        input_values = p.stdout.readline().decode().strip().split()
        if len(input_values) == 0:
            continue
        if input_values[0] == "move":
            direction = input_values[1]
            if direction == "l":
                return Action.WEST.name
            elif direction == "r":
                return Action.EAST.name
            elif direction == "u":
                return Action.NORTH.name
            elif direction == "d":
                return Action.SOUTH.name
            else:
                assert False, f"invalid move {direction}"
        else:
            print(*input_values)
