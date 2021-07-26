from kaggle_environments import make

env = make("hungry_geese", debug=False)
def env_to_rank(env):
    rewards = [goose["reward"] for goose in env.state]
    return rewards
env.reset()
env.run(["mean_cpuct0.25.py", "alphazero_c0.25.py", "alphazero_c0.125.py", "alphazero_c0.5.py"])   ###
res = env_to_rank(env)
print(f"{res},")
