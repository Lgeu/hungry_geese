from kaggle_environments import make

env = make("hungry_geese", debug=False)
def env_to_rank(env):
    rewards = [goose["reward"] for goose in env.state]
    return rewards
env.reset()
env.run(["ucb1.py", "mean.py", "altanate.py", "alphazero.py"])
res = env_to_rank(env)
print(f"{res},")
