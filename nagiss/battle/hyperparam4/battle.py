executables = ['mean_c0.25_cpuct0.25', 'alphazero_c0.25', 'alphakkt_c0.25_kkt0.05', 'alphazero_c0.25_food5']

from kaggle_environments import make

env = make("hungry_geese", debug=False)
def env_to_rank(env):
    rewards = [goose["reward"] for goose in env.state]
    return rewards
env.reset()
env.run([exe + ".py" for exe in executables])
res = env_to_rank(env)
print(f"{res},")
