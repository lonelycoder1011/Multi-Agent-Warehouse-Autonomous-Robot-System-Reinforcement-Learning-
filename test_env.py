from src.environment.warehouse_env import WarehouseEnv

env = WarehouseEnv()
obs, info = env.reset()
print("Env works!")
print("Agents:", env.agents)
print("Num agents:", len(env.agents))
print("Action space:", env.action_space)