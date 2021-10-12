import os
import gym
from stable_baselines3 import PPO, SAC, DDPG, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from bittle_env import BittleEnv
import time
import matplotlib.pyplot as plt
import numpy as np

# Create OpenCatGym environment from class
env = BittleEnv(GUI=True)

#Get path to model
model_path = os.path.join('Training','Saved_Models','PPO_Model_Bittle21')
#Reload the saved model
model = PPO.load(model_path,env=env)

obs = env.reset()

count = 0
x = []
y = []
score = 0
for _ in range(5000):
    action, _state = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    env.render()
    #print(reward)

    x.append(count)
    count+=1
    score+=reward
    y.append(score)

    if done:
        obs = env.reset()
    #time.sleep(1./240.)
    #time.sleep(10)

# plt.scatter(x, y)
# plt.show()

#close environment
env.close()