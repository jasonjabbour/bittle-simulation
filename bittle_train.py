import os
import gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from bittle_env import BittleEnv
from datetime import datetime

#Create OpenAI Gym Environment
env = BittleEnv()
check_env(env)

#Train Model ----------------------------
log_path = os.path.join('Training','Logs')
model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
model.learn(total_timesteps=400000)

#Save the model
PPO_Path = os.path.join('Training','Saved_Models','PPO_Model_Bittle1')
model.save(PPO_Path)

print("Congrats, Done!")
print(datetime.now())

env.close()