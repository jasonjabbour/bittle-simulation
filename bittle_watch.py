import os
import gym
from stable_baselines3 import PPO, SAC, DDPG, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from bittle_env import BittleEnv
import time
import matplotlib.pyplot as plt
import numpy as np

def loadEnv(model_name, vec_norm=False):
    '''Load environment based on model name'''

    # Create OpenCatGym environment from class
    env = BittleEnv(GUI=True)

    #if trained with VecNormalize Wrapper
    if vec_norm:
        #Get Saved Stats
        stats_path = os.path.join('Training','VecNormalize_Stats',model_name+'.pkl')
        env = make_vec_env(lambda: env, n_envs=1)
        env = VecNormalize.load(stats_path, env)
        #  do not update stats at test time
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False

    #Get path to model
    model_path = os.path.join('Training','Saved_Models',model_name)
    #Reload the saved model
    model = PPO.load(model_path,env=env)

    return model, env


def watch(model, env, plot=False):
    obs = env.reset()

    #For plot
    score = 0
    count = 0
    x = []
    y = []

    for _ in range(20000):
        action, _state = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)

        #env.render()
        print(reward)

        x.append(count)
        count+=1
        score+=reward
        y.append(score)

        if done and plot:
            plt.scatter(x, y)
            plt.show()
            break
        elif done:
            obs = env.reset()

    #close environment
    env.close()


if __name__ == '__main__':
    model, env = loadEnv('PPO_Model_Bittle81',vec_norm=True)
    watch(model, env, plot=False)
