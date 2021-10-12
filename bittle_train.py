import os
import gym
from stable_baselines3 import PPO, SAC, DDPG, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from bittle_env import BittleEnv
from datetime import datetime

def loadEnv(GUI=False):
    '''Load and check bittle gym environment'''
    #Create OpenAI Gym Environment
    env = BittleEnv(GUI)
    check_env(env)
    return env

def train(env, episodes=500000, model_name='UNNAMED_MODEL', load_previous=False):
    '''Train new model or load previous model and continue training'''

    #Get path for saving or loading model
    model_path = os.path.join('Training','Saved_Models',model_name)
    log_path = os.path.join('Training','Logs')

    #Continue to train previous model
    if load_previous:
        model = PPO.load(model_path,env=env)
    #Create new model
    else:
        model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)

    #Train model
    model.learn(total_timesteps=episodes)

    #Save the model
    model.save(model_path)

    #Close Environment
    env.close()

if __name__ == '__main__':
    #Load Env
    env = loadEnv(GUI=False)
    #Train Model
    train(env, 4000000, 'PPO_Model_Bittle27', load_previous=True)
    print(f'Training Completed! {datetime.now()}')
