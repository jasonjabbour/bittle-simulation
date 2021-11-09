import os
import gym

from bittle_env import BittleEnv
from datetime import datetime

#Stable Baselines
from stable_baselines3 import PPO, SAC, DDPG, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnRewardThreshold, EvalCallback

NUM_ENV = 5

def loadEnv(GUI, vec_norm, load_previous, load_path, stats_path_load, log_path):
    '''Load and bittle gym environment and initialize model'''
    #Initialize OpenAI Gym Environment
    env = BittleEnv(GUI)
    #env = DummyVecEnv([lambda: env])

    #-- Environment --
    if vec_norm:
        env = make_vec_env(lambda: env, n_envs=NUM_ENV) #for ppo
        #Loading previous vectorized stats
        if load_previous:
            env = VecNormalize.load(stats_path_load, env)
        #Creating new vectorized environment
        else:
            # Automatically normalize the input features and reward
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        print('VecNormalized environment in use')
    else:
        check_env(env)

    #-- Model --
    if load_previous:
        #Continue to train previous model
        model = PPO.load(load_path,env=env)
    else:
        #Create new model
        model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path, seed=6)

    return model, env

def train(episodes=500000, model_name='UNNAMED_MODEL', load_previous=False, model_load='', vec_norm=False, GUI=False):
    '''Train new model or load previous model and continue training'''

    #Make paths even if not needed
    model_path, log_path, load_path, stats_path_save, stats_path_load = make_paths(model_name, model_load)

    #Make environment and model objects
    model, env = loadEnv(GUI, vec_norm, load_previous, load_path, stats_path_load, log_path)

    # Save a checkpoint every 1000 steps
    #checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=model_path,name_prefix=model_name+'_callbacksaved', verbose=1)

    # #Set the reward threshold to stop at
    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=-4, verbose=1)
    # #Every 100000 steps check if passed the reward threshold, if so save it
    # eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=1000, best_model_save_path=model_path+'_callbackSaved', verbose=1)

    #Train model
    model.learn(total_timesteps=episodes)

    #Save the final model
    model.save(model_path)

    #Save stats if needed
    save_stats(vec_norm, env, stats_path_save)

    #Ask user if you want to keep training model
    while True:
        answer1 = input("Would you like to continue training?[Y/N] ")
        if answer1.upper() == 'Y':
            answer2 = input("Enter number timesteps would you like to add: ")
            try:
                answer2 = int(answer2)
                print('Training model for',answer2,'more timesteps')
                #Train model more
                model.learn(total_timesteps=answer2)
                #Save the final model
                model.save(model_path+'_added_'+str(answer2))
                #Save stats if needed
                save_stats(vec_norm, env, model_name+'_added_'+str(answer2))
            except Exception as e:
                print(e)
        elif answer1.upper() == 'N':
            break

    #Close Environment
    env.close()

def make_paths(model_name, model_load):
    '''Make paths to save or load information'''
    #Get path for saving model
    model_path = os.path.join('Training','Saved_Models',model_name)
    #Make path for training logs
    log_path = os.path.join('Training','Logs')
    #Make path to load previous model
    load_path = os.path.join('Training','Saved_Models',model_load)
    #Make path to save statistics to if vec env
    stats_path_save = os.path.join('Training','VecNormalize_Stats',model_name+'.pkl')
    #Make path to load statistics from if vec env
    stats_path_load = os.path.join('Training','VecNormalize_Stats',model_load+'.pkl')
    return model_path, log_path, load_path, stats_path_save, stats_path_load

def save_stats(vec_norm, environment, stats_path_save):
    '''If vec normalized env used, save the stats of the environment'''
    if vec_norm:
        #Save VecNormalize Statistics
        environment.save(stats_path_save)

if __name__ == '__main__':
    #Would you like to vectorize environment?
    vec_norm = True
    #Would you like to view training?
    GUI = False
    #Train Model, provide name for new model and name of previous model if load is needed
    train(8000000, 'PPO_Model_Bittle89', load_previous=False, model_load='', vec_norm=vec_norm, GUI=GUI)
    print(f'Training Completed! {datetime.now()}')
