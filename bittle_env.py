#Import Gym Dependencies
import gym
from gym import Env #Super class
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
#Import Helpers
import numpy as np
import random
import os
from sklearn.preprocessing import normalize
#Import Stable Baselines
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
#Import PyBullet
import pybullet as p
import pybullet_data

MAX_STEP_ANGLE = 1
BOUND_ANGLE = 80
REWARD_FACTOR = 1000
REWARD_WEIGHT_1 = 1.0
REWARD_WEIGHT_2 = 1.0
MAX_EPISODE_LEN = 500  # Number of steps for one training episode

class BittleEnv(Env):
    def __init__(self):
        #Create GUI Window
        p.connect(p.GUI)#, options="--width=960 --height=540 --mp4=\"training.mp4\" --mp4fps=60") # uncommend to create a video
        p.configureDebugVisualizer(p.DIRECT, 1) # Disable rendering during loading
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-10, cameraPitch=-40, cameraTargetPosition=[0.4,0,0])
        p.setGravity(0,0,-9.81)

        #Action Space: 8 joint Angles, Min Angle: -1 rad, Max Angle: 1 rad
        self.action_space = Box(low=-1.4,high=1.4,shape=(8,))
        # The observation space are the torso roll, pitch and the angular velocities and a history of the last 30 joint angles (30*8 = 240 + 6(xyzw v1 av2)  246)
        #NEW: orientation (x,y,z,w) and velocity (linear and angular), and the 8 joint angles = 14
        self.observation_space = Box(low=-1, high=1,shape=(14,))

        self.episode_counter = 0
        self.bound_angle = np.deg2rad(BOUND_ANGLE)

        self.reset()
        self.step(Box(low=-1,high=1,shape=(8,)).sample())

    def step(self, action):

        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        lastPosition = p.getBasePositionAndOrientation(self.bittle_id)[0][0]
        jointAngles = np.asarray(p.getJointStates(self.bittle_id, self.joint_ids), dtype=object)[:,0]
        ds = np.deg2rad(MAX_STEP_ANGLE) # Maximum joint angle derivative (maximum change per step)
        jointAngles += action * ds  # Change per step including agent action

        # Apply new joint angle to the joints. Keep in mind the angle boundaries of +/-1 rad for each joint
        for i in range(len(jointAngles)):
            jointAngles[i] = np.clip(jointAngles[i], -self.bound_angle, self.bound_angle)

        # Set new joint angles
        p.setJointMotorControlArray(self.bittle_id, self.joint_ids, p.POSITION_CONTROL, jointAngles)
        p.stepSimulation()

        # Read robot state (pitch, roll and their derivatives of the torso-link)
        state_robot_pos, state_robot_ang = p.getBasePositionAndOrientation(self.bittle_id)
        state_robot_ang_euler = np.asarray(p.getEulerFromQuaternion(state_robot_ang)[0:2])
        state_robot_vel = np.asarray(p.getBaseVelocity(self.bittle_id)[1])
        state_robot_vel = state_robot_vel[0:2]
        state_robot_vel_norm = normalize(state_robot_vel.reshape(-1,1))

        self.state_robot = np.concatenate((state_robot_ang, state_robot_vel_norm.reshape(1,-1)[0]))

        # Reward is the advance in x-direction
        currentPosition = p.getBasePositionAndOrientation(self.bittle_id)[0][0] # Position in x-direction of torso-link
        #Change in the x direction - y quaternion
        reward = REWARD_WEIGHT_1 * (currentPosition - lastPosition) * REWARD_FACTOR - np.fabs(state_robot_ang_euler[1]) * REWARD_WEIGHT_2
        done = False

        # Stop criteria of current learning episode: Number of steps or robot fell
        self.episode_counter += 1
        if (self.episode_counter > MAX_EPISODE_LEN) or self.is_fallen():
            reward = 0
            done = True

        #Create the observation of the position velocity and angles
        self.observation = np.concatenate((self.state_robot, jointAngles))

        info = {}
        return np.array(self.observation).astype(float), reward, done, info


    def render(self):
        pass

    def reset(self):
        #Reset episode counter
        self.episode_counter = 0

        #Reset Pybullet Environment
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # Disable rendering during loading
        p.setGravity(0,0,-9.81)

        #Load plane into environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, lateralFriction = 1.4)

        #Load bittle into environment
        self.startPos = [0,0,.95]
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.bittle_id = p.loadURDF("models/bittle.urdf",self.startPos, self.startOrientation)

        #Get usable joints and create sliders
        self.joint_ids = self.getJoints()

        #Set the maximum angle for each joint (.52 rad). To make symmetric Legs make right side negative
        jointAngles = np.deg2rad(np.array([30, 30, 30, 30, -30, -30, -30, -30]))

        # Reset joint position to rest pose
        for i in range(len(self.joint_ids)):
            #each joint, reset to its original angle using the joint id
            p.resetJointState(self.bittle_id,self.joint_ids[i], jointAngles[i])

        #Read Robot State
        state_robot_ang = p.getBasePositionAndOrientation(self.bittle_id)[1]
        #Linear and angular velocities
        state_robot_vel = np.asarray(p.getBaseVelocity(self.bittle_id)[1])
        state_robot_vel = state_robot_vel[0:2]
        state_robot_vel = normalize(state_robot_vel.reshape(-1,1))
        #[X,Y,Z,W, Linear Velocity, Angular Velocity]
        self.state_robot = np.concatenate((state_robot_ang, state_robot_vel.reshape(1,-1)[0]))

        # Get current positions of joints
        state_joints = np.asarray(p.getJointStates(self.bittle_id, self.joint_ids), dtype=object)[:,0]
        #State of robot: orientation (x,y,z,w) and velocity (linear and angular), and the 8 joint angles = 14
        self.observation = np.concatenate((self.state_robot, state_joints))

        # Re-activate rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        #return the observation: length of 14
        return np.array(self.observation).astype(float)


    def getJoints(self):
        #Collect the joints that can be manipulated for walking
        joint_ids = []
        #Check every joint of bittle
        for j in range(p.getNumJoints(self.bittle_id)):
            info = p.getJointInfo(self.bittle_id, j)
            jointName = info[1]
            jointType = info[2]
            #Save joints that can be moved (legs and arms)
            if (jointType == p.JOINT_REVOLUTE):
                joint_ids.append(j)
        #Return a list of the joint ids that should be manipulated
        return joint_ids

    def is_fallen(self):
        """ Check if robot is fallen. It becomes "True", when pitch or roll is more than 0.9 rad."""
        position, orientation = p.getBasePositionAndOrientation(self.bittle_id)
        orientation = p.getEulerFromQuaternion(orientation)
        is_fallen = np.fabs(orientation[0]) > 0.9 or np.fabs(orientation[1]) > 0.9
        return is_fallen