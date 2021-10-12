#Import Gym Dependencies
import gym
from gym import Env #Super class
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
#Import Helpers
import numpy as np
import random
#Import PyBullet
import pybullet as p
import pybullet_data

BOUND_ANGLE = 57 #(1 rad)
REWARD_FACTOR = 1000
REWARD_WEIGHT_1 = .1
REWARD_WEIGHT_2 = .1
REWARD_WEIGHT_3 = .3
MAX_EPISODE_LEN = 500  # Number of steps for one training episode
MAX_CHANGE = 15
NUM_JOINTS = 8

class BittleEnv(Env):
    def __init__(self, GUI=False):
        #Create GUI Window
        p.connect(p.GUI) if GUI else p.connect(p.DIRECT)
        #p.connect(p.GUI, options="--width=960 --height=540 --mp4=\"training.mp4\" --mp4fps=60")
        #p.connect(p.GUI, options="--mp4=\"test.mp4\" --mp4fps=240")
        #p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "myvideo.mp4")

        p.configureDebugVisualizer(p.DIRECT, 0) # Disable rendering during loading, 0 for raw GUI
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-10, cameraPitch=-40, cameraTargetPosition=[0.4,0,0])
        p.setGravity(0,0,-9.81)

        #Joint Angle History Queue
        self.history_steps_saved = 1 #10*8 Joint Angles = 80 length
        self.history_joint_queue = np.zeros([self.history_steps_saved*NUM_JOINTS])

        #Action Space: 8 joint Angles, Min Angle: -1 rad, Max Angle: 1 rad
        self.action_space = Box(low=-1,high=1,shape=(NUM_JOINTS,))
        # The observation space are the torso roll, pitch and the angular velocities and a history of the last 30 joint angles (30*8 = 240 + 6(xyzw v1 av2)  246)
        #Position (x,y,z), Orientation (x,y,z), Linear Velocity (x,y,z), Angular Velocity (wx,wy,wz) and the 8 joint angles*10 history = 92
        self.observation_space = Box(low=-50, high=50,shape=(12+NUM_JOINTS*self.history_steps_saved,))

        self.step_counter = 0
        self.bound_angle = np.deg2rad(BOUND_ANGLE)


    def step(self, action):
        '''Step after a given action'''

        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        #Get current information of bittle
        lastPosition = p.getBasePositionAndOrientation(self.bittle_id)[0][0]
        jointAngles = np.asarray(p.getJointStates(self.bittle_id, self.joint_ids), dtype=object)[:,0]

        # Apply new joint angle to the joints.
        change = np.deg2rad(MAX_CHANGE)
        #jointAngles += action * change #11 degrees #+ random.uniform(-.1,.1) #add noise
        jointAngles = action

        #Clip angles that exceed boundaries of +/-1 rad for each joint
        for i in range(len(jointAngles)):
            jointAngles[i] = np.clip(jointAngles[i], -self.bound_angle, self.bound_angle)

        # Set new joint angles
        p.setJointMotorControlArray(self.bittle_id, self.joint_ids, p.POSITION_CONTROL, jointAngles)
        p.stepSimulation()

        # Read robot state
        # Position and Orientation: [x,y,z] positions and [x,y,z] orientations
        state_robot_pos, state_robot_orien = p.getBasePositionAndOrientation(self.bittle_id)
        state_robot_pos = np.asarray(state_robot_pos).reshape(1,-1)[0]
        #Change Orientation to Euler
        state_robot_orien = np.asarray(p.getEulerFromQuaternion(state_robot_orien))
        state_robot_orien = np.asarray(state_robot_orien).reshape(1,-1)[0]
        # Get specific positions
        current_x_position = state_robot_pos[0] # Position in x-direction of torso-link
        current_y_position = state_robot_pos[1]
        current_z_position = state_robot_pos[2] # Position in z-direction

        #Velocity:  Linear velocity [X, Y, Z] and angular velocity [wx,wy,wz]
        state_robot_lin_vel, state_robot_ang_vel = p.getBaseVelocity(self.bittle_id)
        state_robot_lin_vel = np.asarray(state_robot_lin_vel).reshape(1,-1)[0]
        state_robot_ang_vel = np.asarray(state_robot_ang_vel).reshape(1,-1)[0]

        self.state_robot = np.concatenate((state_robot_pos,state_robot_orien,state_robot_lin_vel,state_robot_ang_vel))

        # Reward is the advance in x-direction
        #Change in the x direction + x velocity - z position
        #reward = REWARD_WEIGHT_1 * (current_x_position - lastPosition) + (REWARD_WEIGHT_2*state_robot_lin_vel[0])  - (REWARD_WEIGHT_3 * abs(current_z_position-.95)) - (REWARD_WEIGHT_3 * abs(state_robot_orien[0])) - (REWARD_WEIGHT_3 * abs(state_robot_orien[1]))
        #PPO16:
        #reward = REWARD_WEIGHT_1 * (current_x_position - lastPosition) - (REWARD_WEIGHT_2 * abs(current_z_position-.95)) - (REWARD_WEIGHT_3 * abs(state_robot_lin_vel[2])) - (REWARD_WEIGHT_3 * abs(state_robot_orien[0])) - (REWARD_WEIGHT_3 * abs(state_robot_orien[1]))
        reward = REWARD_WEIGHT_1 * (current_x_position - lastPosition) + (REWARD_WEIGHT_2 * state_robot_lin_vel[0]) - (REWARD_WEIGHT_3 * abs(current_z_position-self.start_height)) - (REWARD_WEIGHT_3 * abs(state_robot_lin_vel[2])) - (REWARD_WEIGHT_3 * abs(state_robot_orien[0])) - (REWARD_WEIGHT_3 * abs(state_robot_orien[1]))
        #reward = (REWARD_WEIGHT_2 * state_robot_lin_vel[0]) - (REWARD_WEIGHT_3 * abs(current_z_position-self.start_height)) - (REWARD_WEIGHT_3 * abs(state_robot_lin_vel[2])) - (REWARD_WEIGHT_3 * abs(state_robot_orien[0])) - (REWARD_WEIGHT_3 * abs(state_robot_orien[1]))

        done = False

        # Stop criteria of current learning episode: Number of steps or robot fell
        self.step_counter += 1
        if (self.step_counter > MAX_EPISODE_LEN) or self.is_fallen():
            reward = 0
            done = True

        #Update history queue
        self.update_queue(jointAngles)

        #Create the observation of the position velocity and angles
        self.observation = np.concatenate((self.state_robot, self.history_joint_queue))

        #p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-10, cameraPitch=-40, cameraTargetPosition=[current_x_position,current_y_position,0])

        info = {}
        return np.array(self.observation).astype(float), reward, done, info

    def render(self):
        '''function required for OpenAI Gym env'''
        pass

    def reset(self):
        '''Reset Bittle environment each episode'''
        #Reset episode counter
        self.step_counter = 0

        #Reset Pybullet Environment
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # Disable rendering during loading
        p.setGravity(0,0,-9.81)

        #Load plane into environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, lateralFriction = 2.4)

        #Load bittle into environment
        self.start_height = .75 if NUM_JOINTS == 4 else .95
        self.startPos = [0,0,self.start_height]
        #Face x direction by rotating about z
        self.startOrientation = p.getQuaternionFromEuler([0,0,-1.5708])
        urdf_name = "bittle_modified_fixed.urdf" if NUM_JOINTS == 4 else "bittle_modified.urdf"
        self.bittle_id = p.loadURDF("models/"+urdf_name,self.startPos, self.startOrientation)

        #Get usable joints and create sliders
        self.joint_ids = self.getJoints()
        #Set the angle for each joint (.52 rad). To make symmetric Legs make right side negative
        jointAngles = np.deg2rad(np.array([30, 30, 30, 30, -30, -30, -30, -30]))

        # Reset joint position to rest position
        for i in range(len(self.joint_ids)):
            #each joint, reset to its original angle using the joint id
            p.resetJointState(self.bittle_id,self.joint_ids[i], jointAngles[i])

        #Reset queue to initial joint angles
        if NUM_JOINTS == 4:
            #choose to store only shoulder joints in observational space, keep knee joints stiff
            self.reset_queue(jointAngles[::2])
            #for all steps use only shoulder joint ids
            self.joint_ids = self.joint_ids[::2]
        #if using all 8 joints pass all angles to store in queue and use all joint ids
        else:
            self.reset_queue(jointAngles)

        # Read robot state ---
        # Position and Orientation: [x,y,z] positions and [x,y,z,w] orientations
        state_robot_pos, state_robot_orien = p.getBasePositionAndOrientation(self.bittle_id)
        state_robot_pos = np.asarray(state_robot_pos).reshape(1,-1)[0]
        #Change Orientation to Euler
        state_robot_orien = np.asarray(p.getEulerFromQuaternion(state_robot_orien))
        state_robot_orien = np.asarray(state_robot_orien).reshape(1,-1)[0]

        #Velocity:  Linear velocity [X, Y, Z] and angular velocity [wx,wy,wz]
        state_robot_lin_vel, state_robot_ang_vel = p.getBaseVelocity(self.bittle_id)
        state_robot_lin_vel = np.asarray(state_robot_lin_vel).reshape(1,-1)[0]
        state_robot_ang_vel = np.asarray(state_robot_ang_vel).reshape(1,-1)[0]

        self.state_robot = np.concatenate((state_robot_pos,state_robot_orien,state_robot_lin_vel,state_robot_ang_vel))

        #Observation: orientation (x,y,z,w) and linear velocity, and the 8 joint angles = 14
        self.observation = np.concatenate((self.state_robot, self.history_joint_queue))

        # Re-activate rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        #p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-10, cameraPitch=-40, cameraTargetPosition=[0.4,0,0])

        #return the observation: length of 14
        return np.array(self.observation).astype(float)

    def getJoints(self):
        '''Collect the joints that can be manipulated for walking'''
        joint_ids = []
        #Check every joint of bittle
        for j in range(p.getNumJoints(self.bittle_id)):
            info = p.getJointInfo(self.bittle_id, j)
            jointName = info[1]
            jointType = info[2]
            #Save joints that can be moved (legs and arms)
            if (jointType == p.JOINT_REVOLUTE) or ((jointType == p.JOINT_FIXED) and (NUM_JOINTS == 4) and ('knee' in jointName.decode('ascii'))):
                joint_ids.append(j)
        #Return a list of the joint ids that should be manipulated
        return joint_ids

    def is_fallen(self):
        '''Check if robot is fallen. True, when pitch or roll is more than 3 rad.'''
        position, orientation = p.getBasePositionAndOrientation(self.bittle_id)
        orientation = p.getEulerFromQuaternion(orientation)
        is_fallen = np.fabs(orientation[0]) > 3 or np.fabs(orientation[1]) > 3
        return is_fallen

    def update_queue(self,jointAngles):
        '''Remove first 8 joints and add 8 new joints to bottom'''
        #remove first 8 joints from beginning
        self.history_joint_queue = self.history_joint_queue[NUM_JOINTS:]
        #add new 8 joints to end
        self.history_joint_queue = np.concatenate((self.history_joint_queue,jointAngles))

    def reset_queue(self,jointAngle):
        '''Since no history at reset, fill queue with start angle position'''
        self.history_joint_queue = np.tile(jointAngle,self.history_steps_saved)