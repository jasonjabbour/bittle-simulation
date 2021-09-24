import pybullet as p
import pybullet_data
import time

class BittleEnv():
    def __init__(self):
        #Create GUI Window
        p.connect(p.GUI)#, options="--width=960 --height=540 --mp4=\"training.mp4\" --mp4fps=60") # uncommend to create a video
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1) # Disable rendering during loading
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-10, cameraPitch=-40, cameraTargetPosition=[0.4,0,0])
        p.setGravity(0,0,-9.81)

        #Load plane into environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, lateralFriction = 1.4)

        #Load bittle into environment
        self.startPos = [0,0,.6]
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.bittle_id = p.loadURDF("models/bittle.urdf",self.startPos, self.startOrientation)

        #Get usable joints and create sliders
        self.joint_ids, self.slider_ids = self.getJoints()

        #Default params
        self.maxangle = .05 #51 degrees
        self.minangle = -.05 #51 degrees
        self.increase = False
        self.initial_angles = [.2,.364,-.2,.364,.2,-.364,-.2,-.364]
        #self.initial_angles = [.2,.364,-.364,.364,.2,-.364,-.364,-.364]
        self.current_angles = [0,0,0,0,0,0,0,0]


    def getJoints(self):
        #Collect the joints that can be manipulated for walking
        joint_ids = []
        slider_ids = []
        #Check every joint of bittle
        for j in range(p.getNumJoints(self.bittle_id)):
            info = p.getJointInfo(self.bittle_id, j)
            jointName = info[1]
            jointType = info[2]
            #Save joints that can be moved (legs and arms)
            if (jointType == p.JOINT_REVOLUTE):
                joint_ids.append(j)
                #addslider. Min and Max angles: 90 degrees
                slider_ids.append(p.addUserDebugParameter(jointName.decode("utf-8"),-1.5708,1.5708,0))

        return joint_ids, slider_ids

    def my_policy(self):
        self.take_position()
        self.move_forward()

    def take_position(self):
        while not self.reached_initial_state():
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
            for i in range(len(self.initial_angles)):
                #go negative
                if (self.initial_angles[i] <= 0) and not (self.current_angles[i] <= self.initial_angles[i]):
                    self.current_angles[i] = self.current_angles[i] - .005
                #go positve
                elif (self.initial_angles[i] >= 0) and not (self.current_angles[i] >= self.initial_angles[i]):
                    self.current_angles[i] = self.current_angles[i] + .005

            p.setJointMotorControlArray(self.bittle_id, self.joint_ids, p.POSITION_CONTROL, self.current_angles)
            p.stepSimulation()

    def reached_initial_state(self):
        count = 0
        for i in self.current_angles:
            if (self.initial_angles[count] >= 0) and i < self.initial_angles[count]:
                return False
            elif (self.initial_angles[count] <= 0) and i > self.initial_angles[count]:
                return False
            count+=1

        return True

    def move_forward(self):
        #Very simple policy to move
        while True:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
            if self.current_angles[0] <= self.maxangle and self.increase:
                self.current_angles[0] =self.current_angles[0] + .004
                self.current_angles[4] =self.current_angles[4] + .004

                self.current_angles[2] =self.current_angles[2] - .004
                self.current_angles[6] =self.current_angles[6] - .004
            elif self.current_angles[0] >= self.maxangle:
                self.increase = False
                self.current_angles[0] =self.current_angles[0] - .004
                self.current_angles[4] =self.current_angles[4] - .004

                self.current_angles[2] =self.current_angles[2] + .004
                self.current_angles[6] =self.current_angles[6] + .004
            elif self.increase == False and self.current_angles[0] >= self.minangle:
                self.current_angles[0] =self.current_angles[0] - .004
                self.current_angles[4] =self.current_angles[4] - .004

                self.current_angles[2] =self.current_angles[2] + .004
                self.current_angles[6] =self.current_angles[6] + .004
            else:
                self.increase = True

            p.setJointMotorControlArray(self.bittle_id, self.joint_ids, p.POSITION_CONTROL, self.current_angles)
            p.stepSimulation()

    def user_input(self):
        #Listen for user input through sliders and change accordingly
        count = 0
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        #Check values of each slider
        for slider in self.slider_ids:
            user_input_angle = p.readUserDebugParameter(slider)
            #Update bittle with new value for joint given by user
            p.setJointMotorControl2(self.bittle_id, self.joint_ids[count], p.POSITION_CONTROL, user_input_angle)
            count+=1
            p.stepSimulation()

if __name__ == "__main__":
    #Initialize Bittle Environment
    bittle = BittleEnv()

    while True:
        bittle.user_input()
        #bittle.my_policy()
