from gym_minigrid.wrappers import *
env = gym.make('MiniGrid-Unlock-v0')


import numpy as np
import tensorflow as tf
from collections import deque
import PIL
import random
import matplotlib.pyplot as plt
import flloat
from flloat.parser.ltlf import LTLfParser


class DQN:
    
    def __init__(self):

        self.discount = 0.95
        self.batch_size = 128
        self.EPS_START = 0.9
        self.EPS_END = 0.5
        self.steps_done = 0
        self.EPS_DECAY = 200
        self.output_size = env.action_space.n
        self.d_len =  100000
        self.dq = deque(maxlen=self.d_len)

        
        with tf.variable_scope(name_or_scope="placeholder",reuse=tf.AUTO_REUSE):
            self.x_input =  tf.placeholder(tf.float32,shape=(None,7,7,4),name="input")
            self.target  =  tf.placeholder(tf.float32,shape=(None,),name="target")
            self.actions =  tf.placeholder(tf.int32,shape=(None,),name="actions")
        
        
        # we need to define two network policy and target network where target is freeze
        
        self.policy_network_output ,self.policy_network_param =  self.architecture("policyNetwork")
        self.target_network_output , self.target_network_param = self.architecture("targetNetwork")
        
        
        # shfiting of the weights 
        # we need to shift the weight from policy to target 
        
        tau = 0.001
        self.copy_weight = []
        for target,source in zip(self.target_network_param,self.policy_network_param):
            self.copy_weight.append(target.assign(target*(1-tau) + source*tau))
            
        self.copy_weight = tf.group(*self.copy_weight)
        
        
        
            
    
    
    def weightMatrix(self,name):
        with tf.variable_scope(name_or_scope=name,reuse=False):
            # defining the weights for it 
            self.conv1  =  tf.get_variable(name="conv1",shape=(8,8,4,16),initializer=tf.initializers.variance_scaling)
            self.conv2  =  tf.get_variable(name="conv2",shape=(4,4,16,32),initializer=tf.initializers.variance_scaling)
            self.dense1 =  tf.get_variable(name="dense1",shape=(32,256),initializer=tf.initializers.variance_scaling)
            self.dense2 =  tf.get_variable(name="dense2",shape=(256,self.output_size),initializer=tf.initializers.variance_scaling)
    
            
            
            
    
    def architecture(self,name):
        
        # first call the architecture 
        self.weightMatrix(name)
        with tf.variable_scope(name,reuse=False):
            
            self.conv = tf.nn.conv2d(self.x_input,self.conv1,strides=[1,4,4,1],padding='SAME')
            self.conv = tf.nn.elu(self.conv)
            
            self.conv = tf.nn.conv2d(self.conv,self.conv2,strides=[1,2,2,1],padding="SAME")
            self.conv =  tf.nn.elu(self.conv)
            
            
            # flatten 
            flatten_data = tf.keras.layers.Flatten()(self.conv)
            
            
            self.dense  = tf.matmul(flatten_data,self.dense1)
            self.dense  = tf.nn.elu(self.dense)
            
            self.output = tf.matmul(self.dense,self.dense2)
            self.output = tf.nn.softmax(self.output)
            trainable_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,name)
            
            return self.output,trainable_param
        
    
    def act(self,state):
        # to get what action to choose 
        
        # we need to do exploration vs exploitation 
        
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        action = None
        action_ = None
        if np.random.rand() < eps_threshold:
            
            action = env.action_space.sample()
            
        else:
            
            
            action_  = self.policy_network_output.eval(feed_dict={self.x_input:[state]})[0]
            action  =  np.argmax(action_)
            
        
        
        return action
        
        
    
    def loss(self):
        
        # we need to substract the target from the actual Q values 
        q_sa = tf.reduce_sum(self.policy_network_output*tf.one_hot(self.actions,self.output_size),axis=-1)
        # now we can substract this value from the target value 
        
        
        self.loss =  tf.squared_difference(self.target,q_sa)
        self.loss  =  tf.reduce_mean(self.loss)
        
        # we only want to use policy network 
        self.grad = tf.gradients(self.loss,self.policy_network_param)
        
        # once i have the gradient wrt to policy net we can apply to network 
        # but to deal with exploding gradient we perform gradient clipping 
        
        clipped_grad = []
        for val in self.grad:
            clipped_grad.append(tf.clip_by_value(val,clip_value_min=-1,clip_value_max=1))
            
        # now we can apply the clipped grad values 
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.optimizer =  self.optimizer.apply_gradients(zip(clipped_grad,self.policy_network_param))
        
        
        
            
    
    def storeData(self,nextObservation,action,reward,observation,done):
        if self.d_len - len(self.dq) <=0:
            self.dq.pop()
            
        
        # now we can store the data 
        self.dq.append((nextObservation,action,reward,observation,done))
        
            
        
    def getData(self,batch_size):
        # we gonna sample the data randomly 
        data  = None
        if len(self.dq) > batch_size:
            data = random.sample(self.dq,batch_size)
            
        return data 
            


class parser:

	def __init__(self):
		
		# parser to parse the ltlf formula

		self.states = {"q1","q2","q3"}

		self.initialState  = "q1"
		self.finalState    = "q3"

		self.transitions()

	def transitions(self):

		self.transition = {

			("q1","q1"):-1,
			("q1","q2"):10,
			("q2","q2"):10,
			("q2","q3"):100,
			("q3","q3"):0
		}


		self.state_dict = {

			(False,False):"q1",
			(True,False):"q2",
			(True,True):"q3"

		}



	def reset(self):
		self.initialState = "q1"


	def trace(self,key,door):

		key_door = (key,door)
		intrinsic_reward  = 0

		if self.state_dict.get(key_door) == None:
			self.reset()
			done = True
		else:
			state  = self.state_dict[key_door]
			
			done  =  False


			if self.transition.get((self.initialState,state)) == None:
				self.reset()
				done = True

			elif self.transition.get((self.initialState,state)) != None:

				intrinsic_reward = self.transition[(self.initialState,state)]
				self.initialState = state 

				

			if done:
				intrinsic_reward = -10

		return intrinsic_reward,done






	
		





def preprocessing(image):
    img =  PIL.Image.fromarray(image).convert("L")
    img = np.array(img).T/255.0
    img = np.expand_dims(img,axis=2)
    img = np.tile(img,4)
    return img


tf.reset_default_graph()
dqn  = DQN()
RA = parser()

dqn.loss()


init   = tf.global_variables_initializer()




key_door = []



updateNetwork = 4
game_loss = 0
with tf.Session() as sess:
    sess.run(init)
    
    
    for each_epispode in range(10000):
        obs  = env.reset()["image"]
        obs  = preprocessing(obs)
        done = False
        in_trinsic  = 0
        total_reward_per_episode = 0
        counter  = 0
        while not done:
            
            plt.imshow(env.render())
            action = dqn.act(obs)

            open_door = env.door.is_open

            if env.carrying == None:
            	key = False
            else:
            	key = True


            intrinsic_reward,failDfa = RA.trace(key,open_door)
            
            
            in_trinsic += intrinsic_reward


            nextObservation,reward,done,_ = env.step(action)
            nextObservation = nextObservation["image"]
            total_reward = reward + intrinsic_reward
            done = failDfa
            dqn.storeData(preprocessing(nextObservation),action,total_reward,obs,done)
            
            obs =  preprocessing(nextObservation)
            
            total_reward_per_episode += reward

            
            
              
            if len(dqn.dq) > 1000 and len(dqn.dq) % updateNetwork== 0:
                # we gonna update 
                
                # retreive the data 
                
                t_rewards ,t_dones,t_obs,t_nextObservations,t_actions = [],[],[],[],[]
                
                for n_obs,n_act,n_reward,nObs,n_done in dqn.getData(dqn.batch_size):
                    t_rewards.append(n_reward)
                    t_dones.append(n_done)
                    t_obs.append(nObs)
                    t_nextObservations.append(n_obs)
                    t_actions.append(n_act)
                    
                
                t_rewards = np.array(t_rewards)
                t_dones = np.array(t_dones)
                t_obs = np.array(t_obs)
                t_nextObservations = np.array(t_nextObservations)
                t_actions = np.array(t_actions)
                
                
                # now we have all the mini batch we can first define our target 
                # we need to send nextObservation 
                
                t_output = dqn.target_network_output.eval(feed_dict={dqn.x_input:t_nextObservations})
                
                target  = []
                
                for i in range(len(t_output)):
                    target.append(t_rewards[i] + dqn.discount*np.max(t_output[i])*(1-t_dones[i]))
                    
                
                
                target = np.array(target)
                
                game_loss,_ = sess.run([dqn.loss,dqn.optimizer],feed_dict={dqn.x_input:t_obs,dqn.actions:t_actions,\
                                                            dqn.target:target
                                                            })
                
                
              
            if counter > 1000:
                sess.run(dqn.copy_weight)
                counter = 0
                
            
            
            counter += 1
        
        
        if each_epispode !=0 and each_epispode % 20 == 0:
            print("After episode ",str(each_epispode)," the game loss is ",str(game_loss)," and reward is ",str(total_reward_per_episode))
            print("Intrinsic Reward ",str(in_trinsic))
            