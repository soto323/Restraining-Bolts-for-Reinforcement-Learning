import numpy as np
import tensorflow as tf
from collections import deque
import PIL
import random
import matplotlib.pyplot as plt
import flloat
from flloat.parser.ltlf import LTLfParser
from tqdm import tqdm
from gym_minigrid.wrappers import *

from models.utils import *

from models.common import *
from models.parser import parser

class DQN:
    'new'
    def __init__(self, env):
        self.env = env
        self.memory_class = memory(100000)
        self.discount = 0.99
        self.batch_size = 128
        self.EPS_START = 0.9
        self.EPS_END = 0.5
        self.steps_done = 0
        self.EPS_DECAY = 100
        self.output_size = self.env.action_space.n
        self.d_len =  100000
        self.dq = deque(maxlen=self.d_len)

        
        with tf.variable_scope(name_or_scope="placeholder",reuse=tf.AUTO_REUSE):
            self.x_input =  tf.placeholder(tf.float32,shape=(None,147),name="input")
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
            self.dense1 = tf.get_variable(name="dense1",shape=(147,128),initializer = tf.initializers.glorot_normal)
            self.dense1Bias = tf.get_variable(name="dense1_bias",shape=(128),initializer=tf.initializers.glorot_normal)

            self.dense2 =  tf.get_variable(name="dense2",shape=(128,128),initializer=tf.initializers.glorot_normal)
            self.dense2Bias = tf.get_variable(name="dense2Bias",shape=(128),initializer=tf.initializers.glorot_normal)

            self.dense3 =  tf.get_variable(name="dense3",shape=(128,128),initializer=tf.initializers.glorot_normal)
            self.dense3Bias = tf.get_variable(name="dense3Bias",shape=(128),initializer=tf.initializers.glorot_normal)

            self.dense4 =  tf.get_variable(name="dense4",shape=(128,self.output_size),initializer=tf.initializers.glorot_normal)
            self.dense4Bias = tf.get_variable(name="dense4Bias",shape=(self.output_size),initializer=tf.initializers.glorot_normal)

        
    
    def architecture(self,name):
        
        # first call the architecture 
        self.weightMatrix(name)
        with tf.variable_scope(name,reuse=False):
            self.dense  = tf.matmul(self.x_input,self.dense1)
            self.dense  = tf.nn.relu(self.dense)
            self.dense  = tf.add(self.dense,self.dense1Bias)

            self.dense  =  tf.matmul(self.dense,self.dense2)
            self.dense  = tf.nn.relu(self.dense)
            self.dense  =  tf.add(self.dense,self.dense2Bias)

            self.dense  = tf.matmul(self.dense,self.dense3)
            self.dense  =  tf.nn.relu(self.dense)
            self.dense  = tf.add(self.dense,self.dense3Bias)
            
            self.output = tf.matmul(self.dense,self.dense4)
            self.output =  tf.add(self.output,self.dense4Bias)
            self.output = tf.nn.softmax(self.output)

            trainable_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,name)
            
            return self.output,trainable_param
        
    
    def act(self,state, sess):
        # to get what action to choose 
        
        # we need to do exploration vs exploitation 
        
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        action = None
        action_ = None
        if np.random.rand() < eps_threshold:
            
            action = self.env.action_space.sample()
            
        else:
            
            action_  = self.policy_network_output.eval(session = sess, feed_dict={self.x_input:state})[0]
            action  =  np.argmax(action_)
            
        
        
        return action
        
        
    
    def huberLoss(self,a,b):
    	error  = a -b
    	result  = tf.cond(tf.reduce_mean(tf.math.abs(error)) > 1.0,lambda:tf.math.abs(error)-0.5,lambda:error*error/2)
    	return result

    def loss(self):

        # we need to substract the target from the actual Q values 
        q_sa = tf.reduce_sum(self.policy_network_output*tf.one_hot(self.actions,self.output_size),axis=-1)
        # now we can substract this value from the target value 


        self.loss =  self.huberLoss(self.target,q_sa)
        self.loss  =  tf.reduce_mean(self.loss)

        # we only want to use policy network 
        self.grad = tf.gradients(self.loss,self.policy_network_param)

        # once i have the gradient wrt to policy net we can apply to network 
        # but to deal with exploding gradient we perform gradient clipping 

        clipped_grad = []
        for val in self.grad:
            clipped_grad.append(tf.clip_by_value(val,clip_value_min=-1,clip_value_max=1))

        # now we can apply the clipped grad values 

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.optimizer =  self.optimizer.apply_gradients(zip(clipped_grad,self.policy_network_param))

    def rememember(self,nextObservation,action,reward,observation,done):
        experiance =  nextObservation,reward,action,observation,done
        self.memory_class.store(experiance)
            
    def sampleData(self):
        
        idx,minibatch,isWeight = self.memory_class.sample(self.batch_size)
        return idx,minibatch,isWeight

    def getData(self,batch_size):
        data  = None
        if len(self.dq) > batch_size:
            data = random.sample(self.dq,batch_size)
            
        return data 

def preprocessing(image):
	return image.reshape(-1,147)