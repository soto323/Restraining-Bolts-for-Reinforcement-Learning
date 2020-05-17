from collections import deque
import random 
import tensorflow as tf
import PIL
import matplotlib.pyplot as plt


from gym_minigrid.wrappers import *
env = gym.make('MiniGrid-Empty-8x8-v0')
env = RGBImgPartialObsWrapper(env)
env = ImgObsWrapper(env)


class DQN:
    
    def __init__(self):

        self.discount = 0.95
        self.batch_size = 32
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.steps_done = 0
        self.EPS_DECAY = 200
        self.output_size = env.action_space.n
        self.d_len =  1000000
        self.dq = deque(maxlen=self.d_len)
        
        with tf.variable_scope(name_or_scope="placeholder",reuse=tf.AUTO_REUSE):
            self.x_input =  tf.placeholder(tf.float32,shape=(None,56,56,4),name="input")
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
            self.dense1 =  tf.get_variable(name="dense1",shape=(1568,256),initializer=tf.initializers.variance_scaling)
            self.dense2 =  tf.get_variable(name="dense2",shape=(256,self.output_size),initializer=tf.initializers.variance_scaling)
    
            
            
            
    
    def architecture(self,name):
        
        # first call the architecture 
        self.weightMatrix(name)
        with tf.variable_scope(name,reuse=False):
            
            self.conv = tf.nn.conv2d(self.x_input,self.conv1,strides=[1,4,4,1],padding='SAME')
            self.conv = tf.nn.relu(self.conv)
            
            self.conv = tf.nn.conv2d(self.conv,self.conv2,strides=[1,2,2,1],padding="SAME")
            self.conv =  tf.nn.relu(self.conv)
            
            
            # flatten 
            flatten_data = tf.keras.layers.Flatten()(self.conv)
            
            
            self.dense  = tf.matmul(flatten_data,self.dense1)
            self.dense  = tf.nn.relu(self.dense)
            
            self.output = tf.matmul(self.dense,self.dense2)
            
            trainable_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,name)
            
            return self.output,trainable_param
        
    
    def act(self,state):
        # to get what action to choose 
        
        # we need to do exploration vs exploitation 
        
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        action = None
        
        if np.random.rand() < eps_threshold:
            
            action = env.action_space.sample()
        else:
            
            
            action  = self.policy_network_output.eval(feed_dict={self.x_input:[state]})[0]
            action  =  np.argmax(action)
            
            
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
        
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
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
            



#  main loop 
def preprocessing(image):
    img =  PIL.Image.fromarray(image).convert("L")
    img = np.array(img).T/255.0
    img = np.expand_dims(img,axis=2)
    img = np.tile(img,4)
    return img


tf.reset_default_graph()
dqn  = DQN()
dqn.loss()

init   = tf.global_variables_initializer()



updateNetwork = 4
game_loss = 0
with tf.Session() as sess:
    sess.run(init)
    
    
    for each_epispode in range(1000):
        obs  = env.reset()
        obs  = preprocessing(obs)
        done = False
        total_reward = 0
        counter  = 0
        while not done:
            
            plt.imshow(env.render())
            # get the action 
            action  = dqn.act(obs)
                  
            # we gonna act in evironment 
            nextObservation,reward,done,_ = env.step(action)

            if reward == 0:
            	reward  = -10
            
            
            #we can now save this information into buffer 
            dqn.storeData(preprocessing(nextObservation),action,reward,obs,done)
            
            # we can now move to next observation 
            obs =  preprocessing(nextObservation)
            
            total_reward += reward
            # now we gonna train the network 
            
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
                
                # now we can feed the data 
                
                
                game_loss,_ = sess.run([dqn.loss,dqn.optimizer],feed_dict={dqn.x_input:t_obs,dqn.actions:t_actions,\
                                                            dqn.target:target
                                                            })
                
                
                
            
            
            if counter > 500:
                sess.run(dqn.copy_weight)
                counter = 0
                
            
            
            counter += 1
        
        
        if each_epispode !=0 and each_epispode % 20 == 0:
            print("After episode ",str(each_epispode)," the game loss is ",str(game_loss)," and reward is ",str(total_reward))
            
            
                
            
            

    