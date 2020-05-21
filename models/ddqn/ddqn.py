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


class DDQN:

    def __init__(self):
        self.memory_class = memory(1000)
        self.discount = 0.95
        self.batch_size = 32
        self.EPS_START = 0.9
        self.EPS_END = 0.5
        self.steps_done = 0
        self.EPS_DECAY = 200
        self.output_size = env.action_space.n
        self.d_len = 10000
        self.dq = deque(maxlen=self.d_len)

        with tf.variable_scope(name_or_scope="placeholder", reuse=tf.AUTO_REUSE):
            self.x_input = tf.placeholder(tf.float32, shape=(None, 147), name="input")
            self.target = tf.placeholder(tf.float32, shape=(None,), name="target")
            self.actions = tf.placeholder(tf.int32, shape=(None,), name="actions")

        # we need to define two network policy and target network where target is freeze

        self.policy_network_output, self.policy_network_param = self.architecture("policyNetwork")
        self.target_network_output, self.target_network_param = self.architecture("targetNetwork")

        # shfiting of the weights
        # we need to shift the weight from policy to target

        tau = 0.001
        self.copy_weight = []
        for target, source in zip(self.target_network_param, self.policy_network_param):
            self.copy_weight.append(target.assign(target * (1 - tau) + source * tau))

        self.copy_weight = tf.group(*self.copy_weight)

    def weightMatrix(self, name):
        with tf.variable_scope(name_or_scope=name, reuse=False):
            # defining the weights for it
            self.dense1 = tf.get_variable(name="dense1", shape=(147, 128), initializer=tf.initializers.variance_scaling)
            self.dense2 = tf.get_variable(name="dense2", shape=(128, 256), initializer=tf.initializers.variance_scaling)
            self.dense3 = tf.get_variable(name="dense3", shape=(256, 128), initializer=tf.initializers.variance_scaling)
            self.dense4 = tf.get_variable(name="dense4", shape=(128, 128), initializer=tf.initializers.variance_scaling)
            # Separating streams into advantage and value networks
            self.dense_adv_net = tf.get_variable(name="adv_net", shape=(128, self.output_size), initializer=tf.initializers.variance_scaling)
            self.dense_val_net = tf.get_variable(name="val_net", shape=(128, 1), initializer=tf.initializers.variance_scaling)

    def architecture(self, name):

        # first call the architecture
        self.weightMatrix(name)
        with tf.variable_scope(name, reuse=False):
            self.dense = tf.matmul(self.x_input, self.dense1)
            self.dense = tf.nn.relu(self.dense)

            self.dense = tf.matmul(self.dense, self.dense2)
            self.dense = tf.nn.relu(self.dense)

            self.dense = tf.matmul(self.dense, self.dense3)
            self.dense = tf.nn.relu(self.dense)

            self.dense = tf.matmul(self.dense, self.dense4)
            self.dense = tf.nn.relu(self.dense)

            # Separating streams into advantage and value networks
            self.adv_net = tf.matmul(self.dense, self.dense_adv_net)
            self.adv_net = tf.nn.relu(self.adv_net)

            self.val_net = tf.matmul(self.dense, self.dense_val_net)
            self.val_net = tf.nn.relu(self.val_net)

            self.output = self.val_net + (self.adv_net - tf.reduce_mean(self.adv_net, reduction_indices=1,
                                                                             keepdims=True))

            #self.output = tf.matmul(self.dense, self.dense4)

            trainable_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)

            return self.output, trainable_param

    def act(self, state):
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

            action_ = self.policy_network_output.eval(feed_dict={self.x_input: state})[0]
            action = np.argmax(action_)

        return action

    def huberLoss(self, a, b):
        error = a - b
        result = tf.cond(tf.reduce_mean(tf.math.abs(error)) > 1.0, lambda: tf.math.abs(error) - 0.5,
                         lambda: error * error / 2)
        return result

    def loss(self):

        # we need to substract the target from the actual Q values
        q_sa = tf.reduce_sum(self.policy_network_output * tf.one_hot(self.actions, self.output_size), axis=-1)
        # now we can substract this value from the target value

        self.loss = self.huberLoss(self.target, q_sa)
        self.loss = tf.reduce_mean(self.loss)

        # we only want to use policy network
        self.grad = tf.gradients(self.loss, self.policy_network_param)

        # once i have the gradient wrt to policy net we can apply to network
        # but to deal with exploding gradient we perform gradient clipping

        clipped_grad = []
        for val in self.grad:
            clipped_grad.append(tf.clip_by_value(val, clip_value_min=-1, clip_value_max=1))

        # now we can apply the clipped grad values

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
        self.optimizer = self.optimizer.apply_gradients(zip(clipped_grad, self.policy_network_param))

    def rememember(self, nextObservation, action, reward, observation, done):
        experiance = nextObservation, reward, action, observation, done
        self.memory_class.store(experiance)

    # def storeData(self,nextObservation,action,reward,observation,done):
    #     if self.d_len - len(self.dq) <=0:
    #         self.dq.pop()

    #     # now we can store the data
    #     self.dq.append((nextObservation,action,reward,observation,done))

    def sampleData(self):

        idx, minibatch, isWeight = self.memory_class.sample(self.batch_size)
        return idx, minibatch, isWeight

    def getData(self, batch_size):
        data = None
        if len(self.dq) > batch_size:
            data = random.sample(self.dq, batch_size)

        return data




def preprocessing(image):
	return image.reshape(-1,147)


# dqn = DDQN()
# RA = parser()

# dqn.loss()

# key_door = []

# updateNetwork = 4
# game_loss = 0


# with tf.Session(config=config) as sess:
#     sess.run(tf.global_variables_initializer())
#     # Saver will help us to save our model
#     saver = tf.train.Saver()

#     with open("models_dueling/dresults.txt", "w") as f:
#         f.write("Ep num \t Loss \t Reward \t Intrinsic Reward \n ")
#         f.close()

#     for each_epispode in range(50000):
#         obs  = env.reset()["image"]
#         obs  = preprocessing(obs)
#         done = False
#         in_trinsic  = 0
#         counter  = 0
#         total_reward_per_episode, total_loss_per_episode = 0, 0
#         steps_episode = 0
#         while not done:
#             #plt.imshow(env.render())
#             action = dqn.act(obs)

#             open_door = env.door.is_open

#             if env.carrying == None:
#                 key = False
#             else:
#                 key = True

#             intrinsic_reward,failDfa = RA.trace(key,open_door)

#             in_trinsic += intrinsic_reward

#             nextObservation,reward,done,_ = env.step(action)
#             reward = reward*10
#             nextObservation = nextObservation["image"]
#             total_reward = reward + intrinsic_reward
#             done = failDfa

#             dqn.rememember(preprocessing(nextObservation), action, total_reward, obs, done)

#             obs = preprocessing(nextObservation)

#             total_reward_per_episode += reward

#             if dqn.memory_class.sumtree.total_priority() > 1000 and dqn.memory_class.sumtree.total_priority() % updateNetwork== 0:
#                 # we gonna update

#                 # retreive the data

#                 t_rewards ,t_dones,t_obs,t_nextObservations,t_actions = [],[],[],[],[]
#                 idx, minibatch, ISWeights = dqn.sampleData()

#                 for n_obs, n_reward, n_act, nObs, n_done in minibatch:
#                     t_rewards.append(n_reward)
#                     t_dones.append(n_done)
#                     t_obs.append(nObs)
#                     t_nextObservations.append(n_obs)
#                     t_actions.append(n_act)

#                 t_rewards = np.array(t_rewards)
#                 t_dones = np.array(t_dones)
#                 t_obs = np.squeeze(np.array(t_obs))
#                 t_nextObservations = np.squeeze(np.array(t_nextObservations))
#                 t_actions = np.array(t_actions)

#                 # now we have all the mini batch we can first define our target
#                 # we need to send nextObservation

#                 t_output = dqn.target_network_output.eval(feed_dict={dqn.x_input:t_nextObservations})

#                 target  = []

#                 for i in range(len(t_output)):
#                     target.append(t_rewards[i] + dqn.discount*np.max(t_output[i])*(1-t_dones[i]))

#                 target = np.array(target)

#                 game_loss, _ = sess.run([dqn.loss, dqn.optimizer],
#                                        feed_dict={dqn.x_input: t_obs, dqn.actions: t_actions, dqn.target: target})

#                 total_loss_per_episode += game_loss

#             if counter > 1000:
#                 sess.run(dqn.copy_weight)
#                 counter = 0
#             counter += 1
#             steps_episode += 1

#         # Save model every 5 episodes
#         if each_epispode % 500 == 0:
#             save_path = saver.save(sess, "models_dueling/dmodel.ckpt")
#             print("Model Saved")

#         # Saving results into txt
#         with open("models_dueling/dresults.txt", "a") as f:
#             f.write(str(each_epispode)+"\t\t"+str(round(total_loss_per_episode/steps_episode,4))+"\t\t"+str(round(total_reward_per_episode,2))+"\t\t"+str(round(in_trinsic,2))+"\n")
#             f.close()

#         if each_epispode !=0 and each_epispode % 20 == 0:
#             print("After episode ",str(each_epispode)," the game loss is ",str(game_loss)," and reward is ",str(total_reward_per_episode))
#             print("Intrinsic Reward ",str(in_trinsic))
