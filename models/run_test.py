import numpy as np


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from collections import deque
import PILip
import random
import matplotlib.pyplot as plt
import flloat
from flloat.parser.ltlf import LTLfParser
from tqdm import tqdm
from gym_minigrid.wrappers import *
import sys
sys.path.append("..")

from models.utils import load_checkpoint, summary

from models.common import sumtree, memory
from models.parser import parser



def run_a2c(sess, env, algo, checkpoints_dir, n_episodes=100000, gui=False, BATCH_SIZE = 32):

    fmt = '\n\n\n[*] Succesfully loaded: {}\n\n\n'.format(algo)
    from models.a2c.a2c import actorCritic, preprocessing
    ac_network = actorCritic(env)
    print(fmt)
    ac_network.actor_critic_loss()
         
    RA = parser()

    """ checkpoints load """    
    try:
        episode_to_restore = load_checkpoint(checkpoints_dir, sess)
        print("[*] restoring from episode {}".format(episode_to_restore))

    except:
        episode_to_restore = 0
        print("[*] failed to load checkpoints")
        sess.run(tf.global_variables_initializer())

    total_reward = 0
    for each_episode in range(20): #numner of videos
        done = False
        in_trinsic  = 0
        state = env.reset()["image"]
        state  = preprocessing(state)
        
        # these are temporary data storage lists 
        save_actions, save_rewards, save_dones = [np.empty([0]) for i in range(3)]
        save_states, save_nextStates = [], [] #TO DO: change this numpy
        #BATCH_SIZE 
        c_loss = 0
        total_loss_actor_critic = 0
        count_batch = 0
        steps=0
        while not done:
            steps+=1
            if gui: 
                plt.imshow(env.render())
        
            # first thing we need to choose the action
            action = ac_network.chooseActionTest(state, sess)
            
            # now i can perform the step in the environment 
            
            next_state,reward,done,_ = env.step(action)
            next_state = next_state["image"]

            open_door = env.door.is_open

            key = False if env.carrying == None else True

            intrinsic_reward,failDfa = RA.trace(key,open_door)
            in_trinsic += intrinsic_reward
            t_reward  = reward +intrinsic_reward

            print(action)
            #print("In %d steps we got %.3f total reward and %.3f instrinsic reward" % (steps, t_reward, intrinsic_reward))

            
            
    
            
                    
            
def run(sess, env, algo, checkpoints_dir, n_episodes=100000, gui=False):

    fmt = '\n\n\n[*] Succesfully loaded: {}\n\n\n'.format(algo)
    if algo=='dqn':
        from models.dqn.dqn import DQN, preprocessing
        dqn  = DQN(env)
        print(fmt)
        dqn.loss()
    elif algo=='ddqn':
        from models.ddqn.ddqn import DDQN, preprocessing
        dqn  = DDQN(env)
        print(fmt)
        dqn.loss()
    # elif algo =='pompdp' or algo =='a2c':
    #     from a2c.a2c import actorCritic, preprocessing
    #     ac_network = actorCritic()
    #     print(fmt)
    #     ac_network.actor_critic_loss()
         
    RA = parser()

    """ checkpoints load """    
    try:
        episode_to_restore = load_checkpoint(checkpoints_dir, sess)
        print("[*] restoring from episode {}".format(episode_to_restore))

    except:
        episode_to_restore = 0
        print("[*] failed to load checkpoints")
        sess.run(tf.global_variables_initializer())


    updateNetwork = 4
    game_loss = 0
    
    for each_episode in range(20): #numner of videos
        obs  = env.reset()["image"]
        obs  = preprocessing(obs)
        total_steps = 0
        total_reward = 0

        done = False
        in_trinsic  = 0
        total_reward_per_episode = 0
        steps  = 0
        while not done:
            steps+=1
            if gui: 
                plt.imshow(env.render())

            action = dqn.act(obs, sess)
            open_door = env.door.is_open

            key = False if env.carrying == None else True

            intrinsic_reward, failDfa = RA.trace(key,open_door)
            
            in_trinsic += intrinsic_reward

            nextObservation,reward,done,_ = env.step(action)
            nextObservation = nextObservation["image"]

            #rewards
            total_reward = reward + intrinsic_reward
            
            done = failDfa

            dqn.rememember(preprocessing(nextObservation),action,total_reward,obs,done)
        
            obs =  preprocessing(nextObservation)
            
            total_reward_per_episode += reward

            print("In %d steps we got %.3f total reward and %.3f instrinsic reward" % (steps, total_reward_per_episode, intrinsic_reward))





















'''
/*
Bits and pieces of this code were taken from
https://github.com/pythonlessons/Reinforcement_Learning/blob/master/05_CartPole-reinforcement-learning_PER_D3QN/PER.py
*/
'''
