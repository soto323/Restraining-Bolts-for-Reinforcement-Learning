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
import sys
sys.path.append("..")

from models.utils import *

from models.common import *
from models.parser import parser

from models.ddqn.ddqn import DDQN
from models.dqn.dqn import DQN

def preprocessing(image):
	return image.reshape(-1,147)


    

def run(sess, env, algo, checkpoints_dir, n_episodes=100000, gui=False):

    fmt = '\n\n\n[*] Succesfully loaded: {}\n\n\n'.format(algo)
    if algo=='dqn':
        dqn  = DQN(env)
        print(fmt)
    elif algo=='ddqn':
         dqn  = DQN(env)
         print(fmt)
         
    RA = parser()
    dqn.loss()


    """ checkpoints load """    
    try:
        episode_to_restore = load_checkpoint(checkpoints_dir, sess)
        print("[*] restoring from episode {}".format(episode_to_restore))

    except:
        episode_to_restore = 0
        print("[*] failed to load checkpoints")
        sess.run(tf.global_variables_initializer())

    #print("[*] creating new graphs")
    #summary_writer = tf.summary.FileWriter('./graphs', sess.graph)
    saver = tf.train.Saver()
    #S_ = tf.Summary()


    updateNetwork = 4
    game_loss = 0

    for each_episode in tqdm(range(episode_to_restore, episode_to_restore+n_episodes), total=n_episodes):
        obs  = env.reset()["image"]
        obs  = preprocessing(obs)

        done = False
        in_trinsic  = 0
        total_reward_per_episode = 0
        counter  = 0
        while not done:
            
            if gui:
                plt.imshow(env.render())

            action = dqn.act(obs, sess)
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
            dqn.rememember(preprocessing(nextObservation),action,total_reward,obs,done)
            
            obs =  preprocessing(nextObservation)
            
            total_reward_per_episode += reward

            
           
              
            if dqn.memory_class.sumtree.total_priority() > 1000 and dqn.memory_class.sumtree.total_priority() % updateNetwork== 0:
                # we gonna update 
                
                # retreive the data 

                t_rewards ,t_dones,t_obs,t_nextObservations,t_actions = [],[],[],[],[]
                idx ,minibatch, ISWeights = dqn.sampleData()

                for n_obs,n_reward,n_act,nObs,n_done in minibatch:
                    t_rewards.append(n_reward)
                    t_dones.append(n_done)
                    t_obs.append(nObs)
                    t_nextObservations.append(n_obs)
                    t_actions.append(n_act)
                    
                
                t_rewards = np.array(t_rewards)
                t_dones = np.array(t_dones)
                t_obs = np.squeeze(np.array(t_obs))
                t_nextObservations = np.squeeze(np.array(t_nextObservations))
                t_actions = np.array(t_actions)


                
                # now we have all the mini batch we can first define our target 
                # we need to send nextObservation 
                
                t_output = dqn.target_network_output.eval(session = sess, feed_dict={dqn.x_input:t_nextObservations})
                
                target  = []
                
                for i in range(len(t_output)):
                    target.append(t_rewards[i] + dqn.discount*np.max(t_output[i])*(1-t_dones[i]))
                    
                
                
                target = np.array(target)
                
                game_loss,_ = sess.run([dqn.loss,dqn.optimizer],feed_dict={dqn.x_input:t_obs,dqn.actions:t_actions,\
                                                            dqn.target:target
                                                            })
                
                
              
            if counter > 500:
                sess.run(dqn.copy_weight)
                counter = 0
                
            
            
            counter += 1
        
        
        if each_episode !=0 and each_episode % 20 == 0:
            print("After episode ",str(each_episode)," the game loss is ",str(game_loss)," and reward is ",str(total_reward_per_episode))
            print("Intrinsic Reward ",str(in_trinsic))


        if each_episode % 100 == 0:
            checkpoint_save_path = saver.save(sess, '{}/Episode_{}.ckpt'.format(checkpoints_dir, each_episode))
            print('Model is saved at {}!'.format(checkpoint_save_path))
'''
/*
Bits and pieces of this code were taken from
https://github.com/pythonlessons/Reinforcement_Learning/blob/master/05_CartPole-reinforcement-learning_PER_D3QN/PER.py
*/
'''