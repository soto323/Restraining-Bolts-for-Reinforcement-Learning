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



def run_a2c(sess, env, algo, checkpoints_dir, n_episodes=100000, gui=False):

    fmt = '\n\n\n[*] Succesfully loaded: {}\n\n\n'.format(algo)
    from models.a2c.a2c import actorCritic, preprocessing
    ac_network = actorCritic(env)
    print(fmt)
    ac_network.actor_critic_loss()
    BATCH_SIZE = 32
         
    RA = parser()

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

    total_reward = 0

    for each_episode in tqdm(range(episode_to_restore, episode_to_restore+n_episodes), total=n_episodes):
        
        # we will use monte carlo approach
        # that is we first record the data 
        # than use it 
        
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

        while not done:
        
            # first thing we need to choose the action
            action = ac_network.chooseAction(state, sess)
            
            # now i can perform the step in the environment 
            
            next_state,reward,done,_ = env.step(action)
            next_state = next_state["image"]

            open_door = env.door.is_open

            key = False if env.carrying == None else True

            intrinsic_reward,failDfa = RA.trace(key,open_door)
            in_trinsic += intrinsic_reward
            t_reward  = reward +intrinsic_reward
            
            # now we gonna store the trajectory 
            
            #save_states = np.append(save_states, state)
            save_states.append(state)
            state = preprocessing(next_state)
            #save_nextStates = np.vstack(save_nextStates, state, axis=0)
            save_nextStates.append(state)
            save_actions = np.append(save_actions, action)
            save_rewards = np.append(save_rewards, t_reward)
            save_dones = np.append(save_dones, done)
            
            total_reward += reward
        
            # done = failDfa

            if count_batch % BATCH_SIZE == 0 and count_batch!=0:
                save_nextStates = np.array(save_nextStates)
                save_states = np.array(save_states)

                value_predicted  = ac_network.act_value.eval(session = sess, feed_dict={ac_network.x_input:save_nextStates})
                target  = []
                save_rewards = (save_rewards  - save_rewards.mean())/(save_rewards.std() + np.finfo(np.float32).eps)

                for i in range(len(value_predicted)):
                    target.append(save_rewards[i] + ac_network.discount*value_predicted[i]*(1-save_dones[i]))

                target = np.array(target)
                save_actions = save_actions.astype(np.int)
                action_convert = save_actions.reshape(-1)
                onehotEncodeAction = np.eye(ac_network.action_size)[action_convert]
                total_loss_actor_critic, _ = sess.run([ac_network.total_loss, ac_network.total_optimizer],
                                            feed_dict  = {ac_network.x_input : save_states,
                                                          ac_network.target : target,
                                                          ac_network.actions : onehotEncodeAction})
                
                count_batch = 0

                #resetting
                save_actions, save_rewards, save_dones = [np.empty([0]) for i in range(3)]
                save_states, save_nextStates = [], [] #TO DO: change this numpy

            count_batch+=1

        
        if each_episode % 20 == 0 and each_episode!=0:
            
            print("After episode ",str(each_episode)," the total loss  ",str(np.sum(total_loss_actor_critic))," And reward ",str(total_reward))
            print("Intrinsic reward after episode ",str(each_episode), "  is ",str(in_trinsic))
            total_reward = 0
            
            
                
            
            
                    
            
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

            ### RB
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

            if dqn.memory_class.sumtree.total_priority() > 1000 and dqn.memory_class.sumtree.total_priority() % updateNetwork== 0:
                # update and retrieve data
                idx, minibatch, ISWeights = dqn.sampleData()

                t_rewards = np.array([i[1] for i in minibatch])
                t_dones = np.array([i[4] for i in minibatch])
                t_actions = np.array([i[2] for i in minibatch])
                t_obs =  np.squeeze(np.array([i[0] for i in minibatch]))
                t_nextObservations = np.squeeze(np.array([i[3] for i in minibatch]))

                # now we have all the mini batch we can first define our target 
                # we need to send nextObservation 
                
                t_output = dqn.target_network_output.eval(session = sess, feed_dict={dqn.x_input:t_nextObservations})
                
                target = np.array([
                    t_rewards[i] + dqn.discount*np.max(t_output[i])*(1-t_dones[i])
                     for i in range(len(t_output))])
                
                game_loss, _ = sess.run([dqn.loss, dqn.optimizer], 
                                        feed_dict={dqn.x_input:t_obs,
                                                   dqn.actions:t_actions,
                                                   dqn.target:target}
                                        )
                
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