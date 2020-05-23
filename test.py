
import os
import tqdm
import re
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import warnings
from datetime import timedelta
import logging
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
import argparse
#tf.get_logger().setLevel('ERROR')

from gym_minigrid.wrappers import *

import numpy as np
from collections import deque
import PIL
import random
import matplotlib.pyplot as plt
import flloat
from flloat.parser.ltlf import LTLfParser

from models.run_test import *

from gym import wrappers


"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow 1.x Deep Reinforcemet Learning using Restraining Bolts [TESTING]"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--env', type=str, default='MiniGrid-Unlock-v0', help='choose gym enviroement')
    parser.add_argument('--gui', type=bool, default=True, help='enable gui (nor recommended for training')
    parser.add_argument('--model_name', type=str, default=None, help='path to model if starting from checkpoint')
    parser.add_argument('--rand_seed', type=int, default=42, help='tf random seed')
    parser.add_argument('--record', type=bool, default=False, help='video record attemps')
    

    return parser.parse_args()




def main(args):

    #algorithm
        # algo -config
    
    # parser -- RB
       # asserttions
    # 
    # train step
    #checkpoitn loading
    #checkpoitn saving

    model_name = args.model_name
    print("[*] proceeding to load model: {}".format(model_name))
    model_dir = model_name

    tf.reset_default_graph()
    tf.random.set_random_seed(args.rand_seed)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)

    """ load env """
    print("[*] attempting to load {} env".format(args.env))

    assert args.env == model_name.split("/")[1].split("_algo")[0].split("env_")[1], "use the same env as the model"
    env = gym.make(args.env)
    print("[*] success")
    if args.record:
        #env = gym.wrappers.Monitor(env, args.record, resume = True)
        env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
        print("recording")
    algo = str(re.search(r'algo_(.*?)_', model_name).group(1))
    print(type(algo))
    supported_algorithms = ['dqn', 'ddqn', 'a2c', 'pompdp']
    assert algo in supported_algorithms, "Unsupported Algorithm! Please choose a supported one: {}".format(*supported_algorithms)
    """ main loop """
    checkpoints_dir = '{}/checkpoints'.format(model_dir)
    if algo in ['dqn', 'ddqn']:
        run(sess=sess, env=env, algo=algo, checkpoints_dir = checkpoints_dir, gui=args.gui)
    else:
        run_a2c(sess=sess, env=env, algo=algo, checkpoints_dir = checkpoints_dir, gui=args.gui)




if __name__ == "__main__":
    args = parse_args()
    if args is None:
        exit()
    main(args)