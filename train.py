
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

from DQNRA import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow 1.x Deep Reinforcemet Learning using Restraining Bolts"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--episodes', type=int, default=10000, help='The number of episodes to run')
    parser.add_argument('--env', type=str, default='MiniGrid-Unlock-v0', help='choose gym enviroement')
    parser.add_argument('--algo', type=str, default='dqn', help='Deep RL algorithm')
    parser.add_argument('--gui', type=bool, default=False, help='enable gui (nor recommended for training')
    parser.add_argument('--model_name', type=str, default=None, help='path to model if starting from checkpoint')
    parser.add_argument('--rand_seed', type=int, default=42, help='tf random seed')

    return parser.parse_args()


def main(args):

    """ saving paths """
    output_dir = "logs"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    if args.model_name is None:
        t = time.strftime('%Y-%m-%d_%H_%M_%S_%z')
        model_name = "env_{}_algo_{}_ep_{}_{}".format(args.env, args.algo, args.episodes, t)
        print("[*] created model folder: {}".format(model_name))
        model_dir = '{}/{}'.format(output_dir, model_name)
    else:
        model_name = args.model_name
        print("[*] proceeding to load model: {}".format(model_name))
        model_dir = model_name
        
    image_dir = '{}/images'.format(model_dir)
    checkpoints_dir = '{}/checkpoints'.format(model_dir)
    for path in [output_dir, model_dir, image_dir, checkpoints_dir]:
        if not os.path.exists(path):
            os.mkdir(path)

    """ tf session definitions """
    tf.reset_default_graph()
    tf.random.set_random_seed(args.rand_seed)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)


    """ load env """
    print("[*] attempting to load {} env".format(args.env))
    env = gym.make(args.env)
    print("[*] success")

    """ main loop """
    
    run(sess, env, checkpoints_dir, n_episodes=args.episodes, gui=args.gui)
     # TO DO: algorithm import
     # algo define run
    


if __name__ == "__main__":
    args = parse_args()
    if args is None:
        exit()
    main(args)