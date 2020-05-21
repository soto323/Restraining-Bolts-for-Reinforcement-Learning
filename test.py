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
tf.get_logger().setLevel('ERROR')



"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow 1.x Deep Reinforcemet Learning using Restraining Bolts [TESTING]"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--env', type=str, default='MiniGrid-Unlock-v0', help='choose gym enviroement')
    parser.add_argument('--gui', type=bool, default=True, help='enable gui (nor recommended for training')
    parser.add_argument('--model_name', type=str, default=None, help='path to model if starting from checkpoint')
    parser.add_argument('--rand_seed', type=int, default=42, help='tf random seed')

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



    pass

if __name__ == "__main__":
    args = parse_args()
    if args is None:
        exit()
    main(args)