import random
import gym
import numpy as np
from gym_minigrid.wrappers import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from collections import deque
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from skimage.transform import resize
import os
import matplotlib.pyplot as plt
import tqdm

env = gym.make('MiniGrid-Unlock-v0')

# I love GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.99


# DISCOUNTED FACTOR
gamma = 0.95

# PARAMS GAME
num_episodes = 1001
buffer_size = 10000
mini_batch_size = 64
steps_per_target_update = 1000

# PARAMS NEURAL NETWKORK
learning_rate = 0.0005
cropped_state_size = [64, 32, 4]
num_input_neurons = cropped_state_size[0]
num_ouptut_neurons = env.action_space.n
common_net_hidden_dimensions = [16, 64]

# For saving in csv file the rewards
import pandas as pd
raw_data = {'episode_number':[], 'total_reward':[]}

class DDDQN:
    def __init__(self,
                 session,
                 scope_name,
                 input_size,
                 hidden_layer_sizes,
                 output_size,
                 learning_rate,
                 state_size):

        self.session = session
        self.scope_name = scope_name
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate

        with tf.variable_scope(self.scope_name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")

            # Input is 100x120x1
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.keras.initializers.glorot_normal(),
                                          name="conv1",
                                          )

            self.conv1_out = tf.nn.relu(self.conv1, name="conv1_out")

            """
            Second convnet: CNN ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.keras.initializers.glorot_normal(),
                                          name="conv2")

            self.conv2_out = tf.nn.relu(self.conv2, name="conv2_out")


            self.flatten = tf.layers.flatten(self.conv2_out)

            # Separating streams into advantage and value networks
            adv_net = tf.layers.dense(self.flatten, 32, activation=tf.nn.relu)
            adv_net = tf.layers.dense(adv_net, self.output_size)

            val_net = tf.layers.dense(self.flatten, 32, activation=tf.nn.relu)
            val_net = tf.layers.dense(val_net, 1)

            self.output = val_net + (adv_net - tf.reduce_mean(adv_net,
                                                              reduction_indices=1,
                                                              keepdims=True))

            # Placeholder for expected q-values
            self.y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

            # Using the loss method provided by tf directly
            self.loss = tf.losses.mean_squared_error(self.y, self.output)

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

    def predict(self, state):
        return self.session.run(self.output,
                                feed_dict={self.inputs_: state})

    def update(self, state, y):
        return self.session.run([self.loss, self.optimizer],
                                feed_dict={
                                    self.inputs_: state,
                                    self.y: y
                                })

    @staticmethod
    def create_copy_operations(source_scope, dest_scope):
        source_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=source_scope)
        dest_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=dest_scope)

        assert len(source_vars) == len(dest_vars)
        result = []

        for source_var, dest_var in zip(source_vars, dest_vars):
            result.append(dest_var.assign(source_var.value()))

        return result


def preprocess_frame(frame):
    # Crop the screen (remove part that contains no information)
    # [Up: Down, Left: right]
    #pred_img = array_to_img(frame)
    #pred_img.save('models/input_frame.jpg')
    cropped_frame = frame[36:-36, 36:-160]

    rgb = resize(cropped_frame, cropped_state_size)

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  # extract luminance

    o = gray.astype('float32') / 128 - 1  # normalize
    o = o.reshape(*o.shape, 1)

    #pred_img = array_to_img(o)
    #pred_img.save('models/output.jpg')
    return o


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((cropped_state_size[0], cropped_state_size[1]), dtype=np.int) for i in range(4)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


def train_dqn(main_dqn, target_dqn, mini_batch):
    """
    param: mini_batch: From the randomly sampled minbatch from replay-buffer,
                       it's a list of experiences in the form of
                       `(state, action, reward, next_state, done)`
    """
    states = np.array([x[0] for x in mini_batch])

    actions = np.array([x[1] for x in mini_batch])
    rewards = np.array([x[2] for x in mini_batch])
    next_states = np.array([x[3] for x in mini_batch])
    done = np.array([x[4] for x in mini_batch])

    # For double DQN: select the best action for next state
    target_Qs_batch = []

    # Calculate Qtarget for all actions that state
    target_output_next_states = target_dqn.predict(np.squeeze(next_states))

    # Get Q values for next_state
    main_output_next_states = main_dqn.predict(np.squeeze(next_states))

    # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a')
    for i in range(0, mini_batch_size):
        terminal = done[i]

        # We got a'
        action = np.argmax(main_output_next_states[i])

        # If we are in a terminal state, only equals reward
        if terminal:
            target_Qs_batch.append(rewards[i])

        else:
            # Take the Qtarget for action a'
            target = rewards[i] + gamma * target_output_next_states[i][action]
            target_Qs_batch.append(target)

    targets_mb = np.array([each for each in target_Qs_batch])

    main_output = main_dqn.predict(np.squeeze(states))
    main_output[np.arange(len(states)), actions] = targets_mb

    loss, optimizer = main_dqn.update(np.squeeze(states), main_output)

    return loss



class RB:
    def __init__(self):
        # parser to parse the ltlf formula
        '''
        State Legend:
            - q1 : ~key & ~nearDorr & ~Door
            - q2 : key &  ~nearDorr & ~Door
            - q3 : key & nearDoor & ~Door
            - q4 : ~key & nearDoor & ~Door
            - q5 : key & nearDoor & Door
        '''
        self.states = {"q1", "q2", "q3", "q4", "q5"}
        self.transitions()

        self.initialState = "q1"
        self.finalState = "q5"
        self.reset()

    def reset(self):
        self.current_state = self.initialState
        self.counter = {transition: 0 for transition in self.transition}

    def transitions(self):
        self.transition = {
            ("q1", "q1"): -0.005,
            ("q1", "q2"): 3.0,
            ("q1", "q4"): 0.0,
            ("q2", "q1"): -2.0,
            ("q2", "q2"): 0.01,
            ("q2", "q3"): 5.0,
            ("q3", "q2"): -0.5,
            ("q3", "q3"): -0.01,
            ("q3", "q4"): -2.0,
            ("q3", "q5"): 10.0,
            ("q4", "q1"): 0.0,
            ("q4", "q3"): 5.0,
            ("q4", "q4"): 0.0,
        }

        # Maps between fluent and DFA state
        self.state_dict = {
            # Legend: key, nearDoor, Door
            (False, False, False): "q1",
            (True, False, False): "q2",
            (True, True, False): "q3",
            (False, True, False): "q4",
            (True, True, True): "q5",
        }

    def trace(self, key, door, near):
        fluents = (key, door, near)
        new_curr = self.state_dict[fluents]
        if (self.current_state, new_curr) in self.transition:
            reward = self.transition[(self.current_state, new_curr)]
            self.counter[(self.current_state, new_curr)] += 1
        else:
            print(self.current_state, new_curr)
            reward = 0.0

        self.current_state = new_curr
        done = self.current_state == self.finalState
        if done:
            self.reset()
        return reward, done


tf.reset_default_graph()

replay_buffer = deque(maxlen=buffer_size)
last_n_rewards = deque(maxlen=num_episodes)
rewards = deque(maxlen=num_episodes)
losses = deque(maxlen=num_episodes)

max_episode_steps = 100

# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((cropped_state_size[0], cropped_state_size[1]), dtype=np.int) for i in range(4)], maxlen=4)


with tf.Session(config = config) as sess:
    main_dqn = DDDQN(session=sess,
                     scope_name="q_main",
                     input_size=num_input_neurons,
                     hidden_layer_sizes=common_net_hidden_dimensions,
                     output_size=num_ouptut_neurons,
                     learning_rate=learning_rate,
                     state_size=cropped_state_size)

    target_dqn = DDDQN(session=sess,
                       scope_name="q_target",
                       input_size=num_input_neurons,
                       hidden_layer_sizes=common_net_hidden_dimensions,
                       output_size=num_ouptut_neurons,
                       learning_rate=learning_rate,
                       state_size=cropped_state_size)

    rb = RB()
    # Saver will help us to save our model
    saver = tf.train.Saver()
    # Load the model
    saver.restore(sess, tf.train.latest_checkpoint('models/models_dddqn/'))
    #sess.run(tf.global_variables_initializer())
    # Make them identical to begin with
    sess.run(DDDQN.create_copy_operations("q_main", "q_target"))
    # Some counter for training cycle
    counter = 1
    try:
        for ep_num in tqdm.tqdm(range(0,num_episodes), total=num_episodes):
            env.reset()
            state = env.render(mode='rgb_array')
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            done = False
            episode_reward, loss, steps = 0, 0, 0
            episode_loss = []
            episode_reward_rb = 0
            episode_reward_env = 0

            # epsilon decay
            epsilon = 1. / ((ep_num / 10) + 1)

            while not done:
                # Get the state
                state = env.render(mode='rgb_array')
                state, stacked_frames = stack_frames(stacked_frames, state, True)

                # select the action
                action = 0
                if np.random.rand() < epsilon:
                    while action == 0 or action == 1:
                        action = random.randint(0, env.action_space.n-1)
                else:
                    state_reshaped = np.squeeze(state)
                    action = np.argmax(main_dqn.predict(state_reshaped.reshape((1, *state_reshaped.shape))))

                # Model checking
                open_door = env.door.is_open
                if env.carrying is None:
                    key = False
                else:
                    key = True

                agent_pos = env.agent_pos
                door_pos = env.door.cur_pos
                if door_pos[0] - 1 == agent_pos[0] and door_pos[1] == agent_pos[1]:
                    near = True
                else:
                    near = False

                reward_rb, done_rb = rb.trace(key, near, open_door)

                # execute the action
                obs, reward, done, info = env.step(action)

                # get next state
                next_state = env.render(mode='rgb_array')
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, True)

                episode_reward_rb += reward_rb
                episode_reward_env += reward

                if ep_num > max_episode_steps:
                    done = True
                    reward = -0.1

                replay_buffer.append((state, action, reward, next_state, done))

                # sample from the buffer and train
                if counter > mini_batch_size:
                    mini_batch = random.sample(replay_buffer, (mini_batch_size))
                    loss = train_dqn(main_dqn, target_dqn, mini_batch)
                    episode_loss.append(loss)

                if steps % steps_per_target_update == 0:
                    sess.run(DDDQN.create_copy_operations("q_main", "q_target"))

                counter += 1
                episode_reward += episode_reward_rb + episode_reward_env
                steps += 1
                state = next_state

            if ep_num % 100 == 0:
                print("Episode number: ", ep_num, " \t rb reward: ", episode_reward_rb, "\t env reward: ", episode_reward_env)
                print("Loss: ", np.mean(episode_loss))

            last_n_rewards.append(episode_reward)
            last_n_avg_reward = np.mean(last_n_rewards)
            rewards.append(episode_reward)
            losses.append(np.mean(episode_loss))

            # Save model every 5 episodes
            if ep_num % 500 == 0:
                save_path = saver.save(sess, "models/models_dddqn/model.ckpt")
                print("Model Saved")
                # Saving results into csv
                raw_data['episode_number'] = np.arange(len(rewards))
                raw_data['total_reward'] = np.array(rewards)
                df = pd.DataFrame(raw_data, columns=raw_data.keys())
                if not os.path.exists("models/models_dddqn/results.csv"):
                    df.to_csv("models/models_dddqn/results.csv", sep="\t")
                else:
                    df_old = pd.read_csv("models/models_dddqn/results.csv", sep = "\t")
                    df = df.append(df_old, ignore_index=True)
    except KeyboardInterrupt:
        print("SIGINT interception")
        pass

raw_data['episode_number'] = np.arange(len(rewards))
raw_data['total_reward'] = np.array(rewards)
data = pd.DataFrame(raw_data, columns=raw_data.keys())

x = data['episode_number']
y = data['total_reward']
plt.plot(x, y)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")

plt.show()