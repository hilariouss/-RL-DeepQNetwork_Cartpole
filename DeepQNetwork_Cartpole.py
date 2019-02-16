import tensorflow as tf
import random
import numpy as np
import gym
from gym import wrappers

from argparse import ArgumentParser

OUT_DIR = 'cartpole-experiment' # default saving directory
MAX_SCORE_QUEUE_SIZE = 100
GAME = 'CartPole-v0' # name of game

# Parameter notation
def get_options():
    parser = ArgumentParser()
    parser.add_argument('--MAX_EPISODE', type=int, default = 3000,
                        help='max number of episodes iteration')
    parser.add_argument('--ACTION_DIM', type=int, default=2,
                        help='number of actions one can take')
    parser.add_argument('--OBSERVATION_DIM', type=int, default=4,
                        help='number of observations one can see')
    parser.add_argument('--GAMMA', type=float, default=0.9,
                        help='discount factor of Q learning')
    parser.add_argument('--INIT_EPSILON', type=float, default=1.0,
                        help='initial probability for randomly sampling action')
    parser.add_argument('--FINAL_EPSILON', type=float, default=1e-5,
                        help='final probability for randomly sampling action')
    parser.add_argument('--EPSILON_DECAY', type=float, default=0.95,
                        help='epsilon decay rate')
    parser.add_argument('--EPSILON_ANNEAL_STEPS', type=int, default=10,
                        help='steps interval to decay epsilon')
    parser.add_argument('--LR', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--MAX_EXPERIENCE', type=int, default=2000,
                        help='size of experience replay buffer')
    parser.add_argument('--BATCH_SIZE', type=int, default=256,
                        help='mini batch size')
    parser.add_argument('--H1_SIZE', type=int, default=128,
                        help='size of hidden layer 1')
    parser.add_argument('--H2_SIZE', type=int, default=128,
                        help='size of hidden layer 2')
    parser.add_argument('--H3_SIZE', type=int, default=128,
                        help='size of hidden layer 3')
    options = parser.parse_args()
    return options

class QAgent:
    def __init__(self, options):
        # hidden layer 1
        self.W1 = self.weight_variable([options.OBSERVATION_DIM, options.H1_SIZE])
        self.b1 = self.bias_variable([options.H1_SIZE])
        # hidden layer 2
        self.W2 = self.weight_variable([options.H1_SIZE, options.H2_SIZE])
        self.b2 = self.bias_variable([options.H2_SIZE])
        # hidden layer 3
        self.W3 = self.weight_variable([options.H2_SIZE, options.H3_SIZE])
        self.b3 = self.bias_variable([options.H3_SIZE])
        # output layer
        self.W4 = self.weight_variable([options.H3_SIZE, options.ACTION_DIM])
        self.b4 = self.bias_variable([options.ACTION_DIM])

    # Define the xavier initializer
    # Input : shape of matrix
    # Output : random values (shaped with input) from a uniform distribution.
    # REASON for this function :

    # 만약 뉴럴 네트워크의 웨이트 값들을 초기에 너무 작게 설정한다면, 각 계층을 따라 signal이 전파되면서
    # 입력값이 너무 작아질 수 있다. 이는 해당 웨이트 값을 사용하기에 너무 작아질 수 있다는 문제가 있다. (즉, 계층이 존재하는
    # 의미가 약해진다.)

    # 반대로, 초기에 너무 크게 설정한다면, 전파되는 신호의 크기가 사용하기 너무 커질 수 있다는 문제가 있다. (즉, 계층의
    # 특성이 과도하게 반영될 수 있다.

    # 따라서, 이러한 문제를 완화하기 위해, 균등 분포를 활용해 웨이트 값들을 초기화 한다. 이는 xavier_initializer를
    # 활용해 구현할 수 있다.
    def xavier_initializer(self, shape):
        dim_sum = np.sum(shape)
        if len(shape) == 1:
            dim_sum+=1
        bound = np.sqrt(6.0 / dim_sum)
        return tf.random_uniform(shape, minval = -bound,maxval = bound)

    # 실제로 상기 xavier_initializer를 활용하여 웨이트를 초기화 하는 코드.
    def weight_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    # 실제로 상기 xavier_initializer를 활용하여 편향을 초기화 하는 코드.
    def bias_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    def add_value_net(self, options): # --> 생성한 layer들 이어주기.
        # Batch size of observation
        observation = tf.placeholder(tf.float32, [None, options.OBSERVATION_DIM])

        h1 = tf.nn.relu(tf.matmul(observation, self.W1)+self.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.W2)+self.b2)
        h3 = tf.nn.relu(tf.matmul(h2, self.W3)+self.b3)
        Q = tf.squeeze(tf.matmul(h3, self.W4) + self.b4)
        return observation, Q
        # observation : state space, Q : logit (prediction)

    # sample action with random rate episode
    def sample_action(self, Q, feed, eps, options):
        act_values = Q.eval(feed_dict=feed)
        if random.random() <= eps:
            action_index = random.randrange(options.ACTION_DIM)
        else:
            action_index = np.argmax(act_values)
        action = np.zeros(options.ACTION_DIM)
        action[action_index] = 1
        return action

def train(env):
    options = get_options() # load arguments
    agent = QAgent(options) # Agent (class which contains the NEURAL NETWORK)
    sess = tf.InteractiveSession() # FIND OUT WHY WE USE THIS INTERACTIVE SESSION

    obs, Q1 = agent.add_value_net(options)

    act = tf.placeholder(tf.float32, [None, options.ACTION_DIM])
    rwd = tf.placeholder(tf.float32, [None, ])

    next_obs, Q2 = agent.add_value_net(options)

    values1 = tf.reduce_sum(tf.multiply(Q1, act), reduction_indices=1) ### reduction_indices???
    values2 = rwd + options.GAMMA * tf.reduce_max(Q2, reduction_indices=1)
    loss = tf.reduce_mean(tf.square(values1 - values2))
    train_step = tf.train.AdamOptimizer(learning_rate=options.LR).minimize(loss)

    sess.run(tf.initialize_all_variables())

    # model saver and loading networks
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state("checkpoints_cartpole")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Successfully loaded:", ckpt.model_checkpoint_path)
    else:
        print("Fail to load old network weights")


    #Some initial local variables
    feed={}
    eps = options.INIT_EPSILON
    global_step = 0
    exp_pointer = 0
    learning_finished = False

    # The replay memory
    obs_queue = np.empty([options.MAX_EXPERIENCE, options.OBSERVATION_DIM])
    act_queue = np.empty([options.MAX_EXPERIENCE, options.ACTION_DIM])
    rwd_queue = np.empty([options.MAX_EXPERIENCE])
    next_obs_queue = np.empty([options.MAX_EXPERIENCE, options.OBSERVATION_DIM])

    # Score cache
    score_queue = []

    # The episode loop
    for i_episode in range(options.MAX_EPISODE):

        observation = env.reset()
        done = False
        score = 0
        sum_loss_value = 0

        # The step loop
        while not done:
            global_step += 1
            if global_step % options.EPSILON_ANNEAL_STEPS == 0 and eps > options.FINAL_EPSILON:
                eps = eps * options.EPSILON_DECAY
            env.render()

            obs_queue[exp_pointer] = observation
            action = agent.sample_action(Q1, {obs: np.reshape(observation, (1, -1))}, eps, options)
            act_queue[exp_pointer] = action
            observation, reward, done, _ = env.step(np.argmax(action))

            score += reward
            reward = score  # Reward will be the accumulative score

            if done and score < 200:
                reward = -500  # If it fails, punish hard
                observation = np.zeros_like(observation)

            rwd_queue[exp_pointer] = reward
            next_obs_queue[exp_pointer] = observation

            exp_pointer += 1
            if exp_pointer == options.MAX_EXPERIENCE:
                exp_pointer = 0  # Refill the replay memory if it is full

            if global_step >= options.MAX_EXPERIENCE:
                rand_indexs = np.random.choice(options.MAX_EXPERIENCE, options.BATCH_SIZE)
                feed.update({obs: obs_queue[rand_indexs]})
                feed.update({act: act_queue[rand_indexs]})
                feed.update({rwd: rwd_queue[rand_indexs]})
                feed.update({next_obs: next_obs_queue[rand_indexs]})
                if not learning_finished:  # If not solved, we train and get the step loss
                    step_loss_value, _ = sess.run([loss, train_step], feed_dict=feed)
                else:  # If solved, we just get the step loss
                    step_loss_value = sess.run(loss, feed_dict=feed)
                # Use sum to calculate average loss of this episode
                sum_loss_value += step_loss_value

        print("===== Episode {} ended with score = {}, avg_loss = {} ======".format(i_episode + 1, score,
                                                                               sum_loss_value / score))
        score_queue.append(score)
        if len(score_queue) > MAX_SCORE_QUEUE_SIZE:
            score_queue.pop(0)
            if np.mean(score_queue) > 195:  # The threshold of being solved
                learning_finished = True
            else:
                learning_finished = False
        if learning_finished:
            print
            "Testing !!!"
        # save progress every 100 episodes
        if learning_finished and i_episode % 100 == 0:
            saver.save(sess, 'checkpoints-cartpole/' + GAME + '-dqn', global_step=global_step)


if __name__ == "__main__":
    env = gym.make(GAME)
    #env.monitor.start(OUT_DIR, force=True) --> deprecated
    env = gym.wrappers.Monitor(env, OUT_DIR, force=True)
    train(env)
    env.monitor.close()