import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import argparse
sys.path.insert(0, '../shared/')
from q_network import QNetwork
from api import API_KEY
from parser import create_parser


def get_reward(done, reward):
    """
    Return a more meaningful reward (i.e. depending on the outcome of every action and not just on winning)
    :param done: bool indicating if the episode if finished
    :param reward: reward returned by the env after taking an action
    :return:
    """
    # Failure
    if done and reward == 0:
        reward = - 100.0
    # Win
    elif done:
        reward = 100.0
    # Move to another case
    else:
        reward = - 1.0
    return reward


def main():
    game = 'FrozenLake-v0'
    save_folder = "results/{}/".format(game)

    parser = create_parser(game=game)
    args = parser.parse_args()
    env = gym.make(game)
    # Number of episodes to train the network on
    MAX_EPISODES = 1000
    # Maximum number of steps before loosing
    MAX_STEPS = env.spec.timestep_limit

    # Create network
    q_network = QNetwork(in_dimension=env.observation_space.n, out_dimension=env.action_space.n, discount_factor=0.99,
                         start_epsilon=0.5, decay_rate=0.99, decay_step=10, learning_rate=0.1)
    q_network.create_network_graph()

    # Monitor episodes
    if args.publish:
        env = gym.wrappers.Monitor(env, save_folder, force=True)

    # init operator
    init = tf.global_variables_initializer()
    # Store the average reward over the 100 last consecutive trials
    avg_last_100_list = []
    with tf.Session() as sess:
        sess.run(init)
        # Store outcome of each episode
        results = []
        for i in range(MAX_EPISODES):
            # Compute the average reward over the 100 last consecutive trials
            if args.stats and len(results) > 0:
                nth_last_element = min(100, len(results))
                avg_last_100 = np.sum(results[len(results) - nth_last_element: len(results)]) / float(nth_last_element)
                avg_last_100_list.append(avg_last_100)
                print("Episode", i, "AVG", avg_last_100, "EPSI", q_network.get_current_epsilon())

            # Reset environment and get first new observation
            obs = env.reset()

            for step in range(0, MAX_STEPS):
                # TODO wrap everything in the q_network class
                # Choose an action by greedily (with e chance of random action) from the Q-network
                # And retrieve scores for all possible actions
                predicted_action, allQ = sess.run([q_network.prediction_op, q_network.Q_out],
                                                  feed_dict={q_network.states: np.identity(q_network.in_dimension)[obs:obs + 1]})

                # Random action ?
                if np.random.rand(1) < q_network.get_current_epsilon():
                    predicted_action = env.action_space.sample()

                # Get new state and reward from environment
                new_obs, reward, done, _ = env.step(predicted_action)
                # Get a more meaningful reward (i.e. depending on the outcome of every action and not just on winning)
                reward = get_reward(done=done, reward=reward)

                # Obtain the outcome probability for the new_obs state by feeding the new state through our network
                all_next_outcomes = sess.run(q_network.Q_out, feed_dict={
                    q_network.states: np.identity(q_network.in_dimension)[new_obs:new_obs + 1]})

                # Obtain the best possible outcome remembered by the network for this state
                best_possible_outcome = np.max(all_next_outcomes)
                targetQ = allQ
                # Set the target value accordingly to the choosen action and the current state (observation)
                targetQ[0, predicted_action] = reward + q_network.discount_factor * best_possible_outcome

                # Train our network using target (taking into account the reward) and predicted Q values
                q_network.train(session=sess, observation=obs, targetQ=targetQ)

                obs = new_obs
                if done:
                    # Progressively reduce epsilon value as the network is learning
                    q_network.end_episode(current_episode=i)
                    results.append(reward > 0)
                    break

        if args.stats:
            plt.plot(avg_last_100_list)
            plt.show()

    if args.publish:
        env.close()
        gym.upload(save_folder, api_key=API_KEY)


if __name__ == "__main__":
    main()
