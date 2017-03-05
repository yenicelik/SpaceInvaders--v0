""" Trains a model depending on  """

import random
import sys

import tensorflow as tf

from helper import preprocess_image

from helper import save_figs

import numpy as np

import datetime
import time


#consider using global variables for this!!!

def train(env, parameter, saver, forward_dict, loss_dict):
    """ Trains the network to the given environment. Saves weights to saver
        In: env (OpenAI gym environment)
        In: parameter (Dictionary with settings)
        In: saver (tf.saver where weights and bias will be saved to)
        Out: rewards_list (reward for instantenous run)
        Out: steps_list (number of steps 'survived' in given episode)
    """

    reward_list = []
    steps_list = []
    
    if parameter['OLD_TF']:
        init = tf.global_variables_initializer()  #replace for newer versions
    else:
        init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for i in xrange(parameter['NUM_EPISODES']):

            start_time = datetime.datetime.now()

            total_reward, steps = run_episode(
                                    env=env,
                                    cur_episode=i, 
                                    sess=sess,
                                    parameter=parameter, 
                                    forward_dict=forward_dict, 
                                    loss_dict=loss_dict
                                )

            end_time = datetime.datetime.now()

            reward_list.append(total_reward)
            steps_list.append(steps)

            percentage = float(i)/parameter['NUM_EPISODES']
            total_time=(end_time - start_time)

            #print (i, parameter['NUM_EPISODES'])
            if i % parameter['SAVE_EVERY'] == 0:
                #save_figs(reward_list, steps_list, parameter)
                saver.save(sess, 'modelSpaceInvader.ckpt', global_step=i)
                
            print "Progress: {0:.3f}%%".format(percentage * 100)
            print "EST. time per episode: " + str(total_time)
            print "Episodes left: {0:d}".format(parameter['NUM_EPISODES'] - i)
            print "Percent of successful episodes: " + str(sum(reward_list)/parameter['NUM_EPISODES']) + "%"
            print

        #save_figs(reward_list, steps_list, parameter)
        saver.save(sess, 'modelSpaceInvader.ckpt', global_step=parameter['NUM_EPISODES'] + 1)

    return reward_list, steps_list
 

def run_episode(env, sess, cur_episode, parameter, forward_dict, loss_dict):
    """ Run one episode of  environment
        In: env
        In: cur_episode
        In: parameter
        Out: total_reward (accumulated over episode)
        Out: steps (needed until termination)
    """

    observation = env.reset()
    total_reward = 0
    done = False
    steps = 1

    while steps < parameter['NUM_STEPS']:
        new_observation, reward, done = step_environment(
                            env=env,
                            observation=observation,
                            sess=sess,
                            eps=parameter['EPS']/(cur_episode+1) + 0.06, #consider turning this into a lambda dictionary function
                            gamma=parameter['GAMMA'],
                            forward_dict=forward_dict, 
                            loss_dict=loss_dict
                        )
        total_reward += reward
        if done:
            break

        observation = new_observation

    return total_reward, steps

#takes about 4 seconds to complete!
def step_environment(env, observation, sess, eps, gamma, forward_dict, loss_dict):
    """ Take one step within the environment """
    """ In: env (OpenAI gym wrapper)
        In: observation (current state of game)
        In: sess (tf graph instance)
        In: eps 
        In: gamma
        Out: new_observation
        Out: reward
        Out: done
    """ 
    ##Policy forward
    p_observation = preprocess_image(observation)
    action, all_Qs = sess.run([forward_dict['predict'], forward_dict['Qout']], feed_dict={forward_dict['input']: p_observation}) #takes about 70% of the running time.. which is fine bcs that's the heart of the calculation
    if np.random.rand(1) < eps:
        action[0] = env.action_space.sample()

    new_observation, reward, done, _ = env.step(action[0])

    ##Max Value forward
    p_new_observation = preprocess_image(new_observation)
    Q_next = sess.run([forward_dict['Qout']], feed_dict={forward_dict['input']: p_new_observation})
    maxQ_next = np.max(Q_next)
    targetQ = all_Qs
    targetQ[0, action[0]] = reward + gamma * maxQ_next

    ##Update to more optimal features
    sess.run([loss_dict['updateModel']], feed_dict={forward_dict['input']:p_new_observation, loss_dict['nextQ']: targetQ})


    return new_observation, reward, done

