

parameter = {
        "GAMMA": .99,
        "EPS": 1.,
        "NUM_EPISODES": 10, #max sohuld be 10 000 or 100 000 for complex tasks #maybe have multiple NUM_EPISODES IF TESTING OR STH ELSE
        "NUM_STEPS": 100, #should be open ended if wrapper is used
        "SAVE_EVERY": 2,
        "OLD_TF": False,
        "SAVE_FIGS": False,
        "X11": True,
        "record": False
    }


env = gym.make('SpaceInvaders-v0')
    if parameter["record"]:
        env = wrappers.Monitor(env, 'SpaceInvaderExperiment')


tf.reset_default_graph()


#######################
# Initialize weights
#######################

Weights = {
    "conv1": tf.Variable(tf.random_normal([8, 8, 1, 16], 0.00, 0.01), name="w_conv1"),
    #output will be of size (1, 21, 21, 16) for stride 4
    "conv2": tf.Variable(tf.random_normal([4, 4, 16, 32], 0.00, 0.01), name="w_conv2"),
    #output will be of size (1, 11, 11, 32) for stride 2
    "affine1": tf.Variable(tf.random_normal([11 * 11 * 32, 256], 0.00, 0.01), name="w_affine1"),
    "affine2": tf.Variable(tf.random_normal([256, 6], 0.00, 0.01), name="w_affine2")
}

bias = {
    "conv1": tf.Variable(tf.random_normal([1, 21, 21, 16], 0.00, 0.01), name="b_conv1"),
    "conv2": tf.Variable(tf.random_normal([1, 11, 11, 32], 0.00, 0.01), name="b_conv2"),
    "affine1": tf.Variable(tf.random_normal([256], 0.00, 0.01), name="b_affine1"),
    "affine2": tf.Variable(tf.random_normal([6], 0.00, 0.01), name="b_affine2")
}

#####################
# Build model
#####################

##Feed Forward
input = tf.placeholder(shape=[84, 84], dtype=tf.float32) #should include 4 pictures instead of one
if verbose:
    print "1 Direct Input: \t\t" + str(input.get_shape())

inputs = tf.reshape(input, [1, 84, 84, 1] )
if verbose:
    print "1.1 Reshape Input: \t\t" + str(inputs.get_shape())


##Conv1
inputs = tf.nn.conv2d(inputs, W['conv1'], strides=[1, 4, 4, 1], padding='SAME') + b['conv1']
inputs = tf.nn.relu(inputs, name=None) #crelu, or something else? they say 'nonlinearity'
if verbose:
    print "2. Conv1 \t\t\t" + str(inputs.get_shape())

##Conv2
inputs = tf.nn.conv2d(inputs, W['conv2'], strides=[1, 2, 2, 1], padding='SAME') + b['conv2']
inputs = tf.nn.relu(inputs, name=None) #crelu, or something else? they say 'nonlinearity'
if verbose:
    print "3. Conv2 \t\t\t" + str(inputs.get_shape())

##Affine1
inputs = tf.reshape(inputs, [1, 11 * 11 * 32])
if verbose:
    print "4.1 Reshape before Affine1 \t" + str(inputs.get_shape())

inputs = tf.matmul(inputs, W['affine1']) + b['affine1']
inputs = tf.nn.relu(inputs, name=None) #crelu, or something else? they say 'nonlinearity'
if verbose:
    print "4.2 Affine1 \t\t\t" + str(inputs.get_shape())

##Affine2 (finally the output)
Qout = tf.matmul(inputs, W['affine2']) + b['affine2']
if verbose:
    print "5. Affine2 \t\t\t" + str(inputs.get_shape())

predict = tf.argmax(Qout, 1) #why exactly this operation?

##Loss 
nextQ = tf.placeholder(shape=[1,6], dtype=tf.float32)
loss =  tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)


####################
# Setup Saver
####################

features = W.copy()
features.update(b)
saver = tf.train.Saver(features)


#####################
# Train Network
#####################
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

        #################
        # Run Episode
        #################
        observation = env.reset()
        total_reward = 0
        done = False
        steps = 1

        while steps < parameter['NUM_STEPS']:

            ###################
            # Step in Episode
            ###################
            ##Policy forward
            p_observation = preprocess_image(observation)
            action, all_Qs = sess.run([predict, Qout], feed_dict={input: p_observation}) #takes about 70% of the running time.. which is fine bcs that's the heart of the calculation
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
            ###################
            # Step in Episode
            ###################


            total_reward += reward
            if done:
                break

            observation = new_observation

        #################
        # Run Episode
        #################

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
