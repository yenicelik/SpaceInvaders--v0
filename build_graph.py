""" Builds graph that is used for the model """

import tensorflow as tf

#Main function
def init_graph():
    """ Modularly initializes graph, using below functions
        In: -
        Out: W (Weights)
        Out: b (bias)
        Out: forward_dict (dictionary containing graph elements used for feed-forwarding)
        Out: loss_dict (dictionary containing graph elements used for loss-retrieval)
    """
    W, b = initialize_parameters()
    input, Qout, predict = build_forward_model(W, b, True)
    nextQ, loss, trainer, updateModel = build_loss_model(W, b, Qout, True)

    forward_dict = {"input":input,
                "Qout":Qout,
                "predict":predict
                }

    loss_dict = {"nextQ": nextQ,
                "loss": loss,
                "trainer":trainer,
                "updateModel":updateModel
                }
                
    return W, b, forward_dict, loss_dict #use unpacking


def initialize_parameters():
    """ Creates weights and biases for the respective model 
        In: -
        Out: Weights, bias
    """
    
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

    return Weights, bias


def build_forward_model(W, b, verbose=True):
    """ Builds the 'logical' and 'recongition' part of the graph used
        In: Weights W
        In: Bias b
        In: verbose (Output dimension on creation) = False
        Out: input
        Out: Qout
        Out: predict
    """
    ##Inputs
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

    #not sure if needed to return
    return input, Qout, predict


def build_loss_model(W, b, Qout, verbose=False):
    """ Builds the 'loss' part of the graph
        In: Weights W
        In: Bias b
        In: Qout (Value for each action given the current state)
        Out: nextQ
        Out: loss
        Out: trainer
        Out: updateModel
        """
    nextQ = tf.placeholder(shape=[1,6], dtype=tf.float32)
    loss =  tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)

    #not sure if needed to return
    return nextQ, loss, trainer, updateModel

