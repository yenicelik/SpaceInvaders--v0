""" """
#incorporate use-input dictionary etc.
from train import train
from setup import setup
from build_graph import init_graph

from helper import setup_saver

import tensorflow as tf


def main():
    """ Main function. All operations start from within here """
    env, parameter = setup()

    tf.reset_default_graph()
    #model_vars
    #W, b, input, Qout, predict, nextQ, loss, trainer, updateModel = init_graph()
    W, b, forward_dict, loss_dict = init_graph()

    # forward_dict = {"input":input,
    #             "Qout":Qout,
    #             "predict":predict
    #             }

    # loss_dict = {"nextQ": nextQ,
    #             "loss": loss,
    #             "trainer":trainer,
    #             "updateModel":updateModel
    #             }


    saver = setup_saver(W, b)

    reward_list, steps_list = train(
                                env=env,
                                parameter=parameter,
                                saver=saver,
                                forward_dict=forward_dict,
                                loss_dict=loss_dict
                            )

if __name__ == '__main__':
    main()
    # import cProfile
    # cProfile.run('main()', 'restats')