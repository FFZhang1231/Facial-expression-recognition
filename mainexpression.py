import tensorflow as tf
from Facial_expression_train import PFER_expression
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

flags = tf.app.flags
flags.DEFINE_integer(flag_name='epoch', default_value=100, docstring='number of epochs')
flags.DEFINE_boolean(flag_name='is_train', default_value=True, docstring='training mode')
flags.DEFINE_string(flag_name='dataset', default_value='MultiPie_train', docstring='dataset name')
flags.DEFINE_string(flag_name='savedir', default_value='PFER',docstring='dir for saving training results')
flags.DEFINE_string(flag_name='testdir', default_value='None', docstring='dir for testing images')
FLAGS = flags.FLAGS


def main(_):

    # print settings
    import pprint
    pprint.pprint(FLAGS.__flags)

    with tf.Session(config=config) as session:
        model = PFER_expression(
            session,  # TensorFlow session
            is_training=FLAGS.is_train,  # flag for training or testing mode
            save_dir=FLAGS.savedir,  # path to save checkpoints, samples, and summary
            dataset_name=FLAGS.dataset  # name of the dataset in the folder ./data
        )
        if FLAGS.is_train:
            print '\n\tTraining Mode'
            model.train(
                num_epochs=FLAGS.epoch,  # number of epochs
            )
        else:
            print '\n\tTesting Mode'
            model.custom_test(
                testing_samples_dir=FLAGS.testdir # test images
            )


if __name__ == '__main__':

    tf.app.run()

