# This code is implemented by tensorflow r0.12
# Date:     Nov. 20th, 2017

from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from scipy.io import savemat
from ops import *
import pdb
import pandas as pd

class PFER_expression(object):
    def __init__(self,
                 session,  # TensorFlow session
                 size_image=224,  # size the input images
                 size_kernel=5,  # size of the kernels in convolution and deconvolution
                 size_batch=36,  # mini-batch size for training and testing, must be square of an integer
                 num_input_channels=3,  # number of channels of input images
                 num_encoder_channels=64,  # number of channels of the first conv layer of encoder
                 num_fx=50,  # number of channels of the layer f(x)
                 num_categories=6,  # number of expressions in the training dataset
                 num_poses =5,      # number of poses in the training dataset
                 num_gen_channels=1024,  # number of channels of the first deconv layer of generator
                 enable_tile_label=True,  # enable to tile the label
                 tile_ratio=1.0,  # ratio of the length between tiled label and fx
                 is_training=True,  # flag for training or testing mode
                 save_dir='./savePFEW',  # path to save checkpoints, samples, and summary
                 ):

        self.session = session
        self.image_value_range = (-1, 1)
        self.size_image = size_image
        self.size_kernel = size_kernel
        self.size_batch = size_batch
        self.num_input_channels = num_input_channels
        self.num_encoder_channels = num_encoder_channels
        self.num_fx = num_fx
        self.num_categories = num_categories
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.save_dir = save_dir
        self.num_poses = num_poses

        # path of the file of trainset. the content style of trainMULTIPIE.txt: name expression-label pose-label
        self.pathtrain = '/path/to/your/data'
        self.file_names = np.loadtxt(self.pathtrain + 'trainMULTIPIE.txt', dtype=bytes, delimiter=' ').astype(str)
        np.random.shuffle(self.file_names)
        self.len_trainset = len(self.file_names)
        self.num_batches = self.len_trainset // self.size_batch

        self.gen_names = np.loadtxt(self.pathtrain + 'genMULTIPIE.txt', dtype=bytes, delimiter=' ').astype(str)
        np.random.shuffle(self.gen_names)
        self.gen_trainset = len(self.gen_names)
        self.num_batches1 = self.gen_trainset // self.size_batch

        self.test_names = np.loadtxt(self.pathtrain + 'testMULTIPIE.txt', dtype=bytes, delimiter=' ').astype(str)
        np.random.shuffle(self.test_names)
        gen = open('testname.txt', 'w')     # name of the testset
        self.len_testset = len(self.test_names)
        self.num_batches2 = self.len_testset // self.size_batch 
        for ii in range(self.test_names.shape[0]):
            gen.write(self.test_names[ii, 0] + '\n')
        gen.close()
        self.len_testset = len(self.test_names)
        self.num_batches2 = self.len_testset // self.size_batch


        #train on BU-3DFE
        # ************ similar to above ************

        #train on SFEW
        # ************ similar to above ************



        # ************************************* input to graph ********************************************************
        self.input_image = tf.placeholder(  #input_image size [36, 224, 224, 3]
            tf.float32,
            [self.size_batch, self.size_image, self.size_image, self.num_input_channels],
            name='input_images'
        )
        self.expression = tf.placeholder(  #expression label for G, D_att, and C_exp. onehot
            tf.float32,
            [self.size_batch, self.num_categories],
            name='expression_labels'
        )

        # if sparse_softmax_cross_entropy_with_logits is used
        # self.expression1 = tf.placeholder(  #expression label for C_exp
        #     tf.int64,
        #     [self.size_batch],
        #     name='expression_labels'
        # )

        self.pose = tf.placeholder( #pose label for G and D_att, and C_exp. onehot
            tf.float32,
            [self.size_batch, self.num_poses],
            name='pose_labels'
        )

        # if sparse_softmax_cross_entropy_with_logits is used
        # self.pose1 = tf.placeholder( #pose label for C_exp
        #     tf.int64,
        #     [self.size_batch],
        #     name='pose_labels'
        # )

        self.f_prior = tf.placeholder( #prior distribution of D_i
            tf.float32,
            [self.size_batch, self.num_fx],
            name='f_prior'
        )

        # ************************************* build the graph *******************************************************
        print '\n\tBuilding graph ...'

        # G_encoder: input image --> f(x)
        self.f = self.Gencoder(
            image=self.input_image
        )

        # G_decoder: f(x) + expression + pose --> generated image
        self.G = self.Gdecoder(
                f=self.f,
                y=self.expression,
                pose=self.pose,
                enable_tile_label=self.enable_tile_label,
                tile_ratio=self.tile_ratio
            )

        # discriminator on identity
        self.D_f, self.D_f_logits = self.discriminator_i(
            f=self.f,
            is_training=self.is_training
        )

        # discriminator on G
        self.D_G, self.D_G_logits= self.discriminator_att(
            image=self.G,
            y=self.expression,
            pose=self.pose,
            is_training=self.is_training
        )

        # discriminator on f_prior
        self.D_f_prior, self.D_f_prior_logits = self.discriminator_i(
            f=self.f_prior,
            is_training=self.is_training,
            reuse_variables=True
        )

        # discriminator on input image
        self.D_input, self.D_input_logits = self.discriminator_att(
            image=self.input_image,
            y=self.expression,
            pose=self.pose,
            is_training=self.is_training,
            reuse_variables=True
        )

        # classifier on original facial images and generated facial images
        self.D_input_ex_logits, self.D_input_pose_logits = self.discriminator_acc(
            image=self.input_image,
            is_training=self.is_training
        )

        # ************************************* loss functions *******************************************************
        # loss function of generator G
        self.EG_loss = tf.reduce_mean(tf.abs(self.input_image - self.G))  # L1 loss

        # loss function of discriminator on identity
        self.D_f_loss_prior = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_f_prior_logits, tf.ones_like(self.D_f_prior_logits))
        )
        self.D_f_loss_f = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_f_logits, tf.zeros_like(self.D_f_logits))
        )
        # loss function of G on identity
        self.E_f_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_f_logits, tf.ones_like(self.D_f_logits))
        )

        # loss function of discriminator on image
        self.D_att_loss_input = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_input_logits, tf.ones_like(self.D_input_logits))
        )
        self.D_att_loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_G_logits, tf.zeros_like(self.D_G_logits))
        )
        # loss function of G on image
        self.G_att_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_G_logits, tf.ones_like(self.D_G_logits))
        )

        # (1) loss function of classifier on image
        self.D_ex_loss_input = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.expression, logits=self.D_input_ex_logits) )
        self.D_pose_loss_input = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.pose, logits=self.D_input_pose_logits) )

        #(2) if sparse_softmax_cross_entropy_with_logits is used
        # self.D_ex_loss_input = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.expression1, logits=self.D_input_ex_logits) )
        # self.D_pose_loss_input = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.pose1, logits=self.D_input_pose_logits) )

        # pdb.set_trace()
        tv_y_size = self.size_image
        tv_x_size = self.size_image
        self.tv_loss = (
            (tf.nn.l2_loss(self.G[:, 1:, :, :] - self.G[:, :self.size_image - 1, :, :]) / tv_y_size) +
            (tf.nn.l2_loss(self.G[:, :, 1:, :] - self.G[:, :, :self.size_image - 1, :]) / tv_x_size)) / self.size_batch

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ACCURACY OPS$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        #Accuracy of expression
        self.d_ex_count = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.D_input_ex_logits, 1), tf.argmax(self.expression,1)), 'int32'))
        #Accuracy of pose
        self.d_pose_count = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.D_input_pose_logits, 1), tf.argmax(self.pose, 1)), 'int32'))

        #(2) if sparse_softmax_cross_entropy_with_logits is used
        # self.d_ex_count = tf.reduce_sum(
        #     tf.cast(tf.equal(tf.argmax(self.D_input_ex_logits, 1), self.expression1), 'int32'))
        # self.d_pose_count = tf.reduce_sum(
        #     tf.cast(tf.equal(tf.argmax(self.D_input_pose_logits, 1), self.pose1), 'int32'))

        # *********************************** trainable variables ****************************************************
        trainable_variables = tf.trainable_variables()

        # print (trainable_variables)
        # variables of G_encoder
        self.E_variables = [var for var in trainable_variables if 'E_' in var.name]
        # variables of G_decoder
        self.G_variables = [var for var in trainable_variables if 'G_' in var.name]
        # variables of discriminator on identity
        self.D_f_variables = [var for var in trainable_variables if 'D_f_' in var.name]
        # variables of discriminator on attributes
        self.D_att_variables = [var for var in trainable_variables if 'D_att_' in var.name]
        # variables of discriminator on expression
        self.D_acc_variables = [var for var in trainable_variables if 'D_acc_' in var.name]

        # ************************************* collect the summary ***************************************
        self.f_summary = tf.summary.histogram('f', self.f)
        self.f_prior_summary = tf.summary.histogram('f_prior', self.f_prior)
        self.EG_loss_summary = tf.summary.scalar('EG_loss', self.EG_loss)
        self.D_f_loss_f_summary = tf.summary.scalar('D_f_loss_f', self.D_f_loss_f)
        self.D_f_loss_prior_summary = tf.summary.scalar('D_f_loss_prior', self.D_f_loss_prior)
        self.E_f_loss_summary = tf.summary.scalar('E_f_loss', self.E_f_loss)
        self.D_f_logits_summary = tf.summary.histogram('D_f_logits', self.D_f_logits)
        self.D_f_prior_logits_summary = tf.summary.histogram('D_f_prior_logits', self.D_f_prior_logits)
        self.D_att_loss_input_summary = tf.summary.scalar('D_att_loss_input', self.D_att_loss_input)
        self.D_att_loss_G_summary = tf.summary.scalar('D_att_loss_G', self.D_att_loss_G)
        self.G_att_loss_summary = tf.summary.scalar('G_att_loss', self.G_att_loss)
        self.D_G_logits_summary = tf.summary.histogram('D_G_logits', self.D_G_logits)
        self.D_input_logits_summary = tf.summary.histogram('D_input_logits', self.D_input_logits)
        self.D_input_ex_logits_summary = tf.summary.histogram('D_input_ex_logits', self.D_input_ex_logits)
        self.D_ex_loss_input_summary = tf.summary.scalar('D_ex_loss_input_summary',self.D_ex_loss_input)
        self.d_ex_count_summary = tf.summary.scalar('d_ex_count', self.d_ex_count)
        self.d_pose_count_summary = tf.summary.scalar('d_pose_count', self.d_pose_count)

        # for saving the graph and variables
        self.saver = tf.train.Saver(max_to_keep=10)

    #get the train data and test data
    def get_batch_train_test(self, enable_shuffle=True, idx=0):
        # # *************************** load file names of images ******************************************************
        if self.is_training:
            if enable_shuffle:
                np.random.shuffle(self.file_names)
            tt_files = self.file_names[idx*self.size_batch: idx*self.size_batch + self.size_batch]
             #path of the traindata
            self.path = self.pathtrain + 'data/MultiPie_train/'
        else:
            tt_files = self.test_names[idx*self.size_batch: idx*self.size_batch + self.size_batch]
             #path of the testdata
            self.path = self.pathtrain + 'data/MultiPie_test/'

        batch_images = np.zeros((self.size_batch, self.size_image, self.size_image, 3))
        for i in range(tt_files.shape[0]):
            sample = [load_image(
                image_path= self.path + tt_files[i, 0],
                image_size=self.size_image,
                image_value_range=self.image_value_range,
                is_gray=(self.num_input_channels == 1),
            )]
            if self.num_input_channels == 1:
                batch_images[i, :, :, :] = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                batch_images[i, :, :, :] = np.array(sample).astype(np.float32)

        tt_label_expression = np.ones(
            shape=(len(tt_files), self.num_categories),
            dtype=np.float32
        ) * self.image_value_range[0]

        tt_label_pose = np.ones(
            shape=(len(tt_files), self.num_poses),
            dtype=np.float32
        ) * self.image_value_range[0]

        # if sparse_softmax_cross_entropy_with_logits is used
        # tt_label_expression1 = np.ones(
        #     shape=(len(tt_files)),
        #     dtype=np.float32
        # ) * self.image_value_range[0]
        #
        # tt_label_pose1 = np.ones(
        #     shape=(len(tt_files)),
        #     dtype=np.float32
        # ) * self.image_value_range[0]

        for i, label in enumerate(tt_files[:, 1]):
            label = tt_files[i, 1].astype('int')
            #changing the expression label as onehot with the target as 1, others as -1;
            tt_label_expression[i, label] = self.image_value_range[-1]

            # if sparse_softmax_cross_entropy_with_logits is used
            # expression label
            # tt_label_expression1[i]=label

            pose = tt_files[i, 2].astype('int')
            if pose == 41:
                pose = 0
            elif pose == 130:
                pose = 1
            elif pose == 50:
                pose = 2
            elif pose == 51:
                pose = 3
            elif pose == 140:
                pose = 4
            # changing the pose label as onehot with the target as 1, others as -1;
            tt_label_pose[i, pose] = self.image_value_range[-1]

            # if sparse_softmax_cross_entropy_with_logits is used
            # pose label
            # tt_label_pose1[i]=pose
        # return batch_images, tt_label_expression, tt_label_pose, tt_label_expression1, tt_label_pose1, tt_files
        return batch_images, tt_label_expression, tt_label_pose, tt_files

    # get the gen data
    def get_batch_gen(self, DIS=True, idx=0):

        if DIS:
            print('dis')
            np.random.shuffle(self.gen_names)
        tt_files = self.gen_names[idx*self.size_batch: idx*self.size_batch + self.size_batch]
        #path of the traindata
        self.path = self.pathtrain + 'data/MultiPie_train/'
        batch_images = np.zeros((self.size_batch, self.size_image, self.size_image, 3))
        for i in range(tt_files.shape[0]):
            sample = [load_image(
                image_path= self.path + tt_files[i, 0],
                image_size=self.size_image,
                image_value_range=self.image_value_range,
                is_gray=(self.num_input_channels == 1),
            )]

            if self.num_input_channels == 1:
                batch_images[i, :, :, :] = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                batch_images[i, :, :, :] = np.array(sample).astype(np.float32)

        tt_label_expression = np.ones(
            shape=(len(tt_files), self.num_categories),
            dtype=np.float32
        ) * self.image_value_range[0]

        tt_label_pose = np.ones(
            shape=(len(tt_files), self.num_poses),
            dtype=np.float32
        ) * self.image_value_range[0]

        # tt_label_expression1 = np.ones(
        #     shape=(len(tt_files)),
        #     dtype=np.float32
        # ) * self.image_value_range[0]
        #
        # tt_label_pose1 = np.ones(
        #     shape=(len(tt_files)),
        #     dtype=np.float32
        # ) * self.image_value_range[0]

        for i, label in enumerate(tt_files[:, 1]):
            label = tt_files[i, 1].astype('int')
            tt_label_expression[i, label] = self.image_value_range[-1]
            # tt_label_expression1[i]=label

            pose = tt_files[i, 2].astype('int')
            if pose == 41:
                pose = 0
            elif pose == 130:
                pose = 1
            elif pose == 50:
                pose = 2
            elif pose == 51:
                pose = 3
            elif pose == 140:
                pose = 4
            tt_label_pose[i, pose] = self.image_value_range[-1]
            # tt_label_pose1[i]=pose
        # return batch_images, tt_label_expression, tt_label_pose, tt_label_expression1, tt_label_pose1, tt_files
        return batch_images, tt_label_expression, tt_label_pose, tt_files

    # get the validation data to validate the generated images
    def get_batch_sample(self, idx=0):

        tt_files = self.test_names[idx*self.size_batch: idx*self.size_batch + self.size_batch]
        batch_images = np.zeros((self.size_batch, self.size_image, self.size_image, 3))
        for i in range(tt_files.shape[0]):
            sample = [load_image(
                 #path of the testdata
                image_path=self.pathtrain + 'data/MultiPie_test/' + tt_files[i,0],
                image_size=self.size_image,
                image_value_range=self.image_value_range,
                is_gray=(self.num_input_channels == 1),
            )]
            if self.num_input_channels == 1:
                batch_images[i, :, :, :] = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                batch_images[i, :, :, :] = np.array(sample).astype(np.float32)

        tt_label_expression = np.ones(
            shape=(len(tt_files), self.num_categories),
            dtype=np.float32
        ) * self.image_value_range[0]

        tt_label_pose = np.ones(
            shape=(len(tt_files), self.num_poses),
            dtype=np.float32
        ) * self.image_value_range[0]

        # tt_label_expression1 = np.ones(
        #     shape=(len(tt_files)),
        #     dtype=np.float32
        # ) * self.image_value_range[0]
        #
        # tt_label_pose1 = np.ones(
        #     shape=(len(tt_files)),
        #     dtype=np.float32
        # ) * self.image_value_range[0]

        for i, label in enumerate(tt_files[:, 1]):
            label = tt_files[i, 1].astype('int')
            tt_label_expression[i, label] = self.image_value_range[-1]

            # tt_label_expression1[i]=label

            pose = tt_files[i, 2].astype('int')
            if pose == 41:
                pose = 0
            elif pose == 130:
                pose = 1
            elif pose == 50:
                pose = 2
            elif pose == 51:
                pose = 3
            elif pose == 140:
                pose = 4
            tt_label_pose[i, pose] = self.image_value_range[-1]

            # tt_label_pose1[i]=pose

        # return batch_images, tt_label_expression, tt_label_pose, tt_label_expression1, tt_label_pose1, tt_files
        return batch_images, tt_label_expression, tt_label_pose, tt_files


    def train(self,
              num_epochs=100,# number of epochs
              learning_rate=0.0002,  # learning rate of optimizer
              #learning_rate=0.00005,  # learning rate of optimizer
              beta1=0.5,  # parameter for Adam optimizer
              decay_rate=0.99,  # learning rate decay (0, 1], 1 means no decay
              #decay_rate=1,  # learning rate decay (0, 1], 1 means no decay
              enable_shuffle=True,  # enable shuffle of the dataset
              use_trained_model=True,# used the saved checkpoint to initialize the model
              ):

        # *********************************** optimizer **************************************************************
        # over all, there are four loss functions, weights may differ from the paper because of different datasets
        # self.loss_EG = self.EG_loss + 0.0005 * self.G_att_loss + 0.0005 * self.E_f_loss + 0.0001 * self.tv_loss+ 0.0001*self.loss_Ex  # slightly increase the params
        self.loss_EG = self.EG_loss + 0.0001 * self.G_att_loss + 0.0001 * self.E_f_loss + 0.0001 * self.tv_loss # slightly increase the params
        self.loss_Df = self.D_f_loss_prior + self.D_f_loss_f
        self.loss_Datt = self.D_att_loss_input + self.D_att_loss_G
        self.loss_Ex = self.D_ex_loss_input+self.D_pose_loss_input

        # set learning rate decay
        self.EG_global_step = tf.Variable(0, trainable=False, name='global_step')
        EG_learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=self.EG_global_step,
            decay_steps=self.len_trainset / self.size_batch * 2,
            decay_rate=decay_rate,
            staircase=True
        )

        # optimizer for G_encoder + G_decoder
        self.EG_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_EG,
            global_step=self.EG_global_step,
            var_list=self.E_variables + self.G_variables
        )

        # optimizer for discriminator on f(x)
        self.D_f_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_Df,
            var_list=self.D_f_variables
        )

        # optimizer for discriminator on attributes
        self.D_att_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_Datt,
            var_list=self.D_att_variables
        )

        # optimizer for discriminator on expression
        self.D_ex_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_Ex,
            var_list=self.D_acc_variables
        )

        # *********************************** tensorboard *************************************************************
        # for visualization (TensorBoard): $ tensorboard --logdir path/to/log-directory
        self.EG_learning_rate_summary = tf.summary.scalar('EG_learning_rate', EG_learning_rate)
        self.summary = tf.summary.merge([
            self.f_summary, self.f_prior_summary,
            self.D_f_loss_f_summary, self.D_f_loss_prior_summary,
            self.D_f_logits_summary, self.D_f_prior_logits_summary,
            self.EG_loss_summary, self.E_f_loss_summary,
            self.D_att_loss_input_summary, self.D_att_loss_G_summary,
            self.G_att_loss_summary, self.EG_learning_rate_summary,
            self.D_G_logits_summary, self.D_input_logits_summary,
            self.D_input_ex_logits_summary,
            self.D_ex_loss_input_summary,
            self.d_ex_count_summary,self.d_pose_count_summary
        ])
        self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'), self.session.graph)

        # ******************************************* training *******************************************************
        print '\n\tPreparing for training ...'

        # initialize the graph
        tf.global_variables_initializer().run()

        # load check point
        if use_trained_model:
            if self.load_checkpoint():
                print("\tSUCCESS ^_^")
            else:
                print("\tFAILED >_<!")


        #if sparse_softmax_cross_entropy_with_logits is used
        # sample_images, sample_label_expression, sample_label_pose, sample_label_expression1, sample_label_pose1, batch_files_name = self.get_batch_sample(0)

        sample_images, sample_label_expression, sample_label_pose, batch_files_name = self.get_batch_sample(0)
        sample_expression_label = map(lambda x: [[i, 0][i < 0] for i in x], sample_label_expression)
        sample_pose_label = map(lambda x: [[i, 0][i < 0] for i in x], sample_label_pose)

        for epoch in range(num_epochs):

            # to save the test results
            trainresult = open('result/'+ str(epoch) + 'a.txt', 'w')
            f1 = open('result/'+str(epoch) + 'test.txt', 'w')
            f2 = open('result/'+str(epoch) + 'index.txt', 'w')

            self.is_training= True
            DIS=True
            enable_shuffle =True

            for ind_batch in range(self.num_batches):
                self.is_training=True

                #if sparse_softmax_cross_entropy_with_logits is used
                # batch_images, batch_label_expression, batch_label_pose, batch_label_expression1, batch_label_pose1, batch_files_name = self.get_batch_train_test(enable_shuffle,ind_batch)
                batch_images, batch_label_expression, batch_label_pose, batch_files_name = self.get_batch_train_test(enable_shuffle,ind_batch)

                #map batch_label_expression and batch_label_pose to onehot with the target as 1, others as 0
                expression_label =map(lambda x:[[i,0][i<0] for i in x], batch_label_expression)
                pose_label = map(lambda x:[[i,0][i<0] for i in x], batch_label_pose)

                enable_shuffle = False
                start_time = time.time()

                # prior distribution on the prior of f [-1,1]
                batch_f_prior = np.random.uniform(
                    self.image_value_range[0],
                    self.image_value_range[-1],
                    [self.size_batch, self.num_fx]
                ).astype(np.float32)

                _, _, _ = self.session.run(
                    fetches=[
                        self.EG_optimizer,
                        self.D_f_optimizer,
                        self.D_att_optimizer
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: batch_label_expression,
                        self.pose: batch_label_pose,
                        self.f_prior: batch_f_prior
                    }
                )

                #(1) if softmax_cross_entropy_with_logits is used
                _ = self.session.run(
                    fetches=[
                        self.D_ex_optimizer
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: expression_label,
                        self.pose: pose_label
                    }
                )

                #(2) if sparse_softmax_cross_entropy_with_logits is used
                # _ = self.session.run(
                #     fetches=[
                #         self.D_ex_optimizer
                #     ],
                #     feed_dict={
                #         self.input_image: batch_images,
                #         self.expression1: batch_label_expression1,
                #         self.pose1: batch_label_pose1
                #     }
                # )

                _ = self.session.run(
                    fetches=[
                        self.EG_optimizer
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: batch_label_expression,
                        self.pose: batch_label_pose,
                        self.f_prior: batch_f_prior
                    }
                )

                _ = self.session.run(
                    fetches=[
                        self.EG_optimizer
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: batch_label_expression,
                        self.pose: batch_label_pose,
                        self.f_prior: batch_f_prior
                    }
                )

                _ = self.session.run(
                    fetches=[
                        self.EG_optimizer
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: batch_label_expression,
                        self.pose: batch_label_pose,
                        self.f_prior: batch_f_prior
                    }
                )

                _ = self.session.run(
                    fetches=[
                        self.EG_optimizer
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: batch_label_expression,
                        self.pose: batch_label_pose,
                        self.f_prior: batch_f_prior
                    }
                )



                #(1) if softmax_cross_entropy_with_logits is used
                Dex_err, Dpose_err, D_ex, D_pose = self.session.run(
                        fetches=[
                            self.D_ex_loss_input,
                            self.D_pose_loss_input,
                            self.d_ex_count,
                            self.d_pose_count
                        ],
                        feed_dict={
                            self.input_image: batch_images,
                            self.expression: expression_label,
                            self.pose: pose_label,
                            self.f_prior: batch_f_prior
                        }
                    )

                #(2) if sparse_softmax_cross_entropy_with_logits is used
                # Dex_err, Dpose_err, D_ex, D_pose = self.session.run(
                #     fetches=[
                #         self.D_ex_loss_input,
                #         # self.D_pose_loss_G,
                #         self.D_pose_loss_input,
                #         self.d_ex_count,
                #         # self.g_ex_count,
                #         self.d_pose_count,
                #         # self.g_pose_count,
                #     ],
                #     feed_dict={
                #         self.input_image: batch_images,
                #         self.expression1: batch_label_expression1,
                #         self.pose1: batch_label_pose1,
                #         self.f_prior: batch_f_prior
                #     }
                # )

                EG_err, Ef_err, Df_err, Dfp_err, Gi_err, DiG_err, Di_err,  TV = self.session.run(
                    fetches = [
                        self.EG_loss,
                        self.E_f_loss,
                        self.D_f_loss_f,
                        self.D_f_loss_prior,
                        self.G_att_loss,
                        self.D_att_loss_G,
                        self.D_att_loss_input,
                        self.tv_loss
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: batch_label_expression,
                        self.pose: batch_label_pose,
                        self.f_prior: batch_f_prior
                    }
                )

                print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\tEG_err=%.4f\tTV=%.4f" %
                    (epoch+1, num_epochs, ind_batch+1, self.num_batches, EG_err, TV))
                print ("\t Accuracy Dex=%.4f /36 \t Dpose =%.4f /36  " % (D_ex, D_pose))

                #using the generated images to train the classifier
                if epoch > 6:

                    #You can change parameter $add$ to add different times of generated images
                    if epoch > 6 and epoch < 10:
                        add = *
                    elif epoch > 10 and epoch < 20:
                        add = *
                    elif epoch > 20 and epoch < 30:
                        add = *
                    elif epoch > 30:
                        add = *

                    for addimg in range(add):

                        # if sparse_softmax_cross_entropy_with_logits is used
                        # gen_images, gen_label_expression, gen_label_pose, gen_label_expression1, gen_label_pose1, gen_files_name = self.get_batch_gen(DIS, ind_batch*add+addimg)

                        gen_images, gen_label_expression, gen_label_pose, gen_files_name = self.get_batch_gen(DIS, ind_batch * add + addimg)
                        DIS = False
                        gen_label_expressiononehot = map(lambda x: [[i, 0][i < 0] for i in x], gen_label_expression)
                        gen_label_poseonehot = map(lambda x: [[i, 0][i < 0] for i in x], gen_label_pose)

                        f, G = self.session.run(
                            [self.f, self.G],
                            feed_dict={
                                self.input_image: gen_images,
                                self.expression: gen_label_expression,
                                self.pose: gen_label_pose
                            }
                        )

                        #(1)
                        _ = self.session.run(
                                fetches=[
                                    self.D_ex_optimizer
                                ],
                                feed_dict={
                                    self.input_image: G,
                                    self.expression: gen_label_expressiononehot,
                                    self.pose: gen_label_poseonehot
                                }
                            )
                        #(2)
                        # _ = self.session.run(
                        #     fetches=[
                        #         self.D_ex_optimizer
                        #     ],
                        #     feed_dict={
                        #         self.input_image: G,
                        #         self.expression1: gen_label_expression1,
                        #         self.pose1: gen_label_pose1
                        #     }
                        # )

                        # (1)
                        Dex_err, Dpose_err, D_ex, D_pose = self.session.run(
                                fetches=[
                                    self.D_ex_loss_input,
                                    self.D_pose_loss_input,
                                    self.d_ex_count,
                                    self.d_pose_count
                                ],
                                feed_dict={
                                    self.input_image: batch_images,
                                    self.expression: expression_label,
                                    self.pose: pose_label,
                                    self.f_prior: batch_f_prior
                                }
                            )

                        #(2)
                        # Dex_err, Dpose_err, D_ex, D_pose = self.session.run(
                        #     fetches=[
                        #         self.D_ex_loss_input,
                        #         # self.D_pose_loss_G,
                        #         self.D_pose_loss_input,
                        #         self.d_ex_count,
                        #         # self.g_ex_count,
                        #         self.d_pose_count,
                        #         # self.g_pose_count,
                        #     ],
                        #     feed_dict={
                        #         self.input_image: batch_images,
                        #         self.expression1: gen_label_expression1,
                        #         self.pose1: gen_label_pose1,
                        #         self.f_prior: batch_f_prior
                        #     }
                        # )

                print("\tEf=%.4f\tDf=%.4f\tDfp=%.4f" % (Ef_err, Df_err, Dfp_err))
                print("\tGi=%.4f\tDi=%.4f\tDiG=%.4f" % (Gi_err, Di_err, DiG_err))
                print("\tDex=%.4f\tDpose=%.4f" % (Dex_err, Dpose_err))
                print ("\t Accuracy DGex=%.4f /36 \t DGpose =%.4f /36  " % (D_ex, D_pose))
                result = 'epoch=' + str(epoch) + '\t' + 'num_batches=' + str(ind_batch) + '\t' + str(D_ex) + '\t'  + '\n'
                trainresult.writelines(result)

                # estimate left run time
                elapse = time.time() - start_time
                time_left = ((num_epochs - epoch - 1) * self.num_batches + (self.num_batches - ind_batch - 1)) * elapse
                print("\tTime left: %02d:%02d:%02d" %
                      (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))

                # add to summary
                # pdb.set_trace()
                # (1)
                summary = self.summary.eval(
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: batch_label_expression,
                        self.pose: batch_label_pose,
                        self.f_prior: batch_f_prior
                    }
                )

                # (2)
                # summary = self.summary.eval(
                #     feed_dict={
                #         self.input_image: batch_images,
                #         self.expression: batch_label_expression,
                #         self.pose: batch_label_pose,
                #         self.expression1: batch_label_expression1,
                #         self.pose1: batch_label_pose1,
                #         self.f_prior: batch_f_prior
                #     }
                # )

                self.writer.add_summary(summary, self.EG_global_step.eval())

            trainresult.close()

            # save sample images for each epoch
            name = '{:02d}.png'.format(epoch+1)
            self.sample(sample_images, sample_label_expression, sample_label_pose, name)
            self.test(sample_images, sample_label_pose, name, sample_label_expression)
			#self.test_acc(sample_images, sample_expression_label, sample_pose_label)
            #print (self.is_training)

            for ind_batch in range(self.num_batches2):
            	self.is_training= False
                # batch_images, batch_label_expression, batch_label_pose, batch_label_expression1, batch_label_pose1, batch_files_name = self.get_batch_train_test(enable_shuffle, ind_batch)

                batch_images, batch_label_expression, batch_label_pose, batch_files_name = self.get_batch_train_test(enable_shuffle, ind_batch)

                batch_label_expressiononehot = map(lambda x: [[i, 0][i < 0] for i in x], batch_label_expression)
                batch_label_poseonehot = map(lambda x: [[i, 0][i < 0] for i in x], batch_label_pose)

                accex, accpose,accindex= self.test_acc(batch_images, batch_label_expressiononehot, batch_label_poseonehot)

                re = str(accex) +'\t' + str(accpose) +'\n'

                #Record the classified labels of each test image
                for jj in range(accindex.shape[0]):
                    resu = accindex[jj]
                    f2.writelines(str(resu)+'\n')
                # Record the number of right test image in each group (each batch_size)
                f1.writelines(re)

            f1.close()
            f2.close()
            # save checkpoint for each 10 epoch
            if np.mod(epoch, 10) == 9:
                self.save_checkpoint()

        # save the trained model
        self.save_checkpoint()
        # close the summary writer
        self.writer.close()

    def Gencoder(self, image, reuse_variables=False):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)

        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'E_conv' + str(i)
            current = conv2d(
                    input_map=current,
                    num_output_channels=self.num_encoder_channels * (2 ** i),
                    size_kernel=self.size_kernel,
                    name=name
                )
            current = tf.nn.relu(current)

        # fully connection layer
        name = 'E_fc'
        current = fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=self.num_fx,
            name=name
        )
        # output
        return tf.nn.tanh(current)

    def Gdecoder(self, f, y, pose, reuse_variables=False, enable_tile_label=True, tile_ratio=1.0):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        if enable_tile_label:
            duplicate = int(self.num_fx * tile_ratio / self.num_categories)
        else:
            duplicate = 1

        f = concat_label(f, y, duplicate=duplicate)
        if enable_tile_label:
            # duplicate = int(self.num_fx * tile_ratio / self.num_poses)
            duplicate = int(self.num_fx * tile_ratio / 2)
        else:
            duplicate = 1
        f = concat_label(f, pose, duplicate=duplicate)

        size_mini_map = int(self.size_image / 2 ** num_layers)

        # fc layer
        name = 'G_fc'
        current = fc(
            input_vector=f,
            num_output_length=self.num_gen_channels * size_mini_map * size_mini_map,
            name=name
        )

        # reshape to cube for deconv
        current = tf.reshape(current, [-1, size_mini_map, size_mini_map, self.num_gen_channels])
        current = tf.nn.relu(current)

        # deconv layers with stride 2
        for i in range(num_layers):
            name = 'G_deconv' + str(i)
            current = deconv2d(
                    input_map=current,
                    output_shape=[self.size_batch,
                                  size_mini_map * 2 ** (i + 1),
                                  size_mini_map * 2 ** (i + 1),
                                  int(self.num_gen_channels / 2 ** (i + 1))],
                    size_kernel=self.size_kernel,
                    name=name
                )
            current = tf.nn.relu(current)

        name = 'G_deconv' + str(i+1)
        current = deconv2d(
            input_map=current,
            output_shape=[self.size_batch,
                          self.size_image,
                          self.size_image,
                          int(self.num_gen_channels / 2 ** (i + 2))],
            size_kernel=self.size_kernel,
            stride=1,
            name=name
        )
        current = tf.nn.relu(current)

        name = 'G_deconv' + str(i + 2)
        current = deconv2d(
            input_map=current,
            output_shape=[self.size_batch,
                          self.size_image,
                          self.size_image,
                          self.num_input_channels],
            size_kernel=self.size_kernel,
            stride=1,
            name=name
        )
        # output
        return tf.nn.tanh(current)

    def discriminator_i(self, f, is_training=True, reuse_variables=False, num_hidden_layer_channels=(64, 32, 16), enable_bn=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        current = f
        # fully connection layer
        for i in range(len(num_hidden_layer_channels)):
            name = 'D_f_fc' + str(i)
            current = fc(
                    input_vector=current,
                    num_output_length=num_hidden_layer_channels[i],
                    name=name
                )
            if enable_bn:
                name = 'D_f_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)
        # output layer
        name = 'D_f_fc' + str(i+1)
        current = fc(
            input_vector=current,
            num_output_length=1,
            name=name
        )
        return tf.nn.sigmoid(current), current

    def discriminator_att(self, image, y, pose, is_training=True, reuse_variables=False, num_hidden_layer_channels=(16, 32, 64, 128), enable_bn=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = len(num_hidden_layer_channels)
        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'D_att_conv' + str(i)
            # pdb.set_trace()
            current = conv2d(
                    input_map=current,
                    num_output_channels=num_hidden_layer_channels[i],
                    size_kernel=self.size_kernel,
                    name=name
                )
            # pdb.set_trace()
            if enable_bn:
                name = 'D_att_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )

            current = tf.nn.relu(current)
            if i == 0:
                current = concat_label(current, y)
                # current = concat_label(current, pose, int(self.num_categories / self.num_poses))
                current = concat_label(current, pose, int(self.num_categories / 2))
        # fully connection layer
        name = 'D_att_fc1'
        current = fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=1024,
            name=name
        )
        current = lrelu(current)
        name = 'D_att_fc2'
        current1 = fc(
            input_vector=current,
            num_output_length=1,
            name=name
        )
        # output
        return tf.nn.sigmoid(current1), current1

    #(1) classifier --VGG19
    def discriminator_acc(self, image, is_training=True, reuse_variables=False, num_hidden_layer_channels=(32, 32, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512), enable_bn=True):
            # pdb.set_trace()
            if reuse_variables:
                tf.get_variable_scope().reuse_variables()
            current = image
            num_hidden_layer_channels = num_hidden_layer_channels
            stride = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            name = 'D_acc_conv' + str(0)
            current = conv2d2(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[0],
                size_kernel=3,
                stride=stride[0],
                name=name
            )
            # alternative - batch normalization -- according to your condition
            # if enable_bn:
            #     name = 'D_acc_bn' + str(0)
            #     current = tf.contrib.layers.batch_norm(
            #         current,
            #         scale=False,
            #         is_training=is_training,
            #         scope=name,
            #         reuse=reuse_variables
            #     )
            current = tf.nn.relu(current)

            name = 'D_acc_conv' + str(1)
            current = conv2d2(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[1],
                size_kernel=3,
                stride=stride[1],
                name=name
            )
            # if enable_bn:
            #     name = 'D_acc_bn' + str(1)
            #     current = tf.contrib.layers.batch_norm(
            #         current,
            #         scale=False,
            #         is_training=is_training,
            #         scope=name,
            #         reuse=reuse_variables
            #     )
            current = tf.nn.relu(current)
            current = max_pool(current)

            # alternative
            # current = avg_pool(current)

            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            name = 'D_acc_conv' + str(2)
            current = conv2d2(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[2],
                size_kernel=3,
                stride=stride[2],
                name=name
            )
            # if enable_bn:
            #     name = 'D_acc_bn' + str(2)
            #     current = tf.contrib.layers.batch_norm(
            #         current,
            #         scale=False,
            #         is_training=is_training,
            #         scope=name,
            #         reuse=reuse_variables
            #     )
            current = tf.nn.relu(current)

            name = 'D_acc_conv' + str(3)
            current = conv2d2(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[3],
                size_kernel=3,
                stride=stride[3],
                name=name
            )
            # if enable_bn:
            #     name = 'D_acc_bn' + str(3)
            #     current = tf.contrib.layers.batch_norm(
            #         current,
            #         scale=False,
            #         is_training=is_training,
            #         scope=name,
            #         reuse=reuse_variables
            #     )
            current = tf.nn.relu(current)
            current = max_pool(current)

            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

            name = 'D_acc_conv' + str(4)
            current = conv2d2(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[4],
                size_kernel=3,
                stride=stride[4],
                name=name
            )
            # if enable_bn:
            #     name = 'D_acc_bn' + str(4)
            #     current = tf.contrib.layers.batch_norm(
            #         current,
            #         scale=False,
            #         is_training=is_training,
            #         scope=name,
            #         reuse=reuse_variables
            #     )
            current = tf.nn.relu(current)

            name = 'D_acc_conv' + str(5)
            current = conv2d2(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[5],
                size_kernel=3,
                stride=stride[5],
                name=name
            )
            # if enable_bn:
            #     name = 'D_acc_bn' + str(5)
            #     current = tf.contrib.layers.batch_norm(
            #         current,
            #         scale=False,
            #         is_training=is_training,
            #         scope=name,
            #         reuse=reuse_variables
            #     )
            current = tf.nn.relu(current)

            name = 'D_acc_conv' + str(6)
            current = conv2d2(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[6],
                size_kernel=3,
                stride=stride[6],
                name=name
            )
            # if enable_bn:
            #     name = 'D_acc_bn' + str(6)
            #     current = tf.contrib.layers.batch_norm(
            #         current,
            #         scale=False,
            #         is_training=is_training,
            #         scope=name,
            #         reuse=reuse_variables
            #     )
            current = tf.nn.relu(current)

            name = 'D_acc_conv' + str(7)
            current = conv2d2(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[7],
                size_kernel=3,
                stride=stride[7],
                name=name
            )
            # if enable_bn:
            #     name = 'D_acc_bn' + str(6)
            #     current = tf.contrib.layers.batch_norm(
            #         current,
            #         scale=False,
            #         is_training=is_training,
            #         scope=name,
            #         reuse=reuse_variables
            #     )
            current = tf.nn.relu(current)
            current = max_pool(current)

            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

            name = 'D_acc_conv' + str(8)
            current = conv2d2(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[8],
                size_kernel=3,
                stride=stride[8],
                name=name
            )
            # if enable_bn:
            #     name = 'D_acc_bn' + str(7)
            #     current = tf.contrib.layers.batch_norm(
            #         current,
            #         scale=False,
            #         is_training=is_training,
            #         scope=name,
            #         reuse=reuse_variables
            #     )
            current = tf.nn.relu(current)

            name = 'D_acc_conv' + str(9)
            current = conv2d2(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[9],
                size_kernel=3,
                stride=stride[9],
                name=name
            )
            # if enable_bn:
            #     name = 'D_acc_bn' + str(8)
            #     current = tf.contrib.layers.batch_norm(
            #         current,
            #         scale=False,
            #         is_training=is_training,
            #         scope=name,
            #         reuse=reuse_variables
            #     )
            current = tf.nn.relu(current)

            name = 'D_acc_conv' + str(10)
            current = conv2d2(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[10],
                size_kernel=3,
                stride=stride[10],
                name=name
            )
            # if enable_bn:
            #     name = 'D_acc_bn' + str(9)
            #     current = tf.contrib.layers.batch_norm(
            #         current,
            #         scale=False,
            #         is_training=is_training,
            #         scope=name,
            #         reuse=reuse_variables
            #     )
            current = tf.nn.relu(current)

            name = 'D_acc_conv' + str(11)
            current = conv2d2(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[11],
                size_kernel=3,
                stride=stride[11],
                name=name
            )
            # if enable_bn:
            #     name = 'D_acc_bn' + str(6)
            #     current = tf.contrib.layers.batch_norm(
            #         current,
            #         scale=False,
            #         is_training=is_training,
            #         scope=name,
            #         reuse=reuse_variables
            #     )
            current = tf.nn.relu(current)

            current = max_pool(current)

            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

            name = 'D_acc_conv' + str(12)
            current = conv2d2(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[12],
                size_kernel=3,
                stride=stride[12],
                name=name
            )
            # if enable_bn:
            #     name = 'D_acc_bn' + str(10)
            #     current = tf.contrib.layers.batch_norm(
            #         current,
            #         scale=False,
            #         is_training=is_training,
            #         scope=name,
            #         reuse=reuse_variables
            #     )
            current = tf.nn.relu(current)

            name = 'D_acc_conv' + str(13)
            current = conv2d2(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[13],
                size_kernel=3,
                stride=stride[13],
                name=name
            )
            # if enable_bn:
            #     name = 'D_acc_bn' + str(11)
            #     current = tf.contrib.layers.batch_norm(
            #         current,
            #         scale=False,
            #         is_training=is_training,
            #         scope=name,
            #         reuse=reuse_variables
            #     )
            current = tf.nn.relu(current)

            name = 'D_acc_conv' + str(14)
            current = conv2d2(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[14],
                size_kernel=3,
                stride=stride[14],
                name=name
            )
            # if enable_bn:
            #     name = 'D_acc_bn' + str(12)
            #     current = tf.contrib.layers.batch_norm(
            #         current,
            #         scale=False,
            #         is_training=is_training,
            #         scope=name,
            #         reuse=reuse_variables
            #     )
            current = tf.nn.relu(current)

            name = 'D_acc_conv' + str(15)
            current = conv2d2(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[15],
                size_kernel=3,
                stride=stride[15],
                name=name
            )
            # if enable_bn:
            #     name = 'D_acc_bn' + str(6)
            #     current = tf.contrib.layers.batch_norm(
            #         current,
            #         scale=False,
            #         is_training=is_training,
            #         scope=name,
            #         reuse=reuse_variables
            #     )
            current = tf.nn.relu(current)

            current = max_pool(current)
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

            name = 'D_acc_fc1'
            current = fc(
                input_vector=tf.reshape(current, [self.size_batch, -1]),
                num_output_length=4096,
                name=name
            )
            current = lrelu(current)
            if self.is_training:
                current = tf.nn.dropout(current, 0.5)

            name = 'D_acc_fc2'
            current = fc(
                input_vector=tf.reshape(current, [self.size_batch, -1]),
                num_output_length=4096,
                name=name
            )
            current = lrelu(current)
            if self.is_training:
                current = tf.nn.dropout(current, 0.5)

            name = 'D_acc_fc3'
            current1 = fc(
                input_vector=tf.reshape(current, [self.size_batch, -1]),
                num_output_length=self.num_categories,
                name=name
            )
            name = 'D_acc_fc4'
            current2 = fc(
                input_vector=tf.reshape(current, [self.size_batch, -1]),
                num_output_length=self.num_poses,
                name=name
            )
            return current1, current2


    # (2) Alternative - classifier --VGG16
#     def discriminator_acc(self, image, is_training=True, reuse_variables=False, num_hidden_layer_channels=(32,32,64,64,128,128,128,256,256,256,512,512,512), enable_bn=True):
#         #pdb.set_trace()
#         if reuse_variables:
#             tf.get_variable_scope().reuse_variables()
#         num_layers = len(num_hidden_layer_channels)
#         current = image
#         num_hidden_layer_channels=num_hidden_layer_channels
#         stride = (1,1,1,1,1,1,1,1,1,1,1,1,1)
#         # conv layers with stride 2
#
#         #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#         name = 'D_acc_conv' + str(0)
#         current = conv2d2(
#                 input_map=current,
#                 num_output_channels=num_hidden_layer_channels[0],
#                 size_kernel=3,
#                 stride=stride[0],
#                 name=name
#             )
#         # if enable_bn:
#         #     name = 'D_acc_bn' + str(0)
#         #     current = tf.contrib.layers.batch_norm(
#         #         current,
#         #         scale=False,
#         #         is_training=is_training,
#         #         scope=name,
#         #         reuse=reuse_variables
#         #     )
#         current = tf.nn.relu(current)
#
#         name = 'D_acc_conv' + str(1)
#         current = conv2d2(
#             input_map=current,
#             num_output_channels=num_hidden_layer_channels[1],
#             size_kernel=3,
#             stride=stride[1],
#             name=name
#         )
#         # if enable_bn:
#         #     name = 'D_acc_bn' + str(1)
#         #     current = tf.contrib.layers.batch_norm(
#         #         current,
#         #         scale=False,
#         #         is_training=is_training,
#         #         scope=name,
#         #         reuse=reuse_variables
#         #     )
#         current = tf.nn.relu(current)
#         current=max_pool(current)
#
# # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#         name = 'D_acc_conv' + str(2)
#         current = conv2d2(
#             input_map=current,
#             num_output_channels=num_hidden_layer_channels[2],
#             size_kernel=3,
#             stride=stride[2],
#             name=name
#         )
#         # if enable_bn:
#         #     name = 'D_acc_bn' + str(2)
#         #     current = tf.contrib.layers.batch_norm(
#         #         current,
#         #         scale=False,
#         #         is_training=is_training,
#         #         scope=name,
#         #         reuse=reuse_variables
#         #     )
#         current = tf.nn.relu(current)
#
#         name = 'D_acc_conv' + str(3)
#         current = conv2d2(
#             input_map=current,
#             num_output_channels=num_hidden_layer_channels[3],
#             size_kernel=3,
#             stride=stride[3],
#             name=name
#         )
#         # if enable_bn:
#         #     name = 'D_acc_bn' + str(3)
#         #     current = tf.contrib.layers.batch_norm(
#         #         current,
#         #         scale=False,
#         #         is_training=is_training,
#         #         scope=name,
#         #         reuse=reuse_variables
#         #     )
#         # current = tf.nn.relu(current)
#         current =max_pool(current)
#
# # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#         name = 'D_acc_conv' + str(4)
#         current = conv2d2(
#             input_map=current,
#             num_output_channels=num_hidden_layer_channels[4],
#             size_kernel=3,
#             stride=stride[4],
#             name=name
#         )
#         # if enable_bn:
#         #     name = 'D_acc_bn' + str(4)
#         #     current = tf.contrib.layers.batch_norm(
#         #         current,
#         #         scale=False,
#         #         is_training=is_training,
#         #         scope=name,
#         #         reuse=reuse_variables
#         #     )
#         current = tf.nn.relu(current)
#
#         name = 'D_acc_conv' + str(5)
#         current = conv2d2(
#             input_map=current,
#             num_output_channels=num_hidden_layer_channels[5],
#             size_kernel=3,
#             stride=stride[5],
#             name=name
#         )
#         # if enable_bn:
#         #     name = 'D_acc_bn' + str(5)
#         #     current = tf.contrib.layers.batch_norm(
#         #         current,
#         #         scale=False,
#         #         is_training=is_training,
#         #         scope=name,
#         #         reuse=reuse_variables
#         #     )
#         current = tf.nn.relu(current)
#
#         name = 'D_acc_conv' + str(6)
#         current = conv2d2(
#             input_map=current,
#             num_output_channels=num_hidden_layer_channels[6],
#             size_kernel=3,
#             stride=stride[6],
#             name=name
#         )
#         # if enable_bn:
#         #     name = 'D_acc_bn' + str(6)
#         #     current = tf.contrib.layers.batch_norm(
#         #         current,
#         #         scale=False,
#         #         is_training=is_training,
#         #         scope=name,
#         #         reuse=reuse_variables
#         #     )
#         current = tf.nn.relu(current)
#         current =max_pool(current)
#
# #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#         name = 'D_acc_conv' + str(7)
#         current = conv2d2(
#             input_map=current,
#             num_output_channels=num_hidden_layer_channels[7],
#             size_kernel=3,
#             stride=stride[7],
#             name=name
#         )
#         # if enable_bn:
#         #     name = 'D_acc_bn' + str(7)
#         #     current = tf.contrib.layers.batch_norm(
#         #         current,
#         #         scale=False,
#         #         is_training=is_training,
#         #         scope=name,
#         #         reuse=reuse_variables
#         #     )
#         current = tf.nn.relu(current)
#
#         name = 'D_acc_conv' + str(8)
#         current = conv2d2(
#             input_map=current,
#             num_output_channels=num_hidden_layer_channels[8],
#             size_kernel=3,
#             stride=stride[8],
#             name=name
#         )
#         # if enable_bn:
#         #     name = 'D_acc_bn' + str(8)
#         #     current = tf.contrib.layers.batch_norm(
#         #         current,
#         #         scale=False,
#         #         is_training=is_training,
#         #         scope=name,
#         #         reuse=reuse_variables
#         #     )
#         current = tf.nn.relu(current)
#
#         name = 'D_acc_conv' + str(9)
#         current = conv2d2(
#             input_map=current,
#             num_output_channels=num_hidden_layer_channels[9],
#             size_kernel=3,
#             stride=stride[9],
#             name=name
#         )
#         # if enable_bn:
#         #     name = 'D_acc_bn' + str(9)
#         #     current = tf.contrib.layers.batch_norm(
#         #         current,
#         #         scale=False,
#         #         is_training=is_training,
#         #         scope=name,
#         #         reuse=reuse_variables
#         #     )
#         current = tf.nn.relu(current)
#         current =max_pool(current)
#
# # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#         name = 'D_acc_conv' + str(10)
#         current = conv2d2(
#             input_map=current,
#             num_output_channels=num_hidden_layer_channels[10],
#             size_kernel=3,
#             stride=stride[10],
#             name=name
#         )
#         # if enable_bn:
#         #     name = 'D_acc_bn' + str(7)
#         #     current = tf.contrib.layers.batch_norm(
#         #         current,
#         #         scale=False,
#         #         is_training=is_training,
#         #         scope=name,
#         #         reuse=reuse_variables
#         #     )
#         current = tf.nn.relu(current)
#
#         name = 'D_acc_conv' + str(11)
#         current = conv2d2(
#             input_map=current,
#             num_output_channels=num_hidden_layer_channels[11],
#             size_kernel=3,
#             stride=stride[11],
#             name=name
#         )
#         # if enable_bn:
#         #     name = 'D_acc_bn' + str(8)
#         #     current = tf.contrib.layers.batch_norm(
#         #         current,
#         #         scale=False,
#         #         is_training=is_training,
#         #         scope=name,
#         #         reuse=reuse_variables
#         #     )
#         current = tf.nn.relu(current)
#
#         name = 'D_acc_conv' + str(12)
#         current = conv2d2(
#             input_map=current,
#             num_output_channels=num_hidden_layer_channels[12],
#             size_kernel=3,
#             stride=stride[12],
#             name=name
#         )
#         # if enable_bn:
#         #     name = 'D_acc_bn' + str(9)
#         #     current = tf.contrib.layers.batch_norm(
#         #         current,
#         #         scale=False,
#         #         is_training=is_training,
#         #         scope=name,
#         #         reuse=reuse_variables
#         #     )
#         current = tf.nn.relu(current)
#         current = max_pool(current)
# # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#         name = 'D_acc_fc1'
#         current = fc(
#             input_vector=tf.reshape(current, [self.size_batch, -1]),
#             num_output_length=2048,
#             name=name
#         )
#         current = lrelu(current)
#         if self.is_training:
#             current = tf.nn.dropout(current, 0.5)
#
#         name = 'D_acc_fc2'
#         current = fc(
#             input_vector=tf.reshape(current, [self.size_batch, -1]),
#             num_output_length=1024,
#             name=name
#         )
#         current = lrelu(current)
#         if self.is_training:
#             current = tf.nn.dropout(current, 0.5)
#
#         name = 'D_acc_fc3'
#         current1 = fc(
#             input_vector=tf.reshape(current, [self.size_batch, -1]),
#             num_output_length=self.num_categories,
#             name=name
#         )
#         name = 'D_acc_fc4'
#         current2 = fc(
#             input_vector=tf.reshape(current, [self.size_batch, -1]),
#             num_output_length=self.num_poses,
#             name=name
#         )
#         return current1, current2


    # (3) Alternative - classifier
    # def discriminator_acc(self, image, is_training=True, reuse_variables=False, num_hidden_layer_channels=(32,64,64,128,128,128,96,192,126,160), enable_bn=True):
    #     if reuse_variables:
    #         tf.get_variable_scope().reuse_variables()
    #     num_layers = len(num_hidden_layer_channels)
    #     current = image
    #     stride = (1,1,2,1,1,2,1,1,2,1,1)
    #     # conv layers with stride 2
    #     for i in range(num_layers):
    #         name = 'D_acc_conv' + str(i)
    #         current = conv2d(
    #                 input_map=current,
    #                 num_output_channels=num_hidden_layer_channels[i],
    #                 size_kernel=3,
    #                 stride=stride[i],
    #                 name=name
    #             )
    #         if enable_bn:
    #             name = 'D_acc_bn' + str(i)
    #             current = tf.contrib.layers.batch_norm(
    #                 current,
    #                 scale=False,
    #                 is_training=is_training,
    #                 scope=name,
    #                 reuse=reuse_variables
    #             )
    #         current = tf.nn.relu(current)
    #     # max_pool(current)
    #     # tf.reshape(current,[self.size_batch, 160])
    #     avg_pool(current)
    #     name = 'D_acc_fc1'
    #     current = fc(
    #         input_vector=tf.reshape(current, [self.size_batch, -1]),
    #         num_output_length=2048,
    #         name=name
    #     )
    #     current = lrelu(current)
    #     if self.is_training:
    #         current = tf.nn.dropout(current, 0.5)
    #
    #     name = 'D_acc_fc2'
    #     current = fc(
    #         input_vector=tf.reshape(current, [self.size_batch, -1]),
    #         num_output_length=1024,
    #         name=name
    #     )
    #     current = lrelu(current)
    #     if self.is_training:
    #         current = tf.nn.dropout(current, 0.5)
    #
    #     name = 'D_acc_fc3'
    #     current1 = fc(
    #         input_vector=tf.reshape(current, [self.size_batch, -1]),
    #         num_output_length=self.num_categories,
    #         name=name
    #     )
    #     name = 'D_acc_fc4'
    #     current2 = fc(
    #         input_vector=tf.reshape(current, [self.size_batch, -1]),
    #         num_output_length=self.num_poses,
    #         name=name
    #     )
    #     return current1, current2

    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(
            sess=self.session,
            save_path=os.path.join(checkpoint_dir, 'model'),
            global_step=self.EG_global_step.eval()
        )

    def load_checkpoint(self):
        print("\n\tLoading pre-trained model ...")
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))
            return True
        else:
            return False

    def sample(self, images, labels, pose, name):
        sample_dir = os.path.join(self.save_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        f, G = self.session.run(
                [self.f, self.G],
                feed_dict={
                    self.input_image: images,
                    self.expression: labels,
                    self.pose: pose
                }
            )
        size_frame = int(np.sqrt(self.size_batch))
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(sample_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[size_frame, size_frame]
        )

    def test(self, images, pose, name, expression):
        # pdb.set_trace()
        test_dir = os.path.join(self.save_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        images = images[:int(np.sqrt(self.size_batch)), :, :, :]

        pose = pose[:int(np.sqrt(self.size_batch)), :]
        size_sample = images.shape[0]
        labels = np.arange(size_sample)
        labels = np.repeat(labels, size_sample)
        query_labels = np.ones(
            shape=(size_sample ** 2, size_sample),
            dtype=np.float32
        ) * self.image_value_range[0]
        for i in range(query_labels.shape[0]):
            query_labels[i, labels[i]] = self.image_value_range[-1]
        # pdb.set_trace()
        query_images = np.tile(images, [self.num_categories, 1, 1, 1])
        query_pose = np.tile(pose, [self.num_categories, 1])

        print ('Generate images with different expressions and poses')
        f, G = self.session.run(
            [self.f, self.G],
            feed_dict={
                self.input_image: query_images,
                self.expression: query_labels,
                self.pose: query_pose
            }
        )
        save_batch_images(
            batch_images=query_images,
            save_path=os.path.join(test_dir, 'input.png'),
            image_value_range=self.image_value_range,
            size_frame=[size_sample, size_sample]
        )
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(test_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[size_sample, size_sample]
        )

    def test_acc(self, images, expression, pose):
        self.is_training = False
        test_images = images
        test_expression = expression
        test_poses = pose

        D_ex_acc, D_pose_acc = self.session.run(
                        [self.d_ex_count, self.d_pose_count],
                        feed_dict={
                            self.input_image: test_images,
                            self.expression: test_expression,
                            self.pose: test_poses
                        }
                    )
        # if sparse_softmax_cross_entropy_with_logits is used
        # D_ex_acc, D_pose_acc= self.session.run(
        #             [self.d_ex_count, self.d_pose_count],
        #             feed_dict={
        #                 self.input_image: test_images,
        #                 self.expression1: test_expression,
        #                 self.pose1: test_poses
        #             }
        #         )

        lo = self.session.run(
            fetches=[self.D_input_ex_logits],
            feed_dict={
                self.input_image: test_images
            }
        )
        re =lo[0]
        index = self.session.run(tf.argmax(re,1))
        print (self.session.run(tf.argmax(re,1)))
        print ("test Accex =%.4f \t ACCpose= %.4f " % (D_ex_acc, D_pose_acc))
        return D_ex_acc, D_pose_acc, index

    #generate
    def custom_test(self,testing_samples_dir):
        pdb.set_trace()
        self.custom_test_names = np.loadtxt(testing_samples_dir + '/test.txt', dtype=bytes, delimiter=' ').astype(str)
        len_testset = len(self.custom_test_names)
        np.random.shuffle(self.custom_test_names)
        test_batches = len_testset // self.size_batch

        for i in range(test_batches):

            test_images, test_label_expression, test_label_pose, test_names= self.get_batch_custom_test(i)
            num_samples = int(np.sqrt(self.size_batch))

            pose_1 = np.ones(
                shape=(num_samples, 5),
                dtype=np.float32
            ) * self.image_value_range[0]
            pose_2 = np.ones(
                shape=(num_samples, 5),
                dtype=np.float32
            ) * self.image_value_range[0]
            pose_3 = np.ones(
                shape=(num_samples, 5),
                dtype=np.float32
            ) * self.image_value_range[0]
            pose_4 = np.ones(
                shape=(num_samples, 5),
                dtype=np.float32
            ) * self.image_value_range[0]
            pose_5 = np.ones(
                shape=(num_samples, 5),
                dtype=np.float32
            ) * self.image_value_range[0]

            for p in range(pose_1.shape[0]):
                pose_1[p, 0] = self.image_value_range[-1]
                pose_2[p, 1] = self.image_value_range[-1]
                pose_3[p, 2] = self.image_value_range[-1]
                pose_4[p, 3] = self.image_value_range[-1]
                pose_5[p, 4] = self.image_value_range[-1]

            if not self.load_checkpoint():
                print("\tFAILED >_<!")
                exit(0)
            else:
                print("\tSUCCESS ^_^")

            self.test(test_images, pose_1, str(i)+'test_as_1.png', test_label_expression,i)
            self.test(test_images, pose_2, str(i)+'test_as_2.png', test_label_expression,i)
            self.test(test_images, pose_3, str(i)+'test_as_3.png', test_label_expression,i)
            self.test(test_images, pose_4, str(i)+'test_as_4.png', test_label_expression,i)
            self.test(test_images, pose_5, str(i)+'test_as_5.png', test_label_expression,i)

            print '\n\tDone! Results are saved as %s\n' % os.path.join(self.save_dir, 'test', 'test_as_xxx.png')

    def get_batch_custom_test(self, idx=0):

        tt_files = self.custom_test_names[idx*self.size_batch: idx*self.size_batch + self.size_batch]
        batch_images = np.zeros((self.size_batch, self.size_image, self.size_image, 3))
        for i in range(tt_files.shape[0]):
            sample = [load_image(
                image_path=self.pathtrain + 'data/FACE_AGING_train/' + tt_files[i,0],
                image_size=self.size_image,
                image_value_range=self.image_value_range,
                is_gray=(self.num_input_channels == 1),
            )]
            if self.num_input_channels == 1:
                batch_images[i, :, :, :] = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                batch_images[i, :, :, :] = np.array(sample).astype(np.float32)

        tt_label_expression = np.ones(
            shape=(len(tt_files), self.num_categories),
            dtype=np.float32
        ) * self.image_value_range[0]

        tt_label_pose = np.ones(
            shape=(len(tt_files), self.num_poses),
            dtype=np.float32
        ) * self.image_value_range[0]

        # tt_label_expression1 = np.ones(
        #     shape=(len(tt_files)),
        #     dtype=np.float32
        # ) * self.image_value_range[0]
        #
        # tt_label_pose1 = np.ones(
        #     shape=(len(tt_files)),
        #     dtype=np.float32
        # ) * self.image_value_range[0]

        for i, label in enumerate(tt_files[:, 1]):
            label = tt_files[i, 1].astype('int')
            tt_label_expression[i, label] = self.image_value_range[-1]

            # tt_label_expression1[i]=label

            pose = tt_files[i, 2].astype('int')
            if pose == 41:
                pose = 0
            elif pose == 130:
                pose = 1
            elif pose == 50:
                pose = 2
            elif pose == 51:
                pose = 3
            elif pose == 140:
                pose = 4
            tt_label_pose[i, pose] = self.image_value_range[-1]

            # tt_label_pose1[i]=pose

        # return batch_images, tt_label_expression, tt_label_pose, tt_label_expression1, tt_label_pose1, tt_files
        return batch_images, tt_label_expression, tt_label_pose, tt_files

