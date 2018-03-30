# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import math
BN_DECAY = 0.9997
BN_EPSILON = 0.001
UPDATE_OPS_COLLECTION = 'update_ops'

class unet_3d_model(object):
    def __init__(self,
                 batch_size=10,
                 input_size=[10,10,10],
                 kernel_size=4,
                 in_channel=1,
                 num_filter = 16,
                 stride = [1,1,1],
                 epochs = 2):
        self.batch_size = batch_size
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.num_filter = num_filter
        self.stride = stride
        self.epochs = epochs

    def build_model(self,input, target, is_training,bn_select,prelu):
        with tf.variable_scope('net', reuse=False) as vs:
            # ---------------------------------conv-------------------------------------------
            conv1 = self.conv3d(input,self.kernel_size,self.in_channel,self.num_filter,'conv1')
            conv1 = self.maxpool3d(conv1)
            if prelu == True:
                relu1 = self.prelu(conv1,'relu1')
            else:
                relu1 = tf.nn.relu(conv1)

            conv2 = self.conv3d(relu1, self.kernel_size, self.num_filter, self.num_filter * 2, 'conv2')
            conv2 = self.maxpool3d(conv2)
            if bn_select == 1:
                bn2 = self.batchnorm(conv2, 'bn2')
            elif bn_select == 2:
                bn2 = self.bn(conv2, is_training, 'bn2')
            else:
                bn2 = conv2
            if prelu == True:
                relu2 = self.prelu(bn2,'relu2')
            else:
                relu2 = tf.nn.relu(bn2)

            conv3 = self.conv3d(relu2, self.kernel_size, self.num_filter * 2, self.num_filter * 4, 'conv3')
            conv3 = self.maxpool3d(conv3)
            if bn_select == 1:
                bn3 = self.batchnorm(conv3, 'bn3')
            elif bn_select == 2:
                bn3 = self.bn(conv3, is_training, 'bn3')
            else:
                bn3 = conv3
            if prelu == True:
                relu3 = self.prelu(bn3, 'relu3')
            else:
                relu3 = tf.nn.relu(bn3)

            # ---------------------------------deconv-------------------------------------------
            conv4 = self.deconv3d(relu3, self.kernel_size, self.num_filter * 2,self.batch_size, 'conv4',conv2)
            if bn_select == 1:
                bn4 = self.batchnorm(conv4, 'bn4')
            elif bn_select == 2:
                bn4 = self.bn(conv4, is_training, 'bn4')
            else:
                bn4 = conv4
            if prelu == True:
                relu4 = self.prelu(tf.concat([bn4,conv2],axis=4), 'relu4')
            else:
                relu4 = tf.nn.relu(tf.concat([bn4,conv2],axis=4))

            conv5 = self.deconv3d(relu4, self.kernel_size, self.num_filter,self.batch_size, 'conv5',conv1)
            if bn_select == 1:
                bn5 = self.batchnorm(conv5, 'bn5')
            elif bn_select == 2:
                bn5 = self.bn(conv5, is_training, 'bn5')
            else:
                bn5 = conv5
            if prelu == True:
                relu5 = self.prelu(tf.concat([bn5,conv1],axis=4), 'relu5')
            else:
                relu5 = tf.nn.relu(tf.concat([bn5,conv1],axis=4))

            output_noise = self.deconv3d(relu5,self.kernel_size,self.in_channel,self.batch_size,'conv6',input)
            output = input - output_noise

            L1_loss_forward = tf.reduce_mean(tf.abs(output - target))
            L2_loss_forward = tf.reduce_mean(tf.square(output - target))
            pixel_num = self.input_size[0] * self.input_size[1]
            #output_flatten = tf.reduce_sum(output,axis=3)
            #tvDiff_loss_forward = \
            #    tf.reduce_mean(tf.image.total_variation(output_flatten)) / pixel_num * 200 / 10000

            tv_lambda = 2000000
            for i in range(self.input_size[2]):
                if i == 0:
                    tvDiff_loss_forward = \
                    tf.reduce_mean(tf.image.total_variation(output[:, :, :, i, :])) / pixel_num * tv_lambda / 10000
                else:
                    tvDiff_loss_forward = tvDiff_loss_forward + \
                                          tf.reduce_mean(tf.image.total_variation(output[:,:,:,i,:])) / pixel_num * 200 / 10000
            for i in range(self.input_size[1]):
                if i == 0:
                    tvDiff_loss_forward = \
                    tf.reduce_mean(tf.image.total_variation(output[:, :, i, :, :])) / pixel_num * tv_lambda / 10000
                else:
                    tvDiff_loss_forward = tvDiff_loss_forward + \
                                          tf.reduce_mean(tf.image.total_variation(output[:,:,i,:,:])) / pixel_num * 200 / 10000
            for i in range(self.input_size[0]):
                if i == 0:
                    tvDiff_loss_forward = \
                    tf.reduce_mean(tf.image.total_variation(output[:, i, :, :, :])) / pixel_num * tv_lambda / 10000
                else:
                    tvDiff_loss_forward = tvDiff_loss_forward + \
                                          tf.reduce_mean(tf.image.total_variation(output[:,i,:,:,:])) / pixel_num * tv_lambda / 10000

            tvDiff_loss_forward = tvDiff_loss_forward / self.input_size[2] / self.input_size[1] / self.input_size[0]
            loss = L1_loss_forward + tvDiff_loss_forward
            loss2 = L2_loss_forward + tvDiff_loss_forward
            del_snr, snr = self.snr(input,output,target)
            input_snr = self.input_snr(input, target)
            with tf.name_scope('summaries'):
                tf.summary.scalar('all loss', loss)
                tf.summary.scalar('L1_loss',L1_loss_forward)
                tf.summary.scalar('tv_loss',tvDiff_loss_forward)
                tf.summary.scalar('snr',snr)
            return output,loss,L1_loss_forward,tvDiff_loss_forward,snr,del_snr,output_noise,input_snr

    def batchnorm(self,input, name):
        with tf.variable_scope(name):
            input = tf.identity(input)
            channels = input.get_shape()[-1:]
            offset = tf.get_variable("gamma", [channels[0]], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)


            scale = tf.get_variable("beta", [channels[0]], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(1, 0.02),trainable=True)
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2, 3], keep_dims=False)
            variance_epsilon = 1e-5
            normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale,
                                                 variance_epsilon=variance_epsilon)
        return normalized

    def input_snr(self, input, target):
        tmp_snr0 = tf.reduce_sum(tf.square(tf.abs(target))) / tf.reduce_sum(tf.square(tf.abs(target - input)))
        out0 = 10.0 * tf.log(tmp_snr0) / tf.log(10.0)  # 输入图片的snr
        return out0

    def bn(self,x,is_training,name):
        with tf.variable_scope(name):
            x_shape = x.get_shape()
            params_shape = x_shape[-1:]

            axis = list(range(len(x_shape) - 1))

            beta = tf.get_variable('beta', params_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', params_shape, dtype=tf.float32, initializer=tf.ones_initializer())

            moving_mean = tf.get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
            moving_variance = tf.get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)

            # these op will only be performed when traing
            mean, variance = tf.nn.moments(x, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
            tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
            tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

            is_training2 = tf.convert_to_tensor(is_training,dtype='bool',name='is_training')

            mean, variance = control_flow_ops.cond(is_training2, lambda :(mean,variance), lambda :(moving_mean,moving_variance))

            x = tf.nn.batch_normalization(x,mean,variance,beta,gamma,BN_EPSILON)
        self.variable_summaries(beta)
        self.variable_summaries(gamma)

        return x


    def snr(self,x,y,y_true):
        tmp_snr = tf.reduce_sum(tf.square(tf.abs(y_true))) / tf.reduce_sum(tf.square(tf.abs(y_true - y)))
        out = 10.0 * tf.log(tmp_snr) / tf.log(10.0)             # 输出图片的snr

        tmp_snr0 = tf.reduce_sum(tf.square(tf.abs(y_true))) / tf.reduce_sum(tf.square(tf.abs(y_true - x)))
        out0 = 10.0 * tf.log(tmp_snr0) / tf.log(10.0)           # 输入图片的snr

        del_snr = out - out0
        return del_snr, out

    def conv3d(self,x,k,in_channel,out_channel,name):
        with tf.variable_scope(name):
            kernel = tf.get_variable('kernel', [k,k,k,in_channel,out_channel],
                                     dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.05),
                                     trainable=True)
            self.variable_summaries(kernel)
            conv = tf.nn.conv3d(x,kernel,strides=[1,1,1,1,1],padding="SAME")
            return conv

    def deconv3d(self,x,k,out_channels,batch_size,name,x2):
        with tf.variable_scope(name):
            in_width, in_height, in_depth, in_channels = [int(d) for d in x.get_shape()[1:]]
            in_width2, in_height2, in_depth2, in_channels2 = [int(d) for d in x2.get_shape()[1:]]
            kernel = tf.get_variable('kernel',[k,k,k,out_channels,in_channels],
                                     dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.05),
                                     trainable=True)
            self.variable_summaries(kernel)
            deconv = tf.nn.conv3d_transpose(x,
                                            kernel,
                                            [batch_size,in_width2,in_height2,in_depth2,out_channels],
                                            strides=[1,2,2,2,1],
                                            padding="SAME")
            return deconv

    def maxpool3d(self,x):
        return tf.nn.max_pool3d(x,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding="SAME")

    def prelu(self,_x,name):
        with tf.variable_scope(name):
            alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
            pos = tf.nn.relu(_x)
            neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg

    def _get_conv_variable(self,input_size):
        out = tf.Variable(tf.truncated_normal(input_size,stddev=0.01,name="weights"))
        return out

    def _get_bias_variable(self,input_size):
        out = tf.Variable(tf.zeros(input_size),name="biases")
        return out

    def variable_summaries(self,var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(var,mean))))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
