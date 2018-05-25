"""
This file contains a implement of Super-resolution by CNN network.
"""

import tensorflow as tf
import os
import cv2 as cv
from time import time
import numpy as np


class TFNetwork:

    def __init__(self, train_path, test_path, train_test_img_size=33, pad_size=None):
        self.train_path = train_path
        self.test_path = test_path

        self.train_test_img_size = train_test_img_size
        if pad_size is None:
            self.pad_size = int(12/2)  # It depends on the network itself
        else:
            self.pad_size = pad_size

    def pre_data(self, train_sample_number, test_sample_number, crop_stride, factor):

        init = time()  # Record init time

        print('Start prepare training data.....')

        # Read original images
        train_filenames = os.listdir(self.train_path)
        train_filenames = sorted(train_filenames)

        test_filenames = os.listdir(self.test_path)
        test_filenames = sorted(test_filenames)

        train_sample = []
        test_sample = []

        for i in range(train_sample_number):

            if i % 10 == 0:
                print('Reading No.[%d] train image. Total number: [%d]. Current time consume of preparation:[%.1f] s' %
                      (i+1, train_sample_number, time()-init))

            image = cv.cvtColor(cv.imread(self.train_path + train_filenames[i]), cv.COLOR_BGR2YCrCb)
            image = image[:, :, 0]  # Super-resolution only Y channel
            image = image.astype(np.float32) / 255.  # Normalization
            train_sample.append(image)

        for i in range(test_sample_number):

            if i % 5 == 0:
                print('Reading No.[%d] test image. Total number: [%d]. Current time consume of preparation: [%.1f] s'
                      % (i+1, test_sample_number, time()-init))

            image = cv.cvtColor(cv.imread(self.test_path + test_filenames[i]), cv.COLOR_BGR2YCrCb)
            image = image[:, :, 0]  # Super-resolution only Y channel
            image = image.astype(np.float32) / 255.  # Normalization
            test_sample.append(image)

        # Crop image
        train_sample_label = []
        train_sample_input = []
        test_sample_label = []
        test_sample_input = []

        for i in range(train_sample_number):

            if i % 10 == 0:
                print('Processing No.[%d] train image. Total number: [%d]. '
                      'Current time consume of preparation: [%.1f] s' %
                      (i+1, train_sample_number, time()-init))

            # Make shape of image a rectangle
            h, w = train_sample[i].shape
            label = train_sample[i]
            if h > w:
                label = label[:w, :]
            else:
                label = label[:, :h]

            # Downsample image. Then upsamle the same the downsampled image
            input_ = cv.resize(label, (int(label.shape[0]/factor), int(label.shape[1]/factor)))
            input_ = cv.resize(input_, (label.shape[0], label.shape[1]))

            for x in range(0, label.shape[0] - self.train_test_img_size + 1, crop_stride):
                for y in range(0, label.shape[1] - self.train_test_img_size + 1, crop_stride):

                    sub_input = input_[x:x + self.train_test_img_size, y:y + self.train_test_img_size]
                    sub_label = label[x:x + self.train_test_img_size, y:y + self.train_test_img_size]
                    sub_label = sub_label[self.pad_size:int(-1*self.pad_size), self.pad_size:int(-1*self.pad_size)]

                    sub_input = sub_input.reshape([self.train_test_img_size, self.train_test_img_size, 1])
                    sub_label = sub_label.reshape([self.train_test_img_size - int(2*self.pad_size),
                                                   self.train_test_img_size - int(2*self.pad_size),
                                                   1])

                    train_sample_input.append(sub_input)
                    train_sample_label.append(sub_label)

        for i in range(test_sample_number):

            if i % 5 == 0:
                print('Processing No.[%d] test image. Total number: [%d]. '
                      'Current time consume of preparation: [%.1f] s' %
                      (i+1, test_sample_number, time()-init))

            # Make shape of image a rectangle
            h, w = test_sample[i].shape
            label = test_sample[i]
            if h > w:
                label = label[:w, :]
            else:
                label = label[:, :h]

            # Downsample image. Then upsamle the same the downsampled image
            input_ = cv.resize(label, (int(label.shape[0]/factor), int(label.shape[1]/factor)))
            input_ = cv.resize(input_, (label.shape[0], label.shape[1]))

            for x in range(0, label.shape[0] - self.train_test_img_size + 1, crop_stride):
                for y in range(0, label.shape[1] - self.train_test_img_size + 1, crop_stride):

                    sub_input = input_[x:x + self.train_test_img_size, y:y + self.train_test_img_size]
                    sub_label = label[x:x + self.train_test_img_size, y:y + self.train_test_img_size]
                    sub_label = sub_label[self.pad_size:int(-1*self.pad_size), self.pad_size:int(-1*self.pad_size)]

                    sub_input = sub_input.reshape([self.train_test_img_size, self.train_test_img_size, 1])
                    sub_label = sub_label.reshape([self.train_test_img_size - int(2*self.pad_size),
                                                   self.train_test_img_size - int(2*self.pad_size),
                                                   1])

                    test_sample_input.append(sub_input)
                    test_sample_label.append(sub_label)

        print('Total number of train set: [%d], Total number of test set: [%d]' %
              (train_sample_input.__len__(), test_sample_input.__len__()))

        print('Start converting data from list to numpy.....')

        train_sample_input = np.array(train_sample_input)
        train_sample_label = np.array(train_sample_label)
        test_sample_input = np.array(test_sample_input)
        test_sample_label = np.array(test_sample_label)

        print('Convert is done, Current time consume of preparation: [%.1f] s' % (time() - init))

        return train_sample_input, train_sample_label, test_sample_input, test_sample_label, time()-init

    def train(self, n_epoch, batch_size, train_sample_number, test_sample_number):

        init = time()

        print('Starting training network. Begin to read train data.....')

        # !!I don't use test data
        train_sample_input, train_sample_label, _, _, _ = \
            self.pre_data(train_sample_number=train_sample_number, test_sample_number=test_sample_number,
                          crop_stride=20, factor=4)

        print('Read train data is done. Current time consume: [%.1f] s' % (time() - init))

        input_images = tf.placeholder(dtype=tf.float32, shape=[None, self.train_test_img_size, self.train_test_img_size, 1])
        label_images = tf.placeholder(dtype=tf.float32, shape=[None, self.train_test_img_size - int(2*self.pad_size),
                                                               self.train_test_img_size - int(2*self.pad_size), 1])

        b1 = tf.Variable(tf.zeros([64]), name='b1')
        b2 = tf.Variable(tf.zeros([32]), name='b2')
        b3 = tf.Variable(tf.zeros([1]), name='b3')

        w1 = tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1')
        w2 = tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2')
        w3 = tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')

        conv1 = tf.nn.relu(tf.nn.conv2d(input=input_images, filter=w1, strides=[1, 1, 1, 1], padding='VALID',
                                        name='conv1') + b1)
        conv2 = tf.nn.relu(tf.nn.conv2d(input=conv1, filter=w2, strides=[1, 1, 1, 1], padding='VALID',
                                        name='conv2') + b2)
        conv3 = tf.nn.conv2d(input=conv2, filter=w3, strides=[1, 1, 1, 1], padding='VALID',
                             name='conv3') + b3

        loss_op = tf.reduce_mean(tf.square(conv3 - label_images))
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss_op)

        print('Have built network. Current time consume: [%.1f] s' % (time() - init))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        loss_record = []
        batch_epoch = int(train_sample_input.shape[0]/batch_size)

        for i in range(n_epoch):

            for j in range(batch_epoch):

                train_batch_input = train_sample_input[j:j+batch_size, :, :, :]
                train_batch_label = train_sample_label[j:j+batch_size, :, :, :]

                loss_cur, _ = sess.run([loss_op, train_op],
                                                     feed_dict={input_images: train_batch_input,
                                                                label_images:train_batch_label})
                loss_record.append(loss_cur)
                print('In No.[%d] epoch: No.[%d] batch. Total number of batch: [%d]. '
                      'Total number of epoch: [%d]. Current value of loss function: : [%.5e] ' %
                      (i+1, j+1, batch_epoch, n_epoch, loss_cur))

        saver = tf.train.Saver({
            'b1': b1, 'b2': b2, 'b3': b3,
            'w1': w1, 'w2': w2, 'w3': w3,
        })

        # Save model to a fixed location
        saver.save(sess, './srcnn_model/model.ckpt')
        sess.close()

        print('Starting writing value changes of loss function to a .mat file.....')
        import scipy.io as sio

        # Save vaule changes of loss function(to a .mat file)
        sio.savemat('./result/srcnn_loss.mat', {
            'srcnn_loss': np.array(loss_record),
        })

        print('Train is done. Total time consume: [%.1f] s' % (time() - init))

    def predict(self, input_video, mat_path):

        from inputvideo import VideoSRContinuous

        # It is very pity that I fix the resolution of input video, which make using other video file hard. :(
        print('Start loading video.....')
        lr_raw_y = VideoSRContinuous(path=input_video, channel=0, batchframe_size=(150, 150)).totalframes()
        hr_raw_y = VideoSRContinuous(path=input_video, channel=0, batchframe_size=(612, 612)).totalframes()
        lr_raw_cr = VideoSRContinuous(path=input_video, channel=1, batchframe_size=(150, 150)).totalframes()
        hr_raw_cr = VideoSRContinuous(path=input_video, channel=1, batchframe_size=(612, 612)).totalframes()
        lr_raw_cb = VideoSRContinuous(path=input_video, channel=2, batchframe_size=(150, 150)).totalframes()
        hr_raw_cb = VideoSRContinuous(path=input_video, channel=2, batchframe_size=(612, 612)).totalframes()

        hr = np.zeros([600, 600, 3, hr_raw_y.shape[2]], dtype=np.uint8)
        hr_cur = np.zeros([600, 600, 3], dtype=np.uint8)

        lr_y = np.zeros([612, 612, lr_raw_y.shape[2]])
        lr_cr = np.zeros([600, 600, lr_raw_y.shape[2]])
        lr_cb = np.zeros([600, 600, lr_raw_y.shape[2]])

        lr = np.zeros([600, 600, 3, hr_raw_y.shape[2]], dtype=np.uint8)
        lr_cur = np.zeros([600, 600, 3], dtype=np.uint8)

        out = np.zeros([600, 600, lr_raw_y.shape[2]])

        # Only super-resolution the Y channel!!!!
        for i in range(hr_raw_y.shape[2]):
            hr_cur[:, :, 0] = hr_raw_y[6:-6, 6:-6, i]
            hr_cur[:, :, 1] = hr_raw_cr[6:-6, 6:-6, i]
            hr_cur[:, :, 2] = hr_raw_cb[6:-6, 6:-6, i]
            hr[:, :, :, i] = cv.cvtColor(hr_cur, cv.COLOR_YCR_CB2RGB)

            lr_y[:, :, i] = cv.resize(lr_raw_y[:, :, i], (612, 612))
            lr_cr_cur = cv.resize(lr_raw_cr[:, :, i], (612, 612))
            lr_cr[:, :, i] = lr_cr_cur[6:-6, 6:-6]
            lr_cb_cur = cv.resize(lr_raw_cb[:, :, i], (612, 612))
            lr_cb[:, :, i] = lr_cb_cur[6:-6, 6:-6]

            lr_y_cur = cv.resize(lr_raw_y[:, :, i], (612, 612), interpolation=cv.INTER_NEAREST)
            lr_cur[:, :, 0] = lr_y_cur[6:-6, 6:-6]
            lr_cr_cur = cv.resize(lr_raw_cr[:, :, i], (612, 612), interpolation=cv.INTER_NEAREST)
            lr_cur[:, :, 1] = lr_cr_cur[6:-6, 6:-6]
            lr_cb_cur = cv.resize(lr_raw_cb[:, :, i], (612, 612),  interpolation=cv.INTER_NEAREST)
            lr_cur[:, :, 2] = lr_cb_cur[6:-6, 6:-6]
            lr[:, :, :, i] = cv.cvtColor(lr_cur, cv.COLOR_YCR_CB2RGB)

        lr_y = lr_y.astype(np.float32) / 255.

        print('Load video success. Start predict the network.....')
        init = time()

        tf.reset_default_graph()
        input_images = tf.placeholder(dtype=tf.float32, shape=[None, 612, 612, 1])

        # Rebuild the network
        b1 = tf.Variable(tf.zeros([64]), name='b1')
        b2 = tf.Variable(tf.zeros([32]), name='b2')
        b3 = tf.Variable(tf.zeros([1]), name='b3')

        w1 = tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1')
        w2 = tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2')
        w3 = tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')

        conv1 = tf.nn.relu(
            tf.nn.conv2d(input=input_images, filter=w1, strides=[1, 1, 1, 1], padding='VALID', name='conv1') + b1)
        conv2 = tf.nn.relu(
            tf.nn.conv2d(input=conv1, filter=w2, strides=[1, 1, 1, 1], padding='VALID', name='conv2') + b2)
        conv3 = tf.nn.conv2d(input=conv2, filter=w3, strides=[1, 1, 1, 1], padding='VALID', name='conv3') + b3

        saver = tf.train.Saver({
            'b1': b1, 'b2': b2, 'b3': b3,
            'w1': w1, 'w2': w2, 'w3': w3,
        })

        sess = tf.Session()
        saver.restore(sess, './srcnn_model/model.ckpt')

        for i in range(hr_raw_y.shape[2]):
            print('Processing No.[%d] frame. Total number of frames: [%d]' % (i+1, out.shape[2]))
            in_cur = lr_y[:, :, i]
            in_cur.shape = [1, 612 ,612, 1]
            out_cur = sess.run(conv3, feed_dict={input_images: in_cur})
            out_cur = out_cur * 255.

            out_cur[out_cur > 255] = 255
            out_cur[out_cur < 0] = 0

            out_cur = out_cur.astype(np.uint8)
            out_cur.shape = [600, 600]
            out[:, :, i] = out_cur

        sess.close()
        time_consume = time() - init

        ycrcb_cur = np.zeros([600, 600, 3], dtype=np.uint8)
        cnn_result = np.zeros([600, 600, 3, hr_raw_y.shape[2]], dtype=np.uint8)

        for i in range(hr_raw_y.shape[2]):
            ycrcb_cur[:, :, 0] = out[:, :, i].astype(np.uint8)
            ycrcb_cur[:, :, 1] = lr_cr[:, :, i].astype(np.uint8)
            ycrcb_cur[:, :, 2] = lr_cb[:, :, i].astype(np.uint8)
            cnn_result[:, :, :, i] = cv.cvtColor(ycrcb_cur, cv.COLOR_YCR_CB2RGB)

        print('Test network is done!. Staring Writing results to a .mat file.....')
        # Here I svae color result, while I save 3-channel result separately in maptv.py
        import scipy.io as sio
        sio.savemat(mat_path, {
            'hr': hr,
            'lr': lr,
            'cnn_result': cnn_result,
        })

        print('Predict is done! Time Consume of predict phase: [%d s]' % time_consume)


if __name__ == '__main__':

    # This is a demo of srcnn.py

    # Where your trainning data store
    train_path = './dataset/t_HR/'
    # Where your test data store. Should be noticed that I do not make use of test data.
    test_path = './dataset/v_HR/'
    # Where your input video store
    input_video = './dataset/Person.avi'
    # Where your test result(.mat file) store
    mat_path = './result/Person-srcnn.mat'

    net = TFNetwork(train_path=train_path, test_path=test_path, train_test_img_size=33)
    # net.train(n_epoch=5, batch_size=1024, train_sample_number=500, test_sample_number=50)  # Train network
    net.predict(input_video=input_video, mat_path=mat_path)  # Predict network
