"""
This file contains a implement of Super-resolution by MAP&Total variation.
"""

import motion
import numpy as np
import tensorflow as tf
import cv2 as cv
import time


class MAPTV:

    def __init__(self, iter_times, lam, alpha, factor):
        self.iter_times = iter_times
        self.lam = lam
        self.alpha = alpha
        self.factor = factor

    def solver(self, ys, ys_true, key_index=None):

        ys_height, ys_width, frames = ys.shape
        z_height = int(ys_height * self.factor)
        z_width = int(ys_width * self.factor)

        if key_index is None:
            key_index = int(frames/2)

        # Motion Estimation
        src = []
        sta = []
        src_cur = cv.resize(ys[:, :, key_index], (z_height, z_width))
        src_cur = src_cur.astype(np.uint8)
        m_matrix = []

        for i in range(frames):
            sta_cur = cv.resize(ys[:, :, i], (z_height, z_width))
            sta_cur = sta_cur.astype(np.uint8)
            sta.append(sta_cur)
            src.append(src_cur)

        # Calculate motion matrix
        motione_time_init = time.time()
        m_matrix_all = motion.motion_compensation(src=src, sta=sta)
        motione_time = time.time() - motione_time_init

        tf.reset_default_graph()

        with tf.Session() as sess:

            # Construct matrix M
            for i in range(frames):
                m_matrix.append(tf.constant(m_matrix_all[i], dtype=tf.float32))

            z = tf.Variable(initial_value=cv.resize(ys[:, :, key_index], (z_height, z_width),
                                                    interpolation=cv.INTER_NEAREST), dtype=tf.float32)

            loss_f = 0

            # Construct matrix B
            b_filter = np.ndarray((4, 4))
            b_filter[:] = 1 / 16
            b_filter.shape = [4, 4, 1, 1]

            for i in range(frames):
                m_z = tf.matmul(m_matrix[i], z)
                m_z = tf.reshape(m_z, shape=[1, z_height, z_width, 1])
                b_m_z = tf.nn.conv2d(m_z, filter=b_filter, strides=[1, 1, 1, 1], padding='SAME')

                # Construct matrix D
                d_b_m_z = tf.nn.avg_pool(b_m_z, ksize=[1, self.factor, self.factor, 1],
                                         strides=[1, self.factor, self.factor, 1], padding='VALID')

                d_b_m_z = tf.reshape(d_b_m_z, shape=[ys_height, ys_width])
                loss_f += tf.norm(d_b_m_z - tf.constant(ys[:, :, i], dtype=tf.float32), 2) * 0.5

            # Calculate Total variation
            z_tv = tf.image.total_variation(tf.reshape(z, shape=[z_height, z_width, 1]))
            loss_r = self.lam * tf.norm(z_tv, 2)
            loss = loss_f + loss_r
            train = tf.train.AdamOptimizer(self.alpha).minimize(loss)

            z_output = tf.cast(z, tf.uint8)
            sess.run(tf.global_variables_initializer())

            map_time_init = time.time()

            for i in range(self.iter_times):

                if i % 10 == 0:
                    print('MAP : No.[%d] step. Total number of step: [%d]. '
                          'Current value of loss function: : [%.5e] ' %
                          (i, self.iter_times, sess.run(loss)))

                sess.run(train)

            map_time = time.time() - map_time_init

            time_consume = {
                'map_time': map_time,   # Time consume of MAP
                'motione_time': motione_time  # Time consue of motion estimation
            }

            z_output = sess.run(z_output)

        return z_output, time_consume


if __name__ == '__main__':

    input_video = './dataset/Person.avi'
    mat_path = './result/Person-map.mat'

    # Resolution of high-resolution video(hr_shape*hr_shape)
    hr_shape = 600
    # Resolution of low-resolution video(hr_shape*hr_shape)
    lr_shape = 150

    from inputvideo import VideoSRContinuous

    maptv_ = MAPTV(100, 1200, 0.3, 4)

    print('Start loading video.....')
    # Load Video
    hr_raw_r = VideoSRContinuous(input_video, 0, (hr_shape, hr_shape), batchframe_num=5)
    hr_raw_g = VideoSRContinuous(input_video, 1, (hr_shape, hr_shape), batchframe_num=5)
    hr_raw_b = VideoSRContinuous(input_video, 2, (hr_shape, hr_shape), batchframe_num=5)
    lr_raw_r = VideoSRContinuous(input_video, 0, (lr_shape, lr_shape), batchframe_num=5)
    lr_raw_g = VideoSRContinuous(input_video, 1, (lr_shape, lr_shape), batchframe_num=5)
    lr_raw_b = VideoSRContinuous(input_video, 2, (lr_shape, lr_shape), batchframe_num=5)

    # Store final reuslt of video
    out_map_r = np.zeros(shape=[hr_shape, hr_shape, lr_raw_r.batches.__len__()])
    out_map_g = np.zeros(shape=[hr_shape, hr_shape, lr_raw_g.batches.__len__()])
    out_map_b = np.zeros(shape=[hr_shape, hr_shape, lr_raw_b.batches.__len__()])

    # Store final reuslt of video
    out_map_r_time_consume = []
    out_map_g_time_consume = []
    out_map_b_time_consume = []

    print('Start MAP&Total variation.....')

    for i in range(lr_raw_r.batches.__len__()):
        out_map_r[:, :, i], traindata = maptv_.solver(ys=lr_raw_r.batches[i].data, ys_true=hr_raw_r.batches[i].data)
        out_map_r_time_consume.append(traindata)
        out_map_g[:, :, i], traindata = maptv_.solver(ys=lr_raw_g.batches[i].data, ys_true=hr_raw_g.batches[i].data)
        out_map_g_time_consume.append(traindata)
        out_map_b[:, :, i], traindata = maptv_.solver(ys=lr_raw_b.batches[i].data, ys_true=hr_raw_b.batches[i].data)
        out_map_b_time_consume.append(traindata)

        print('Frame : No.[%d] frame. Total number of frames: [%d]' % (i, lr_raw_r.batches.__len__()))

    # Here I svae 3-channel result separately, while I save data combines 3-channel in maptv.py
    import scipy.io as sio
    sio.savemat(mat_path, {
        'hr_raw_r': hr_raw_r.totalframes(),
        'hr_raw_g': hr_raw_g.totalframes(),
        'hr_raw_b': hr_raw_b.totalframes(),
        'lr_raw_r': lr_raw_r.totalframes(),
        'lr_raw_g': lr_raw_g.totalframes(),
        'lr_raw_b': lr_raw_b.totalframes(),
        'out_map_r': out_map_r,
        'out_map_g': out_map_g,
        'out_map_b': out_map_b,
        'out_map_r_traindata': out_map_r_time_consume,
        'out_map_g_traindata': out_map_g_time_consume,
        'out_map_b_traindata': out_map_b_time_consume,
    })
