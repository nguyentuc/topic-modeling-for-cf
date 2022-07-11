import const
import numpy as np
import utils
import numpy.random as random
import math
import datetime
import sys

from data_shuffler import DataShuffer
import nn
import tensorflow as tf
import time

from keras.layers import Input, Dense
from keras.models import Model


class Param():
    a = 1
    b = 0.01
    lambda_u = 0.01
    lambda_v = 100
    learning_rate = -1
    alpha_smooth = 0
    max_inter = 30


class CTR_OLD():
    def __int__(self):
        self.m_beta = None
        self.m_theta = None
        self.m_U = None
        self.m_V = None
        self.m_num_factors = 0
        self.m_num_items = 0
        self.m_num_users = 0

    def read_init_information(self, theta_path, corpus, alpha_smooth):
        num_topics = self.m_num_factors
        vtx = 1.0 - const.EPS
        print "Loading theta..."

        # self.m_theta = np.ndarray((self.m_num_items,
        #                          self.m_num_factors),
        #                         dtype=float)
        self.m_theta = utils.load_matrix_np(theta_path)
        self.m_theta += alpha_smooth

        print "Normalizing..."
        self.m_theta = utils.norm_matrix(self.m_theta)

    def set_model_parameters(self, num_factors, num_users, num_items):
        self.m_num_factors = num_factors
        self.m_num_items = num_items
        self.m_num_users = num_users

    def init_model2(self, ctr_run=0):
        print "Initializing model..."
        # self.m_U = np.ndarray((self.m_num_users,self.m_num_factors),dtype=float)
        # self.m_U.fill(0.0)
        # self.m_V = np.ndarray((self.m_num_items,self.m_num_factors),dtype=float)
        # self.m_V.fill(0.0)

        if ctr_run:
            print "In Ctr run"
            # self.m_theta = np.random.uniform(size=(self.m_num_items,self.m_num_factors))
            # self.m_theta = utils.norm_matrix(self.m_theta)

            self.m_U = np.random.uniform(size=(self.m_num_users, self.m_num_factors))
            self.m_V = np.random.uniform(size=(self.m_num_items, self.m_num_factors))
            self.m_V = utils.norm_matrix(self.m_V)
        else:
            self.m_theta = np.ndarray((self.m_num_items, self.m_num_factors), dtype=float)
            self.m_theta.fill(0)
            self.m_U = np.random.uniform(size=(self.m_num_users, self.m_num_factors))
            self.m_V = np.random.uniform(size=(self.m_num_items, self.m_num_factors))

    def opt_v_dropx(self, v, theta, param, sumu, sphi):
        sumud = np.copy(sumu)
        d2 = np.ndarray(self.m_num_factors, dtype=float)
        d2.fill(0)
        sz = theta.shape[0]
        for i in xrange(sz):
            d2[i] = theta[i]

        d1 = np.copy(sphi)
        detal = np.ndarray(self.m_num_factors, dtype=float)
        detal.fill(0)

        if param.lambda_v > 0.00001:
            d2 *= -1.0 * param.lambda_v
            d2 += sumud

            d1 *= (4 * param.lambda_v)
            # if np.any(d1<0):
            #     print "Fatal error here"
            #     exit(-1)

            for i in xrange(self.m_num_factors):
                detal[i] = (d2[i] * d2[i])
            detal += d1
            for i in xrange(self.m_num_factors):
                detal[i] = math.sqrt(detal[i])
            d2 *= -1
            detal += d2
            detal *= 1.0 / (2 * param.lambda_v)
        else:
            for i in xrange(self.m_num_factors):
                detal[i] = d1[i] / sumud[i]

        utils.numpy_copy_vector(detal, v)

    def learn_pctr(self, users, items, corpus, param, directory):
        self.init_model2(param.ctr_run)
        tmp_U = np.ndarray((self.m_num_users, self.m_num_factors), dtype=float)
        tmp_V = np.ndarray((self.m_num_items, self.m_num_factors), dtype=float)

        se = np.ndarray(self.m_num_factors, dtype=float)
        se.fill(1e-40)
        num_items = len(items.m_vec_len)
        x = np.ndarray(self.m_num_factors, dtype=float)

        n_drops = 0
        it = 0

        time_start = datetime.datetime.now()
        THRES_HOLD = 1.0 - param.learning_rate;

        print "Starting learning..."
        with tf.Graph().as_default():
            if param.nn == 1:  # trainning using neural network for change dimension of item.
                mv_embedding = DataShuffer.load_inputs(data_name=param.data_name)
                # print 'mv_embedding shape: ',mv_embedding.shape # (3883, 100)

                sess = tf.Session()
                input_placeholder, target_placeholder = nn.get_placeholder()

                outputs = nn.forward(input_placeholder)
                print input_placeholder.shape, target_placeholder.shape, outputs.shape
                loss_f = nn.loss(outputs, target_placeholder)
                train_op = nn.training(loss_f, nn.FLAGS.learning_rate)
                initiliazer = tf.global_variables_initializer()

                sess.run(initiliazer)
            else:
                print "Skip neural network"
            # loading mv embedding

            while True:
                if it >= param.max_iter:
                    break
                it += 1
                # likelihood_old = likelihood
                tmp_U.fill(0.0)
                tmp_V.fill(0.0)
                print "\r\tCalculating...",
                sys.stdout.flush()
                for i in xrange(self.m_num_users):
                    item_ids = users.m_vec_data[i]
                    n = users.m_vec_len[i]
                    if n > 0:
                        uu = self.m_U[i]
                        for ii in xrange(n):
                            r = random.uniform()
                            if r > THRES_HOLD:
                                n_drops += 1
                                continue
                            iv = item_ids[ii]
                            if (iv >= num_items):
                                continue
                            v = self.m_V[iv]
                            # x = np.copy(v)
                            utils.numpy_copy_vector(v, x)
                            x *= uu
                            x = utils.vnormalize(x)

                            utils.numpy_add_row(tmp_U, x, i)
                            utils.numpy_add_row(tmp_V, x, iv)

                sum_u = np.sum(tmp_U, axis=0)
                for v in xrange(self.m_num_items):
                    if items.m_vec_len[v] > 0:
                        view_theta = self.m_theta[v]
                        vv = self.m_V[v]
                        sphi = tmp_V[v]
                        self.opt_v_dropx(vv, view_theta, param, sum_u, sphi)

                x = np.sum(self.m_V, axis=0)
                x += se
                se.fill(param.lambda_u)

                tmp_U += se
                x += se

                se.fill(1e-40)

                tmp_U /= x
                self.m_U, tmp_U = utils.swap(self.m_U, tmp_U)

                # Training neural network

                if param.nn == 1:
                    data_shuffler = DataShuffer(inputs=mv_embedding, outputs=self.m_V)  # object

                    print "Begin nn training"
                    try:
                        for step in xrange(100):
                            start_time = time.time()
                            # next batch
                            feed_dict = nn.fill_train_feed_dict(data_shuffler,
                                                                input_placeholder,
                                                                target_placeholder)
                            # print 'Feed dict: '
                            # print feed_dict
                            # exit()

                            _, loss_value = sess.run([train_op, loss_f],
                                                     feed_dict=feed_dict)

                            duration = time.time() - start_time
                            if step % 10 == 0:
                                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                    except:
                        print "Error in NN module"
                        exit(-1)

                    # Update theta:

                    thetax = nn.do_forword(sess, outputs, mv_embedding, input_placeholder)
                    print "SUM", np.sum(thetax)
                    self.m_theta = thetax

                time_current = datetime.datetime.now()
                elapsed = time_current - time_start
                print "\r Iter: %4s %10s" % (it, elapsed)

        print "\n---------------------------"
        print "NDROP - AVERAGE %s %s\n" % (n_drops, n_drops / param.max_iter)

        np.savetxt("%s/final-U.dat" % directory, self.m_U)
        np.savetxt("%s/final-V.dat" % directory, self.m_V)

    # add by tucng (for change input into bags of words)
    def learn_pctr_bagsofword(self, users, items, corpus, param, directory):
        self.init_model2(param.ctr_run)
        tmp_U = np.ndarray((self.m_num_users, self.m_num_factors), dtype=float)
        tmp_V = np.ndarray((self.m_num_items, self.m_num_factors), dtype=float)

        se = np.ndarray(self.m_num_factors, dtype=float)
        se.fill(1e-40)
        num_items = len(items.m_vec_len)
        x = np.ndarray(self.m_num_factors, dtype=float)

        n_drops = 0
        it = 0

        time_start = datetime.datetime.now()
        THRES_HOLD = 1.0 - param.learning_rate;

        print "Starting learning..."
        with tf.Graph().as_default():
            if param.nn == 1:  # trainning using neural network for change dimension of item.
                mv_embedding = DataShuffer.load_inputs_base_on_bagsofword_word2vec(data_name=param.data_name)

                # print 'embedding dim: ', mv_embedding.shape
                # exit()

                sess = tf.Session()
                input_placeholder, target_placeholder = nn.get_placeholder()

                outputs = nn.forward(input_placeholder)  # tao model
                print input_placeholder.shape, target_placeholder.shape, outputs.shape
                loss_f = nn.loss(outputs, target_placeholder)  # define loss
                train_op = nn.training(loss_f, nn.FLAGS.learning_rate)  # define optimize
                initiliazer = tf.global_variables_initializer()  # Initializing the variables
                sess.run(initiliazer)
            else:
                print "Skip neural network"
            # loading mv embedding

            while True:  # bat dau hoc voi max_iter lan cap nhat.
                if it >= param.max_iter:
                    break
                it += 1
                # likelihood_old = likelihood
                tmp_U.fill(0.0)
                tmp_V.fill(0.0)
                print "\r\tCalculating...",
                sys.stdout.flush()
                for i in xrange(self.m_num_users):
                    item_ids = users.m_vec_data[i]
                    n = users.m_vec_len[i]
                    if n > 0:
                        uu = self.m_U[i]
                        for ii in xrange(n):
                            r = random.uniform()
                            if r > THRES_HOLD:
                                n_drops += 1
                                continue
                            iv = item_ids[ii]
                            if (iv >= num_items):
                                continue
                            v = self.m_V[iv]
                            # x = np.copy(v)
                            utils.numpy_copy_vector(v, x)
                            x *= uu
                            x = utils.vnormalize(x)

                            utils.numpy_add_row(tmp_U, x, i)
                            utils.numpy_add_row(tmp_V, x, iv)

                sum_u = np.sum(tmp_U, axis=0)
                for v in xrange(self.m_num_items):
                    if items.m_vec_len[v] > 0:
                        view_theta = self.m_theta[v]
                        vv = self.m_V[v]
                        sphi = tmp_V[v]
                        self.opt_v_dropx(vv, view_theta, param, sum_u, sphi)

                x = np.sum(self.m_V, axis=0)
                x += se
                se.fill(param.lambda_u)

                tmp_U += se
                x += se

                se.fill(1e-40)

                tmp_U /= x
                self.m_U, tmp_U = utils.swap(self.m_U, tmp_U)  # return tmp_U, m_U.

                # Training neural network

                if param.nn == 1:
                    data_shuffler = DataShuffer(inputs=mv_embedding, outputs=self.m_V)  # object

                    print "Begin nn training"  # trainning mang neural da duoc dinh nghia phia tren

                    try:
                        for step in xrange(100):  # Loop over all batches
                            start_time = time.time()
                            # next minibatch
                            feed_dict = nn.fill_train_feed_dict(data_shuffler,
                                                                input_placeholder,
                                                                target_placeholder)

                            #  kieu dictionary gom 2 phan thu: input_placeholder, output_placeholder moi phan tu la ma tran 100x 100
                            # print 'Feed dict: '
                            # print type(feed_dict)
                            # print feed_dict

                            # run optimize backprop train_op and loss_f (cost) (to get loss value)
                            _, loss_value = sess.run([train_op, loss_f],
                                                     feed_dict=feed_dict)  # float number.

                            duration = time.time() - start_time
                            if step % 10 == 0:
                                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                    except:
                        print "Error in NN module"
                        exit(-1)

                    # Update theta:

                    thetax = nn.do_forword(sess, outputs, mv_embedding, input_placeholder)
                    # print "thetax: ", thetax.shape -> 3883 x 100
                    # exit()

                    print "SUM", np.sum(thetax)
                    self.m_theta = thetax

                time_current = datetime.datetime.now()
                elapsed = time_current - time_start
                print "\r Iter: %4s %10s" % (it, elapsed)

        print "\n---------------------------"
        print "NDROP - AVERAGE %s %s\n" % (n_drops, n_drops / param.max_iter)

        np.savetxt("%s/final-U.dat" % directory, self.m_U)
        np.savetxt("%s/final-V.dat" % directory, self.m_V)

    # add by tucng (with input = bagofwords + word2vec => autoencoder)
    def learn_pctr_autoencoder(self, users, items, corpus, param, directory):
        self.init_model2(param.ctr_run)
        tmp_U = np.ndarray((self.m_num_users, self.m_num_factors), dtype=float)
        tmp_V = np.ndarray((self.m_num_items, self.m_num_factors), dtype=float)

        se = np.ndarray(self.m_num_factors, dtype=float)
        se.fill(1e-40)
        num_items = len(items.m_vec_len)
        x = np.ndarray(self.m_num_factors, dtype=float)

        n_drops = 0
        it = 0

        time_start = datetime.datetime.now()
        THRES_HOLD = 1.0 - param.learning_rate;

        print "Starting learning..."
        with tf.Graph().as_default():
            if param.nn == 1:  # trainning using neural network for change dimension of item.
                mv_embedding = DataShuffer.load_inputs_base_on_bagsofword_word2vec(data_name=param.data_name)

                print 'Embedding dim: ', mv_embedding.shape  # 3883 x 3499
                print 'Beginning trainning Autoencoder'
                lenth_data = len(mv_embedding)
                train_autoencode = mv_embedding[:- int(lenth_data / 10)]
                valid_autoencode = mv_embedding[- int(lenth_data / 10):]

                # tien hanh dua vao auto encoder: de dua ve 3883 x 100
                # tach ra lam tap train + validation

                encoding_dim = 100
                input_dim = Input(shape=(3499,))
                # Encoder Layers
                encoded1 = Dense(2000, activation='relu')(input_dim)
                encoded2 = Dense(500, activation='relu')(encoded1)
                encoded3 = Dense(100, activation='relu')(encoded2)

                decoded1 = Dense(500, activation='relu')(encoded3)
                decoded2 = Dense(2000, activation='relu')(decoded1)
                decoded3 = Dense(3499, activation='sigmoid')(decoded2)

                # Combine Encoder and Deocder layers
                autoencoder = Model(inputs=input_dim, outputs=decoded3)
                # Compile the Model
                autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
                # trainning auto encoder
                autoencoder.fit(train_autoencode, train_autoencode, epochs=10, batch_size=32, shuffle=False,
                                validation_data=(valid_autoencode, valid_autoencode))

                # Use Encoder level to reduce dimension of train and test data
                encoder = Model(inputs=input_dim, outputs=encoded3)
                encoded_data = encoder.predict(mv_embedding)

                print "Encodeed data: ", encoded_data.shape


                # dua vao feed_forward
                sess = tf.Session()
                input_placeholder, target_placeholder = nn.get_placeholder()

                outputs = nn.forward(input_placeholder)  # tao model
                print input_placeholder.shape, target_placeholder.shape, outputs.shape
                loss_f = nn.loss(outputs, target_placeholder)  # define loss
                train_op = nn.training(loss_f, nn.FLAGS.learning_rate)  # define optimize
                initiliazer = tf.global_variables_initializer()  # Initializing the variables
                sess.run(initiliazer)
            else:
                print "Skip neural network"
            # loading mv embedding

            while True:  # bat dau hoc voi max_iter lan cap nhat.
                if it >= param.max_iter:
                    break
                it += 1
                # likelihood_old = likelihood
                tmp_U.fill(0.0)
                tmp_V.fill(0.0)
                print "\r\tCalculating...",
                sys.stdout.flush()
                for i in xrange(self.m_num_users):
                    item_ids = users.m_vec_data[i]
                    n = users.m_vec_len[i]
                    if n > 0:
                        uu = self.m_U[i]
                        for ii in xrange(n):
                            r = random.uniform()
                            if r > THRES_HOLD:
                                n_drops += 1
                                continue
                            iv = item_ids[ii]
                            if (iv >= num_items):
                                continue
                            v = self.m_V[iv]
                            # x = np.copy(v)
                            utils.numpy_copy_vector(v, x)
                            x *= uu
                            x = utils.vnormalize(x)

                            utils.numpy_add_row(tmp_U, x, i)
                            utils.numpy_add_row(tmp_V, x, iv)

                sum_u = np.sum(tmp_U, axis=0)
                for v in xrange(self.m_num_items):
                    if items.m_vec_len[v] > 0:
                        view_theta = self.m_theta[v]
                        vv = self.m_V[v]
                        sphi = tmp_V[v]
                        self.opt_v_dropx(vv, view_theta, param, sum_u, sphi)

                x = np.sum(self.m_V, axis=0)
                x += se
                se.fill(param.lambda_u)

                tmp_U += se
                x += se

                se.fill(1e-40)

                tmp_U /= x
                self.m_U, tmp_U = utils.swap(self.m_U, tmp_U)  # return tmp_U, m_U.

                # Training neural network

                if param.nn == 1:
                    data_shuffler = DataShuffer(inputs=encoded_data, outputs=self.m_V)  # object

                    print "Begin nn training"  # trainning mang neural da duoc dinh nghia phia tren

                    try:
                        for step in xrange(100):  # Loop over all batches
                            start_time = time.time()
                            # next minibatch
                            feed_dict = nn.fill_train_feed_dict(data_shuffler,
                                                                input_placeholder,
                                                                target_placeholder)

                            #  kieu dictionary gom 2 phan thu: input_placeholder, output_placeholder moi phan tu la ma tran 100x 100
                            # print 'Feed dict: '
                            # print type(feed_dict)
                            # print feed_dict

                            # run optimize backprop train_op and loss_f (cost) (to get loss value)
                            _, loss_value = sess.run([train_op, loss_f],
                                                     feed_dict=feed_dict)  # float number.

                            duration = time.time() - start_time
                            if step % 10 == 0:
                                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                    except:
                        print "Error in NN module"
                        exit(-1)

                    # Update theta:

                    thetax = nn.do_forword(sess, outputs, encoded_data, input_placeholder)
                    # print "thetax: ", thetax.shape -> 3883 x 100
                    # exit()

                    print "SUM", np.sum(thetax)
                    self.m_theta = thetax

                time_current = datetime.datetime.now()
                elapsed = time_current - time_start
                print "\r Iter: %4s %10s" % (it, elapsed)

        print "\n---------------------------"
        print "NDROP - AVERAGE %s %s\n" % (n_drops, n_drops / param.max_iter)

        np.savetxt("%s/final-U.dat" % directory, self.m_U)
        np.savetxt("%s/final-V.dat" % directory, self.m_V)
