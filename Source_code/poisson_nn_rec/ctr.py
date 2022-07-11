import const
import numpy as np
import utils
import numpy.random as random
#import cmath as cmath
import math
import datetime
import sys

from data_shuffler import DataShuffer
import nn
import tensorflow as tf
import time


class Param():
    a = 1
    b = 0.01
    lambda_u = 0.01
    lambda_v = 100
    learning_rate = -1
    alpha_smooth = 0
    max_inter = 30


class CTR():
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
               # print 'Detal: ', detal[i]
                detal[i] = math.sqrt(abs(detal[i]))

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
        # sum_u = np.ndarray(self.m_num_factors,dtype=float)
        # a_minus_b = param.a - param.b
        n_drops = 0
        it = 0
        # likelihood_old = 0
        # likelihood = 0

        time_start = datetime.datetime.now()
        THRES_HOLD = 1.0 - param.learning_rate;

        print "Starting learning..."

        with tf.Graph().as_default():
            if param.nn == 1:
                # NN Elements
                mv_embedding = DataShuffer.load_inputs(data_name=param.data_name)

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

                # Save non-zero rating drops of all users
                allRatingsItemDrops = []

                for i in xrange(self.m_num_users):
                    item_ids = users.m_vec_data[i]
                    n = users.m_vec_len[i]
                    ratingItemDrops = []
                    # Append for each user
                    allRatingsItemDrops.append(ratingItemDrops)
                    if n > 0:
                        uu = self.m_U[i]
                        for ii in xrange(n):
                            r = random.uniform()
                            iv = item_ids[ii]

                            if r > THRES_HOLD:
                                # Store dropped ids
                                ratingItemDrops.append(iv)
                                n_drops += 1
                                continue

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

                # New update with drop on zero-ratings
                for i in xrange(self.m_num_users):
                    n = users.m_vec_len[i]
                    marker = np.zeros((num_items), dtype=int)
                    v = np.copy(x)

                    # Mark ratings...
                    for j in xrange(n):
                        iv = item_ids[ii]
                        marker[iv] = 1
                    # Remove  dropped zero ratings
                    for j in xrange(num_items):
                        if marker[j] == 0:
                            r = random.uniform()
                            if r > THRES_HOLD:
                                v -= self.m_V[j]
                    # Remove dropped non-zero ratings
                    for j in allRatingsItemDrops[i]:
                        v -= self.m_V[j]

                    tmp_U[i] /= v

                del allRatingsItemDrops

                # tmp_U /= x
                self.m_U, tmp_U = utils.swap(self.m_U, tmp_U)

                # Training neural network

                if param.nn == 1:

                    data_shuffler = DataShuffer(inputs=mv_embedding, outputs=self.m_V)

                    print "Begin nn training"
                    try:
                        for step in xrange(100):
                            start_time = time.time()

                            feed_dict = nn.fill_train_feed_dict(data_shuffler,
                                                                input_placeholder,
                                                                target_placeholder)

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
