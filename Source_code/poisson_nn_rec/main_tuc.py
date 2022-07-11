import optparse
from ctr_tuc import Param, CTR_TUC
from data import Data
import const
import os
import nn

import sys


def run():
    optparser = optparse.OptionParser()
    optparser.add_option("-D", "--directory", default=".")
    optparser.add_option("-U", "--user")
    optparser.add_option("-I", "--item")
    optparser.add_option("-A", "--a", type="float", default=const.a)
    optparser.add_option("-B", "--b", type="float", default=const.b)
    optparser.add_option("-X", "--lambda_u", type="float", default=0.01)
    optparser.add_option("-Y", "--lambda_v", type="float", default=100)
    optparser.add_option("-M", "--max_iter", type="int", default=30)
    optparser.add_option("-C", "--num_factors", type="int", default=100)
    optparser.add_option("-T", "--mult")
    optparser.add_option("-Z", "--beta_init")
    optparser.add_option("-F", "--num_hidden", type="int", default=100)
    optparser.add_option("-E", "--theta_init")
    optparser.add_option("-R", "--learning_rate", type="float")
    optparser.add_option("-S", "--alpha_smooth", type="float", default=0.0)
    optparser.add_option("-O", "--random_seed", type="int")
    optparser.add_option("-N", "--nn", type="int", default=1)
    optparser.add_option(
        "-P",
        '--nn_learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    optparser.add_option(
        "-Q",
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )

    opts = optparser.parse_args()[0]
    # cac bo tham so khac nhau dung de chay mo hinh
    # doi voi mo hinh chay tren bo du lieu citeulike cac tham so khong doi chi thay doi num_hidden
    print 'Parameters: ', opts

    nn.FLAGS.batch_size = opts.batch_size
    nn.FLAGS.learning_rate = opts.nn_learning_rate # learning rate neural networks

    directory = opts.directory
    learning_rate = opts.learning_rate
    out_dir = "%s-%s-%s-%s" % (directory, opts.num_hidden, opts.num_factors, opts.learning_rate)

    saving_folder = "results/%s" % (out_dir)

    print 'Saving folder: ', saving_folder

    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    print "------------------------------"

    a = 1.0
    b = 1.0
    ctr_run = 0

    if len(opts.mult) > 1:
        ctr_run = 1
    param = Param()
    param.a = a
    param.b = b
    param.lambda_u = opts.lambda_u
    param.lambda_v = opts.lambda_v
    param.learning_rate = opts.learning_rate # gan lai learning_rate vao param
    param.alpha_smooth = opts.alpha_smooth
    param.random_seed = opts.random_seed
    param.max_iter = opts.max_iter
    param.num_factors = opts.num_factors
    param.directory = saving_folder
    param.ctr_run = ctr_run
    param.data_name = directory
    param.nn = opts.nn
    param.n_hidden = opts.num_hidden
    # Run

    num_factors = param.num_factors
    # nn.init(param.n_hidden)
    const.N_LATENT_DIM = num_factors
    const.N_HIDDEN = param.n_hidden

    print "PARAMS factor = laten dimenson, hidden: ", num_factors, const.N_LATENT_DIM, const.N_HIDDEN

    users = Data()
    print "Loading users..."
    users.read_data(opts.user) # doc tu users_train.dat bao moi dong tuong ung voi user_id tuong ung va gia tri rating tuong ung
    # ket thuc read duoc 2 mang tuong ung nhau
    # voi moi thanh phan cua mang m_vec_data la vector k chieu voi cac phan tu la movies_id
    # thanh phan tuong ung ben m_vec_len la k (do dai cua vector tuong ung tren m_vec_data.

    num_users = len(users.m_vec_len) # do m_vec_len: la mang cac so luong item tuong ung voi moi user.
    # num_users: so luong cac user, cac phan tu trong m_vec_len.

    items = Data()
    print "Loading items..."
    items.read_data(opts.item)
    num_items = len(items.m_vec_len)# tuong tu doi voi doi tuong items.

    # EDIT BY TUCNV FOR RUN WITH ALL NEURAL NETWORK
    if num_items > 10000 or num_users > 10000:# so luong user or item qua lon khong su dung neural network
        param.nn = 0
    # RUN ALL WITH NEURAL NETWORKS

    ctr = CTR_TUC()
    ctr.set_model_parameters(num_factors, num_users, num_items)

    # nn.FLAGS.max_steps = num_items / nn.FLAGS.batch_size * 1
    nn.FLAGS.max_steps = 100

    if len(opts.mult) > 1:
        ctr.read_init_information(opts.theta_init, None, opts.alpha_smooth) # theta -> dung cho movie features.
    # hoac theta co the dung trong neuralnet after train.

    # print "Param:::"
    # print param.learning_rate
    # exit()
    ctr.learn_pctr(users, items, None, param, saving_folder)  # hoc va luu ra file
    # ctr.learn_pctr_bagsofword_concat_word2vec(users, items, None, param, saving_folder)  # hoc va luu ra file


if __name__ == "__main__":
    run()
