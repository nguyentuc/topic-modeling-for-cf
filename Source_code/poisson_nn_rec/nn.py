import const
import tensorflow as tf
import numpy as np

Wo = np.ndarray((const.N_EMBED_DIM, const.N_LATENT_DIM), dtype=np.float32)
Wo.fill(0.33)
bo = np.ndarray(const.N_LATENT_DIM, dtype=np.float32)
bo.fill(0.0)


class FLAGS:
    pass


def get_placeholder():
    input_placeholder = tf.placeholder(tf.float32, shape=(None, const.N_EMBED_DIM))
    output_placeholder = tf.placeholder(tf.float32, shape=(None, const.N_LATENT_DIM))
    return input_placeholder, output_placeholder


def fill_train_feed_dict(data_shuffler, input_placeholder, output_placeholder):
    try:
        datas = data_shuffler.next_minibatch(FLAGS.batch_size)
        # tra ra mot tuple gom 2 mang, moi mang co kich thuoc 100 x 100.

        # print 'TucNg datatype: ',type(datas)
        # print 'Length: ', len(datas)
        # print 'Data 1: ',datas[0].shape,' - ',datas[0]
        # print 'Data 2: ',datas[1].shape,'-', datas[1]


    except:
        print "Fatal"
        exit(-1)
    feed_dict = {
        input_placeholder: datas[0],
        output_placeholder: datas[1]
    }
    return feed_dict


def fill_infer_feed_dict(inputs, input_placeholder):
    feed_dict = {
        input_placeholder: inputs
    }
    return feed_dict


def init(s1=50, s2=100):
    print "IInit ", s1
    with tf.name_scope("ns"):
        print const.N_EMBED_DIM, const.N_HIDDEN, const.N_LATENT_DIM
        W = tf.get_variable("w1", initializer=tf.constant(0.3, shape=[const.N_EMBED_DIM, const.N_HIDDEN]))
        b = tf.get_variable("b1", initializer=tf.constant(0.0, shape=[con]))
        W2 = tf.get_variable("w2", initializer=tf.constant(0.3, shape=[50, const.N_LATENT_DIM]))
        b2 = tf.get_variable("b2", initializer=tf.constant(0., shape=[const.N_LATENT_DIM]))


def forward(inputs):
    with tf.name_scope("ns"):
        print "Params embeded dimension: ", const.N_EMBED_DIM
        print "Params hidden_1 dimension: ", const.N_HIDDEN
        print "Params hidden_2, latent dimension: ", const.N_LATENT_DIM

        W = tf.get_variable("w1", initializer=tf.constant(0.3, shape=[const.N_EMBED_DIM, const.N_HIDDEN]))
        b = tf.get_variable("b1", initializer=tf.constant(0.0, shape=[const.N_HIDDEN]))
        W2 = tf.get_variable("w2", initializer=tf.constant(0.3, shape=[const.N_HIDDEN, const.N_LATENT_DIM]))
        b2 = tf.get_variable("b2", initializer=tf.constant(0., shape=[const.N_LATENT_DIM]))

        # Batch-size * Embedding_DIM to Input_DIM * Bath_size * Embedding_DIM
        # print inputs.get_shape()
        # batch_size =  tf.shape(inputs)[0]

        # input2 = tf.transpose(inputs,[1,0,2])

        # print "BS",batch_size        #input2 = tf.reshape(input2,[const.N_INPUT_DIM,batch_size * const.N_EMBED_DIM])
        product = tf.matmul(inputs, W)
        print product.get_shape()

        # product = tf.reshape(product,[batch_size,const.N_EMBED_DIM])
        output = tf.add(product, b)
        output = tf.nn.relu(output)
        product = tf.matmul(output, W2)
        output = tf.add(product, b2)

        # print "p1"
        # product1 = tf.matmul(inputs,W)
        # print "p1 shape",product1.get_shape()
        # output1 = tf.add(product1,b)
        # output1 = tf.nn.relu(output1)
        # product2 = tf.matmul(output1,W2)
        # output2 = tf.add(product2,b2)
        return tf.nn.relu(output)


def forward_by_tucnv(inputs):
    with tf.name_scope("ns"):
        # print "Params embeded dimension: ", const.N_EMBED_DIM
        # print "Params hidden_1 dimension: ", const.N_HIDDEN
        # print "Params hidden_2, latent dimension: ", const.N_LATENT_DIM
        # exit()

        # W = tf.get_variable("w1", initializer=tf.constant(0.3, shape=[const.N_EMBED_DIM, const.N_HIDDEN]))
        # b = tf.get_variable("b1", initializer=tf.constant(0.0, shape=[const.N_HIDDEN]))
        # W2 = tf.get_variable("w2", initializer=tf.constant(0.3, shape=[const.N_HIDDEN, const.N_LATENT_DIM]))
        # b2 = tf.get_variable("b2", initializer=tf.constant(0., shape=[const.N_LATENT_DIM]))

        W = tf.get_variable("w1", initializer=tf.constant(0.3, shape=[const.N_EMBED_DIM, const.N_HIDDEN]))
        b = tf.get_variable("b1", initializer=tf.constant(0.0, shape=[const.N_HIDDEN]))
        W2 = tf.get_variable("w3", initializer=tf.constant(0.3, shape=[const.N_HIDDEN, const.N_LATENT_DIM]))
        b2 = tf.get_variable("b3", initializer=tf.constant(0., shape=[const.N_LATENT_DIM]))

        product = tf.matmul(inputs, W)
        print product.get_shape()

        # product = tf.reshape(product,[batch_size,const.N_EMBED_DIM])
        output = tf.add(product, b)
        output = tf.nn.relu(output)
        product = tf.matmul(output, W2)
        output = tf.add(product, b2)

        return tf.nn.relu(output)


def do_forword(sess, forwarder, inputs, input_placehodler):# input = mv_embedding, forwarder = output.
    with tf.Graph().as_default():
        feed_dict = fill_infer_feed_dict(inputs, input_placehodler)
        # return feed_dict la kieu du lieu dict.
        # feed_dict = {
        #     input_placeholder: inputs
        # }
        return sess.run(forwarder, feed_dict=feed_dict)


def loss(outputs, targets):
    return tf.reduce_mean(tf.pow(outputs - targets, 2))


def training(loss, learning_rate):
    # tf.summary.scalar('loss', loss)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    optimizer = tf.train.AdagradOptimizer(learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op
