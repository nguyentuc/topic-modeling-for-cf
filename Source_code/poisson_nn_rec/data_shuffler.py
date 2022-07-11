import numpy as np
import copy
import const


class DataShuffer():
    def __init__(self, inputs, outputs):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = 0

        self.inputs = inputs
        self.outputs = outputs
        self._num_examples = len(outputs)

    @staticmethod
    def load_inputs(data_root="../../data", data_name=""):
        print "Loading inputs...."
        embedding = np.loadtxt("%s/%s/embedding.dat" % (data_root, data_name))
        inputs = []
        f_onehot = open("%s/%s/one_hot_cate.dat" % (data_root, data_name), "r")
        while True:
            line = f_onehot.readline()
            if line == "":
                break
            one_hots = line.strip().split(" ")

            input = np.ndarray((const.N_EMBED_DIM), dtype=float)  # N_EMBED_DIM : la so chieu cua bo word2vec
            input.fill(0)
            for hot in one_hots:
                ind = int(hot)
                input += embedding[ind]
            input /= len(one_hots)  # mang 100 phan tu
            inputs.append(input)
        f_onehot.close()
        inputs = np.asarray(inputs)  # mang gom 3883 phan tu tuong ung voi cac movie

        return inputs

    # add by tucng for change input data into 3883 x 100 => 3883 x 18 (load_inputs version 2)
    @staticmethod
    def load_inputs_into_bags_of_word(data_root="../../data", data_name=""):
        print "Loading inputs...."
        inputs = []
        f_onehot = open("%s/%s/z_one_hot_content.dat" % (data_root, data_name), "r")
        while True:
            line = f_onehot.readline()
            if line == "":
                break
            one_hots = line.strip().split(" ")

            input = np.ndarray((const.N_EMBED_DIM), dtype=float)  # do tu vung (so chu de ) la 18
            input.fill(0)

            for hot in one_hots:
                ind = int(hot)
                input[ind] = 1
            inputs.append(input)
        f_onehot.close()
        inputs = np.asarray(inputs)  # mang gom 3883 phan tu tuong ung voi cac movie

        return inputs

    # add by tucnv for using word2vec base on new content
    # bieu dien moi phim dua tren title va content
    # su dung moi tu bieu dien dang vector 100 chieu
    # moi phim tinh trung binh dua tren cac tu cua no xuat hien trong bo tu dien vaf bd dang vector 100 chieu.
    @staticmethod
    def load_inputs_on_new_content(data_root="../../data", data_name=""):
        print "Loading inputs...."
        embedding = np.loadtxt("%s/%s/z_content_embedding.dat" % (data_root, data_name))
        inputs = []
        f_onehot = open("%s/%s/z_one_hot_content.dat" % (data_root, data_name), "r")
        while True:
            line = f_onehot.readline()
            if line == "":
                break
            one_hots = line.strip().split(" ")

            input = np.ndarray((const.N_EMBED_DIM), dtype=float)  # N_EMBED_DIM : la so chieu cua bo word2vec
            input.fill(0)
            for hot in one_hots:
                ind = int(hot)
                input += embedding[ind]
            input /= len(one_hots)  # mang 100 phan tu
            inputs.append(input)
        f_onehot.close()
        inputs = np.asarray(inputs)  # mang gom 3883 phan tu tuong ung voi cac movie

        return inputs

    # add by tucnv for using word2vec + bagofword
    # bieu dien moi phim dua tren title va content dang bagofword (l1)# co so chieu bang tu dien
    # bieu dien moi phim dua tren trung binh cua word2vec (l1)# co so chieu bang 100 (l2)
    # noi lai voi nhau duoc vector v+ 100 chieu

    @staticmethod
    def load_inputs_base_on_bagsofword_word2vec(data_root="../../data", data_name=""):
        print "Loading inputs...."
        embedding = np.loadtxt("%s/%s/z_content_embedding.dat" % (data_root, data_name))
        inputs = []
        f_onehot = open("%s/%s/z_one_hot_content.dat" % (data_root, data_name), "r")
        while True:
            line = f_onehot.readline()
            if line == "":
                break
            one_hots = line.strip().split(" ")

            input1 = np.ndarray(100, dtype=float)  # N_EMBED_DIM : la so chieu cua bo word2vec
            input1.fill(0)
            for hot in one_hots:
                ind = int(hot)
                input1 += embedding[ind]
            input1 /= len(one_hots)  # mang 100 phan tu

            # concat voi tu dien one hot
            input2 = np.ndarray(3399, dtype=float)  # do tu vung (so chu de ) la 3399
            input2.fill(0)

            for hot in one_hots:
                ind = int(hot)
                input2[ind] = 1
            input = np.concatenate((input1, input2), axis=None)
            inputs.append(input)  # duoc mot mang tuong ung 3883 movies moi movie la vector 3499 chieu
        f_onehot.close()
        inputs = np.asarray(inputs)  # mang gom 3883 phan tu tuong ung voi cac movie
        # print "TUCCC: ", inputs.shape # (3883, 3499)
        # exit()
        return inputs

    def next_minibatch(self, batch_size):
        start = self._index_in_epoch
        # print start,batch_size,self._num_examples,self.inputs.shape
        if self._epochs_completed == 0 and start == 0:
            # print "D First..."
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._inputs = self.inputs[perm0]
            self._outputs = self.outputs[perm0]

        if start + batch_size > self._num_examples:
            # print "D Shift...",start,self._num_examples,batch_size
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            inputs_rest_part = self.inputs[start:self._num_examples]
            outputs_rest_part = self.outputs[start:self._num_examples]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            inputs_new_part = self.inputs[start:end]
            outputs_new_part = self.outputs[start:end]
            v = np.concatenate((inputs_rest_part, inputs_new_part), axis=0)
            return (np.concatenate((inputs_rest_part, inputs_new_part), axis=0),
                    np.concatenate((outputs_rest_part, outputs_new_part), axis=0))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return (self._inputs[start:end], self._outputs[start:end])
