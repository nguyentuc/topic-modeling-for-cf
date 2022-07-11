import numpy as np
class Data():
    def __init__(self):
        self.m_vec_data = []
        self.m_vec_drop = []
        self.m_vec_len = []
        self.m_vec_ndrop = []

    def read_data(self,path):
        length = 0
        n = 0
        id = 0
        total = 0
        f = open(path,"r")
        while True:
            line = f.readline()
            if line == "":
                break
            line = line.strip()
            parts = line.split(" ")
            if len(parts) < 1:
                continue
            length = int(parts[0])
            ids = np.ndarray(length,dtype=int)
            for i in xrange(length):
                ids[i] = int(parts[i+1])
            self.m_vec_data.append(ids)
            self.m_vec_len.append(length)
            total += length
        f.close()
        print "Read %s vector with %s entries"%(len(self.m_vec_len),total)
    def release_drop(self):
        self.m_vec_drop = None
        self.m_vec_ndrop = None
