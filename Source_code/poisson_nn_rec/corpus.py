import numpy as np
from io import open
class Document:
    def __init__(self,length=0):
        self.m_length = length
        self.m_words = np.ndarray(length,dtype=int)
        self.m_counts = np.ndarray(length,dtype=int)
        self.m_total = 0

class Corpus():
    def __init__(self):
        self.m_num_docs = 0
        self.m_size_vocab = 0
        self.m_num_total_words = 0
        self.m_docs = []

    def max_corpus_length(self):
        max_length = 0
        for d in self.m_docs:
            if max_length < d.m_length:
                max_length = d.m_length
        return max_length

    def read_data(self,path):
        nd = 0
        nw = 0
        f = open(path,"r")
        while True:
            line = f.readline()
            if line == "":
                break
            line = line.strip()
            parts = line.split(" ")
            length = int(parts[0])
            doc = Document(length)
            for i in xrange(length):
                pair = parts[i+1].split(":")
                word = int(pair[0])
                count = int(pair[1])
                doc.m_words[i] = word
                doc.m_counts[i] = count
                doc.m_total += count
                if word >= nw:
                    nw = word + 1

            self.m_num_total_words += doc.m_total
            self.m_docs.append(doc)
            nd += 1
        f.close()
        self.m_num_docs = nd
        self.m_size_vocab = nw
        print "Number of docs: %s"%self.m_num_docs
        print "Number of terms: %s"%self.m_size_vocab
        print "Number of total words: %s"%self.m_num_total_words