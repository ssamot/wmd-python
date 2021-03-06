__author__ = 'ssamot'

import numpy as np, string
from emd import emd
import gensim


class WMDQA:
    def __init__(self, model="default", vecsize=300):
        if (model == "default"):
            self.model = gensim.models.Word2Vec.load_word2vec_format(
                '/home/ssamot/projects/wmd-python/GoogleNews-vectors-negative300.bin', binary=True)
        else:
            self.model = model

        self.vecsize = vecsize

    # read datasets line by line
    def _read_line_by_line(self, lines, C, vec_size):
        # get stop words (except for twitter!)
        SW = set()
        for line in open('/home/ssamot/projects/wmd-python/stop_words.txt'):
            line = line.strip()
            if line != '':
                SW.add(line)

        stop = list(SW)

        if len(C) == 0:
            C = np.array([], dtype=np.object)
        num_lines = len(lines)

        X = np.zeros((num_lines,), dtype=np.object)
        BOW_X = np.zeros((num_lines,), dtype=np.object)
        count = 0
        the_words = np.zeros((num_lines,), dtype=np.object)
        for line in lines:
            # print '%d out of %d' % (count+1, num_lines)
            line = str(line)
            line = line.strip()
            line = line.translate(string.maketrans("", ""), string.punctuation)

            # print line
            W = line.split()
            F = np.zeros((vec_size, len(W)))
            inner = 0
            word_order = np.zeros((len(W)), dtype=np.object)
            bow_x = np.zeros((len(W),))
            for word in W[0:len(W)]:
                try:
                    test = self.model[word]
                    if word in stop:
                        word_order[inner] = ''
                        continue
                    if word in word_order:
                        IXW = np.where(word_order == word)
                        bow_x[IXW] += 1
                        word_order[inner] = ''
                    else:
                        word_order[inner] = word
                        bow_x[inner] += 1
                        F[:, inner] = self.model[word]
                except KeyError, e:
                    # print 'Key error: "%s"' % str(e)
                    word_order[inner] = ''
                inner = inner + 1
            Fs = F.T[~np.all(F.T == 0, axis=1)]
            word_orders = word_order[word_order != '']
            bow_xs = bow_x[bow_x != 0]
            X[count] = Fs.T
            the_words[count] = word_orders
            BOW_X[count] = bow_xs
            count = count + 1;
        return (X, BOW_X, C, the_words)

    def _process_data(self, dataset, vec_size):
        (X, BOW_X, C, words) = self._read_line_by_line(dataset, [], vec_size)

        n = np.shape(X)
        n = n[0]
        for i in xrange(n):
            bow_i = BOW_X[i]
            bow_i = bow_i / np.sum(bow_i)
            bow_i = bow_i.tolist()
            BOW_X[i] = bow_i
            X_i = X[i].T
            X_i = X_i.tolist()
            X[i] = X_i
        return X, BOW_X

    def _distance(self, x1, x2):
        return np.sqrt(np.sum((np.array(x1) - np.array(x2)) ** 2))

    def _get_wmd(self, args):
        X, BOW_X, i, j = args
        return emd((X[i], BOW_X[i]), (X[j], BOW_X[j]), self._distance)
        # return Di

    def processqawmd(self, question, data):

        lines = [question] + data
        X, BOW_X = self._process_data(lines, self.vecsize)

        sims = []
        for i in range(0, len(lines)):
            if (lines[i] == "___"):
                sim = np.inf
            else:
                sim = self._get_wmd([X, BOW_X, 0, i])
            sims.append(sim)

        sims = np.array(sims)[1:]
        # sims[np.isnan(sims)] = np.inf
        # print sims
        return sims

    def getTopRelativeMemories(self, question, data, hops):
        argmins = []
        data = list(data[:])
        hops = min(hops, len(data))
        question = str(question)
        words_explored = []
        for i in range(hops):
            sims = self.processqawmd(question, data)
            # sims = processqawcd(question, data)
            argmin = sims.argmin()
            # print data[argmin]
            question = question + data[argmin]
            data[sims.argmin()] = "___"
            argmins.append(argmin)
        argmins = np.array(argmins)
        argmins.sort()
        return argmins


if __name__ == "__main__":
    data = """ John took the football there .
    John left the football .
    Mary went back to the bathroom .
    Sandra got the football there .
    Daniel journeyed to the kitchen .
    John journeyed to the bedroom .
    Sandra dropped the football .
    Mary moved to the kitchen .
    Daniel grabbed the apple there .
     Mary went to the bathroom .
     Mary moved to the bedroom .
     John went back to the bathroom .
     Daniel discarded the apple there .
     Mary travelled to the kitchen .
     Sandra picked up the football there .
     Sandra left the football there .
     Mary went to the hallway .
     Daniel got the apple there .
     Daniel travelled to the hallway .
     Mary took the football there .
     John travelled to the hallway .
     Mary put down the football."""
#hallway 20 21
    data = data.split("\n")

    print data

    question = "Where is the football ?"

    wmdqa = WMDQA()
    top_mems = wmdqa.getTopRelativeMemories(question, data, 5)
    print top_mems
    data = np.array(data)[top_mems]
    print data
