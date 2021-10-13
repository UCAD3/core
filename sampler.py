import numpy as np

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, SEED):
    def sample():

        user = np.random.randint(0, usernum)
        user=list(user_train.keys())[user]
        while len(user_train[user]) <= 1:
            user = np.random.randint(0, usernum)
            user = list(user_train.keys())[user]

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)  #产生不在ts的 negtive item index,
            nxt = i
            idx -= 1
            if idx == -1: break
        #pos-- positive -- the prediction target
        return (user, seq, pos, neg)

    np.random.seed(SEED)
    one_batch = []
    for i in range(batch_size):
        one_batch.append(sample())

    return (zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.User=User
        self.usernum=usernum
        self.itemnum=itemnum
        self.batch_size=batch_size
        self.maxlen=maxlen

    def next_batch(self):
        return sample_function(self.User, self.usernum, self.itemnum,
                               self.batch_size, self.maxlen, np.random.randint(2e9))
