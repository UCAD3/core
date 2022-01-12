import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import copy
import random
from collections import defaultdict
import pandas as pd
from sklearn.metrics import roc_curve,roc_auc_score

def partition_data(datadict, pro=0.8): #used in autoencoder; return 的只有值 没有sess_key
    whole_data=[it for _,it in datadict.items()]
    whole_data=np.array(whole_data)
    print(whole_data[0][0])
    print("Whole dataset size: ", whole_data.shape)

    train_num=round(len(whole_data)*pro)
    train_data=whole_data[:train_num]
    test_data=whole_data[train_num:]

    print("validate session keys: ", list(datadict.keys())[train_num:])
    return train_data, test_data

def dict2list(datadict):
    whole_data = [it for _, it in datadict.items()]
    whole_data = np.array(whole_data)
    print("Test session keys: ", list(datadict.keys()))
    return whole_data

def display_rec(x_test, decoded_imgs):
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def txt2dict(txtfname, saveto, de_dupli=False):
    fname='data/{}'.format(txtfname)
    num_sessions=0
    ddict={}
    dset=set()
    with open(fname, 'r') as f:
        for line in f.readlines():
            line=list(map(int, line.strip().split()))
            if de_dupli:
                if tuple(line) in dset:
                    continue
            ddict[num_sessions]=line
            num_sessions+=1
            dset.add(tuple(line))

    np.save('data/'+saveto+'.npy', ddict)
    print("【Saving】 {} to data/{}".format(txtfname, saveto))
    print("data length: ", len(ddict))

def data_partition(npyfname, pro=0.8):  # used in transformer
    datadict = np.load('data/train/%s.npy' % npyfname, allow_pickle=True).item()
    sess_train = {}
    sess_test = {}
    sess_num = len(datadict)
    train_num = round(sess_num * pro)
    query_num = 0
    cnt = 0
    for sess, seq in datadict.items():
        nfeedback = len(seq)
        if nfeedback < 2: continue
        if len(set(seq)) == 1:
            # print(seq)
            continue
        query_num = max(query_num, max(seq))
        if cnt < train_num:
            sess_train[sess] = seq
        else:
            sess_test[sess] = seq
        cnt += 1
    print("\nNumber of training sessions: {}".format(len(sess_train)))
    print("Number of testing sessions: {}".format(len(sess_test)))
    return [sess_train, sess_test, train_num, query_num+1]

def add_noise(ddict, noise_pro):
    total_data=len(ddict)
    add_num=round(total_data*noise_pro)
    ano_sess=np.load('data/test/rdm_cmb.npy', allow_pickle=True).item()
    ano_sess=list(ano_sess.values())
    cc=0
    for k,v in ddict.items():
        if add_num==cc: break
        ddict[k]=ano_sess[cc%len(ano_sess)]
        cc+=1
    return ddict

def sessdict2Matrix(ddict, dim):
    # transform session into fix length
    mat=[]
    for k,v in ddict.items():
        vec=[0]*dim
        for q in v:
            vec[q-1]+=1
        mat.append(vec)
    return mat

def data_partition_ori(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, data, itemnum, args, sess, stage='train'):
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    keys = list(data.keys())
    for u in keys:
        if len(data[u]) < 1 or len(data[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        # idx-=1
        for i in reversed(data[u][:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(data[u])
        rated.add(0)
        target = data[u][-1]

        item_idx=[i for i in range(1, itemnum + 1)]
        
        predictions = -model.predict(sess, [u], [seq], item_idx)  # sort from low to high, so using negative.
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[target-1]  # 0 target-1

        valid_user += 1
        if rank < args.top_n:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        else:
            if stage == 'test':
                print("【Detect as anomaly】:session {}".format(u))

    return NDCG, HT, valid_user


def visual_att(model, data, args, sess):
    keys = list(data.keys())
    for u in keys:
        if len(data[u]) < 1 or len(data[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        print("\ninput seq: ", data[u][:-1])
        # idx-=1
        for i in reversed(data[u][:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        #print("\ninput seq: ", seq)
        target = data[u][-1]
        print("target: ", target)
        attentions = model.get_att(sess, [u], [seq])  # 6layers * 8-head * 100*100
        i,j=1,0
        for layer in attentions:
            att=np.mean(layer, axis=0)

            leth=len(data[u][:-1])
            att=att[-leth:, -leth:]
            print("attention is: ", att)
            name=data[u]
            if len(name)>101:
                name=data[u][-101:]
            plot_heat(np.array(att), name, u, i)
            i+=1

import seaborn as sns
def softmax(m):
    p=np.zeros(m.shape)
    for i in range(len(m)):
        p[i,:]=np.exp(m[i,:])/np.sum(np.exp(m[i,:]))
    return p

def plot_heat(matrix, name,u, i):
    matrix=softmax(matrix)
    print("matrix is :", matrix)
    ax = sns.heatmap(matrix, cmap="YlGnBu")
    #ax.set_xticklabels(name[:-1])
    #ax.set_yticklabels(name[1:])
    ax.set_title("Layer {}".format(i))
    plt.savefig('./attention/{}_layer_{}'.format(u, i))
    plt.show()

def plot_history(history):
    losses1 = [x['val_loss1'] for x in history]
    losses2 = [x['val_loss2'] for x in history]
    plt.plot(losses1, '-x', label="loss1")
    plt.plot(losses2, '-x', label="loss2")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.show()

def histogram(y_test, y_pred, fn):
    plt.figure(figsize=(12, 6))
    plt.hist(y_pred[y_test == 1], color='#EC7063', label=fn)
    plt.hist(y_pred[y_test == 0], color='#82E0AA', label='Normal')  #y_pred[y_test == 1]]
    plt.title("Results", size=20)
    plt.grid()
    plt.legend()
    plt.show()

def ROC(y_test, y_pred):
    fpr, tpr, tr = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    idx = np.argwhere(np.diff(np.sign(tpr - (1 - fpr)))).flatten()

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.plot(fpr, 1 - fpr, 'r:')
    plt.plot(fpr[idx], tpr[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    return tr[idx]


def confusion_matrix(target, predicted, perc=False):
    data = {'y_Actual': target,
            'y_Predicted': predicted
            }
    df = pd.DataFrame(data, columns=['y_Predicted', 'y_Actual'])
    confusion_matrix = pd.crosstab(df['y_Predicted'], df['y_Actual'], rownames=['Predicted'], colnames=['Actual'])
    mat=confusion_matrix.to_numpy()
    return mat

if __name__ == '__main__':
    #data_partition('query_seq')
    dataset_name=['hdfs', 'bgl', 'openstack']
    for d_name in dataset_name:
        txt2dict('train/{}_train.txt'.format(d_name), 'train/{}'.format(d_name))
        txt2dict('test/{}_test_normal.txt'.format(d_name), 'test/{}_normal'.format(d_name), de_dupli=True)
        txt2dict('test/{}_test_abnormal.txt'.format(d_name), 'test/{}_abnormal'.format(d_name))

    '''
    data=np.load('data/test/hdfs_normal.npy', allow_pickle=True).item()
    dset=set()
    for ln in data.values():
        dset.add(tuple(ln))
    print("data length: ", len(dset))
    '''
