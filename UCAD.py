import os
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import UCAD
from tqdm import tqdm
from util import *
from multiprocessing import freeze_support

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def generate(npyfname, window_size, pro=0.8):
    datadict = np.load('data/train/%s.npy' % npyfname, allow_pickle=True).item()
    sess_whole=[]
    data_list=list(datadict.values())
    cul_i=0
    for line in data_list:
        for i in range(len(line) - window_size-1):
            sess_whole.append(tuple(line[i:i + window_size+1]))

    # remove duplicates
    sess_whole=list(set(sess_whole))
    sess_whole=[list(v) for v in sess_whole]

    ndatadict={k:v for k,v in enumerate(sess_whole)}

    sess_train = {}
    sess_test = {}
    sess_num = len(ndatadict)
    print("Sliding windows: ", sess_num)

    train_num = round(sess_num * pro)
    query_num = 0
    cnt = 0
    for sess, seq in ndatadict.items():
        nfeedback = len(seq)
        if nfeedback < 2: continue

        query_num = max(query_num, max(seq))
        if cnt < train_num:
            sess_train[sess] = seq
        else:
            sess_test[sess] = seq
        cnt += 1
    print("\nNumber of training sessions: {}".format(len(sess_train)))
    print("Number of testing sessions: {}".format(len(sess_test)))
    return [sess_train, sess_test, train_num, query_num + 1]


def evaluate_each(model, data, itemnum, args, sess):
    pred_abn=0.0
    valid_user = len(data)

    keys = list(data.keys())
    for u in keys:
        if len(data[u]) < 1 or len(data[u]) < 1: continue
        pad_data=[0]*(args.maxlen+1-len(data[u]))+data[u]

        for i in range(len(pad_data)-args.maxlen):
            seq = pad_data[i:i+args.maxlen]
            target=pad_data[i+args.maxlen]

            item_idx = [i for i in range(1, itemnum + 1)]

            predictions = -model.predict(sess, [u], [seq], item_idx)
            predictions = predictions[0]
            rank = predictions.argsort().argsort()[target-1]  # 0 target-1

            if rank > args.top_n:
                pred_abn += 1
                break

    HT=valid_user-pred_abn
    return HT, valid_user

def eval_data(model, fname, querynum, args, sess):
    seq1=np.load('data/test/{}.npy'.format(fname), allow_pickle=True).item()
    print('\n【Evaluating on {} dataset, size is {}】'.format(fname, len(seq1)))
    s_valid = evaluate_each(model, seq1, querynum, args, sess)
    print('HR@%d: %d / %d = %.4f)' % (
        args.top_n, s_valid[0], s_valid[1], s_valid[0] / s_valid[1]))

    return s_valid

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='bgl')
parser.add_argument('--train_dir', default='train_log')
parser.add_argument('--model_dir', default='model/bgl/')
parser.add_argument('--with_pos', default=False, type=bool, help="model with or without position embedding.")
parser.add_argument('--loss_mode', default='triple_add_logits')
parser.add_argument('--add_noise', default=0, type=float, help="noise percentage, e.g., 0.2")

parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--maxlen', default=10, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=6, type=int)
parser.add_argument('--num_epochs', default=101, type=int)
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--warmup_steps', default=50.0, type=float)
parser.add_argument('--margin', default=0.5, type=int)  # top_n=150 on bgl
parser.add_argument('--top_n', default=150, type=int, help="top n candidate for evaluating.")
parser.add_argument('--pro', default=0.8, type=float, help="train and testing data split.")
parser.add_argument('--isTrain', default=False, type=bool)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()


dataset = generate(args.dataset, args.maxlen, pro=args.pro)
[session_train, session_valid, sessionnum, querynum] = dataset
querynum=querynum+1
if args.add_noise!=0:
    session_train=add_noise(session_train, args.add_noise)

num_batch = len(session_train) // args.batch_size
sampler = WarpSampler(session_train, sessionnum, querynum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=1)
cc = 0.0
for u in session_train:
    cc += len(session_train[u])
print( 'average sequence length: %.2f' % (cc / len(session_train)))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

model = UCAD(sessionnum, querynum, args)
writer = tf.summary.FileWriter('logs/', sess.graph)
sess.run(tf.initialize_all_variables())

saver=tf.train.Saver()

T = 0.0
t0 = time.time()

try:
    if args.isTrain:
        f.write("Training:"+'\n')

        for epoch in range(1, args.num_epochs + 1):
            print('epoch: ',epoch)
            for step in tqdm(range(num_batch)):
                u, seq, pos, neg = sampler.next_batch()
                auc, loss, _, _summ= sess.run([model.auc, model.loss, model.train_op, model.merged],
                                        {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                         model.is_training: True})
                writer.add_summary(_summ)

            if epoch % 5 == 0:
                print ('Evaluating',)
                t_test = evaluate_each(model, session_train, querynum, args, sess)
                print ('\nepoch:%d, HR@%d: %d / %d = %.4f)' % (
                epoch, args.top_n, t_test[1], t_test[2], t_test[1]/t_test[2])) # valid (NDCG@10: %.4f, HR@10: %.4f), t_valid[0], t_valid[1],

                f.write(str(t_test) + '\n')
                f.flush()

        saver.save(sess, args.model_dir+'model.ckpt', global_step=args.num_epochs)
        print("Model saved in path: %s \n" % (args.model_dir+'model.ckpt'))
    ## only testing
    else:
        ckpt=tf.train.get_checkpoint_state(args.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore from model {}.".format(args.model_dir))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No ckpt.....")
except:
    f.close()
    exit(1)


print('\n【Evaluating on validation dataset, size is {}】'.format(len(session_valid)))
s_valid=evaluate_each(model, session_valid, querynum, args, sess)
print("args.with_pos : {}, args.loss_mode: {}.".format(args.with_pos, args.loss_mode))
print('Validation on normal session, HR@%d: %d / %d = %.4f)' % (
        args.top_n, s_valid[0], s_valid[1], s_valid[0] / s_valid[1]))

FP=0
TN=0

file_nor=['{}_normal'.format(args.dataset)]
for fn in file_nor:
    s_valid=eval_data(model, fn, querynum, args, sess)
    FP+=s_valid[1]-s_valid[0]
    TN+=s_valid[0]

TP, FN=0,0
file_abn=['{}_abnormal'.format(args.dataset)]
for fn in file_abn:
    s_valid=eval_data(model, fn, querynum, args, sess)
    TP+= s_valid[1] - s_valid[0]
    FN+= s_valid[0]

P = 100 * TP / (TP + FP)
R = 100 * TP / (TP + FN)
F1 = 2 * P * R / (P + R)

print("FP:", FP)
print("FN:", FN)
print("TP:",TP)
print("TN:",TN)
print('Precision:{}, Recall:{}, F1:{}'.format(P, R, F1))
print('Finished Predicting')

f.close()
t1 = time.time()
print("Done")
print(t1-t0)

