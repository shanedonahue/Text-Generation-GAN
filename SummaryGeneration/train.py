import tensorflow as tf
import sys
import numpy as np
from data import Data
from model import SummaryModel
import argparse
import os

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity('ERROR')

parser = argparse.ArgumentParser(description = 'Train/Test summarization model', formatter_class = argparse.ArgumentDefaultsHelpFormatter)

# Import Setting
parser.add_argument("--doc_file", type = str, default = './data/doc.p', help = 'path to document file')
parser.add_argument("--vocab_file", type = str, default = './data/vocab.p', help = 'path to vocabulary file')
parser.add_argument("--emb_file", type = str, default = './data/emb.p', help = 'path to embedding file')
parser.add_argument("--src_time", type = int, default = 200, help = 'maximal # of time steps in source text')
parser.add_argument("--sum_time", type = int, default = 50, help = 'maximal # of time steps in summary')
parser.add_argument("--max_oov_bucket", type = int, default = 280, help = 'maximal # of out-of-vocabulary word in one summary')
parser.add_argument("--train_ratio", type = float, default = 0.8, help = 'ratio of training data')
parser.add_argument("--seed", type = int, default = 888, help = 'seed for spliting data')

# Saving Setting
parser.add_argument("--log", type = str, default = './log/', help = 'logging directory')
parser.add_argument("--save", type = str, default = './model/', help = 'model saving directory')
parser.add_argument("--checkpoint", type = str, help = 'path to checkpoint point')
parser.add_argument("--autosearch", type = bool, default = False, help = "[NOT AVAILABLE] Set 'True' if searching for latest checkpoint")
parser.add_argument("--save_interval", type = int, default = 1250, help = "Save interval for training")

# Hyperparameter Setting
parser.add_argument("--batch_size", type = int, default = 16, help = 'number of samples in one batch')
parser.add_argument("--gen_lr", type = float, default = 1e-3, help = 'learning rate for generator')
parser.add_argument("--dis_lr", type = float, default = 1e-3, help = 'learning rate for discriminator')
parser.add_argument("--cov_weight", type = float, default = 1e-3, help = 'learning rate for coverage')

if __name__ == '__main__':

    params = vars(parser.parse_args(sys.argv[1:]))
    print(params)

    model = SummaryModel(**params)
    data = Data(**params)

    test_generator = data.get_next_epoch_test()

    for i in range(1):
        print(f'Training Epoch {i}...')
        generator = data.get_next_epoch()
        # without coverage
        model.train_one_epoch(generator, data.n_train_batch, coverage_on=False)
        # with coverage
        # model.train_one_epoch(generator, data.n_train_batch, coverage_on = True, model_name = 'with_coverage')
        # Pre-train Disciminator
        #
        # model.train_one_epoch_pre_dis(train_data, data.n_train_batch, coverage_on=True)
        # model.train_one_epoch(generator, data.n_train_batch)
    train_data = data.get_next_epoch()
    test_data = data.get_next_epoch_test()

    for feed_dict in test_data:
        tokens, scores, attens = model.beam_search(feed_dict)
        src, ref, gen = data.id2word(feed_dict, tokens)
        gt_attens = model.sess.run(model.atten_dist, feed_dict=feed_dict)
        x = 0
        print("".join(src[x]), end='\n\n')
        print("".join(ref[x]), end='\n\n')
        print("".join(gen[x]), end='\n\n')
        print(scores[x])
        break
