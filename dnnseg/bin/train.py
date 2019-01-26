import sys
import os
import shutil
import time
import numpy as np
import pickle
import gzip
import argparse

sys.setrecursionlimit(2000)

from dnnseg.config import Config
from dnnseg.data import AcousticDataset, score_segmentation
from dnnseg.kwargs import UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS, UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS, UNSUPERVISED_WORD_CLASSIFIER_BAYES_INITIALIZATION_KWARGS
from dnnseg.plot import plot_acoustic_features

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains a DNN-Seg model from a config file.
    ''')
    argparser.add_argument('config', help='Path to configuration file.')
    argparser.add_argument('-p', '--preprocess', action='store_true', help='Preprocess data (even if saved data object exists in the model directory)')
    argparser.add_argument('-r', '--restart', action='store_true', help='Restart training even if model checkpoint exists (this will overwrite existing checkpoint)')
    args = argparser.parse_args()

    p = Config(args.config)

    if not p['use_gpu_if_available']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    t0 = time.time()
    data_name = 'data_f%s_d%s.obj' %(p['n_coef'], p['order'])
    if not args.preprocess and os.path.exists(p.train_data_dir + '/' + data_name):
        sys.stderr.write('Loading saved training data...\n')
        sys.stderr.flush()
        with open(p.train_data_dir + '/' + data_name, 'rb') as f:
            train_data = pickle.load(f)
    else:
        train_data = AcousticDataset(
            p.train_data_dir,
            n_coef=p['n_coef'],
            order=p['order'],
        )
        if p.save_preprocessed_data:
            sys.stderr.write('Saving preprocessed training data...\n')
            with open(p.train_data_dir + '/' + data_name, 'wb') as f:
                pickle.dump(train_data, f, protocol=2)

    # train_data_feed = train_data.get_streaming_data_feed(500,50,50,minibatch_size=100,filter='vad',randomize=True)
    # inputs, left_targets, right_targets, file_ix, time_ix, feats = next(train_data_feed)
    # timestamps = np.zeros(inputs.shape[:-1])
    # plot_acoustic_features(
    #     inputs,
    #     left_targets,
    #     right_targets,
    #     segmentation_probs=timestamps[..., None],
    #     segmentations=timestamps[..., None],
    #     hard_segmentations=True,
    #     prefix='resamp_test'
    # )
    #
    # # for i in range(10):
    # #     inputs, left_targets, right_targets = next(train_data_feed)
    # #     print(inputs)
    # #     print(left_targets)
    # #     print(right_targets)
    # #     print(inputs.shape)
    # #     print(left_targets.shape)
    # #     print(right_targets.shape)
    # #     input()
    # exit()

    sys.stderr.write('=' * 50 + '\n')
    sys.stderr.write('TRAINING DATA SUMMARY\n\n')
    sys.stderr.write(train_data.summary(indent=2))
    sys.stderr.write('=' * 50 + '\n\n')

    if p.train_data_dir != p.val_data_dir:
        if not args.preprocess and os.path.exists(p.val_data_dir + '/' + data_name):
            sys.stderr.write('Loading saved validation data...\n')
            sys.stderr.flush()
            with open(p.val_data_dir + '/' + data_name, 'rb') as f:
                val_data = pickle.load(f)
        else:
            val_data = AcousticDataset(
                p.val_data_dir,
                n_coef=p['n_coef'],
                order=p['order'],
            )
            if p.save_preprocessed_data:
                sys.stderr.write('Saving preprocessed dev data...\n')
                with open(p.val_data_dir + '/' + data_name, 'wb') as f:
                    pickle.dump(val_data, f, protocol=2)

            sys.stderr.write('=' * 50 + '\n')
            sys.stderr.write('CROSS-VALIDATION DATA SUMMARY\n\n')
            sys.stderr.write(val_data.summary(indent=2))
            sys.stderr.write('=' * 50 + '\n\n')

    else:
        val_data = None

    t1 = time.time()

    sys.stderr.write('Data loaded in %ds\n\n' %(t1-t0))
    sys.stderr.flush()

    if p['segtype'] == 'rnd':
        train_data.initialize_random_segmentation(7.4153)
        if val_data is not None:
            val_data.initialize_random_segmentation(7.4153)

    sys.stderr.write('Initializing encoder-decoder...\n\n')

    if args.restart and os.path.exists(p.outdir + '/tensorboard'):
        shutil.rmtree(p.outdir + '/tensorboard')

    kwargs = {}

    for kwarg in UNSUPERVISED_WORD_CLASSIFIER_INITIALIZATION_KWARGS:
        kwargs[kwarg.key] = p[kwarg.key]

    if p['network_type'] == 'mle':
        from dnnseg.model import AcousticEncoderDecoderMLE

        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_MLE_INITIALIZATION_KWARGS:
            kwargs[kwarg.key] = p[kwarg.key]

        dnnseg_model = AcousticEncoderDecoderMLE(
            p['k'],
            train_data,
            **kwargs
        )
    else:
        from dnnseg.model_bayes import AcousticEncoderDecoderBayes

        for kwarg in UNSUPERVISED_WORD_CLASSIFIER_BAYES_INITIALIZATION_KWARGS:
            kwargs[kwarg.key] = p[kwarg.key]

        dnnseg_model = AcousticEncoderDecoderBayes(
            p['k'],
            train_data,
            **kwargs
        )

    dnnseg_model.build(len(train_data.segments(segment_type=p['segtype'])), outdir=p.outdir, restore=not args.restart)

    sys.stderr.write('Fitting encoder-decoder...\n\n')

    dnnseg_model.fit(
        train_data,
        val_data=val_data,
        n_iter=p['n_iter'],
        ix2label=train_data.ix2label(p['segtype']),
    )
