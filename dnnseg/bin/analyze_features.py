import sys
import os
import math
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree  import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pydot
import argparse

from dnnseg.data import get_random_permutation

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains a DNN-Seg model from a config file.
    ''')
    argparser.add_argument('data', help='Path to data containing predicted and gold classification labels.')
    argparser.add_argument('gold_cols', nargs='+', help='Names of column in data set to use as regression target.')
    argparser.add_argument('-d', '--direction', type=str, default='pred2gold', help='Direction of classification. One of ["gold2pred", "pred2gold"].')
    argparser.add_argument('-M', '--max_depth', type=float, default=None, help='Maximum permissible tree depth.')
    argparser.add_argument('-m', '--min_impurity_decrease', type=float, default=0., help='Minimum impurity decrease necessary to make a split.')
    argparser.add_argument('-n', '--n_estimators', type=int, default=100, help='Number of estimators (trees) in random forest.')
    argparser.add_argument('-f', '--n_folds', type=int, default=5, help='Number of folds in cross-validation (>= 2).')
    argparser.add_argument('-v', '--verbose', action='store_true', help='Report progress to standard error.')
    argparser.add_argument('-o', '--outdir', default=None, help='Output directory.')
    argparser.add_argument('-t', '--type', type=str, default=None, help='Type of classification. If left out, all possible features will be classified. One of ["quartetts", "biphones", "features", "phonloc", "featloc"]')
    args = argparser.parse_args()

    is_embedding_dimension = re.compile('d([0-9]+)')

    if not args.outdir:
        directory = os.path.dirname(args.data)
        if args.type:
            path = directory + '/' + args.type + '/'
        else:
            path = directory
    elif not os.path.exists(args.outdir):
        path = args.outdir
        
    if not os.path.exists(path):
        os.makedirs(path)


    df = pd.read_csv(args.data)

    ## reduce df only to relevant features to be evaluated
    d = ['index','fileID','start','end','speaker','label','d1','d2','d3','d4','d5','d6','d7','d8']

    if args.type == 'quartetts':
        ##todo: cut df only to columns concerning to respective classification types
        feats = ['1','2','3','4','5','6','7','8']

    elif args.type == 'biphones':
        ##only take into account columns that refer to biphones
        feats = ['bU','bY','dE','gY','kI','m@','mE','n@','nE','rE','SI','SU','vU','zE','zU',
                 '@l','@m','El','EN','Er','Es','Ex','Ir','Ix','Ul','Un','UN','Ur','Yl','Yr']

    elif args.type == 'features':
        ##only take into account columns that refer to features contained in the label
        feats = ['art_Frik','art_Lat','art_Nas','art_Plo','art_Trill',
                      'ort_Alv','ort_Bilab','ort_LabDent','ort_Pal','ort_PostAlv','ort_Uvul','ort_Vel','aschwa']

    elif args.type == 'featloc':
        ##only take into account columns that refer to features contained in the label by position
        feats = ['onset_art_Frik','onset_art_Nas','onset_art_Plo','onset_art_Trill',
                        'onset_ort_Alv','onset_ort_Bilab','onset_ort_LabDent','onset_ort_PostAlv','onset_ort_Uvul',
                        'onset_ort_Vel','onset_stimmhaft','onset_stimmlos',
                        'coda_art_Frik','coda_art_Lat','coda_art_Nas','coda_art_Trill',
                        'coda_ort_Alv','coda_ort_Bilab','coda_ort_Pal','coda_ort_Uvul','coda_ort_Vel','coda_aschwa',
                        'coda_stimmhaft','coda_stimmlos']

    elif args.type == 'phonloc':
        ##only take into account columns that refer to phonemes contained in the label by position
        feats = ['onset_z','onset_v','onset_k','onset_S','onset_m','onset_d','onset_n','onset_r','onset_b','onset_g',
                       'nuc_U','nuc_I','nuc_E','nuc_@','nuc_Y',
                       'coda_Ng','coda_r','coda_x','coda_m','coda_l','coda_s','coda_n',
                       'high_freq','low_freq','onset_stimmhaft','onset_stimmlos']

    df = df[df.columns.intersection(d + feats)]

    latent_dim_names = [c for c in df.columns if is_embedding_dimension.match(c)]

    precision = {}
    recall = {}
    f1 = {}
    accuracy = {}

    if args.gold_cols == ['english']:
        gold_cols = ['syllabic', 'consonantal', 'sonorant', 'continuant', 'delayed_release', 'approximant', 'nasal', 'voice', 'spread_glottis', 'labial', 'round', 'labiodental', 'coronal', 'anterior', 'distributed', 'strident', 'lateral', 'dorsal', 'high', 'low', 'front', 'back', 'tense', 'stress', 'diphthong']
    elif args.gold_cols == ['xitsonga']:
        gold_cols = ['syllabic', 'consonantal', 'sonorant', 'continuant', 'delayed_release', 'approximant', 'trill', 'nasal', 'voice', 'spread_glottis', 'constricted_glottis', 'labial', 'round', 'labiodental', 'coronal', 'anterior', 'distributed', 'strident', 'lateral', 'dorsal', 'high', 'low', 'front', 'back', 'tense', 'implosive']
    elif args.gold_cols == ['german']:
        gold_cols = feats
    elif args.gold_cols == ['spanish']:
        print("Not yet implemented for Spanish")
    else:
        gold_cols = args.gold_cols

    if args.direction.lower() == 'pred2gold':
        X = df[latent_dim_names] > 0.5

        for gold_col in gold_cols:
            perm, perm_inv = get_random_permutation(len(X))
            fold_size = math.ceil(float(len(X)) / args.n_folds)
            y = df[gold_col]

            score = 0
            predictions = []
            gold = []

            for i in range(0, len(X), fold_size):
                classifier = RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    criterion='entropy',
                    class_weight='balanced',
                    max_depth=args.max_depth,
                    min_impurity_decrease=args.min_impurity_decrease
                )

                train_select = np.zeros(len(X)).astype('bool')
                train_select[i:i+fold_size] = True
                train_select = train_select[perm_inv]

                cv_select = np.ones(len(X)).astype('bool')
                cv_select[i:i+fold_size] = False
                cv_select = cv_select[perm_inv]


                X_train = X[train_select]
                y_train = y[train_select]
                X_cv = X[cv_select]
                y_cv = y[cv_select]

                classifier.fit(X_train, y_train)
                predictions.append(classifier.predict(X_cv))
                gold.append(y_cv)

            predictions = np.concatenate(predictions, axis=0)
            gold = np.concatenate(gold, axis=0)
            precision[gold_col] = precision_score(gold, predictions)
            recall[gold_col] = recall_score(gold, predictions)
            f1[gold_col] = f1_score(gold, predictions)
            accuracy[gold_col] = accuracy_score(gold, predictions)

            if args.verbose:
                sys.stderr.write('Cross-validation F1 for variable "%s": %.4f\n' % (gold_col, f1[gold_col]))

            tree_ix = np.random.randint(args.n_estimators)

            graph = export_graphviz(
                classifier[tree_ix],
                feature_names=latent_dim_names,
                class_names=['-%s' % gold_col, '+%s' % gold_col],
                rounded=True,
                proportion=False,
                precision=2,
                filled=True
            )

            (graph,) = pydot.graph_from_dot_data(graph)

            outfile = path + '/decision_tree_%s.png' % gold_col
            graph.write_png(outfile)

        outfile = path + '/decision_tree_scores.txt'
        with open(outfile, 'w') as f:
            f.write('feature precision recall f1 accuracy\n')
            for c in sorted(list(f1.keys())):
                f.write('%s %s %s %s %s\n' % (c, precision[c], recall[c], f1[c], accuracy[c]))

    elif args.direction.lower() == 'gold2pred':
        X = df[gold_cols] > 0.5

        for latent_dim in latent_dim_names:
            perm, perm_inv = get_random_permutation(len(X))
            fold_size = math.ceil(float(len(X)) / args.n_folds)
            y = df[latent_dim]

            score = 0
            predictions = []
            gold = []

            for i in range(0, len(X), fold_size):
                classifier = RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    criterion='entropy',
                    class_weight='balanced',
                    max_depth=args.max_depth,
                    min_impurity_decrease=args.min_impurity_decrease
                )

                train_select = np.zeros(len(X)).astype('bool')
                train_select[i:i+fold_size] = True
                train_select = train_select[perm_inv]

                cv_select = np.ones(len(X)).astype('bool')
                cv_select[i:i+fold_size] = False
                cv_select = cv_select[perm_inv]


                X_train = X[train_select]
                y_train = y[train_select]
                X_cv = X[cv_select]
                y_cv = y[cv_select]

                classifier.fit(X_train, y_train)
                predictions.append(classifier.predict(X_cv))
                gold.append(y_cv)

            predictions = np.concatenate(predictions, axis=0)
            gold = np.concatenate(gold, axis=0)
            precision[latent_dim] = precision_score(gold, predictions)
            recall[latent_dim] = recall_score(gold, predictions)
            f1[latent_dim] = f1_score(gold, predictions)
            accuracy[latent_dim] = accuracy_score(gold, predictions)

            if args.verbose:
                sys.stderr.write('Cross-validation F1 for latent dimension "%s": %.4f\n' % (latent_dim, f1[latent_dim]))

            tree_ix = np.random.randint(args.n_estimators)

            graph = export_graphviz(
                classifier[tree_ix],
                feature_names=gold_cols,
                class_names=['-%s' % latent_dim, '+%s' % latent_dim],
                rounded=True,
                proportion=False,
                precision=2,
                filled=True
            )

            (graph,) = pydot.graph_from_dot_data(graph)

            outfile = path + '/decision_tree_%s.png' % latent_dim
            graph.write_png(outfile)

        outfile = path + '/decision_tree_scores.txt'
        with open(outfile, 'w') as f:
            f.write('feature precision recall f1 accuracy\n')
            for c in sorted(list(f1.keys())):
                f.write('%s %s %s %s %s\n' % (c, precision[c], recall[c], f1[c], accuracy[c]))

    else:
        raise ValueError('Direction parameter %s not recognized.' % args.direction)
