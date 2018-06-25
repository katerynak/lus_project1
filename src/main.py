#!/usr/bin/python3

import train_test
import os
from subprocess import check_output
import numpy as np
import heapq
from operator import itemgetter


def max_val(l, i):
    return max(enumerate(map(itemgetter(i), l)), key=itemgetter(1))


def cross_validate(k, trainFiles, valFiles, valFunc, smoothing, order):
    """
    execute valFunc on each couple of (trainFiles, valFiles) for each param
    returns param giving best average score and list of scores if all cofigurations
    """

    scores = []
    for s in smoothing:
        for o in order:
            scores_sum = 0
            i = 0
            for train, val in zip(trainFiles, valFiles):
                scores_sum += float(valFunc(smoothing=s, ngram_order=o, trainFile=train, testFile=val, fold=str(i)))
                i += 1
            scores.append([scores_sum / k, s, o])
    return scores[max_val(scores, 0)[0]], scores


def cross_validate_improvement(k, trainFiles, valFiles, valFunc, smoothing, order):
    """
    execute valFunc on each couple of (trainFiles, valFiles) for each param
    returns param giving best average score and list of scores if all cofigurations
    """

    scores = []
    for s in smoothing:
        for o in order:
            scores_sum = 0
            i = 0
            for train, val in zip(trainFiles, valFiles):
                add_info_files = train_test.add_info_to_data_words(trainFile=train, testFile=val)
                scores_sum += float(valFunc(smoothing=s, ngram_order=o, trainFile=add_info_files['train'],
                                            testFile=add_info_files['test'], fold=str(i), working_dir="improvement/",
                                            improvement=True))
                i += 1
            scores.append([scores_sum / k, s, o])
    return scores[max_val(scores, 0)[0]], scores


def max_cross_val(workingDir):
    file_list = []
    for file in os.listdir(workingDir):
        f = workingDir + file
        file_list.append(f)
    file_list.sort()
    configs = [file_list[x:x + 10] for x in range(0, len(file_list), 10)]
    scores = []
    for config in configs:
        config_scores = []
        for f in config:
            f1_score = check_output("awk '{print $8}' " + "{0} |sed '2q;d'".format(f), shell=True).decode("utf-8")
            config_scores.append(float(f1_score))
        scores.append(config_scores)
    means = [np.mean(score) for score in scores]
    max_idx = [means.index(val) for val in heapq.nlargest(10, means)]


if __name__ == "__main__":

    """
    baseline evaluation: results of other evaluation metrics are saved 
                         in folder 'results/baseline'
    """

    print("Computing baseline models results")

    f1_baseline = train_test.baseline_random()
    print("baseline random result: " + f1_baseline)

    f1_baseline = train_test.baseline_majority()
    print("baseline majority result: " + f1_baseline)

    f1_baseline = train_test.baseline_random_data_dist()
    print("baseline random data distribution result: " + f1_baseline)

    """
    basic model evaluation: all possible configurations
    """

    print("Computing minimum model results")

    smoothing = ["absolute", "katz", "kneser_ney", "presmoothed", "witten_bell"]
    order = [1, 2, 3, 4, 5]
    freq_cut_off = [1, 2, 3, 4, 5]
    basic_scores = []
    for s in smoothing:
        scores_order = []
        for o in order:
            # no cut_off
            f1_score = train_test.train_test_minimum(smoothing=s, ngram_order=o)
            basic_scores.append([f1_score, s, o])
            # with cut_off
            for f in freq_cut_off:
                train_test.train_test_minimum(smoothing=s, ngram_order=o, cut_off=True, cut_off_freq=f)
                basic_scores.append([f1_score, s, o, f])

    """
    testing basic model with 5-fold cross validation
    """

    k = 5

    # create cross val files
    valFiles, trainFiles = train_test.k_folds(k)

    smoothing = ["absolute", "katz", "kneser_ney", "presmoothed", "witten_bell"]
    order = [2, 3, 4, 5]

    max_score, scores = cross_validate(k, valFiles=valFiles,
                                       trainFiles=trainFiles,
                                       valFunc=train_test.train_test_minimum,
                                       smoothing=smoothing,
                                       order=order)

    """
    final evaluation with configurations giving max result in cross validation
    """

    f1_minimum = train_test.train_test_minimum(smoothing=max_score[1], ngram_order=max_score[2], fold='final_eval')
    print("result minimum: " + f1_minimum)

    """
    improvement 1: substitute 'O' tags with concatenation of unique lemmas and pos tags
    """

    # create test/train files for improvements/additional info
    add_info_files = train_test.add_info_to_data_pos_lemma()

    """
    evaluate basic model with improved test and train, and best params from cross validation
    """

    f1_score = train_test.train_test_minimum(trainFile=add_info_files['train'],
                                             testFile=add_info_files['test'],
                                             smoothing='kneser_ney', ngram_order=4,
                                             working_dir='unique_pos_tags/', improvement=True)

    """
    improvement 2: substitute 'O' tags with tokens
    """

    # create test/train files for improvements/additional info
    add_info_files = train_test.add_info_to_data_words()

    smoothing = ["absolute", "katz", "kneser_ney", "presmoothed", "witten_bell"]
    order = [2, 3, 4, 5]

    improve_scores = []

    for s in smoothing:
        for o in order:
            f1_score = train_test.train_test_minimum(trainFile=add_info_files['train'],
                                                     testFile=add_info_files['test'],
                                                     smoothing=s, ngram_order=o,
                                                     working_dir='improvement_all_data/', improvement=True)
            improve_scores.append([f1_score, s, o])

    """
        testing improved model with 10-fold cross validation
    """

    k = 10

    # create cross val files
    valFiles, trainFiles = train_test.k_folds(k)

    smoothing = ["absolute", "katz", "kneser_ney", "presmoothed", "witten_bell"]
    order = [2, 3, 4, 5]

    max_score, scores = cross_validate_improvement(k, valFiles=valFiles,
                                                   trainFiles=trainFiles,
                                                   valFunc=train_test.train_test_minimum,
                                                   smoothing=smoothing,
                                                   order=order)

    f1_score = train_test.train_test_minimum(trainFile=add_info_files['train'],
                                             testFile=add_info_files['test'],
                                             smoothing=max_score[1], ngram_order=max_score[2],
                                             working_dir='best_result/', improvement=True)

    train_test.printConfusionMatrix('../test_out/improvement/pred_data.txt')
