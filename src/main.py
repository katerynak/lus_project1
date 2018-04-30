#!/usr/bin/python3

import train_test

if __name__== "__main__":

   # smoothing = ["absolute", "katz", "kneser_ney", "presmoothed", "katz_frac", "witten_bell"]
    smoothing = [ "witten_bell"]
    order = [2,3,4,5]
    scores_smoothing = []
    scores_order = []
    for s in smoothing:
        scores_order = []
        for o in order:
            scores_order.append(train_test.train_test_minimum(smoothing=s, ngram_order=o))
        scores_smoothing.append(scores_order)

    # f1_baseline = train_test.baseline_random()
    # print("baseline random result: " + f1_baseline)
    #
    # f1_baseline = train_test.baseline_majority()
    # print("baseline majority result: " + f1_baseline)
    #
    # f1_baseline = train_test.baseline_random_normal_dist()
    # print("baseline random normal distribution result: " + f1_baseline)

