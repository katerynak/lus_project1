from concept_tagger import ConceptTagger
from stats import Stat
from fileNames import FileNames
import os
from subprocess import call, check_output
import pandas as pd
import itertools
import random
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix


def train_test_minimum(trainFile="", testFile="", smoothing = "witten_bell", ngram_order = 3,
                       working_dir="", improvement=False, fold="", cut_off=False, cut_off_freq=2):

    """
    function tests a minimum requirement model
    outputs f1 score of the model
    input : trainFile, testFile
    if retOutFiles is set to true function returns names of files where results are written
    """

    #default working dir + test/train files
    minimum_dir = 'minimum/'

    if working_dir:
        minimum_dir = working_dir

    fst_out_dir = FileNames.FST_DIR.value + minimum_dir
    if not os.path.exists(fst_out_dir):
        os.makedirs(fst_out_dir)

    unigram_taggerFstFile= fst_out_dir + FileNames.UNIGRAM_CONCEPT_FST.value
    ngram_lmFile = fst_out_dir + FileNames.NGRAM_LM.value

    res_out_dir = FileNames.RESULTS_DIR.value + minimum_dir
    if not os.path.exists(res_out_dir):
        os.makedirs(res_out_dir)

    res_out_file = res_out_dir + FileNames.RESULT.value

    test_out_dir = FileNames.TEST_OUT_DIR.value + minimum_dir
    if not os.path.exists(test_out_dir):
        os.makedirs(test_out_dir)

    test_out_file = test_out_dir + FileNames.TEST_OUT.value

    st = Stat(trainFile=trainFile)
    if testFile:
        st.testFile = testFile
    st.load_data()
    st.count_tokens()
    st.count_tags()
    st.count_tokens_tags()
    st.probs_token_tags()
    if (cut_off):
        st.create_cut_off_unk_table(cutoff_freq=cut_off_freq)
    else:
        st.create_unk_table()
    st.write_token_unk_pos_probs()
    st.create_lexicon()
    st.write_sentences_tags()

    ct = ConceptTagger()

    ct.create_unigram_tagger(st.lexiconFile, st.unigram_conc_unk, unigram_taggerFstFile)
    # default smoothing and order for now
    ct.create_language_model(st.lexiconFile, st.sentecesTagsFile, ngram_lmFile, smoothing=smoothing, order=ngram_order)

    test_tokens_sentences, test_tags, test_tokens = st.read_sentences_tokens_tags()
    out_list = []
    cnt = 0
    tot = len(test_tokens_sentences)

    # for each sentence creates acceptor, concatenates it with unigram tagger and language model,
    # parses out lex result and appends it to results
    for string in test_tokens_sentences:
        cnt += 1
        print('{0}/{1}'.format(cnt, tot))
        print(string)
        accFile = ct.create_acceptor(string, st.lexiconFile)
        ct.composeFsts(accFile, unigram_taggerFstFile, 'tmp.fst')
        ct.composeFsts('tmp.fst', ngram_lmFile, 'tmp2.fst')
        ct.shortestPath('tmp2.fst', 'out.fst')
        iob_tags = ct.parseOut(st.lexiconFile,'out.fst').split('\n')
        out_list.append(iob_tags)

    call('rm tmp.fst; rm tmp2.fst; rm out.fst; rm 1.fst', shell=True)
    out_list = list(itertools.chain(*out_list))
    df = pd.DataFrame({'col': out_list})
    y_pred = df['col']
    pred_data = pd.DataFrame([test_tokens, test_tags, y_pred])
    pred_data = pred_data.transpose()
    pred_data.to_csv(test_out_file, index=None, header=None, sep=' ', mode='w')

    co_suffix = ""
    if (cut_off):
        co_suffix = "__cut_off_freq_" + str(cut_off_freq)

    evalFile = res_out_file + '__'+ smoothing + "__ngram_size_" + str(ngram_order) + co_suffix + str(fold)

    if not improvement:
        call('./conlleval.pl < {0} > {1}'.format(test_out_file, evalFile), shell=True)
    else:
        eval_unique_pos_tags(test_out_file, evalFile)

    f1_score = check_output("awk '{print $8}' "+"{0} |sed '2q;d'".format(evalFile), shell=True).decode("utf-8")

    return f1_score


def baseline_random(trainFile="", testFile=""):
    baseline_dir = "baseline/"

    res_out_dir = FileNames.RESULTS_DIR.value + baseline_dir
    if not os.path.exists(res_out_dir):
        os.makedirs(res_out_dir)

    res_out_file = res_out_dir + FileNames.RESULT.value + '_random'

    test_out_dir = FileNames.TEST_OUT_DIR.value + baseline_dir
    if not os.path.exists(test_out_dir):
        os.makedirs(test_out_dir)

    test_out_file = test_out_dir + FileNames.TEST_OUT.value + '_random'

    st = Stat(trainFile=trainFile)
    if testFile:
        st.testFile = testFile
    st.load_data()
    tags = st.get_tags()
    test_tokens_sentences, test_tags, test_tokens = st.read_sentences_tokens_tags()
    random_tags = []

    for token in test_tokens:
        if token == '':
            random_tags.append(token)
        else:
            random_tags.append(tags[random.randint(0,tags.size-1)])

    df = pd.DataFrame({'col': random_tags})
    y_pred = df['col']
    pred_data = pd.DataFrame([test_tokens, test_tags, y_pred])
    pred_data = pred_data.transpose()
    pred_data.to_csv(test_out_file, index=None, header=None, sep=' ', mode='w')

    call('./conlleval.pl < {0} > {1}'.format(test_out_file, res_out_file), shell=True)

    f1_score = check_output(
        "awk '{print $8}' " + "{0} |sed '2q;d'".format(res_out_file),
        shell=True).decode("utf-8")

    return f1_score


def baseline_majority(trainFile="", testFile=""):
    baseline_dir = "baseline/"

    res_out_dir = FileNames.RESULTS_DIR.value + baseline_dir
    if not os.path.exists(res_out_dir):
        os.makedirs(res_out_dir)

    res_out_file = res_out_dir + FileNames.RESULT.value + '_majority'

    test_out_dir = FileNames.TEST_OUT_DIR.value + baseline_dir
    if not os.path.exists(test_out_dir):
        os.makedirs(test_out_dir)

    test_out_file = test_out_dir + FileNames.TEST_OUT.value + '_majority'

    st = Stat(trainFile=trainFile)
    if testFile:
        st.testFile = testFile
    st.load_data()
    test_tokens_sentences, test_tags, test_tokens = st.read_sentences_tokens_tags()
    random_tags = []

    majority_tag = st.get_majority_tag()

    for token in test_tokens:
        if token == '':
            random_tags.append(token)
        else:
            random_tags.append(majority_tag)

    df = pd.DataFrame({'col': random_tags})
    y_pred = df['col']
    pred_data = pd.DataFrame([test_tokens, test_tags, y_pred])
    pred_data = pred_data.transpose()
    pred_data.to_csv(test_out_file, index=None, header=None, sep=' ', mode='w')

    call('./conlleval.pl < {0} > {1}'.format(test_out_file, res_out_file), shell=True)

    f1_score = check_output(
        "awk '{print $8}' " + "{0} |sed '2q;d'".format(res_out_file),
        shell=True).decode("utf-8")

    return f1_score


def baseline_random_data_dist(trainFile="", testFile=""):
    baseline_dir = "baseline/"

    res_out_dir = FileNames.RESULTS_DIR.value + baseline_dir
    if not os.path.exists(res_out_dir):
        os.makedirs(res_out_dir)

    res_out_file = res_out_dir + FileNames.RESULT.value + '_random_data_dist'

    test_out_dir = FileNames.TEST_OUT_DIR.value + baseline_dir
    if not os.path.exists(test_out_dir):
        os.makedirs(test_out_dir)

    test_out_file = test_out_dir + FileNames.TEST_OUT.value + '_random_data_dist'

    st = Stat(trainFile=trainFile)
    if testFile:
        st.testFile = testFile
    st.load_data()
    tags = st.get_tags()
    test_tokens_sentences, test_tags, test_tokens = st.read_sentences_tokens_tags()
    random_tags = []
    tags_data_dist = st.get_tag_distribution()
    for token in test_tokens:
        if token == '':
            random_tags.append(token)
        else:
            random_tags.append(np.random.choice(tags_data_dist.index, 1, p=tags_data_dist.tolist()).tolist()[0])

    df = pd.DataFrame({'col': random_tags})
    y_pred = df['col']
    pred_data = pd.DataFrame([test_tokens, test_tags, y_pred])
    pred_data = pred_data.transpose()
    pred_data.to_csv(test_out_file, index=None, header=None, sep=' ', mode='w')

    call('./conlleval.pl < {0} > {1}'.format(test_out_file, res_out_file), shell=True)

    f1_score = check_output(
        "awk '{print $8}' " + "{0} |sed '2q;d'".format(res_out_file),
        shell=True).decode("utf-8")

    return f1_score


def k_folds(k = 5, trainFile="", testFile="", featsTrainFile = "", featsTestFile = ""):

    """
    function splits test and train file in different files to be used for cross validation,
    split is based on number of phrases and not number of lines
    """

    if not trainFile:
        trainFile = FileNames.TRAIN.value
    if not testFile:
        testFile = FileNames.TEST.value
    if not featsTrainFile:
        featsTrainFile = FileNames.ADD_TRAIN.value
    if not featsTestFile:
        featsTestFile = FileNames.ADD_TEST.value

    test = pd.read_csv(testFile, sep="\s+", header=None, skip_blank_lines=False)
    test.columns = ['tokens', 'tags']
    delimiters = []
    delim = [0]
    tmp = pd.isnull(test).any(1).nonzero()[0].tolist()
    for d in tmp:
        delim.append(d)
        delimiters.append(delim)
        delim = [d]

    # shuffle phrases
    delimiters = shuffle(delimiters)

    # count test phrases
    fold_size = int(len(delimiters)/k)

    folds_delimiters = [[] for _ in range(k)]

    for i in range(k-1):
        folds_delimiters[i] = delimiters[i*fold_size:(i+1)*fold_size]
    folds_delimiters[k-1] = delimiters[(k-1) * fold_size:]

    folds = [pd.DataFrame() for _ in range(k)]

    for i in range(k):
        for id in folds_delimiters[i]:
            folds[i] = folds[i].append(test.loc[id[0]:id[1]-1, :],  ignore_index=True)

    valFiles = []
    trainFiles = []
    # now for each fold save it as validation and save others as training set
    for i in range(k):
        valFiles.append('../data/validation'+str(i))
        trainFiles.append('../data/training'+str(i))
        folds[i].iloc[1:].to_csv('../data/validation'+str(i), index=None,  header=None, sep='\t', mode='w')
        pd.concat(folds[:i] + folds[i+1:]).iloc[1:].to_csv('../data/training'+str(i), index=None,  header=None, sep='\t', mode='w')

    for v in valFiles:
        with open(v,mode='a') as f:
            f.write('\n')

    for t in trainFiles:
        with open(t,mode='a') as f:
            f.write('\n')

    return valFiles, trainFiles


def add_info_to_data(trainFile="", testFile="", featsTrainFile = "", featsTestFile = ""):
    """
    function substitutes O tag with concatenation of unique token and pos tag
    and writes new test and train files with fixed names
    """
    if not trainFile:
        trainFile = FileNames.TRAIN.value
    if not testFile:
        testFile = FileNames.TEST.value
    if not featsTrainFile:
        featsTrainFile = FileNames.ADD_TRAIN.value
    if not featsTestFile:
        featsTestFile = FileNames.ADD_TEST.value


    train_data = pd.read_csv(trainFile, sep='\t', header=None, skip_blank_lines=False)
    train_data.columns = ['tokens', 'tags']

    test_data = pd.read_csv(testFile, sep='\t', header=None, skip_blank_lines=False)
    test_data.columns = ['tokens', 'tags']

    train_data_feats = pd.read_csv(featsTrainFile, sep='\t', header=None, skip_blank_lines=False)
    train_data_feats.columns = ['tokens', 'pos_tags', 'unique_tokens']

    test_data_feats = pd.read_csv(featsTestFile, sep='\t', header=None, skip_blank_lines=False)
    test_data_feats.columns = ['tokens', 'pos_tags', 'unique_tokens']


    newUniquePOSTrainFile = "../data/pos.unique.train.data"
    newUniquePOSTestFile = "../data/pos.unique.test.data"

    unique_tokens_train = pd.DataFrame()

    for token, tag, unique_tag, pos_tag in zip(train_data['tokens'], train_data['tags'], train_data_feats['unique_tokens'], train_data_feats['pos_tags']):
        if tag=='O':
            t = "_" + unique_tag + pos_tag
            #t = "_" + pos_tag
        else:
            t = tag
        unique_tokens_train = unique_tokens_train.append([[token, t]])

    unique_tokens_test = pd.DataFrame()

    for token, tag, unique_tag, pos_tag in zip(test_data['tokens'], test_data['tags'],
                                               test_data_feats['unique_tokens'], test_data_feats['pos_tags']):
        if tag == 'O':
            t = "_" + unique_tag + pos_tag
            #t = "_" + pos_tag
        else:
            t = tag
        unique_tokens_test = unique_tokens_test.append([[token, t]])

    unique_tokens_train.to_csv(newUniquePOSTrainFile, index=None,  header=None, sep='\t', mode='w')
    unique_tokens_test.to_csv(newUniquePOSTestFile, index=None,  header=None, sep='\t', mode='w')

    return {'train': newUniquePOSTrainFile, 'test':newUniquePOSTestFile}


def add_info_to_data_pos_lemma(trainFile="", testFile="", featsTrainFile = "", featsTestFile = ""):
    """
    function substitutes O tag with concatenation of unique token and pos tag
    and writes new test and train files with fixed names
    """
    if not trainFile:
        trainFile = FileNames.TRAIN.value
    if not testFile:
        testFile = FileNames.TEST.value
    if not featsTrainFile:
        featsTrainFile = FileNames.ADD_TRAIN.value
    if not featsTestFile:
        featsTestFile = FileNames.ADD_TEST.value

    train_data = pd.read_csv(trainFile, sep='\t', header=None, skip_blank_lines=False)
    train_data.columns = ['tokens', 'tags']

    test_data = pd.read_csv(testFile, sep='\t', header=None, skip_blank_lines=False)
    test_data.columns = ['tokens', 'tags']

    train_data_feats = pd.read_csv(featsTrainFile, sep='\t', header=None, skip_blank_lines=False)
    train_data_feats.columns = ['tokens', 'pos_tags', 'unique_tokens']

    test_data_feats = pd.read_csv(featsTestFile, sep='\t', header=None, skip_blank_lines=False)
    test_data_feats.columns = ['tokens', 'pos_tags', 'unique_tokens']

    newUniquePOSTrainFile = "../data/pos.unique.train.data"
    newUniquePOSTestFile = "../data/pos.unique.test.data"

    unique_tokens_train = pd.DataFrame()

    for token, tag, unique_tag, pos_tag in zip(train_data['tokens'], train_data['tags'], train_data_feats['unique_tokens'], train_data_feats['pos_tags']):
        if tag=='O':
            t = "_" + unique_tag + pos_tag
            #t = "_" + pos_tag
        else:
            t = tag
        unique_tokens_train = unique_tokens_train.append([[token, t]])

    unique_tokens_test = pd.DataFrame()

    for token, tag, unique_tag, pos_tag in zip(test_data['tokens'], test_data['tags'],
                                               test_data_feats['unique_tokens'], test_data_feats['pos_tags']):
        if tag == 'O':
            t = "_" + unique_tag + pos_tag
            #t = "_" + pos_tag
        else:
            t = tag
        unique_tokens_test = unique_tokens_test.append([[token, t]])

    unique_tokens_train.to_csv(newUniquePOSTrainFile, index=None,  header=None, sep='\t', mode='w')
    unique_tokens_test.to_csv(newUniquePOSTestFile, index=None,  header=None, sep='\t', mode='w')

    return {'train': newUniquePOSTrainFile, 'test':newUniquePOSTestFile}


def add_info_to_data_words(trainFile="", testFile=""):
    """
    function substitutes O tag with concatenation of unique token and pos tag
    and writes new test and train files with fixed names
    """
    if not trainFile:
        trainFile = FileNames.TRAIN.value
    if not testFile:
        testFile = FileNames.TEST.value

    train_data = pd.read_csv(trainFile, sep='\t', header=None, skip_blank_lines=False)
    train_data.columns = ['tokens', 'tags']

    test_data = pd.read_csv(testFile, sep='\t', header=None, skip_blank_lines=False)
    test_data.columns = ['tokens', 'tags']

    newUniquePOSTrainFile = "../data/token.train.data"
    newUniquePOSTestFile = "../data/token.test.data"

    unique_tokens_train = pd.DataFrame()

    for token, tag in zip(train_data['tokens'], train_data['tags']):
        if tag=='O':
            t = "_" + token
        else:
            t = tag
        unique_tokens_train = unique_tokens_train.append([[token, t]])

    unique_tokens_test = pd.DataFrame()

    for token, tag in zip(test_data['tokens'], test_data['tags']):
        if tag == 'O':
            t = "_" + token
        else:
            t = tag
        unique_tokens_test = unique_tokens_test.append([[token, t]])

    unique_tokens_train.to_csv(newUniquePOSTrainFile, index=None,  header=None, sep='\t', mode='w')
    unique_tokens_test.to_csv(newUniquePOSTestFile, index=None,  header=None, sep='\t', mode='w')

    return {'train': newUniquePOSTrainFile, 'test':newUniquePOSTestFile}


def eval_unique_pos_tags(predDataFile, resutlsFile):

    """
    function reads and overwrites predDataFile substituting tags starting with '_' with tags 'O',
    evaluate results of new out file, overwrites the evaluation results and returns f1 score
    """

    pred_data = pd.read_csv(predDataFile, sep='\s+', header=None, skip_blank_lines=False)
    pred_data.columns = ['token', 'pred_tag', 'true_tag']
    pred_data['pred_tag'] = pred_data['pred_tag'].str.replace(r'^_.*$', 'O')
    pred_data['true_tag'] = pred_data['true_tag'].str.replace(r'^_.*$', 'O')

    pred_data.to_csv(predDataFile, index=None, header=None, sep=' ', mode='w')

    call('./conlleval.pl < {0} > {1}'.format(predDataFile, resutlsFile), shell=True)

    f1_score = check_output(
        "awk '{print $8}' " + "{0} |sed '2q;d'".format(resutlsFile),
        shell=True).decode("utf-8")

    return f1_score


def printConfusionMatrix(predFile):
    """
    function prints confusion matrix of predicted concept tags
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    pred_data = pd.read_csv(predFile, sep='\s+', header=None, skip_blank_lines=True)
    pred_data.columns = ['tokens', 'test_tags', 'pred_tags']

    # pred_data['pred_tags'] = pred_data['pred_tags'].apply(lambda x: x if (x=='O') else x[2:])
    # pred_data['test_tags'] = pred_data['test_tags'].apply(lambda x: x if (x=='O') else x[2:])

    pred = pred_data['pred_tags'].tolist()

    y_test = pred_data['test_tags'].tolist()

    labels = pd.unique(pred_data[['test_tags', 'pred_tags']].values.ravel('K'))

    yticklabels = ['B-award.ceremony','I-award.ceremony', 'B-award.category', 'I-award.category',
              'B-director.nationality', 'B-movie.gross_revenue', 'I-movie.gross_revenue']

    #labels = pred_data['test_tags'].unique().tolist()
    cm = confusion_matrix(y_test, pred, xticklabels=labels, yticklabels = yticklabels)
    fig = plt.figure(figsize=(15, 8))
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, vmin=0, vmax=50)  # annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.show()
    fig.savefig("../plots/confusion_matrix" + ".pdf", format='pdf')