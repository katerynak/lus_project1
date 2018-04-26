from concept_tagger import ConceptTagger
from stats import Stat
from fileNames import FileNames
import os
from subprocess import call
import pandas as pd
import itertools

def train_test_minimum(trainFile=""):

    #default working dir + test/train files

    minimum_dir = 'minimum/'
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
    st.load_data()
    st.count_tokens()
    st.count_tags()
    st.count_tokens_tags()
    st.probs_token_tags()
    st.create_unk_table()
    st.write_token_unk_pos_probs()
    st.create_lexicon()
    st.write_sentences_tags()

    ct = ConceptTagger()

    ct.create_unigram_tagger(st.lexiconFile, st.unigram_conc_unk, unigram_taggerFstFile)
    # default smoothing and order for now
    ct.create_language_model(st.lexiconFile, st.sentecesTagsFile, ngram_lmFile)


    test_tokens_sentences, test_tags, test_tokens = st.read_sentences_tokens_tags()
    out_list = []
    cnt = 0
    tot = len(test_tokens_sentences)

    #for each sentence creates acceptor, concatenates it with unigram tagger and language model,
    #parses out lex result and appends it to results
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

    call('./conlleval.pl < {0} > {1}'.format(test_out_file, res_out_file), shell=True)

