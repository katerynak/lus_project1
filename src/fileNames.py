from aenum import Enum
"""
aenum is useful for accessing data both from string and alias
"""

class FileNames(Enum):

    TRAIN = '../data/NLSPARQL.train.data'
    TEST = '../data/NLSPARQL.test.data'
    STATS_DIR = "../stats/"
    TOK_POS_PROBS = "tok_pos_probs.txt"
    LEXICON =  "train.lex"
    PROB_UNK =  "train_unk.txt"
    LEX_TRANS_TXT =  'lexicon_transducer.txt'
    LEX_TRANS =  'lexicon_transducer.fst'
    UNIGRAM_CONCEPT_UNK =  'unigram_unk.txt'
    SENT_TAGS = 'sentences_tags.txt'

    #transducer components file names
    FST_DIR = '../models/'
    UNIGRAM_CONCEPT_FST = "unigram.fst"
    UNK_FST =  "unk.fst"
    UNIGRAM_CONCEPT_UNK_FST =  "unigram_unk.fst"
    NGRAM_LM = "ngram.lm"

    #test_out file names
    TEST_OUT_DIR = "../test_out/"
    TEST_OUT = "pred_data.txt"

    #results file names
    RESULTS_DIR = "../results/"
    RESULT = "eval_result"
