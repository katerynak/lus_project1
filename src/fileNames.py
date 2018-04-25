from aenum import Enum
"""
aenum is useful for accessing data both from string and alias
"""

class FileNames(Enum):

    TRAIN = './data/NLSPARQL.train.data'
    TEST = './data/NLSPARQL.test.data'
    STATS_DIR = "./stats/"
    TOK_POS_PROBS = STATS_DIR + "tok_pos_probs.txt"
    LEXICON = STATS_DIR + "train.lex"
    PROB_UNK = STATS_DIR + "train_unk.txt"
    LEX_TRANS_TXT = STATS_DIR + 'lexicon_transducer.txt'
    LEX_TRANS = STATS_DIR + 'lexicon_transducer.fst'
    UNIGRAM_CONCEPT_UNK = STATS_DIR + 'unigram_unk.txt'
    SENT_TAGS = STATS_DIR + 'sentences_tags.txt'

    #transducer components file names
    UNIGRAM_CONCEPT_FST = STATS_DIR + "unigram.fst"
    UNK_FST = STATS_DIR + "unk.fst"
    UNIGRAM_CONCEPT_UNK_FST = STATS_DIR + "unigram_unk.fst"
