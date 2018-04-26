import pandas as pd
import numpy as np
import os
from subprocess import call
from fileNames import FileNames


#stat loads all default filenames

#for different test/train just set trainFile and testFile, in this case
#all files in stats will be overwritten, change working directory if you don't
#want to overwrite current statistics

class Stat(object):

    def __init__(self, directory="", trainFile=""):

        if not directory:
            self.directory = FileNames.STATS_DIR.value
            print(self.directory)
        else:
            self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        if not trainFile:
            self.trainFile = FileNames.TRAIN.value
        else:
            self.trainFile = trainFile
        self.testFile = FileNames.TEST.value
        self.tok_pos_probsFile = self.directory + FileNames.TOK_POS_PROBS.value
        self.lexiconFile = self.directory + FileNames.LEXICON.value
        self.prob_unkFile = self.directory + FileNames.PROB_UNK.value
        self.sentecesTagsFile = self.directory + FileNames.SENT_TAGS.value
        self.unigram_conc_unk = self.directory + FileNames.UNIGRAM_CONCEPT_UNK.value

    def load_data(self):
        self.data = pd.read_csv(self.trainFile, sep='\t', header=None)
        self.data.columns = ['tokens', 'tags']

        # remove $ character from tags
        self.data['tags'] = self.data['tags'].map(lambda x: x.rstrip('$'))

    def count_tokens(self):
        self.tok_counts = pd.DataFrame(self.data['tokens'].value_counts().reset_index())
        self.tok_counts.columns = ['tokens', 'tokens_counts']

    def count_tags(self):
        self.pos_counts = pd.DataFrame(self.data['tags'].value_counts().reset_index())
        self.pos_counts.columns = ['tags', 'tags_counts']

    def count_tokens_tags(self):
        self.tok_pos_counts = self.data.groupby(['tokens', 'tags']).size().reset_index()
        self.tok_pos_counts.columns = ['tokens', 'tags', 'tokens_tags_counts']

    def probs_token_tags(self):
        self.tok_pos_probs = pd.merge(self.tok_pos_counts, self.pos_counts, on='tags', how='outer')
        self.tok_pos_probs[
            'tokens_given_tags_probs'] = self.tok_pos_probs.tokens_tags_counts / self.tok_pos_probs.tags_counts
        self.tok_pos_probs['neg_log_tokens_given_tags_probs'] = -np.log(self.tok_pos_probs.tokens_given_tags_probs)
        self.tok_pos_probs['from_state'] = '0'
        self.tok_pos_probs['to_state'] = '0'

    def create_unk_table(self):
        self.tok_prob_unk = 1 / len(self.pos_counts['tags'])
        self.unk_probs = pd.DataFrame(self.pos_counts['tags'].copy())
        self.unk_probs.columns = ['tags']
        self.unk_probs['from_state'] = '0'
        self.unk_probs['to_state'] = '0'
        self.unk_probs['tokens'] = '<unk>'
        self.unk_probs['neg_log_tokens_given_tags_probs'] = -np.log(self.tok_prob_unk)

    def write_token_unk_pos_probs(self):
        cols_to_keep = ['from_state', 'to_state', 'tokens', 'tags', 'neg_log_tokens_given_tags_probs']
        self.tok_pos_probs_to_keep = pd.DataFrame(self.tok_pos_probs[cols_to_keep])
        self.tok_pos_probs_to_keep.to_csv(self.unigram_conc_unk, index=None,  header=None,sep='\t', mode='w')
        self.unk_probs[cols_to_keep].to_csv(self.unigram_conc_unk, header=None, index = None, sep = '\t', mode='a')
        with open(self.unigram_conc_unk, "a") as f:
            f.write('0')

    def create_lexicon(self):
        call('ngramsymbols < ' + self.trainFile + ' > ' + self.lexiconFile, shell=True)
        print(self.lexiconFile)

    def write_sentences_tags(self, fileOut=""):
        if not fileOut:
            fileOut = self.sentecesTagsFile
        sentences = pd.read_csv(self.trainFile, sep='\t', header=None, skip_blank_lines=False)
        sentences.columns = ['tokens', 'tags']

        sentences_tags = sentences['tags']

        list_of_lists = []
        list = []
        for t in sentences_tags:
            if pd.isnull(t):
                list_of_lists.append(list)
                list = []
            else:
                list.append(t)

        thefile = open(fileOut, 'w')
        for list in list_of_lists:
            for l in list:
                thefile.write("%s\t" % l)
            thefile.write("\n")
        thefile.close()

#change testFile name if different testFile is used

    def read_sentences_tokens_tags(self):
        sentences = pd.read_csv(self.testFile, sep="\s+", header=None, skip_blank_lines=False)
        sentences.columns = ['tokens', 'tags']
        sentences_tags = sentences['tags']
        sentences_tokens = sentences['tokens']
        tags_list_of_lists = []
        tok_list_of_lists = []
        list_tag = []
        list_token = []
        for tag, tok in zip(sentences_tags, sentences_tokens):
            if pd.isnull(tag):
                tags_list_of_lists.append(' '.join(list_tag))
                tok_list_of_lists.append(' '.join(list_token))
                list_tag = []
                list_token = []
            else:
                list_tag.append(tag)
                list_token.append(tok)
        return tok_list_of_lists, sentences_tags.fillna(value=''), sentences_tokens.fillna(value='')