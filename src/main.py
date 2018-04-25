#!/usr/bin/python3

from stats import Stat

class Main(object):

    st = Stat()
    test_tokens_sentences, test_tags, test_tokens = st.read_sentences_tokens_tags()

    def processTrainData(self):
        self.st.load_data()
        self.st.count_tokens()
        self.st.count_tags()
        self.st.count_tokens_tags()
        self.st.probs_token_tags()
        self.st.create_unk_table()
        self.st.write_token_unk_pos_probs()
        self.st.create_lexicon()
        self.st.write_sentences_tags()

if __name__== "__main__":
    m = Main()
    m.processTrainData()