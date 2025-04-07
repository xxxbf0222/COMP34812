from nltk import RegexpTokenizer, pos_tag
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

class POS_tokenizer():
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return wn.NOUN
    
    def tokenize(self, text):
        # sentence-tokenisation -> word-tokenisation -> POS tagging -> case folding -> stopwords removal -> lemmatisation
        sentences = sent_tokenize(text)
        words = []
        lemmatizer = WordNetLemmatizer()
        for sent in sentences:
            tokenizer = RegexpTokenizer(r'<unk>|[a-zA-Z0-9]+')
            tokens = tokenizer.tokenize(sent)
            for word,tag in pos_tag(tokens):
                word = word.lower()
                if word not in self.stop_words:
                    words.append(lemmatizer.lemmatize(word,pos=self.get_wordnet_pos(tag)))
        return words