import re 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def read_stopwords(fn):
    return set([line.strip() for line in open(fn, encoding='utf-8') if len(line.strip()) != 0])

class preprocessor(object):
    def __init__(self, stopwords):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = stopwords

    def tokenize(self, text):
        return ' '.join(word_tokenize(text))

    def remove_stop(self, text):
        return ' '.join([w for w in text.split() if w not in self.stopwords])

    def remove_nonalph(self, text):
        return ' '.join([w for w in text.split() if re.search('\W+', w) == None and len(w) > 1])

    def remove_dig(self, text):
        return ' '.join([w for w in text.split() if w.isdigit() != True])

    def stem(self, text):
        return ' '.join([self.lemmatizer.lemmatize(w) for w in text.split()])

    def preprocess(self, text):
        s = text.lower()
        s = self.tokenize(s)
        s = self.remove_stop(s)
        s = self.remove_nonalph(s)
        s = self.remove_dig(s)
        s = self.stem(s)
        return s

if __name__ == '__main__':
    stopwords = read_stopwords("./dataset/stopwords.en.txt")
    preprocessor = preprocessor(stopwords)

    from sklearn.datasets import fetch_20newsgroups
    corpus = fetch_20newsgroups(shuffle=True, random_state=1,remove=('headers', 'footers', 'quotes'))
    texts = corpus.data
    print (texts[0])

    for s in texts:
        print (preprocessor.preprocess(s))
        break