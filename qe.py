from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import re

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile, 'r', encoding='utf-8')
    model = defaultdict(lambda: np.zeros(300))
    for line in tqdm(f):
        splitLine = line.split()
        word = splitLine[0]
        model[word] = np.array([float(val) for val in splitLine[1:]])
    print ('Done.', len(model), 'words loaded!')
    return model

class QueryExpansion(object):
    def __init__(self, alpha=1e-6, phi=0.5, emb_path = './QDTM/glove_embedding/glove.6B.300d.txt'):
        self.alpha = alpha
        self.phi = phi
        self.gloveModel = loadGloveModel(emb_path)

    def initialize(self, processed_texts = [], n_features = 0):
        self.texts = processed_texts
        self.tf_vectorizer = CountVectorizer(max_df = 0.95, min_df=3, max_features=n_features, token_pattern="\w+\$*\d*")
        self.vectors = self.tf_vectorizer.fit_transform(processed_texts).toarray()
        self.ptMc = self.vectors.sum(0)/(self.vectors.sum())
        temp = self.vectors.sum(1)
        temp[temp==0] = 1
        self.ptMd = self.vectors/(temp.reshape((-1,1)))
        self.dlm = self.alpha * self.ptMc + (1 - self.alpha) * self.ptMd
        self.term_matrix = np.zeros((self.dlm.shape[1], 300))
        for i in range(self.dlm.shape[1]):
            self.term_matrix[i] = self.gloveModel[self.tf_vectorizer.get_feature_names()[i]]
        

    def suggest(self, processed_query, num_words = 100, rule="AND", mode="KLD"):
        ptd = self.dlm

        query_index = []
        for i, t in enumerate(processed_query.split()):
            query_index.append(self.tf_vectorizer.vocabulary_[t])
        query_index = np.array(query_index)

        pqd = np.ones(ptd.shape[0])
        for i in query_index:
            pqd *= ptd[:,i]
        
        for i in range(len(self.texts)):
            if rule == 'AND' and len(set(re.sub("\$\d+", "", w) for w in self.texts[i].split()).intersection(set(processed_query.split()))) < len(processed_query.split()):
                pqd[i] = 0.0
            elif rule == 'OR' and len(set(self.texts[i].split()).intersection(set(processed_query.split()))) == 0:
                pqd[i] = 0.0
                
        topdocs = np.array(self.texts)[np.argsort(pqd)[::-1]]
        topscores = np.array(pqd)[np.argsort(pqd)[::-1]]

        for i in range(len(topdocs)):
            if topscores[i] == 0.:
                topdocs = topdocs[:i]
                top_docs = i
                break
            
        if mode == "REL":
            pqd_2 = np.array(pqd)[np.argsort(pqd)[:-top_docs - 1:-1]]
            ptd_2 = np.array(ptd)[np.argsort(pqd)[:-top_docs - 1:-1]]
            ptq = (ptd_2 * pqd_2.reshape((-1,1))).sum(0)
            if ptq.sum() != 0:
                ptq = ptq / ptq.sum() # normalize
            else:
                ptq = np.zeros(ptd.shape[1]) + 1 / ptd.shape[1]

            query_matrix = np.zeros(ptq.shape[0])
            for w in processed_query.split():
                query_matrix[self.tf_vectorizer.vocabulary_[w]] += 1

            cand_query = self.term_matrix @ self.term_matrix.T @ query_matrix
            qp = np.zeros(ptq.shape[0])
            qp[np.argsort(cand_query)[:-num_words-1:-1]] = cand_query[np.argsort(cand_query)[:-num_words-1:-1]]
            qp = qp / qp.sum()

            ptq = (1-self.phi) * ptq + self.phi * qp
            top_words = np.array(self.tf_vectorizer.get_feature_names())[np.argsort(ptq)[:-num_words - 1:-1]]

        if mode == "FRE":
            temp = self.vectors[np.argsort(pqd)[:-top_docs-1:-1]].sum(0)
            temp = temp/temp.sum()
            top_words = np.array(self.tf_vectorizer.get_feature_names())[np.argsort(temp)[:-num_words - 1:-1]]

        if mode == "KLD":
            temp = np.array(self.vectors)[np.argsort(pqd)[:-top_docs - 1:-1]]
            temp = temp.sum(0)/(temp.sum())
            temp = self.alpha * self.ptMc + (1 - self.alpha) * temp
            temp = temp * np.log2(temp/self.ptMc)
            top_words = np.array(self.tf_vectorizer.get_feature_names())[np.argsort(temp)[:-num_words - 1:-1]]

       
        return top_words, np.argsort(pqd)[:]

if __name__ == '__main__':

    import preprocessor
    from preprocessor import read_stopwords, preprocessor

    stopwords = read_stopwords("./dataset/stopwords.en.txt")
    preprocessor = preprocessor(stopwords)

    from sklearn.datasets import fetch_20newsgroups
    corpus = fetch_20newsgroups(shuffle=True, 
                            subset='all',
                            random_state=1, 
                            categories=['alt.atheism', 'comp.graphics',
                                        'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                                        'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 
                                        'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 
                                        'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast'], 
                            remove=('headers', 'footers', 'quotes'))
    texts = corpus.data
    processed_texts = []
    for s in tqdm(texts):
        temp_text = preprocessor.preprocess(s)
        if temp_text == '' or len(temp_text.split())>300 or len(temp_text.split())<3:
            continue
        processed_texts.append(temp_text)

    query = 'atheism'
    processed_query = preprocessor.preprocess(query)

    qe = QueryExpansion(emb_path = './QDTM/glove_embedding/glove.6B.300d.txt')
    qe.initialize(processed_texts, 5000)
    print (processed_query)
    top_words, top_docs = qe.suggest(processed_query, num_words=100, mode="FRE")
    print (top_words, top_docs)
    top_words, top_docs = qe.suggest(processed_query, num_words=100, mode="REL")
    print (top_words, top_docs)
    top_words, top_docs = qe.suggest(processed_query, num_words=100, mode="KLD")
    print (top_words, top_docs)