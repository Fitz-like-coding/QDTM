from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

import numpy as np
from tqdm import tqdm

from qe import QueryExpansion
import preprocessor
from preprocessor import read_stopwords, preprocessor
import re
import os

def get20News(processor):
    if not os.path.exists("./dataset/20news.txt"):
        data = fetch_20newsgroups(shuffle=True, 
                                subset='all',
                                random_state=1, 
                                categories=['alt.atheism', 'comp.graphics',
                                            'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                                            'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 
                                            'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 
                                            'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast'], 
                                remove=('headers', 'footers', 'quotes'))

        print ("preparing dataset...")
        processed_texts = []
        labels = []
        for i in tqdm(range(len(data.data))):
            temp_text = processor.preprocess(data.data[i])
            if temp_text == '' or len(temp_text.split())>300 or len(temp_text.split())<3:
                continue
            processed_texts.append(temp_text)
            labels.append(data.target[i])
        with open("./dataset/20news.txt", 'w') as file:
            for i in range(len(processed_texts)):
                file.write(processed_texts[i] + '\t' + str(labels[i]) + '\n')
    else:
        processed_texts = []
        labels = []
        with open("./dataset/20news.txt", 'r') as file:
            for line in file.readlines():
                temp = line.split('\t')
                processed_texts.append(temp[0])
                labels.append(int(temp[1]))
    return processed_texts, labels

def getTagmynews(processor):
    data = []
    targets = []
    with open("./dataset/TagMyNews.txt", "r") as file:
        c = 0
        temp = ''
        for line in tqdm(file.readlines()):
            if c % 8 == 0 or c % 8 == 1:
                temp += line.strip() + ' '
                c += 1
                continue
            elif c % 8 == 2:
                temp = re.sub("u\.s\.", "usa", temp)
                temp = re.sub("\\bu\.s\\b", "usa", temp)
                temp = re.sub("\\bus\\b", "usa", temp)
                data.append(temp.strip())
                temp = ''
                c += 1
            elif c % 8 == 6:
                targets.append(line.strip())
                c += 1
            else:
                c += 1
                continue

    print ("preprocessing dataset...")
    processed_texts = []
    labels = []
    for i in tqdm(range(len(data))):
        temp_text = processor.preprocess(data[i])
        if temp_text == '' or len(temp_text.split())>300 or len(temp_text.split())<3:
            continue
        if (targets[i] != 'us' and targets[i] != 'world'):
            processed_texts.append(temp_text)
            labels.append(["business","entertainment","health","sci_tech","sport"].index(targets[i]))
    labels = np.array(labels)
    return processed_texts, labels

def getSearchsnippets(processor):
    data = []
    targets = []
    with open("./dataset/SearchSnippets.txt", "r") as file:
        for line in tqdm(file.readlines()):
            data.append(line.strip())
    with open("./dataset/SearchSnippets_label.txt", "r") as file:
        for line in tqdm(file.readlines()):
            targets.append(line.strip())

    print ("preprocessing dataset...")
    processed_texts = []
    labels = []
    for i in tqdm(range(len(data))):
        temp_text = processor.preprocess(data[i])
        if temp_text == '' or len(temp_text.split())>300 or len(temp_text.split())<3:
            continue
        processed_texts.append(temp_text)
        labels.append(int(targets[i]) - 1)
    labels = np.array(labels)
    return processed_texts, labels

if __name__ == '__main__':
    stopwords = read_stopwords("./dataset/stopwords.en.txt")
    processor = preprocessor(stopwords)

    dataset = "tagmynews"
    assert dataset == "20news" or dataset == "tagmynews" or dataset == "searchsnippets"

    if dataset == "20news":
        processed_texts, labels = get20News(processor)
        query = ['atheism', 'graphics', 'pc hardware', 'mac hardware', 
         'for sale', 'automobile', 'motorcycles', 'baseball', 
         'hockey', 'encrypt', 'electronics', 'medicine', 
         'space', 'christian', 'guns', 'middle east']
        rule = ['AND'] * 16

    if dataset == "tagmynews":
        processed_texts, labels = getTagmynews(processor)
        query = ['business', 'entertainment', 'health', 'technology', 'sports game']
        rule = ['AND','AND','AND','AND','OR']

    if dataset == "searchsnippets":
        processed_texts, labels = getSearchsnippets(processor)
        query = ['Business', 'Computers', 'Culture Art Entertainment', 'Education Science', 
                'Car Engineering', 'Health', 'Politics Society', 'Sports']
        rule = ['AND','AND','OR','OR','OR','AND','AND','AND']


    n_features = 5000 # use top 5000 vocabulary
    qe = QueryExpansion(emb_path = './QDTM/glove_embedding/glove.6B.300d.txt')
    qe.initialize(processed_texts, n_features = n_features)

    print ("extract concept words for each query...")
    processed_query = []
    for q in query:
        processed_query.append(processor.preprocess(q))
    topdoc_index = []
    expanded_query = []
    for i, q in enumerate(processed_query):
        temp = qe.suggest(q, num_words=100, mode="KLD", rule=rule[i])
        expanded_query.append(temp[0])
        topdoc_index.append(temp[1])

    #get top 10 distinct concept words for each query.
    allConceptWords = [w for e in expanded_query for w in e]
    allqueryWords = [w for q in processed_query for w in q.split()]
    conceptWords = []
    for i in range(len(query)):
        q = expanded_query[i]
        conceptWords.append(' '.join(set([w for w in q if allConceptWords.count(w) == 1][:10] + 
                                            [w for w in processed_query[i].split() if allqueryWords.count(w) == 1]
                                        )))
    for i in range(len(query)):
        print("query:", query[i], "; Concept words:", conceptWords[i])

    #convert data to the format accepted by our model
    data_samples = conceptWords + processed_texts
    vectorizer = CountVectorizer(min_df=3, max_features=n_features)
    vectors = vectorizer.fit_transform(data_samples)
    vobs = vectorizer.get_feature_names()
    with open('./QDTM/input/data_' + dataset + '_test.txt', 'w', encoding='utf-8') as file:
        for line in tqdm(vectors):
            doc = []
            doc.append(str(len(line.indices)))
            count = 0 
            for index, value in zip(line.indices, line.data):
                doc.append(str(index)+":"+str(value))
                count += 1
            assert len(line.indices) == count
            doc = ' '.join(doc)
            file.write(doc + '\n')
    with open('./QDTM/input/vobs_' + dataset + '_test.txt', 'w', encoding='utf-8') as file:
        for v in vobs:
            file.write(v + '\n')