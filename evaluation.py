import numpy as np
from numpy.linalg import norm
import preprocessor
from preprocessor import read_stopwords, preprocessor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from palmettopy.palmetto import Palmetto
import re
from collections import Counter
from numpy import dot
from numpy.linalg import norm

from prepareText import *
from qe import loadGloveModel
palmetto = Palmetto("http://localhost:7777/palmetto-webapp/service/")
gloveModel = loadGloveModel("./QDTM/glove_embedding/glove.6B.300d.txt")

def getDocTopicDistribtuion(path):
    doc_word = np.zeros((vectors.shape[0], feature))
    with open(path+'.txt') as file:
        for line in file.readlines():
            temp = line.strip().split()
            if temp[0] == 'd' or int(temp[0]) <= target-1:
                continue
            doc_word[int(temp[0])-target, int(temp[1])] += 1

    topic_word = np.zeros((1000, feature))
    with open(path + '_nzw.txt', 'r') as file:
        z = 0
        for line in file.readlines():
            w = 0
            for n in line.strip().split():
                topic_word[z,w] = n
                w += 1
            z += 1
            
    pzw = topic_word/topic_word.sum(0)
    results = (doc_word @ pzw.T)

    temp = results + 1e-20
    zd = temp / temp.sum(1).reshape((temp.shape[0], 1))
    return zd

def getTopicWordDistribtuion(path):
    zw = np.zeros((1000, feature))
    with open(path, 'r') as file:
        z = 0
        for line in file.readlines():
            w = 0
            for n in line.strip().split():
                zw[z,w] = n
                w += 1
            z += 1
    return zw[:z]

def getAccuracy(X, Y):
    clf = LogisticRegression(max_iter=500)
    scores = cross_val_score(clf, X, Y, scoring="accuracy", cv=5)
    return np.mean(scores)

def getPrecisionAtK(X, Y):
    label_count = Counter(Y)
    tru = np.asarray(Y)
    p_results = []
    for topic_index in range(target):
        top_index = np.argsort(X[:, topic_index])[::-1][:label_count[topic_index]]
        precision = (tru[top_index] == topic_index).sum()/label_count[topic_index]
        # print(precision)
        p_results.append(precision)
    return np.mean(p_results)

def getCoherence(zw, vobs):
    local_coherence = []
    for z in zw[:target]:
        top_words = [re.sub("\$\d*", "", vobs[i]) for i in z.argsort()[:-10 - 1:-1]]
        print (top_words)
        local_coherence.append(palmetto.get_coherence(top_words))
    return np.mean(local_coherence)

def getDiversity(zw, vobs, type_tracker):
    tp_index = []
    diversity = []
    for t in range(target):
        topic_words = []
        for i, z in enumerate(zw):
            percent = (zw.sum(1) / zw.sum())[i]*100
            top_words = [re.sub("\$\d*", "", vobs[i]) for i in z.argsort()[:-10 - 1:-1]]
            if (percent > 0.5 and type_tracker[i] == t):
                print ("{0:.2f}%".format(percent) , type_tracker[i], top_words)
                topic_words.append([re.sub("\$\d*", "", vobs[i]) for i in z.argsort()[:-25 - 1:-1]])
                tp_index.append(i)
                
        ave_diff = []
        for topic in topic_words:
            other_topic = [i for i in topic_words if i != topic]
            other_topic = [w for i in other_topic for w in i]
            diff = len(set(topic)-set(other_topic))/25
            ave_diff.append(diff)
        diversity.append(np.mean(ave_diff)) 
    return np.mean(diversity)

def getCohesion(parent_zw, sub_zw, vobs):
    overall_topic = parent_zw[:target] + 0.5
    overall_topic = overall_topic / overall_topic.sum(1).reshape((overall_topic.shape[0], 1))
    subtopics = sub_zw + 0.5
    subtopics = subtopics / subtopics.sum(1).reshape((subtopics.shape[0], 1))

    cohesion = []
    for i_topic, topic in enumerate(overall_topic):
        topic_embedding = np.zeros(300)
        for index in topic.argsort()[: -10 - 1: -1]:
            topic_embedding += gloveModel[vobs[index]] * topic[index]
        ave = []
        for i_subtopic, subtopic in enumerate(subtopics):
            percent = (sub_zw.sum(1) / sub_zw.sum())[i_subtopic]*100
            if (percent > 0.5 and type_tracker[i_subtopic] == i_topic):
                subtopic_embedding = np.zeros(300)
                for index in subtopic.argsort()[: -10 - 1: -1]:
                    subtopic_embedding += gloveModel[vobs[index]] * subtopic[index]
                
                cos_sim = dot(topic_embedding, subtopic_embedding)/(norm(topic_embedding)*norm(subtopic_embedding))
                ave.append(cos_sim)
    #             print (i_topic, i_subtopic, cos_sim)
        print ('Topic ' + str(i_topic) + ' cohesion %.2f' % np.mean(ave))
        cohesion.append(np.mean(ave))
    return np.mean(cohesion)

if __name__ == '__main__':
    stopwords = read_stopwords("./dataset/stopwords.en.txt")
    processor = preprocessor(stopwords)

    dataset = "tagmynews"
    assert dataset == "20news" or dataset == "tagmynews" or dataset == "searchsnippets"

    if dataset == "20news":
        processed_texts, labels = get20News(processor)
        target = 16

    if dataset == "tagmynews":
        processed_texts, labels = getTagmynews(processor)
        target = 5

    if dataset == "searchsnippets":
        processed_texts, labels = getSearchsnippets(processor)
        target = 8

    n_features = 5000
    vectorizer = CountVectorizer(min_df=3, max_features=n_features)
    vectors = vectorizer.fit_transform(processed_texts).toarray()
    feature = len(vectorizer.get_feature_names())

    acc = []
    p_k = []

    path = './QDTM/results/QDTM-' + dataset + '-parent'
    # document classification measure
    X = getDocTopicDistribtuion(path)
    Y = labels
    accnracy = getAccuracy(X, Y)
    print ("acc:", accnracy)
    
    # document retrieval measure
    precisionAtK = getPrecisionAtK(X, Y)
    print ("p@k:", precisionAtK)

    # coherence measure
    path = './QDTM/results/QDTM-' + dataset + '-parent_nzw.txt'
    zw = getTopicWordDistribtuion(path)
    vobs = []
    with open('./QDTM/input/vobs_' + dataset + '_test.txt', 'r') as file:
        for line in file.readlines():
            vobs.append(line.strip())
    coherence = getCoherence(zw, vobs)
    print ('ave coherence', coherence)

    # diversity measure
    path = './QDTM/results/QDTM-' + dataset + '-sub.txt'
    type_tracker = {}
    with open(path) as file:
        for line in file.readlines():
            temp = line.strip().split()
            if temp[0] == 'd' or int(temp[0]) <= target-1:
                continue
            type_tracker[int(temp[-2])] = int(temp[-1])
    path = './QDTM/results/QDTM-' + dataset + '-sub_nzw.txt'
    zw = getTopicWordDistribtuion(path)
    vobs = []
    with open('./QDTM/input/vobs_' + dataset + '_test.txt', 'r') as file:
        for line in file.readlines():
            vobs.append(line.strip())
    diversity = getDiversity(zw, vobs, type_tracker)
    print ('ave diversity:', diversity)

    # cohesion measure
    path = './QDTM/results/QDTM-' + dataset + '-parent_nzw.txt'     
    parent_zw = getTopicWordDistribtuion(path)
    path = './QDTM/results/QDTM-' + dataset + '-sub_nzw.txt'
    sub_zw = getTopicWordDistribtuion(path)
    cohesion = getCohesion(parent_zw, sub_zw, vobs)
    print ('ave cohesion:', cohesion)


    
