# QDTM - Query-Driven Topic Model

This is the source code for the ACL finding paper: 

* **Z. Fang, Y. He and R. Procter. Query-Driven Topic Model, ACL Findings, 2021.**

Our model is built based on the jave implementation of Gibbs sampler HDP here: https://github.com/arnim/HDP

# Setup:

* Download or clone the repo. Denote the repo location as SOURCE_DIR
* Download and express **glove.6B.zip** from https://nlp.stanford.edu/projects/glove/ to the folder: 
    * SOURCE_DIR/QDTM/glove_embedding*
* You need to create a conda python 3.8 environment and install necessary packages.
* To calculate the coherence of the topics, you will need to install the **Palmetto** package. 
    * The details about Palmetto can be found here: https://github.com/dice-group/Palmetto

# Pre-process data:

* from SOURCE_DIR, run "python prepareText.py" to preprocesse dataset. 
* Processed dataset will be dumped to "./QDTM/input". 
* There will be two dumped files (for 20newsgroup dataset):

    1. data_20news_test.txt
    2. vob_20news_test.txt

# Training the model:

* from SOURCE_DIR/QDTM/src, type "javac QDTM.java" to complie the code
* then, type "java QDTM" to train the model
* the model will output four files to "./QDTM/results":

    1. QDTM-20news-parent.txt       # word parent-topic assignment for each document
    2. QDTM-20news-parent_nzw.txt   # word distribution for each parent-topic 
    3. QDTM-20news-sub.txt          # word sub-topic assignment for each document
    4. QDTM-20news-sub_nzw.txt      # word distribution for each sub-topic 

# evaluate the model:

* from SOURCE_DIR, run "python evaluation.py" to evaluate the performance.
