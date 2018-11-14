# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random
import random
from random import shuffle

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

import logging
import sys
import os

# Setup logger
log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

# Globals`
SAMPLE_SIZE = 50000
SPLIT_SIZE  = 25000
DIM_SIZE    = 500
MODEL_FILE  = './model.d2v'

# Doc2Vec model using PV-DBOW
def train():
    
    # PV-DBOW
    model = Doc2Vec(dm=0, min_count=0, window=10, vector_size=DIM_SIZE, hs=1, epoch=20, sample=0.0, negative=5, workers=5)

    # Convert the sources into d2v TaggedDocument
    sources = {'neg.txt':'NEG', 'pos.txt':'POS'}

    sentences = []
    for source, prefix in sources.items():
        with utils.smart_open(source) as fin:
            for item_no, line in enumerate(fin):
                sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))

    log.info('Building vocabulary...')
    model.build_vocab(sentences)

    alpha = 0.025
    min_alpha = 0.001
    num_epochs = 20
    alpha_delta = (alpha - min_alpha) / num_epochs

    log.info('Training doc2vec model...')
    for epoch in range(num_epochs):
        model.alpha = alpha
        model.min_alpha = alpha
        shuffle(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        alpha -= alpha_delta
        
    log.info('Saving to model file...')
    model.save(MODEL_FILE)

# At some point we'll switch to inferring from text read from a file
def infer():
    log.info('Loading model file...')
    model = Doc2Vec.load(MODEL_FILE)

    docvecs = numpy.zeros((SAMPLE_SIZE, DIM_SIZE))
    labels = numpy.zeros(SAMPLE_SIZE)

    log.info('Preparing training data...')
    for count in range(SPLIT_SIZE):
        docvecs[count] = model.docvecs['NEG_' + str(count)]
        docvecs[SPLIT_SIZE + count] = model.docvecs['POS_' + str(count)]
        labels[count] = 0
        labels[SPLIT_SIZE + count] = 1


    log.info('Fitting classifier...')
    clf = LogisticRegression()
    clf.fit(docvecs, labels)

    # Checking inference with one sample
    #pred_sam = "fantastic pleasurable comedy attractive"
    #pred_sam = "The best software I have ever used in my life"
    pred_sam = "Well everything looks good so far I am just going to have physical therapy - no surgeries Yay I am attaching a rough outline so you can see what I think will work for completing this particular app Let me know what you think Susan"
    log.info('Predicting on: %s' % pred_sam)
    #pred = clf.predict(model.infer_vector(pred_sam.split(" ")).reshape(1, -1))
    #log.info(pred)
    pred_lbl = clf.predict_proba(model.infer_vector(pred_sam.split(" ")).reshape(1, -1))
    percent_neg = str('%.2f' % (pred_lbl[0,0]*100))
    percent_pos = str('%.2f' % (pred_lbl[0,1]*100))

    log.info(pred_lbl)
    log.info(clf.classes_)
    if percent_neg > percent_pos: log.info('Sentiment: Negative ' + percent_neg + '%')
    else: log.info('Sentiment: Positive ' + percent_pos + '%')

def main():
    if len(sys.argv) == 2:
        if sys.argv[1] == 'train':
            train()
        elif sys.argv[1] == 'infer':
            infer()
    else:
        print('')
        print('Usage: ./sentiment.py <train>|<infer>')
        print('')

        # Train d2v model if not already
        if not os.path.isfile("./model.d2v"):
            log.info('Model file not found...')
            train()
        else:
            log.info('Model file found...')
        infer()

if __name__ == "__main__": main()