# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# others
from random import shuffle
import glob
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

# globals
from globals import DIM_SIZE
from globals import MODEL_FILE
from globals import META_FILE
from globals import log
from globals import sys
from globals import logging
from globals import pickle
from globals import transform
from globals import update_progress

# Doc2Vec PV-DBOW
def train():    
    
    model = Doc2Vec(dm=0, min_count=0, window=10, vector_size=DIM_SIZE, hs=1, epoch=20, sample=0.0, negative=5, workers=5)

    # Convert the sources into d2v TaggedDocument
    log.info('Compiling data sources...')
    exDict = get_srcs()
    sources = exDict["sources"]
    tot_size = exDict["neg_size"] + exDict["pos_size"]

    sentences = []
    neg_count, pos_count, prefix_count = 0, 0, 0
    log.info('Processing data sources...')
    for source, prefix in sources.items():
        if prefix == 'NEG': prefix_count = neg_count
        elif prefix == 'POS': prefix_count = pos_count
        else:
            log.error('Unknown prefix found: '+prefix+'. Exiting...')
            sys.exit()
        with utils.smart_open(source) as fin:
            for line_no, line in enumerate(fin):
                words = transform(utils.to_unicode(line))
                sentences.append(TaggedDocument(words, [prefix + '_%s' % prefix_count]))
                prefix_count += 1
                update_progress(float((neg_count+pos_count+line_no+1)/tot_size))
        if prefix == 'NEG': neg_count = prefix_count
        elif prefix == 'POS': pos_count = prefix_count

    log.info('Building vocabulary...')
    model.build_vocab(sentences)

    alpha = 0.025
    min_alpha = 0.001
    num_epochs = 20
    alpha_delta = (alpha - min_alpha) / num_epochs

    log.info('Training doc2vec model...')
    log.setLevel(logging.WARNING)
    for epoch in range(num_epochs):
        update_progress(float((epoch+1)/num_epochs))
        model.alpha = alpha
        model.min_alpha = alpha
        shuffle(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        alpha -= alpha_delta
    
    log.setLevel(logging.INFO)

    # score
    score(model, exDict["neg_size"], exDict["pos_size"])

    log.info('Saving to model file...')
    try:
        model.save(MODEL_FILE)
    except:
        log.error('Error saving model')
        sys.exit()
    try:
        log.info('Saving model metadata...')
        pickle.dump(exDict, open(META_FILE, 'wb'))
        log.info('Saved '+META_FILE)
    except:
        log.error('Error saving model metadata')
        sys.exit()

# get all data sources
def get_srcs():
    
    sources = {}
    pos_size = 0
    neg_size = 0
    dat_files = glob.glob('./datasets/*.dat')
    log.info('Found ' + str(len(dat_files)) + ' data files: '+str(dat_files))

    for dat_file in dat_files:
        if "neg_" in dat_file: 
            sources[dat_file] = "NEG"
            neg_size += len(open(dat_file).readlines())  # A necessary evil for now
        elif "pos_" in dat_file:
            sources[dat_file] = "POS"
            pos_size += len(open(dat_file).readlines())  # A necessary evil for now

    log.info('Sample Size:' + str(neg_size+pos_size) + ' -ve:' + str(neg_size) + ' +ve:' + str(pos_size))

    return {"sources":sources, "neg_size":neg_size, "pos_size":pos_size}

# let's score the accuracy
def score(model, neg_size, pos_size):
#def score():
    log.info('Scoring with LogisticRegression...')

    # we'll use 80/20 for train/test
    ntrain_size = int(neg_size*0.8)
    ptrain_size = int(pos_size*0.8)

    ntest_size = neg_size - ntrain_size
    ptest_size = pos_size - ptrain_size

    # initialize the arrays    
    train_docvecs = numpy.zeros((ntrain_size+ptrain_size, DIM_SIZE))
    train_labels = numpy.zeros(ntrain_size+ptrain_size)

    test_docvecs = numpy.zeros((ntest_size+ptest_size, DIM_SIZE))
    test_labels = numpy.zeros(ntest_size+ptest_size)

    for count in range(ntrain_size+ntest_size):
        if count < ntrain_size:
            train_docvecs[count] = model.docvecs['NEG_' + str(count)]
            train_labels[count] = 0
        else:
            test_docvecs[count - ntrain_size] = model.docvecs['NEG_' + str(count)]
            test_labels[count - ntrain_size] = 0

    for count in range(ptrain_size+ptest_size):
        if count < ptrain_size:
            train_docvecs[ntrain_size + count] = model.docvecs['POS_' + str(count)]
            train_labels[ntrain_size + count] = 1
        else:
            test_docvecs[ntest_size + count - ptrain_size] = model.docvecs['POS_' + str(count)]
            test_labels[ntest_size + count - ptrain_size] = 1

    log.info('Fitting classifier...')
    clf = LogisticRegression()
    #clf = MLPClassifier()
    clf.fit(train_docvecs, train_labels)

    log.info('Score: '+str(clf.score(test_docvecs, test_labels)))


if __name__ == "__main__": train()