# gensim modules
from gensim.models import Doc2Vec

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

# globals
from globals import DIM_SIZE
from globals import MODEL_FILE
from globals import META_FILE
from globals import log
from globals import sys
from globals import pickle
from globals import transform
from globals import update_progress

def infer():

    model = None
    exDict = None
    try:
        log.info('Loading model file...')
        model = Doc2Vec.load(MODEL_FILE)
    except:
        log.error('Error loading '+MODEL_FILE+'. Try running train.py')
        sys.exit()
    try:
        log.info('Loading meta file...')
        exDict = pickle.load(open(META_FILE, 'rb'))
    except:
        log.error('Error loading '+META_FILE+'. Try running train.py')
        sys.exit()

    log.info('Preparing training data...')
    neg_size = exDict["neg_size"]
    pos_size = exDict["pos_size"]
    tot_size = neg_size + pos_size
    log.info('Sample Size:' + str(tot_size) + ' -ve:' + str(neg_size) + ' +ve:' + str(pos_size))

    # initialize the arrays    
    docvecs = numpy.zeros((tot_size, DIM_SIZE))
    labels = numpy.zeros(tot_size)

    for count in range(neg_size):
        docvecs[count] = model.docvecs['NEG_' + str(count)]
        labels[count] = 0

    for count in range(pos_size):
        docvecs[neg_size + count] = model.docvecs['POS_' + str(count)]
        labels[neg_size + count] = 1

    log.info('Fitting classifier...')
    clf = LogisticRegression()
    clf.fit(docvecs, labels)

    # Checking inference with one sample
    filename = 'infer.txt'
    file = open(filename, 'rt')
    text = file.read()
    file.close()

    pred_sam = transform(text)
    log.info('Predicting on: %s' % pred_sam)
    pred_lbl = clf.predict_proba(model.infer_vector(pred_sam).reshape(1, -1))
    percent_neg = str('%.2f' % (pred_lbl[0,0]*100))
    percent_pos = str('%.2f' % (pred_lbl[0,1]*100))

    log.info(pred_lbl)
    log.info(clf.classes_)
    if percent_neg > percent_pos: log.info('Sentiment: Negative ' + percent_neg + '%')
    else: log.info('Sentiment: Positive ' + percent_pos + '%')

if __name__ == "__main__": infer()