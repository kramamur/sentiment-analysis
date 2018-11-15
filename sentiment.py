# gensim modules
from gensim.models import Doc2Vec

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

# globals
from globals import SAMPLE_SIZE
from globals import SPLIT_SIZE
from globals import DIM_SIZE
from globals import MODEL_FILE
from globals import log
from globals import transform
from globals import update_progress

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
    filename = 'infer.txt'
    file = open(filename, 'rt')
    text = file.read()
    file.close()

    pred_sam = transform(text)
    log.info('Predicting on: %s' % pred_sam)
    #pred = clf.predict(model.infer_vector(pred_sam.split(" ")).reshape(1, -1))
    #log.info(pred)
    #pred_lbl = clf.predict_proba(model.infer_vector(pred_sam.split(" ")).reshape(1, -1))
    pred_lbl = clf.predict_proba(model.infer_vector(pred_sam).reshape(1, -1))
    percent_neg = str('%.2f' % (pred_lbl[0,0]*100))
    percent_pos = str('%.2f' % (pred_lbl[0,1]*100))

    log.info(pred_lbl)
    log.info(clf.classes_)
    if percent_neg > percent_pos: log.info('Sentiment: Negative ' + percent_neg + '%')
    else: log.info('Sentiment: Positive ' + percent_pos + '%')

def main():
    infer()

if __name__ == "__main__": main()