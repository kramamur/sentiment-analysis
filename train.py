# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# others
from random import shuffle
import glob

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

if __name__ == "__main__": train()