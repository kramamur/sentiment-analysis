# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random
from random import shuffle

# globals
from globals import SAMPLE_SIZE
from globals import SPLIT_SIZE
from globals import DIM_SIZE
from globals import MODEL_FILE
from globals import log
from globals import logging
from globals import transform
from globals import update_progress

# Doc2Vec PV-DBOW
def train():    
    
    model = Doc2Vec(dm=0, min_count=0, window=10, vector_size=DIM_SIZE, hs=1, epoch=20, sample=0.0, negative=5, workers=5)

    # Convert the sources into d2v TaggedDocument
    sources = {'./datasets/neg.txt':'NEG', './datasets/pos.txt':'POS'}

    sentences = []
    count = 0
    log.info('Processing data sources...')
    for source, prefix in sources.items():
        with utils.smart_open(source) as fin:
            for item_no, line in enumerate(fin):
                count += 1
                update_progress(float(count/SAMPLE_SIZE))
                words = transform(utils.to_unicode(line))
                sentences.append(TaggedDocument(words, [prefix + '_%s' % item_no]))
                
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
    model.save(MODEL_FILE)

def main():
    train()

if __name__ == "__main__": main()