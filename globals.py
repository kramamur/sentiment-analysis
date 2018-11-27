# nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# others
import logging
import sys
import string
import pickle

# Setup logger
log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

# Globals
DIM_SIZE    = 152
MODEL_FILE  = './model.d2v'
META_FILE = MODEL_FILE + '.pickle'

# clean and transform
def transform(text):
    # split into words and to lowercase
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]

    # remove punctuation from each word
    table = str.maketrans('','', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # filter out stop words
    #stop_words = set(stopwords.words('english'))
    #words = [w for w in words if not w in stop_words]

    return words

# show a progress bar
def update_progress(progress):

    # lets gets the bar length to be reasonable
    barLength = 50
    status = ""

    # check if we are done
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"

    block = int(round(barLength*progress))
    text = "\rProgress: [{0}] {1}% {2}".format( "="*block + " "*(barLength-block), int(progress*100), status)
    sys.stdout.write(text)
    sys.stdout.flush()
