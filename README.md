### Sentiment Analysis
The idea here is to extract a high level "Positive" or "Negative" sentiment from a given piece of text. This can be useful in predicting customer churn for instance by analysing email interactions of ongoing support queries. *The project is still an experiment*.

At the moment, I am using gensim's [doc2vec](https://rare-technologies.com/doc2vec-tutorial/) library based on Mikolov and Le's [paper](https://arxiv.org/pdf/1405.4053v2.pdf). I am training a doc2vec model using a dataset of movie reviews from imdb and then using simple logistic regression to predict a crude positive/negative probability. I combined the test and train dataset of phrases from imdb (since other experiments have already proven good accuracy). The data was already curated from my source, but I removed stopwords using nltk. Still experimenting with different d2v models but at the moment it seems like pv-dbow gives good results for this dataset (dm=0). Negative sampling is turned off (hs=1) and this seems to make some difference. Perhaps removing the stopwords compromises the context I am not sure yet (but pv-dbow should ignore context words already?).

### Install and Run
- Install Anaconda python3 from [here](https://www.anaconda.com/download/) if you don't already have it
- Fork or clone the repo `git clone https://github.com/kramamur/sentiment-analysis.git`
- Dependencies - `conda install gensim nltk numpy scikit-learn` (`numpy` and `scikit` come with conda bjic)
- Download nltk data - `nltk.download('punkt')` and `nltk.download('stopwords')`
- Train the model - `python train.py`
- Insert your favorite text into `infer.txt`- Yelp reviews give fairly accurate results
- Predict - `python sentiment.py`

