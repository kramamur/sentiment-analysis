### Sentiment Analysis
The idea here is to extract a high level "Positive" or "Negative" sentiment from a given piece of text. This can be useful in predicting customer churn for instance by analysing email interactions of ongoing support queries. *The project is still an experiment*.

At the moment, I am using gensim's [doc2vec](https://rare-technologies.com/doc2vec-tutorial/) library based on Mikolov and Le's paper[1]. I am training a doc2vec model using a training dataset and then using simple logistic regression to predict a 2-way coarse-grained positive/negative probability. I combined the test and train dataset of phrases from imdb and added cornell's sentiment polarity v2.0 dataset to the mix. The data was already curated from my source, but I removed stopwords using nltk. Still experimenting with different d2v models but at the moment it seems like pv-dbow gives good results for this dataset (dm=0). Negative sampling is turned off (hs=1) and this seems to make some difference. Perhaps removing the stopwords compromises the context I am not sure yet (but pv-dbow should ignore context words already?).

### Install and Run
- Install Anaconda python3 from [here](https://www.anaconda.com/download/) if you don't already have it
- Fork or clone the repo `git clone https://github.com/kramamur/sentiment-analysis.git`
- Dependencies - `conda install gensim nltk numpy scikit-learn` (`numpy` and `scikit` come with conda bjic)
- Download nltk data - `nltk.download('punkt')` and `nltk.download('stopwords')`
- Train the model - `python train.py`
- Insert your favorite text into `infer.txt`- Yelp reviews give fairly accurate results
- Predict - `python sentiment.py`

---
```
Citations
1. Distributed Representations of Sentences and Documents
   tmikolov@,qlv@google.com, (https://arxiv.org/pdf/1405.4053v2.pdf)
2. Large Movie Review Dataset, amaas@cs.stanford.edu (http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.bib)
3. Cornell sentiment polarity dataset, v2.0 (http://www.cs.cornell.edu/people/pabo/movie-review-data/)
```
