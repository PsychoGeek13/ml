import sys

def SDGRegressionExample():
    import numpy as np
    from sklearn.datasets import load_boston
    from sklearn.linear_model import SGDRegressor
    from sklearn.cross_validation import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_validation import train_test_split
    data = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(data.data,data.target)
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)
    X_test = X_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test)
    regressor = SGDRegressor(loss='squared_loss')
    scores = cross_val_score(regressor, X_train, y_train, cv=5)
    print 'Cross validation r-squared scores:', scores
    print 'Average cross validation r-squared score:', np.mean(scores)
    regressor.fit_transform(X_train, y_train)
    print 'Test set r-squared score', regressor.score(X_test, y_test)

def categoricalFeatures():
    from sklearn.feature_extraction import DictVectorizer
    onehot_encoder= DictVectorizer()
    instances=[
        {'city':'New York'},
        {'city':'San Francisco'},
        {'city': 'Chapel Hill'}
    ]
    print onehot_encoder.fit_transform(instances).toarray()

def lemmatize(token, tag):
    from nltk.stem.wordnet import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    if tag[0].lower() in ['n', 'v']:
        return lemmatizer.lemmatize(token, tag[0].lower())
    return token

def bagOfWordsModel():
    #simple vectorization example
    from sklearn.feature_extraction.text import CountVectorizer
    corpus = [
        'UNC played Duke in basketball',
        'Duke lost the basketball game',
        'I ate a sandwich'
    ]
    vectorizer = CountVectorizer()
    print vectorizer.fit_transform(corpus).todense()
    print vectorizer.vocabulary_
    #viewing the euclidean distance between features vectors
    from sklearn.metrics.pairwise import   euclidean_distances
    counts = [[0, 1, 1, 0, 0, 1, 0, 1],[0, 1, 1, 1, 1, 0, 0, 0],[1, 0, 0, 0, 0, 0, 1, 0]]
    print ('Distances between 1st and 2nd documents:',euclidean_distances(counts[0],counts[1]))
    print ('Distances between 1st and 3rd documents:',euclidean_distances(counts[0],counts[2]))
    print ('Distances between 2nd and 3rd documents:',euclidean_distances(counts[1],counts[2]))
    #filtering stop words
    vectorizer = CountVectorizer(stop_words='english')
    print vectorizer.fit_transform(corpus).todense()
    print vectorizer.vocabulary_
    # stemming and lemmatization
    """stemming =  removes all patterns of characters that appear to be affixes,resulting in a token that is not necessarily a valid word.  and lemmatization = finding the roots of a word ex jumping becomes jump
     Lemmatization frequently
    requires a lexical resource, like WordNet, and the word's part of speech. Stemming
    algorithms frequently use rules instead of lexical resources to produce stems and can
    operate on any token, even without its context.

    stem mesh hayfara2 been gathering as a noun and gathering as a verb w hay2lebhom homma el etneen le gather
    lemmatization bey7tag el context 3ashan yeraga3 el verbs lel root w el nouns zay ma heya
    stemming uses rules to remove characters that appear as zyadat fa momken yebawaz kelma ex: was>= wa, lemmatization uses el context
    """
    from nltk import word_tokenize
    from nltk.stem import PorterStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk import pos_tag
    wordnet_tags = ['n','v']

    corpus = [
 'He ate the sandwiches',
 'Every sandwich was eaten by him'
 ]
    stemmer = PorterStemmer()
    print 'Stemmed:', [[stemmer.stem(token) for token in word_tokenize(document)] for document in corpus]

    lemmatizer = WordNetLemmatizer()
    tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
    print 'Lemmatized:', [[lemmatize(token, tag) for token, tag in
                           document] for document in tagged_corpus]


    #TF-IDF => the frequencies of the tokens are put into considerations
    from sklearn.feature_extraction.text import  CountVectorizer
    corpus = ['The dog ate a sandwich, the wizard transfigured a sandwich, and I ate a sandwich']
    vectorizer = CountVectorizer(stop_words='english')
    """The binary argument is defaulted to False,so instead of a binary representation
     we get a number of occurences for each token"""
    print vectorizer.fit_transform(corpus).todense()


def LogisticRegressionSMSFilteringExample():
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model.logistic import LogisticRegression
    from sklearn.cross_validation import train_test_split, cross_val_score
    df = pd.read_csv('C:/Users/Ahmad/Documents/Mastering ML with Scikitlearn/ml/DataSets/smsspamcollection/SMSSpamCollection', delimiter='\t',header=None)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1],df[0])
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    for i in xrange(0,5):
        print X_test_raw.values.tolist()[i],"\r\n Classification: ", predictions[i]
    #for i, prediction in enumerate(predictions[:5]):
     #   print 'Prediction: %s. Message: %s' % (prediction, X_test_raw[i])


def kaggleTitanic():
    import pandas as pd
    from sklearn.linear_model.logistic import LogisticRegression
    from sklearn.cross_validation import train_test_split, cross_val_score
    df = pd.read_csv('C:/Users/Ahmad/Documents/Mastering ML with Scikitlearn/ml/DataSets/Titanic/train.csv', delimiter='\t',header=None)
    #X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1],df[0])
    features=df.drop(['Survived','PassengerId','Name'], axis=1)


if __name__ == '__main__':
    LogisticRegressionSMSFilteringExample()
    #SDGRegressionExample()
    #if len(sys.argv) != 3:
        #print('usage: knapsack.py [the Path to the file containing the tasks ] [max allowed number of Days]')
        #sys.exit(1)
    #tasksFileName = sys.argv[1]
    #with open(tasksFileName) as tasksFile:
        #tasksLines = tasksFile.readlines()

    #maxNumberOfDays = int(sys.argv[2])
    #tasks = [map(int, taskLine.split()) for taskLine in tasksLines[1:]]

    #bestValue, reconstruction = knapsack(tasks, maxNumberOfDays)

    #print('The max fees total is: {0}'.format(bestValue))
    #print('The tasks are as follows:')
    #for fee, day in reconstruction:
     #   print('fee: {0}, day: {1}'.format(fee, day))