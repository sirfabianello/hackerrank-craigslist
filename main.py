# Enter your code here. Read input from STDIN. Print output to STDOUT
import sys
import json

if __name__ == '__main__':

    # Reading training data
    categ = [] # list of categories
    heads = [] # list of text samples
    jsonFlag = 0 # for the first line (not a json)
    
    f = open('training.json', 'rt')
    for line in f:
        if jsonFlag == 0:
            N = float(line)
            jsonFlag = 1
        else :
            data = json.loads(line) # as a dict
            categ.append(data['category']) 
            heads.append(data['city']+' '+ data['section']+' '+ data['heading'])               
    f.close()
    
    # Creating bag-of-words
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer(stop_words='english', lowercase=True) # tokenizing and filtering of stopwords
    X_train_counts = count_vect.fit_transform(heads)
    
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    # Trainig classifier
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB().fit(X_train_tfidf, categ)
    
    # Reading input data
    heads_to_predict = []
    jsonFlag = 0
    
    #fp = open('sample-test.in.json', 'rt')
    #for line in fp:
    for line in sys.stdin:
        if jsonFlag == 0:
            N2 = float(line)
            jsonFlag = 1
        else :
            data = json.loads(line) 
            heads_to_predict.append(data['city']+' '+ data['section']+' '+ data['heading'])
            
    #fp.close()
    
    # Predicting categories
    X_new_counts = count_vect.transform(heads_to_predict)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    
    predicted = clf.predict(X_new_tfidf)
    
    # Printing to sdtout
    for cat in predicted:
        print cat 