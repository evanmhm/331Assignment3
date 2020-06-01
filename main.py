import numpy as np
from collections import Counter
import math
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import csv
import re

def main():
    training = []
    with open("trainingSet.txt", 'r') as file:
        line = file.readline()
        i = 0
        while line:
            training.append(line.split('\t'))
            line = file.readline()
            training[i][1] = int(training[i][1][1])
            i += 1

    testing = []
    with open("testSet.txt", 'r') as file:
        line = file.readline()
        i = 0
        while line:
            testing.append(line.split('\t'))
            testing[i][1] = int(testing[i][1][1])
            line = file.readline()
            i += 1

    print(training)
    print(testing)


    # this vectorizer will skip stop words
    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text,
        max_features=4000,
        min_df=30
    )

    # fit the vectorizer on the text
    vectorizer.fit([i[0] for i in training])

    # get the vocabulary
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
    pos_text = []
    neg_text = []
    pos_count = 0
    neg_count = 0
    for review in training:
        if (review[1] == 1):
            for word in list(set(vocabulary) & set(clean_text(review[0]).split())):
                pos_text.append(word)
            pos_count += 1
        else:
            for word in list(set(vocabulary) & set(clean_text(review[0]).split())):
                neg_text.append(word)
            neg_count += 1

    pos_dict = Counter(pos_text)
    neg_dict = Counter(neg_text)
    prob_pos = pos_count/(pos_count+neg_count)
    prob_neg = neg_count/(pos_count+neg_count)


    correct = naive_bayes([i[0] for i in training], pos_dict, neg_dict, pos_text, neg_text, prob_pos, prob_neg, [i[1] for i in training], vocabulary, 1, 0)
    print("Testing Accuracy: ", correct)

    # predictions = naive_bayes_predict(imdb_data['review'][40000:50000], pos_dict, neg_dict, pos_text, neg_text, prob_pos, prob_neg, vocabulary, 1.0)
    # with open("test-predictions3.csv", 'w') as file:
    #     for p in predictions:
    #         file.write(str(p)+"\n")

def naive_bayes(reviews, pos_dict, neg_dict, pos_text, neg_text, prob_pos, prob_neg, labels, vocabulary, alpha, offset):
    correct = 0
    for i, review in enumerate(reviews):
        conditional_pos = 0.
        conditional_neg = 0.
        for word in clean_text(review).split():

            conditional_pos += math.log(((pos_dict[word]+alpha)/(len(pos_text)+len(vocabulary)*alpha)) * prob_pos)
            conditional_neg += math.log(((neg_dict[word]+alpha)/(len(neg_text)+len(vocabulary)*alpha)) * prob_neg)

        if ((conditional_pos > conditional_neg and labels[i+offset] == 1) or (conditional_pos < conditional_neg and labels[i+offset] == 0)):
            correct += 1

    return correct

def naive_bayes_predict(reviews, pos_dict, neg_dict, pos_text, neg_text, prob_pos, prob_neg, vocabulary, alpha):
    predictions = []
    for i, review in enumerate(reviews):
        conditional_pos = 0.
        conditional_neg = 0.
        for word in clean_text(review).split():
            conditional_pos += math.log(((pos_dict[word]+alpha)/(len(pos_text)+len(vocabulary)*alpha)) * prob_pos)
            conditional_neg += math.log(((neg_dict[word]+alpha)/(len(neg_text)+len(vocabulary)*alpha)) * prob_neg)

        if (conditional_pos > conditional_neg):
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions

def clean_text(text):
    print(text)
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    #pattern = r'[^a-zA-z0-9\s]'
    #text = re.sub(pattern, '', text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text



if __name__ == '__main__':
    main()
