import nltk

nltk.download('movie_reviews')
nltk.download('stopwords')

from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.stem import PorterStemmer
import re

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
punctuation = [".", '!', ',', '?', "'", ';', '(', ')', '/', ':']

neg_review_ids = list(movie_reviews.fileids('neg'))
pos_review_ids = list(movie_reviews.fileids('pos'))

neg_train = list()
pos_train = list()
neg_test = list()
pos_test = list()
total_train = list()


#negative review words split up
for i in range(0,700):  #70% of data for training
    neg_review = movie_reviews.words(neg_review_ids[i])
    for j in neg_review:
        match = re.match('[0-9]+', j)
        if j not in punctuation and j not in stop_words and not match:
                neg_train.append(stemmer.stem(j))
                total_train.append(stemmer.stem(j))

for i in range(700,1000):   #30% of data for testing
    neg_review = movie_reviews.words(neg_review_ids[i])
    for j in neg_review:
        match = re.match('[0-9]+', j)
        if j not in punctuation and j not in stop_words and not match:
            neg_test.append(stemmer.stem(j))

#positive review words split up
for i in range(0,700):  #70% of data for training
    pos_review = movie_reviews.words(pos_review_ids[i])
    for j in pos_review:
        match = re.match('[0-9]+', j)
        if j not in punctuation and j not in stop_words and not match:
            pos_train.append(stemmer.stem(j))
            total_train.append(stemmer.stem(j))

for i in range(700,1000):   #30% of data for testing
    pos_review = movie_reviews.words(pos_review_ids[i])
    for j in pos_review:
        match = re.match('[0-9]+', j)
        if j not in punctuation and j not in stop_words and not match:
            pos_test.append(stemmer.stem(j))


#analyze reviews in training set (Naive Bayes Classifier):
#probability of each class
p_neg = (0.7*len(neg_review_ids))/(len(neg_review_ids)+len(pos_review_ids))
p_pos = (0.7*len(neg_review_ids))/(len(neg_review_ids)+len(pos_review_ids))

#total vocabulary count
total_freq = FreqDist(total_train)


#conditional probabilities for each word in training set
#class negative:
neg_freq = FreqDist(neg_train)
neg_prob = list()
neg_prob_words = list()

for i in neg_freq: #done for all words in training set (neg)
    neg_prob.append((neg_freq[i]+1)/(len(neg_train)+len(total_freq)))
    neg_prob_words.append(i)

#class positive:
pos_freq = FreqDist(pos_train)
pos_prob = list()
pos_prob_words = list()

for i in pos_freq: #done for all words in training set (pos)
    pos_prob.append((pos_freq[i]+1)/(len(pos_train)+len(total_freq)))
    pos_prob_words.append(i)

print('done for the more official part.', end = "\n")
#anything past here is experimental since final probability values are too small to be registered

#taking the top 20 words for each to use for testing?:
top_neg_prob = list()
top_pos_prob = list()
top_neg_prob_words = list()
top_pos_prob_words = list()

temp_neg_prob = list()
temp_pos_prob = list()

#copying over probs temporarily to be manipulated:
for i in neg_prob:
    temp_neg_prob.append(i)

for i in pos_prob:
    temp_pos_prob.append(i)



#classifying each word in testing set:
#(just testing currently with random single sample)
n_prob = p_neg
p_prob = p_pos
pos_test_r = list()

pos_review = movie_reviews.words(pos_review_ids[900])
for j in range(500):
    match = re.match('[0-9]+', pos_review[j])
    if pos_review[j] not in punctuation and pos_review[j] not in stop_words and not match:
        pos_test_r.append(stemmer.stem(pos_review[j]))

for i in pos_test_r:
    for j in range(len(neg_prob)):
        if i == neg_prob_words[j]:
            n_prob = n_prob * neg_prob[j]
    for k in range(len(pos_prob)):
        if i == pos_prob_words[k]:
            p_prob = p_prob * pos_prob[k]

print('done for experimental part:')
print('Negative class probaility for sample:', end = " ")
print(n_prob)
print('Positive class probability for sample:',end = " ")
print(p_prob)

print('conclusion thus far:')
print('python will not register tiny float numbers past a point, so values are 0.0')
