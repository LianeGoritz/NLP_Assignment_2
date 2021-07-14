import nltk

nltk.download('brown')

from nltk.corpus import brown
from nltk import bigrams
from nltk import trigrams
from nltk import FreqDist
from nltk import ConditionalFreqDist
from nltk import ConditionalProbDist
from nltk import MLEProbDist
import random

#generate random words using Shannon Visualization Method: bigrams
words = brown.words()

def bigram_sentence_generator(words):
    bigram_words = list(bigrams(words))
    start_words = random.choice(bigram_words)
    sentence = list()

    punctuation = ['.', '?', '!', ',', "``", ";"]
    check = False

    for i in punctuation:
        while check == False:
            if start_words[0] == i:
                start_words = random.choice(bigram_words)
            else:
                check = True

    sentence.append(start_words[0])
    sentence.append(start_words[1])

    #count of frequency of certain words with other words (bigram):
    bigram_counts = ConditionalFreqDist(bigram_words)


    #calculating the probabilities for all bigrams:
    bigram_probs = ConditionalProbDist(bigram_counts, MLEProbDist) #P(w_i|w_i-1)

    #add words to list based on maximum probabilities for the most recent word added to sentence:
    end_check = False
    count = 0
    while end_check == False:
        punct = False
        for p in punctuation:
            if sentence[-1] == p:
                punct = True
                if sentence[-1] != ',':
                    count = count + 1 #counts up to a maximum of three sentences after each ending sentence punctuation token

        if punct == False:
            next_word = bigram_probs[sentence[-1]].max()
            sentence.append(next_word)

        else:
            punct_check = False
            new_start_words = random.choice(bigram_words)
            while punct_check == False:
                for p in punctuation:
                    if new_start_words[0] == p:
                        new_start_words = random.choice(bigram_words)
                    else:
                        punct_check = True
            sentence.append(new_start_words[0])
            sentence.append(new_start_words[1])


        if sentence[-1] == "''" or count == 3:
            end_check = True

    for i in sentence:
        print(i, end = " ")
    print(end = '\n')


text = bigram_sentence_generator(words)

def trigram_sentence_generator(words):
    trigram_words = list(trigrams(brown.words()))
    start_words = random.choice(trigram_words)
    sentence = list()

    punctuation = ['.', '?', '!', ',', "``", ";"]
    check = False

    for i in punctuation:
        while check == False:
            if start_words[0] == i:
                start_words = random.choice(trigram_words)
            else:
                check = True

    sentence.append(start_words[0])
    sentence.append(start_words[1])
    sentence.append(start_words[2])

    #count of frequency of trigrams:
    trigram_counts = nltk.FreqDist(trigram_words)
    trigram_probs = nltk.KneserNeyProbDist(trigram_counts)

    print(sentence[0], end = " ")
    print(sentence[1], end = " ")
    print(sentence[2], end = " ")

    #while sentence[-1] != "''":
        #next_word = trigram_probs[sentence[2


text = trigram_sentence_generator(words)
