# ECE467 Project 1: Text Categorization
# By Colin Hwang

from math import log, inf
# citing that I learned most of how to use nltk from https://realpython.com/nltk-nlp-python/ 
# used nltk for word tokenization, stop words, stemming, POS tagging
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
Stemmer = PorterStemmer()

def main():
    # retrieve input and output file name from users
    trainingFile = input("Enter the name of the training file: ")
    testingFile = input("Enter the name of the testing file: ")
    outFile = input("Enter the name of the output file: ")

    # creating the list of training documents
    trainingOpen = open(trainingFile, 'r') # read file
    splitTraining = trainingFile.split("/")[:-1] #eliminate last part of input path (corpus#_train.labels)
    trainingPath = "/".join(splitTraining)
    trainingList = filter(lambda x: x != "", trainingOpen.read().split("\n"))  # Forming a list of the training docs
    trainingList = map(lambda x: trainingPath + x[1:], trainingList)
    trainingList = list(map(lambda x: x.split(" "), trainingList))  # gets the file path and its designated categorization
    trainingOpen.close()

    # same idea as the training documents
    testingOpen= open(testingFile, 'r')
    splitTesting = testingFile.split("/")[:-1]
    testingPath = "/".join(splitTesting)
    testingList = list(filter(lambda x: x != "", testingOpen.read().split("\n"))) # list of testing documents to preserve format when writing to output
    testingListPaths = list(map(lambda x: testingPath + x[1:], testingList)) # for retrieving paths to testing docs
    testingOpen.close()

    # naiveBayes
    predictionList = naiveBayes(trainingList, testingListPaths)

    # Writes predictions to output file
    output(testingList, predictionList, outFile)

def naiveBayes(trainingList, testingListPaths):
    smoothing = .07 # to avoid 0 probability on words not seen
    tokenList = set()  # keep track of all tokens seen
    categoryDocs = {}  # keep track of the number of documents per category
    tokenFreq = {}  # keep track of the frequency of each token per category
    tokenCount = {}  # keep track of the number of tokens in each category
    predictions = []  # predictions for each testing document

    # go through all the training docs and their categories
    # tokenize document, stem tokens, and keep track of docs per category, 
    for [path, category] in trainingList:
        # keep track of how many docs are in each category
        if category in categoryDocs:
            categoryDocs[category] += 1
        else:
            categoryDocs[category] = 1
        trainingDoc = open(path, 'r')
        
        # tokenize docs
        tokenizedDoc = word_tokenize(trainingDoc.read()) #from nltk
        # removing stopwords and words not in the english alphabet
        tokenizedDoc = [word for word in tokenizedDoc if word not in stop_words and word.isalpha()] 
        
        #FOR LATER, TO IMPROVE CONSIDER ON DOING .CASEFOLD()!!!!!! *****

        # iterates through each token in the new tokenized doc
        for token in tokenizedDoc:
            # again, from nltk, stems token to its root
            # token frequency in each category
            stemToken = Stemmer.stem(token)
            if category not in tokenFreq:
                tokenFreq[category] = {}
            if stemToken in tokenFreq[category]:
                tokenFreq[category][stemToken] += 1
            else:
                tokenFreq[category][stemToken] = 1

            # number of tokens per category
            if category in tokenCount:
                tokenCount[category] += 1
            else:
                tokenCount[category] = 1

            # keep track of each token
            tokenList.add(stemToken)

        trainingDoc.close()

    # based off Sable's slides...
    # calculate prior probabilities with: P(c) = # of docs in category c / total # of docs
    # calculate maximum liklihood estimate with: P(t|c) = # of docs in category c containing token t / total docs in category c
    category_priors = {} 
    token_category_conditional = {}
    category_list = categoryDocs.keys() # list of all categories 
    num_training_docs = sum(categoryDocs.values()) # total # of training documents

    # iterate through each category
    for category in category_list:
        # prior probabilities
        category_priors[category] = categoryDocs[category] / num_training_docs

        # MLE + smoothing
        token_category_conditional[category] = {}
        denominator = tokenCount[category] + (smoothing * len(tokenList))
        for token in tokenList:
            if token in tokenFreq[category].keys():
                token_category_conditional[category][token] = (tokenFreq[category][token] + smoothing) / denominator
            else:
                token_category_conditional[category][token] = smoothing / denominator

    # Now go through the testing documents
    for path in testingListPaths:
        testFreq = {}  # keep track of frequency of tokens in each document

        testingDoc = open(path, 'r')
        tokenizedDoc = word_tokenize(testingDoc.read()) # tokenize each testing doc, nltk
        tokenizedDoc = [word for word in tokenizedDoc if word not in stop_words and word.isalpha()] # no stop words or non words
        # iterate through tokens in the tokenized doc
        for token in tokenizedDoc:
            # stem token to its root
            stemToken = Stemmer.stem(token)
            # keep track of token frequency
            if stemToken not in testFreq:
                testFreq[stemToken] = 1
            else:
                testFreq[stemToken] += 1
        testingDoc.close()

        predictCat = ("", -inf)  # will store (predicted category, probability)
        # go through each category and calculate probabilities
        # prob estimates are small, so use log probs
        # probability = argmax (prior prob + sum(conditional prob))
        for category in category_list:
            prior_prob = log(category_priors[category])
            conditional_prob = 0
            for token in testFreq.keys():
                if token in tokenList:
                    conditional_prob += log(token_category_conditional[category][token]) * testFreq[token]
            prob = prior_prob + conditional_prob
            if prob > predictCat[1]:
                predictCat = (category, prob) # picking the highest probability to be the most likely category

        predictions.append(predictCat[0])

    return predictions
    
def output(testingList, predictions, outFile):
    outFileRead = open(outFile, 'w')
    # put predicitions into testingList
    for (path, prediction) in zip(testingList, predictions):
        outFileRead.write(path + " " + prediction + '\n')

    outFileRead.close()

main()
#END