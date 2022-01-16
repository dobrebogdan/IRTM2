# stopwords and xgboost, accuracy = 0.26-0.28
# Importing libraries
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import nltk
import xgboost as xgb
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import csv
import numpy as np


lemmatizer = WordNetLemmatizer()
english_stopwords = stopwords.words('english')

# removing punctuation and lemmatizing
def clean_sentence(sentence):
    sentence = sentence.lower()
    for char in sentence:
        if (not char.isalpha()) and char != ' ':
            sentence = sentence.replace(char, ' ')
    tokens = sentence.split()
    good_tokens = []
    for token in tokens:
        if token not in english_stopwords:
            good_tokens.append(lemmatizer.lemmatize(token))
    sentence = ' '.join(good_tokens)
    return sentence


classes_number = 10
epochs_number = 15
splits_number = 5

labels_to_class = {
    'Metal': 0,
    'Jazz': 1,
    'Pop': 2,
    'Electronic': 3,
    'R&B': 4,
    'Hip-Hop': 5,
    'Rock': 6,
    'Indie': 7,
    'Country': 8,
    'Folk': 9
}

# Reading the train and test data
train_examples = []
train_labels = []

test_examples = []
test_labels = []

# headers were removed from file
with open('./Dataset/Lyrics-Genre-Train.csv') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        train_examples.append(clean_sentence(row[4]))
        train_labels.append(labels_to_class[row[3]])

with open('./Dataset/Lyrics-Genre-Test-GroundTruth.csv') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        test_examples.append(clean_sentence(row[4]))
        test_labels.append(labels_to_class[row[3]])

train_examples = np.array(train_examples)
train_labels = np.array(train_labels)

test_examples = np.array(test_examples)
test_labels = np.array(test_labels)

def get_model():
    xgb_classifier = xgb.XGBClassifier()
    return xgb_classifier


def get_words_from_examples(examples):
    words = {}
    for example in examples:
        tokens = nltk.word_tokenize(example)
        for token in tokens:
            if token in words.keys():
                words[token] = words[token] + 1
            else:
                if len(token) > 1 and token[0].islower():
                    words[token] = 1
    return words


def get_words_from_example(example):
    tokens = nltk.word_tokenize(example)
    words = {}
    for token in tokens:
        if token in words.keys():
            words[token] = words[token] + 1
        else:
            if len(token) > 1 and token[0].islower():
                words[token] = 1
    return words


def get_stop_word_rankings():
    stop_words = []
    stop_words_ranking = {}
    words = get_words_from_examples(train_examples)

    words_list = []
    for (word, cnt) in words.items():
        words_list.append((cnt, word))
    words_count = sorted(words_list, reverse=True)
    # 13
    top_words_counts = words_count[0:20]
    rank = 1
    for (_, word) in top_words_counts:
        stop_words.append(word)
        stop_words_ranking[word] = rank
        rank += 1
    return stop_words, stop_words_ranking


(stop_words, stop_words_ranking) = get_stop_word_rankings()

train_examples_vectorized = []
test_examples_vectorized = []

# Vectorizing train and test data
for train_example in train_examples:
    words = get_words_from_example(train_example)
    words_list = []
    for (word, cnt) in words.items():
        words_list.append((cnt, word))
    words_count = sorted(words_list, reverse=True)

    # print(words_count)
    # compute the stop words if not already existing

    curr_ranking = [0.0] * len(stop_words)
    curr_word_rank = 0.0
    sum = 0
    for (_, word) in words_count:
        if word in stop_words:
            curr_word_rank += 1
            poz = int(stop_words_ranking[word] - 1)
            curr_ranking[poz] = curr_word_rank

    train_examples_vectorized.append(curr_ranking)

train_examples_vectorized = np.array(train_examples_vectorized)

for test_example in test_examples:
    words = get_words_from_example(test_example)
    words_list = []
    for (word, cnt) in words.items():
        words_list.append((cnt, word))
    words_count = sorted(words_list, reverse=True)

    curr_ranking = [0.0] * len(stop_words)
    curr_word_rank = 0.0
    sum = 0
    for (_, word) in words_count:
        if word in stop_words:
            curr_word_rank += 1
            poz = int(stop_words_ranking[word] - 1)
            curr_ranking[poz] = curr_word_rank

    test_examples_vectorized.append(curr_ranking)


def train_and_predict():
    model = get_model()
    model.fit(train_examples_vectorized, train_labels)
    predicted_labels = model.predict(test_examples_vectorized)
    print(f'Accuracy score for validation is {accuracy_score(test_labels, predicted_labels)}')
    print(confusion_matrix(test_labels, predicted_labels))

def n_fold_cross_validation():
    # Using KFold to split in 5 parts for cross validation
    kfold = KFold(n_splits=splits_number, shuffle=True)
    kfold_split = kfold.split(train_examples, train_labels)
    step = 0
    accuracies = []
    # iterating through the different splits
    for curr_train, curr_test in kfold_split:
        step += 1
        model = get_model()
        # training the model
        model.fit(train_examples_vectorized[curr_train], train_labels[curr_train])

        # getting the evaluation results
        predicted_labels = model.predict(train_examples_vectorized[curr_test])
        # printing the loss and accuracy
        accuracy = accuracy_score(train_labels[curr_test], predicted_labels)
        print(f'Step {step}: Accuracy - {accuracy}')
        accuracies.append(accuracy)
    # writing the results to an output file
    with open('results.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        for accuracy in accuracies:
            writer.writerow(str(accuracy))


train_and_predict()
n_fold_cross_validation()
