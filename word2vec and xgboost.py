# word2vec and xgboost, model accuracy = 0.28-0.30
# Importing libraries
import csv

import gensim
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


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



batch_size = 32
imag_height = 128
imag_width = 55
training_directory = './train'
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

print(set(test_labels))

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))


def get_model():
    xgb_classifier = xgb.XGBClassifier()

    return xgb_classifier

train_examples = np.array(train_examples)
train_labels = np.array(train_labels).astype('float32')

def get_embedding(sentence, word2vec):
    sum = None
    num = 0
    for word in sentence:
        if sum is None:
            try:
                sum = word2vec.wv[word]
                num = 1
            except:
                pass
        else:
            try:
                sum = sum + word2vec.wv[word]
                num += 1
            except:
                pass
    sum = sum / float(num)
    return sum


def train_and_predict():
    model = get_model()
    word2vec = gensim.models.Word2Vec(train_examples, min_count=1, window=5, sg=0)
    vectorized_train_examples = [get_embedding(train_example, word2vec) for train_example in train_examples]
    model.fit(vectorized_train_examples, train_labels)

    vectorized_test_examples = [get_embedding(test_example, word2vec) for test_example in test_examples]
    predicted_labels = model.predict(vectorized_test_examples)
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
        word2vec = gensim.models.Word2Vec(train_examples[curr_train], min_count=1, window=5, sg=0)
        vectorized_train_examples = [get_embedding(train_example, word2vec) for train_example in train_examples[curr_train]]
        # training the model
        model.fit(vectorized_train_examples, train_labels[curr_train])

        # getting the evaluation results
        vectorized_test_examples = [get_embedding(train_example, word2vec) for train_example in train_examples[curr_test]]
        predicted_labels = model.predict(vectorized_test_examples)
        # printing the loss and accuracy
        accuracy = accuracy_score(train_labels[curr_test], predicted_labels)
        print(f'Step {step}: Accuracy - {accuracy}')
        accuracies.append(accuracy)
    # writing the results to an output file
    with open('results.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        for accuracy in accuracies:
            writer.writerow(str(accuracy))


#train_and_predict()
n_fold_cross_validation()
