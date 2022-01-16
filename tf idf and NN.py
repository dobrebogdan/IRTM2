# tf idf and neural networks, model accuracy = 0.32-0.34
# Importing libraries
import csv
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf
import tensorflow_hub as hub
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


def get_model(curr_train_examples, curr_train_labels):
    train_dataset = tf.data.Dataset.from_tensor_slices((curr_train_examples, curr_train_labels))


    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    VOCAB_SIZE = 1000
    encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, output_mode="tf-idf")
    encoder.adapt(train_dataset.map(lambda text, label: text))
    model = tf.keras.models.Sequential([
        encoder,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(classes_number)
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

train_examples = np.array(train_examples)
train_labels = np.array(train_labels).astype('float32')

def train_and_predict():
    model = get_model(train_examples, train_labels)
    model.fit(train_examples, train_labels, batch_size=64, epochs=20)

    predicted_labels = model.predict(test_examples)
    predicted_labels = [predicted_label.argmax() for predicted_label in predicted_labels]
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
        model = get_model(train_examples[curr_train], train_labels[curr_train])
        # training the model
        model.fit(train_examples[curr_train], train_labels[curr_train], batch_size=batch_size, epochs=epochs_number)
        # getting the evaluation results
        results = model.evaluate(train_examples[curr_test], train_labels[curr_test])
        # printing the loss and accuracy
        print(f'Step {step}: Loss - {results[0]}, Accuracy - {results[1]}')
        accuracies.append(results[1])
    # writing the results to an output file
    with open('results.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        for accuracy in accuracies:
            writer.writerow(str(accuracy))


# train_and_predict()
n_fold_cross_validation()
