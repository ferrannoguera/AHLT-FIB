from itertools import chain
import json
import nltk
from nltk.tokenize import word_tokenize
import os
import pycrfsuite
import re
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import string
from time import time
from tqdm import tqdm
from xml.dom.minidom import parse, parseString

from nltk.stem.snowball import SnowballStemmer  # stem
from nltk import pos_tag  # partof-speech
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
snowballstem = SnowballStemmer("english")

print(sklearn.__version__)

nltk.download('punkt')

# CONSTANTS & PARAMETERS
TRAIN_DIR = "data/Train"
DEVEL_DIR = "data/Devel"
TEST_DIR = "data/Test-NER"
INPUT_DIR = DEVEL_DIR
VERSION = "000"
MODEL_DIR = "models/"
MODEL_FILE = MODEL_DIR + "model_ml_Goal1_" + VERSION + ".crfsuite"
OUTPUT_DIR = "out/ml/Goal1/"
OUTPUT_FILE = OUTPUT_DIR + "task9.1_ArnauCanyadell_FerranNoguera_" + VERSION + ".txt"


def parseXML(file):
    dom = parse(file)
    doc = dom.childNodes[0]
    return doc.getElementsByTagName("sentence")


def get_entities_from_xml(sentence):
    xml_entities = sentence.getElementsByTagName("entity")
    entities = [[]] * len(xml_entities)
    for i, xml_entity in enumerate(xml_entities):
        x = dict(xml_entity.attributes.items())
        offset = x['charOffset'].split(';')[0].split('-')  # .split(';')[0] = ignorem els casos de NEs separats ToDo
        entities[i] = (x['type'], int(offset[0]), int(offset[1]))
    # print('entities', entities)
    return entities


def get_sentence_info(sentence):
    x = dict(sentence.attributes.items())
    # print(x['text'])
    entities = get_entities_from_xml(sentence)
    return x['id'], x['text'], entities


def tokenize(text):
    t = word_tokenize(text)
    i = 0  # position in sentence
    j = 0  # number of token
    tokens = []
    while len(tokens) < len(t):
        # print(text, len(text), i)
        # print(t, len(t), j)
        if text[i] == ' ':
            i += 1
        # Override stupid NLTK " substitution
        if text[i] == '"':
            t[j] = '"'
        tokens.append((t[j], i, i + len(t[j]) - 1))
        i += len(t[j])
        j += 1
    return tokens


def get_labels(tokens, entities):
    i = 0
    j = 0
    labels = [[]] * len(tokens)
    while i < len(tokens) and j < len(entities):
        if entities[j][1] == tokens[i][1]:
            labels[i] = "B-" + entities[j][0]
            if entities[j][2] == tokens[i][2]:
                j += 1
        elif entities[j][1] < tokens[i][1]:
            labels[i] = "I-" + entities[j][0]
            if entities[j][2] == tokens[i][2]:
                j += 1
        else:
            labels[i] = "O"
        i += 1
    while i < len(tokens):
        labels[i] = "O"
        i += 1
    return labels


punctuation = set(string.punctuation)


def is_punctuation(token):
    return token in punctuation


def is_numeric(token):
    try:
        float(token.replace(",", ""))
        return True
    except:
        return False


def uppercases(w):
    if w.isupper():
        return "allCaps"
    elif w.islower():
        return "lowerCaps"
    else:
        return "mixedCaps"


def word_type(w):
    if is_numeric(w):
        return "numeric"
    elif is_punctuation(w):
        return "punctuation"
    elif not w.isalnum():
        return "specialSymbol"
    elif w.isalpha():
        return "onlyLetters"
    else:
        return "lettersAndNumbers"

def extract_features(s):
    # Input: Receives a tokenized sentence s (list of triples (word, offsetFrom, offsetTo) ).
    # Output: Returns a list of binary feature vectors, one per token in s
    # s = [(word, offsetFrom, offsetTo), ...]
    # return [[feat1_1, feat1_2, .. feat1_n1], [feat2_1, feat2_2, .. feat2_n2], .. [featn_1, featn_2, .. featn_nn]]
    features = [[]] * len(s)

    word = ""
    for token in s:
        word += str(token[0]) + " "
    partOfSpeech = pos_tag(word_tokenize(word))

    for i, token in enumerate(s):
        l = [
            "word=" + token[0],
            "lemma=" + lemmatizer.lemmatize(token[0]),
            "PoS=" + partOfSpeech[i][1],
            "suff=" + token[0][-4:],
            "uppers=" + uppercases(token[0]),
            "istitle=" + str(token[0].istitle()),
            "type=" + word_type(token[0]),
            "length=" + str(len(token[0])),
        ]
        if i > 0:
            l.append("-1:stem=" + snowballstem.stem(s[i - 1][0]))
            l.append("-1:PoS=" + partOfSpeech[i - 1][1])
            l.append("-1:uppers=" + uppercases(s[i - 1][0]))
            l.append("-1:istitle=" + str(s[i - 1][0].istitle()))
            l.append("-1:type=" + word_type(s[i - 1][0]))
        else:
            l.append("BOS")  # Beginning Of Sentence
        """
        if i - 1 > 0:
            l.append("-2:stem=" + snowballstem.stem(s[i - 2][0]))
            l.append("-2:PoS=" + partOfSpeech[i - 2][1])
            l.append("-2:uppers=" + uppercases(s[i - 2][0]))
            l.append("-2:istitle=" + str(s[i - 2][0].istitle()))
            l.append("-2:type=" + word_type(s[i - 2][0]))

        if i - 2 > 0:
            l.append("-3:word=" + s[i - 3][0])
            l.append("-3:lemma=" + lemmatizer.lemmatize(s[i - 3][0]))
            l.append("-3:stem=" + porterstem.stem(s[i - 3][0]))
            l.append("-3:PoS=" + partOfSpeech[i - 3][1])
            l.append("-3:position=" + str(i - 3))
            l.append("-3:suff=" + s[i - 3][0][-4:])
            l.append("-3:uppers=" + uppercases(s[i - 3][0]))
            l.append("-3:type=" + word_type(s[i - 3][0]))
        """

        if i + 1 < len(s):
            l.append("+1:stem=" + snowballstem.stem(s[i + 1][0]))
            l.append("+1:PoS=" + partOfSpeech[i + 1][1])
            l.append("+1:uppers=" + uppercases(s[i + 1][0]))
            l.append("+1:istitle=" + str(s[i + 1][0].istitle()))
            l.append("+1:type=" + word_type(s[i + 1][0]))
        else:
            l.append("EOS")  # End Of Sentence
        """
        if i + 2 < len(s):
            l.append("+2:stem=" + snowballstem.stem(s[i + 2][0]))
            l.append("+2:PoS=" + partOfSpeech[i + 2][1])
            l.append("+2:uppers=" + uppercases(s[i + 2][0]))
            l.append("+2:istitle=" + str(s[i + 2][0].istitle()))
            l.append("+2:type=" + word_type(s[i + 2][0]))

        if i + 3 < len(s):
            l.append("+3:word=" + s[i + 3][0])
            l.append("+3:lemma=" + lemmatizer.lemmatize(s[i + 3][0]))
            l.append("+3:stem=" + porterstem.stem(s[i + 3][0]))
            l.append("+3:PoS=" + partOfSpeech[i + 3][1])
            l.append("+3:position=" + str(i + 3))
            l.append("+3:suff=" + s[i + 3][0][-4:])
            l.append("+3:uppers=" + uppercases(s[i + 3][0]))
            l.append("+3:type=" + word_type(s[i + 3][0]))
        """
        # print(l)
        # print("---------------")
        features[i] = l
    return features


def output_features(id, tokens, features):
    # Input: Receives a sentence id, a tokenized sentence, and list of binary feature vectors (one per token)
    # Output: Prints to stdout the feature vectors in the following format: one line per token, one blank line after
    # each sentence. Each token line contains tab-separated fields: sent_id, token, span_start, span_end, gold_class,
    # feature1, feature2, ...
    for token, token_features in zip(tokens, features):
        print("\t".join([id, token[0], str(token[1]), str(token[2])] + token_features))


def output_entities(id, tokens, classes, out_file):
    current_ne = None
    for token, tag in zip(tokens, classes):
        if tag[0] == "B" or tag[0] == "I" and (current_ne is None or tag[2:] != current_ne[3]):
            if current_ne is not None:
                out_file.write(id + '|' + str(current_ne[1]) + '-' + str(current_ne[2]) + '|' + current_ne[0] + '|' +
                               current_ne[3] + '\n')
            current_ne = list(token) + [tag[2:]]
        elif tag[0] == "I":
            # if current_ne is None:
            #     raise Exception("Tokens syntax incorrect. I-class is not preceeded by a B-class")
            # elif tag[2:] != current_ne[3]:
            #     raise Exception("Tokens syntax incorrect. I-classA is preceeded by a B-classB, where classA != classB")
            # else:
            current_ne[2] = token[2]
            current_ne[0] += " " + token[0]
        elif tag[0] == "O":
            if current_ne is not None:
                out_file.write(id + '|' + str(current_ne[1]) + '-' + str(current_ne[2]) + '|' + current_ne[0] + '|' +
                               current_ne[3] + '\n')
                current_ne = None


# output_entities("DDI-DrugBank.d553.s0", [("Ascorbic",0,7), ("acid",9,12), (",",13,13), ("aspirin",15,21), (",",22,22),
# 										 ("and",24,26), ("the",28,30),("common",32,37), ("cold",39,42)],
# 				["B-drug", "I-drug", "O", "B-brand", "O", "O", "O", "O", "O"], sys.stdout)


def evaluate(input_dir, output_file):
    os.system("java -jar eval/evaluateNER.jar "
              + input_dir + " " + output_file)


def train(input_dir, model_file):
    trainer = pycrfsuite.Trainer(verbose=True)
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 200,
        'feature.possible_transitions': True
    })
    input_files = os.listdir(input_dir)
    # output_file = open(output_file, 'w')
    print('Reading files')
    for file in tqdm(input_files):
        tree = parseXML(input_dir + '/' + file)
        features_list = []
        labels_list = []
        for sentence in tree:
            (id, text, entities) = get_sentence_info(sentence)
            tokens = tokenize(text)
            # entities = get_entities_from_xml(sentence)
            labels = get_labels(tokens, entities)
            features = extract_features(tokens)
            features_list.append(features)
            labels_list.append(labels)
            # output_features(id, tokens, features)
            # classes = [] # ToDo
            # output_entities(id, tokens, classes, output_file)
            # print('features', features)
            # print('labels', labels)
            trainer.append(features, labels)
    print('--- Start training')
    start_time = time()
    trainer.train(model_file)
    print("--- %s seconds ---" % (time() - start_time))


def nerc(input_dir, model_file, output_file):
    input_files = os.listdir(input_dir)
    output_file = open(output_file, 'w')
    crf_tagger = pycrfsuite.Tagger()
    crf_tagger.open(model_file)
    for file in input_files:
        tree = parseXML(input_dir + '/' + file)
        for sentence in tree:
            (id, text, _) = get_sentence_info(sentence)
            tokens = tokenize(text)
            features = extract_features(tokens)
            # output_features(id, tokens, features)
            classes = crf_tagger.tag(features)
            output_entities(id, tokens, classes, output_file)


if __name__ == '__main__':
    train(TRAIN_DIR, MODEL_FILE)
    nerc(INPUT_DIR, MODEL_FILE, OUTPUT_FILE)
    evaluate(INPUT_DIR, OUTPUT_FILE)
