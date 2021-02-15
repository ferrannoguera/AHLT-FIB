import argparse
from xml.dom.minidom import parse, parseString
import os
from nltk.parse.corenlp import CoreNLPDependencyParser
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--text", help="path of the train input directory", default="False")
args = parser.parse_args()

try:
    my_parser = CoreNLPDependencyParser(url="http://localhost:9000")
    text = "Hello, my name is Ferran, a pleasure to meet you!"
    analyze, = my_parser.raw_parse(str(args.text))
    print(analyze)
except (ConnectionError, ConnectionRefusedError) as e:
    print("Loading parser\n")
    print("Error while trying to connect to CorNLP server. Try running:\n")
    print("\tcd stanford-corenlp-full-2018-10-05")
    print("\tjava -mx4g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer")
    exit()

