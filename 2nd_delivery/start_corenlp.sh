#!/bin/bash
reset
cd stanford-corenlp-full-2018-10-05/
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
