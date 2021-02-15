# 2nd Delivery folder - AHLT 2020
By Arnau Canyadell & Ferran Noguera

Instructions:
1. `wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip` & `unzip stanford-corenlp-full-2018-10-05.zip`
2. `cd stanford-corenlp-full-2018-10-05`
3. `java -mx4g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer`
4. In a new terminal, depending on the task, run:
  - `python src/rule-based/ddi.py`
  - `python src/ml/ddi.py`
  
  Use `python src/XXX/ddi.py --help` to see more options
