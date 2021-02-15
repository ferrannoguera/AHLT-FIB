from xml.dom.minidom import parse, parseString
import nltk
from nltk.tokenize import word_tokenize
import os
import re
import json

nltk.download('punkt')

## PARAMETERS
VERSION = "010"

TRAIN_DIR = "data/Train"
DEVEL_DIR = "data/Devel"
TEST_DIR = "data/Test-NER"
OUTPUT_DIR = "out/"
OUTPUT_FILE = "task9.1_ArnauCanyadell_FerranNoguera_"+VERSION+".txt"

inputdir = DEVEL_DIR

# x = text
# y = NEs (drug, drug_n, group, brand)


def parseXML(file):
	dom = parse(file)
	doc = dom.childNodes[0]
	return doc.getElementsByTagName("sentence")


def get_sentence_info(sentence):
	x = dict(sentence.attributes.items())
	return x['id'], x['text']


def get_sentence_entities(sentence):
	# x = dict(sentence.attributes.items())
	entities = sentence.getElementsByTagName("entity")
	x = [dict(entity.attributes.items()) for entity in entities]
	xx = [(elem['type'], elem['text']) for elem in x]
	return xx


def tokenize(text):
	t = word_tokenize(text)
	i = 0 # position in sentence
	j = 0 # number of token
	tokens = []
	while len(tokens) < len(t):
		#print(text, len(text), i)
		#print(t, len(t), j)
		if text[i] == ' ':
			i += 1
		# Override stupid NLTK " substitution
		if text[i] == '"':
			t[j] = '"'
		tokens.append((t[j], i, i+len(t[j]) - 1))
		i += len(t[j])
		j += 1
	return tokens


# These suffixes were taken from the train dataset
DRUG_SUFFIXES = ["oin", "ine","ide", "cin", "fil", "ion", "mil", "tal", "rin", "lin", "ium", "ril", "nol","lyn",
				"hol", "ole","mic","xic","xib","vir","mab","vec","ast","lax","sin","pam","tan"]


DRUG_COMPOSED_WORDS = ['acid','sodium','alkaloids','hydrochloride','f2alpha','edisylate','iodide','sulfate','acetate'
						'antiinflammatory','mustard','hcl','mofetil','gallate','cations','nitrate']

GROUP_COMPOSED_LAST_WORDS = [
	'adjuvant',
	'agonist',
	'alkaloid',
	'antibiotic',
	'antidepressent',
	'anti-inflammatory',
	'blocker',
	'class',
	'compound',
	'depressants',
	'diuretic',
	'drug',
	'hormone',
	'inhibitor',
	'medication',
	'product',
	'solution',
	'steroid',
	'vaccine',
	'vasodilator'
]

GROUP_COMPOSED_MIDDLE_WORDS = [
	'oxidase',
	'channel',
	'anti-inflammatory',
	'blocking',
	'reuptake',
	'serotonin',
	'reductase',
	'pump',
	'depressant',
	'anhydrase'
]
ALL_COMPOSED_WORDS = DRUG_COMPOSED_WORDS + GROUP_COMPOSED_LAST_WORDS

# Increase F1: alkaloid/s (most impact),
# Decrease F1: agent,
# No impact on F1: adjuvant, agonist


def append_token(ret, name, offset1, offset2, classtype):
	ret.append(dict(name=name, offset=str(offset1) + '-' + str(offset2), type=classtype))
	return ret


def match_words(list_words_1, list_words_2):
	for i in range(min(len(list_words_1), len(list_words_2))):
		if list_words_1[i][0] != list_words_2[i]:
			return False
	return True


def match_suffix(token, suffixes):
	for suf in suffixes:
		if re.search(".+" + suf + "$", token):
			return True
	return False

"""
# TOO INEFFICIENT. DOES NOT WORK
# This function is O(n), where n=number of instances in dict. Could be O(log n) if the dict was ordered and the search
# optimized. However, the whole program does not take long by now so we can keep it like that.
def token_in_dict(token_list, i, dict):
	for elem in dict:
		words = word_tokenize(elem)
		j = 0
		are_equal = True
		while j < len(words) and i + j < len(token_list) and are_equal:
			are_equal = words[j] == token_list[i + j]
		if are_equal:
			return dict[elem], len(words) # tag, num words in NE
	return None, 0

		# If token is any word that appears in the training data
		if not appended:
			ne, tag, ne_length = find_token_in_dict(token_list, i, training_words_dict)
			if ne is not None:
				appended = True
				skipwords = ne_length - 1
				append_token(ret, ne, token[1], token_list[i+skipwords][2], tag)
"""

def extract_entities(token_list):
	ret = []
	with open('entities/trainingFeatures/entity_dict.json', 'r') as f:
		training_words_dict = json.load(f)
	skipwords = 0
	for i,token in enumerate(token_list):
		appended = False
		if skipwords != 0:
			skipwords -= 1
			appended = True
		# If token is any word that appears in the training data
		if not appended:
			if token[0] in training_words_dict:
				words = training_words_dict[token[0]][0]
				tag = training_words_dict[token[0]][1]
				appended = match_words(token_list[i:], words)
				if appended:
					skipwords = len(words)
					word_appended = token[0]
					if len(words) > 0:
						word_appended += ' ' + ' '.join(words)
					append_token(ret, word_appended, token[1], token_list[i+skipwords][2], tag)
		# If token finishes with any suffix in DRUG_SUFFIXES
		"""	
		with open('entities/trainingFeatures/suffix_dict.json', 'r') as f:
			training_suffix_dict = json.load(f)
		if not appended and token[0][-4:] in training_suffix_dict:
			appended = True
			append_token(ret,token[0],token[1],token[2],training_suffix_dict[token[0][-4:]])
		"""
		# If the token matches any custom affix
		if not appended:
			if match_suffix(token[0].lower(), DRUG_SUFFIXES):
				if len(token_list) > i+1:
					#If the following token is in DRUG_COMPOSED_WORDS both tokens will be a drug
					for words in DRUG_COMPOSED_WORDS:
						if token_list[i+1][0].lower() == words:
							appended = True
							append_token(ret,token[0]+" "+token_list[i+1][0],token[1],token_list[i+1][2],"drug")
				if not appended:
					appended = True
					append_token(ret,token[0],token[1],token[2],"drug")
		# Vitamins C, E, B*, D* are drugs
		if not appended and token[0].lower() == 'vitamin' and len(token_list) > i+1 and re.search("^(C|E|[BD][\w-]+)$", token_list[i+1][0]):
			appended = True
			append_token(ret,token[0]+" "+token_list[i+1][0],token[1],token_list[i+1][2],"drug")
		# Vitamins D*, K*, A are group
		if not appended and token[0].lower() == 'vitamin' and len(token_list) > i+1:
			if len(token_list) > i+2 and\
					(token_list[i+1][0]=='D' and (token_list[i+2][0].lower() == 'analogue' or token_list[i+2][0].lower() == 'preparations') or\
					(token_list[i+1][0] == 'K' and token_list[i+2][0].lower() == 'antagonists')):
				appended = True
				append_token(ret,token[0]+" "+token_list[i+1][0]+" "+token_list[i+2][0],token[1],token_list[i+2][2],"group")
			elif token_list[i+1][0] == 'A' or token_list[i+1][0] == 'D' or token_list[i+1][0] == 'K':
				appended = True
				append_token(ret,token[0]+" "+token_list[i+1][0],token[1],token_list[i+1][2],"group")
		#Capital letters most probably are brands
		## Extremely useful
		if not appended and token[0].isupper() and len(token[0]) > 4:
			appended = True
			append_token(ret,token[0],token[1],token[2],"brand")
		"""
		# If first letter is in capital not following a dot
		if not appended and i-1 >= 0 and token_list[i-1][0] != '.' and token_list[i-1][0] != ':' and token[0][0].isupper():
			appended = True
			append_token(ret,token[0],token[1],token[2],"brand")
		"""
		#Composed words belonging to group
		## This rule creates a lot of hits and a lot of misses (not very precise)
		if not appended:
			for words in GROUP_COMPOSED_LAST_WORDS:
				if i-1 >= 0 and re.search(words + "s?$", token[0]):
					stop = False
					j = 0
					while(not stop and i-j >= 0):
						stop = True
						j += 1
						for mid_words in GROUP_COMPOSED_MIDDLE_WORDS:
							if (token_list[i-j][0] == mid_words):
								stop = False
					appended = True
					append_token(ret,token_list[i-j][0]+" "+token[0],token_list[i-j][1],token[2],"group")
		"""
		# If contains "toxin" ==> drug_n
		## NOTE: this rule has 0 effect in TEST/DEVEL :(
		if not appended and re.search("toxin", token[0]):
			appended = True
			append_token(ret,token[0],token[1],token[2],"drug_n")
		"""
		"""
		# Consider most common drug_n words
		if not appended:
			j = 0
			while not appended and j < len(DRUG_N_COMMON_WORDS):
				if token[0] == DRUG_N_COMMON_WORDS[j]:
					appended = True
					ret.append({'name': token[0], 'offset': str(token[1]) + '-' + str(token[2]), 'type': "drug_n"})
				j += 1
		"""
		"""
		# If contains '-' ==> drug_n (last rule)
		## Bad performance (increases drug_n hits and misses, but overall drug_n F1 score is increased. However it increases misses for other)
		if not appended:
			if re.search("-", token[0]):
				appended = True
				append_token(ret, token[0], token[1], token[2], "drug_n")
		# If contains '-' && next word is not COMPOUND ==> drug_n (last rule)
		if not appended:
			if re.search("-", token[0]) and (i+1 == len(token_list) or token_list[i+1] not in ALL_COMPOSED_WORDS):
				appended = True
				append_token(ret, token[0], token[1], token[2], "drug_n")
		"""
	return ret


def output_entities(id, entities, outputfile):
	for ent in entities:
		outputfile.write(id + "|" + ent['offset'] + "|" + ent['name'] + "|" + ent['type'] + '\n')


def evaluate(inputdir, outputfile):
	os.system("java -jar eval/evaluateNER.jar "
			  + inputdir + " " + outputfile)


def nerc(inputdir, outputfile) :
	input_files = os.listdir(inputdir)
	output_file = open(outputfile, 'w')
	for file in input_files :
		tree = parseXML(inputdir + '/' + file)
		for sentence in tree :
			(id, text) = get_sentence_info(sentence)
			token_list = tokenize(text)
			entities = extract_entities(token_list)
			output_entities(id, entities, output_file)
	evaluate(inputdir, outputfile)


nerc(inputdir, OUTPUT_DIR + OUTPUT_FILE)
