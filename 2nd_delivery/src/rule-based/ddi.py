import argparse
from xml.dom.minidom import parse, parseString
import os
from nltk.parse.corenlp import CoreNLPDependencyParser

# DEFAULT VALUES
TRAIN = "Train"
DEVEL = "Devel"
TEST = "Test-DDI"
DEF_INPUT_DIR = "data/" + DEVEL
DEF_GROUP_NAME = "ArnauCanyadell_FerranNoguera"
DEF_VERSION = "002"


def analyze(s):
    """
    Input: Receives a sentence text s, and sends it to
    CoreNLP to obtain the tokens, tags, and dependency tree.
    It also adds the start/end offsets to each token

    :param s:
    :return: Returns the nltk DependencyGraph object produced by CoreNLP, enriched with token offsets.
    """
    try:
        # Clean string the "/" and "-" might be too restrictive, recheck
        s = s.replace("\r", " ").replace("\n", " ").replace("/", " ").replace("-", " ")
        mytree, = my_parser.raw_parse(s)
        index = 0
        for leaf in range(len(mytree.nodes)):
            # mytree's element is acting as the root of the tree
            if leaf > 0:
                word = mytree.nodes[leaf]['word']
                # Consider space distance
                while index < len(s) and s[index] == ' ':
                    index = index + 1
                # Special cases with ( ) [ ] as they are translated to -LRB- for instance
                if mytree.nodes[leaf]['rel'] == 'punct' and len(word) > 2 and word[1:-1].isalnum():
                    word_size = 1
                else:
                    word_size = len(word)
                mytree.nodes[leaf].update([('start', str(index))])
                mytree.nodes[leaf].update([('end', str(index + word_size - 1))])
                index = index + word_size
        return mytree
    except Exception:
        print("Inside analyze function\n")
        print("Error while trying to connect to CorNLP server. Try running:\n")
        print("\tcd stanford-corenlp-full-2018-10-05")
        print("\tjava -mx4g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer")
        exit()
    return ""


# Returns the highest position of each element in the dependency graph
def positions_on_tree(analysis, e1_start, e1_end, e2_start, e2_end):
    e1_pos = 0
    e2_pos = 0
    for leaf in range(len(analysis.nodes)):
        if e1_pos == 0 or e2_pos == 0:
            # e1
            if leaf > 0 and analysis.nodes[leaf]['start'] == e1_start:
                e1_pos = leaf
                e1_head_next = analysis.nodes[e1_pos]['head']
                while e1_head_next != 0 and int(e1_start) <= int(analysis.nodes[e1_head_next]['start']) \
                        and int(analysis.nodes[e1_head_next]['end']) <= int(e1_end):
                    e1_pos = e1_head_next
                    e1_head_next = analysis.nodes[e1_head_next]['head']
            # e2
            if leaf > 0 and analysis.nodes[leaf]['start'] == e2_start:
                e2_pos = leaf
                e2_head_next = analysis.nodes[leaf]['head']
                while e2_head_next != 0 and int(e2_start) <= int(analysis.nodes[e2_head_next]['start']) \
                        and analysis.nodes[e2_head_next]['end'] <= e2_end:
                    e2_pos = e2_head_next
                    e2_head_next = analysis.nodes[e2_head_next]['head']
        else:
            break
    return e1_pos, e2_pos


# True if pos_1 under pos_2, False otherwise
def get_from_root(analysis, pos_e1, pos_e2):
    relation = "independent"
    # Analyze e1
    leaf = analysis.nodes[pos_e1]['head']
    stop = analysis.nodes[leaf]['tag'] == 'TOP'
    e1_to_root = [pos_e1]
    while not stop:
        e1_to_root.append(leaf)
        leaf = analysis.nodes[leaf]['head']
        stop = analysis.nodes[leaf]['tag'] == 'TOP'
    # e1 under e2
    for pos in e1_to_root:
        if pos == pos_e2:
            relation = "e1_under_e2"
            break
    # Analyze e2
    leaf = analysis.nodes[pos_e2]['head']
    stop = analysis.nodes[leaf]['tag'] == 'TOP'
    e2_to_root = [pos_e2]
    while not stop:
        e2_to_root.append(leaf)
        leaf = analysis.nodes[leaf]['head']
        stop = analysis.nodes[leaf]['tag'] == 'TOP'
    # e2 under e1
    if relation == "independent":
        for pos in e2_to_root:
            if pos == pos_e1:
                relation = "e2_under_e1"
                break

    return e1_to_root, e2_to_root, relation


EFFECT_REL = ["effect"]

EFFECT_INTERACTION_VERBS = ["produce",
                            "appear",
                            "reduce",
                            "increase",
                            "be",
                            "conclude"]

MECHANISM_REL = ["reduce",
                 "increase",
                 "decrease",
                 "depress",
                 "long",
                 "accelerate"]

INT_REL = ["interaction",
           "interact"]

ADVISE_REL = ["concurrent",
              "concurrently",
              "together",
              "combination",
              "combine",
              "recommend",
              "avoid"]

ADVISE_AUX_REL = ["may",
              "might",
              "could"]


# def contains_aux_advise(word):
#     return "advise" if word in ADVISE_AUX_REL else "null"
#
#
# def contains_effect(word):
#     return word.lower() in EFFECT_REL
#
#
# def contains_interaction_effect(word):
#     return word.lower() in EFFECT_INTERACTION_VERBS


# Checks effect (CHECK THAT VERB ON TOP IS AN INTERACTION VERB)
def check_effect(e1_to_root, e2_to_root):
    # Only one encounters effect NN
    one_effect = any([analysis.nodes[elem]['lemma'].lower() in EFFECT_REL for elem in e1_to_root])
    two_effect = any([analysis.nodes[elem]['lemma'].lower() in EFFECT_REL for elem in e2_to_root])
    if one_effect != two_effect:
        return "effect"
    else:
        return "null"


# Check in between mechanism clue word
def check_mechanism(e_to_root):
    for elem in e_to_root:
        if analysis.nodes[elem]['lemma'] in MECHANISM_REL:
            return "mechanism"
    return "null"


# def contains_int(word_leaf):
#     for word in INT_REL:
#         if word_leaf == word:
#             return "int"
#     return "null"


def check_int(e_to_root):
    for elem in e_to_root:
        if analysis.nodes[elem]['lemma'] in INT_REL:
            return "int"
    return "null"


# def contains_adv(word_leaf):
#     for word in ADVISE_REL:
#         if word_leaf == word:
#             return "advise"
#     return "null"


def check_adv(e_to_root):
    for elem in e_to_root:
        if analysis.nodes[elem]['lemma'] in ADVISE_REL:
            return "advise"
    return "null"


def check_interaction(analysis, entities, e1, e2, s, f):
    """

    :param analysis: DependencyGraph object with all
    sentence information
    :param entities: list of all entities in the sentence
    (id and offsets)
    :param e1: id of the first entity to be checked
    :param e2: id of the second entity to be checked
    :return: Returns a 0/1 value indicating whether the
        sentence states an interaction between entities e1 and e2,
        and the type of interaction (null if there is none).
    """

    do_analysis = True
    if not entities[e1][0].isalnum() or not entities[e1][1].isalnum() \
            or not entities[e2][0].isalnum() or not entities[e2][1].isalnum():
        do_analysis = False
    else:
        e1_pos, e2_pos = positions_on_tree(analysis, entities[e1][0], entities[e1][1],
                                             entities[e2][0], entities[e2][1])
        if e1_pos == 0 or e2_pos == 0:
            do_analysis = False
            print(f)
            print(s)
            print("Error in the xml probably.")
        # If its the same word it has no interaction at all
        elif analysis.nodes[e1_pos]['lemma'] == analysis.nodes[e2_pos]['lemma']:
            do_analysis = False

    ddi_type = "null"

    if do_analysis:
        e1_to_root, e2_to_root, relation = get_from_root(analysis, e1_pos, e2_pos)
        # Eliminated (unnecessary)
        # if relation == "e1_under_e2" or relation == "e2_under_e1":
        #     # Check effect
        #     if ddi_type == "null":
        #         ddi_type = check_effect(e1_to_root, e2_to_root)
        #
        #     # # Check in between mechanism clue word (not effective)
        #     # if ddi_type == "null":
        #     #     ddi_type = check_mechanism(e1_to_root)
        #     # if ddi_type == "null":
        #     #     ddi_type = check_mechanism(e2_to_root)
        #
        # else:
        # Check effect
        if ddi_type == "null":
            ddi_type = check_effect(e1_to_root, e2_to_root)

        # Check advise
        if ddi_type == "null":
            ddi_type = check_adv(e1_to_root)
        if ddi_type == "null":
            ddi_type = check_adv(e2_to_root)

        # Check int
        if ddi_type == "null":
            ddi_type = check_int(e1_to_root)
        if ddi_type == "null":
            ddi_type = check_int(e2_to_root)

        # Check in between mechanism clue word
        if ddi_type == "null":
            ddi_type = check_mechanism(e1_to_root)
        if ddi_type == "null":
            ddi_type = check_mechanism(e2_to_root)
            """
            # Get root
            connection = 0
            if ddi_type == "null":
                for elem_e1 in e1_to_root:
                    for elem_e2 in e2_to_root:
                        if elem_e1 == elem_e2:
                            connection = elem_e1
                            break
                    if connection != 0:
                        break
            if ddi_type == "null":
                for aux in analysis.nodes[connection]['deps']['aux']:
                    ddi_type = contains_aux_advise(analysis.nodes[aux]['lemma'])

            # If the union verb is interact
            for elem_e1 in e1_to_root:
                if analysis.nodes[elem_e1]['lemma'] == "recommend":
                    ddi_type = "advise"
    
            for elem_e2 in e2_to_root:
                if analysis.nodes[elem_e2]['lemma'] == "recommend":
                    ddi_type = "advise"
            """
            # if ddi_type == "null" and analysis.nodes[connection]['lemma'] == "interact":
            #    ddi_type = "int"

            # If the union verb is recommend
            # if ddi_type == "null" and analysis.nodes[connection]['lemma'] == "recommend":
            #    ddi_type = "advise"

    if ddi_type == "null":
        is_ddi = "0"
    else:
        is_ddi = "1"

    return is_ddi, ddi_type


def evaluate(inputdir, outputfile):
    """
    Input: Receives a data directory and the filename for the
    results to evaluate. inputdir is the folder containing
    original XML (with the ground truth). outputfile is the
    file name with the entities produced by your system.

    Output: Prints statistics about the predicted entities in
    the given output file.

    Note: outputfile must match the pattern: task9.2_NAME_NUMBER.txt
    (where NAME may be any string and NUMBER any natural number).
    You can use this to encode the program version that produced the file.

    :param inputdir:
    :param outputfile:
    """
    os.system("java -jar eval/evaluateDDI.jar " + inputdir + " " + outputfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir", help="path of the input directory", default=DEF_INPUT_DIR)
    # parser.add_argument("datadir", help="path to the data directory", default=DEF_DATA_DIR)
    parser.add_argument("--groupname", help="path of the output file", default=DEF_GROUP_NAME)
    parser.add_argument("--version", help="", default=DEF_VERSION)
    args = parser.parse_args()

    output_file = "out/task9.2_" + args.groupname + "_" + args.version + ".txt"

    # connect to your CoreNLP server
    try:
        my_parser = CoreNLPDependencyParser(url="http://localhost:9000")
    except (ConnectionError, ConnectionRefusedError) as e:
        print("Loading parser\n")
        print("Error while trying to connect to CorNLP server. Try running:\n")
        print("\tcd stanford-corenlp-full-2018-10-05")
        print("\tjava -mx4g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer")
        exit()

    # process each file in directory
    with open(output_file, 'w') as outf:
        for f in os.listdir(args.inputdir):
            # parse XML file, obtaining a DOM tree
            tree = parse(args.inputdir + "/" + f)
            # process each sentence in the file
            sentences = tree.getElementsByTagName("sentence")
            for s in sentences:
                sid = s.attributes["id"].value  # get sentence id
                stext = s.attributes["text"].value  # get sentence text

                # load sentence entities into a dictionary
                entities = {}
                ents = s.getElementsByTagName("entity")
                for e in ents:
                    id = e.attributes["id"].value
                    offs = e.attributes["charOffset"].value.split("-")
                    entities[id] = offs

                #stext = "It may be necessary to adjust the dosage of oral anticoagulants upon beginning or stopping disulfiram. since disulfiram may prolong prothrombin time."
                # Tokenize, tag, and parse sentence
                if len(stext) > 0:
                    analysis = analyze(stext)
                # for each pair in the sentence, decide whether it is DDI and its type
                pairs = s.getElementsByTagName("pair")
                for p in pairs:
                    id_e1 = p.attributes["e1"].value
                    id_e2 = p.attributes["e2"].value
                    (is_ddi, ddi_type) = check_interaction(analysis, entities, id_e1, id_e2, stext, f)
                    print("|".join([sid, id_e1, id_e2, is_ddi, ddi_type]), file=outf)

    # get performance score
    evaluate(args.inputdir, output_file)
