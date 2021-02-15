import argparse
from xml.dom.minidom import parse, parseString
import os
from nltk.parse.corenlp import CoreNLPDependencyParser
import sys

# DEFAULT VALUES
TRAIN = "Train"
DEVEL = "Devel"
TEST = "Test-DDI"
DEF_TRAIN_DIR = "data/" + TRAIN
DEF_DEVEL_DIR = "data/" + DEVEL
DEF_TEST_DIR = "data/" + TEST
DEF_GROUP_NAME = "ArnauCanyadell_FerranNoguera"
DEF_VERSION = "101"


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='',
                     decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


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


def get_path(analysis, address):
    current_address = address
    path = [current_address]
    while current_address != 0:
        current_address = analysis.nodes[current_address]['head']
        path.append(current_address)
    path.reverse()
    return path


def common_path(path1, path2):
    i = 0
    while i < len(path1) and i < len(path2) and path1[i] == path2[i]:
        i += 1
    if i == 0:
        return []
    path = path1[i:]
    path.reverse()
    return path + path2[i - 1:]


def features_lemmas_before_between_after(analysis, e1_addr, e2_addr):
    features = []
    d = {
        0: 'lb1=',
        1: 'lib=',
        2: 'la2='
    }
    i = 0
    for addr in range(1, len(analysis.nodes)):
        if addr == e1_addr or addr == e2_addr:
            i += 1
        else:
            if analysis.nodes[addr]['rel'] != 'punct':
                features.append(d[i] + analysis.nodes[addr]['lemma'])
    return features


def extract_features(analysis, entities, e1, e2):
    """

    :param analysis: analyzed sentence tree
    :param entities: entities present in the sentence
    :param e1: id of the first target entity
    :param e2: id of the second target entity
    :return: list of binary features, preceeded by [sic]
    """
    features = []
    if not entities[e1][0].isalnum() or not entities[e1][1].isalnum() \
            or not entities[e2][0].isalnum() or not entities[e2][1].isalnum():
        return ['root-error']
        # likely an error in xml
    else:
        e1_addr, e2_addr = positions_on_tree(analysis, entities[e1][0], entities[e1][1],
                                             entities[e2][0], entities[e2][1])
    if e1_addr == 0 or e2_addr == 0:
        return ['root-error']
        # likely an error in xml

    if analysis.nodes[e1_addr]['lemma'] == analysis.nodes[e2_addr]['lemma']:
        features = features + ["sameMedicine"]
    else:
        e1_path = get_path(analysis, e1_addr)
        e2_path = get_path(analysis, e2_addr)
        e1_e2_path = common_path(e1_path, e2_path)
        # True if e1 under e2
        relation_e1_e2 = [True for elem in e1_path if elem == e2_addr]
        # True if e2 under e1
        relation_e2_e1 = [True for elem in e2_path if elem == e1_addr]
        # 1under2
        if len(relation_e1_e2) > 0:
            for elem in e1_e2_path[1:-1]:
                features = features + ["tag" + "=" + analysis.nodes[elem]['tag']]
                features = features + ["dep" + "=" + analysis.nodes[elem]['rel']]
            features = features + ['dep' + "=" + analysis.nodes[e1_e2_path[0]]['rel']]
        # 2under1
        elif len(relation_e2_e1) > 0:
            for elem in e1_e2_path[1:-1]:
                features = features + ["tag" + "=" + analysis.nodes[elem]['tag']]
                features = features + ["dep" + "=" + analysis.nodes[elem]['rel']]
            features = features + ['dep' + "=" + analysis.nodes[e1_e2_path[-1]]['rel']]
        # independent
        else:
            connection = 0
            e1_path.reverse()
            e2_path.reverse()
            for elem_e1 in e1_path:
                for elem_e2 in e2_path:
                    if elem_e1 == elem_e2:
                        connection = elem_e1
                        break
                if connection != 0:
                    break

            for elem in e1_path[1:-1]:
                if elem == connection:
                    break
                else:
                    features = features + ["tag" + "=" + analysis.nodes[elem]['tag']]
                    features = features + ["dep" + "=" + analysis.nodes[elem]['rel']]
            features = features + ['dep' + "=" + analysis.nodes[e1_e2_path[0]]['rel']]
            for elem in e2_path[1:-1]:
                if elem == connection:
                    break
                else:
                    features = features + ["tag" + "=" + analysis.nodes[elem]['tag']]
                    features = features + ["dep" + "=" + analysis.nodes[elem]['rel']]
            features = features + ['dep' + "=" + analysis.nodes[e1_e2_path[-1]]['rel']]

            features = features + ["independent"]

        features = features + features_lemmas_before_between_after(analysis, e1_addr, e2_addr)

    return features


def output_features(id, e1, e2, type, features, isMegam=False, file=sys.stdout):
    """
    Prints to stdout the feature vector in the following format: one single line per vector, with tab-separated fields:
    `sent_id, ent_id1, ent_id2, gold_class, feature1, feature2, ...`
    If isMegam, output is formatted like this:

    :param id: sentence id
    :param e1: first entity
    :param e2: second entity
    :param type: type of DDI (gold class or prediction)
    :param features: list of binary features
    :param isMegam: True if output must be fed to Megam algorithm
    :return:
    """
    if not isMegam:
        l = [id, e1, e2, type] + features
    else:
        l = [str(class_name_to_number(type))] + features
    file.write("\t".join(l) + "\n")


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


def class_name_to_number(class_name):
    if class_name == "null":
        return 0
    elif class_name == "mechanism":
        return 1
    elif class_name == "effect":
        return 2
    elif class_name == "advise":
        return 3
    elif class_name == "int":
        return 4
    raise ValueError("class name should be one of {null, mechanism, effect, advise, int}, bit instead found: "
                     + str(class_name))


def create_features_file(input_dir, features_file, features_megan_file):
    outf = open(features_file, 'w')
    outf_megan = open(features_megan_file, 'w')
    print("Creating features for " + input_dir.split("/")[1] + " directory.")
    l = len(os.listdir(input_dir))
    printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)
    for i, f in enumerate(os.listdir(input_dir)):
        # parse XML file, obtaining a DOM tree
        tree = parse(input_dir + "/" + f)
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

            # Tokenize, tag, and parse sentence
            if len(stext) > 0:
                analysis = analyze(stext)
            # for each pair in the sentence, decide whether it is DDI and its type
            pairs = s.getElementsByTagName("pair")
            for p in pairs:
                id_e1 = p.attributes["e1"].value
                id_e2 = p.attributes["e2"].value
                # is_ddi = p.attributes["ddi"].value
                ddi_type = "null"
                if p.attributes["ddi"].value == "true":
                    try:
                        ddi_type = p.attributes["type"].value
                    except (KeyError, ValueError):  # XML annotation error
                        pass  # is_ddi = "false"
                features = extract_features(analysis, entities, id_e1, id_e2)
                output_features(sid, id_e1, id_e2, ddi_type, features, file=outf)
                output_features(sid, id_e1, id_e2, ddi_type, features, file=outf_megan, isMegam=True)
        printProgressBar(i + 1, l, prefix='Progress:', suffix='Complete', length=50)
    outf.close()
    outf_megan.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdirtrain", help="path of the train input directory", default=DEF_TRAIN_DIR)
    parser.add_argument("--inputdirdevel", help="path of the devel input directory", default=DEF_DEVEL_DIR)
    parser.add_argument("--inputdirtest", help="path of the test input directory", default=DEF_TEST_DIR)
    parser.add_argument("--createtest", help="create test features", choices=["True", "False"], default="False")
    parser.add_argument("--groupname", help="path of the output file", default=DEF_GROUP_NAME)
    parser.add_argument("--version", help="version of the algorithm", default=DEF_VERSION)
    args = parser.parse_args()

    features_files = ["features/features_train_" + args.version + ".txt",
                      "features/features_devel_" + args.version + ".txt",
                      "features/features_test_" + args.version + ".txt"]

    features_megan_files = ["features/features_megan_train_" + args.version + ".txt",
                            "features/features_megan_devel_" + args.version + ".txt",
                            "features/features_megan_test_" + args.version + ".txt"]

    # connect to your CoreNLP server
    try:
        my_parser = CoreNLPDependencyParser(url="http://localhost:9000")
    except (ConnectionError, ConnectionRefusedError) as e:
        print("Loading parser\n")
        print("Error while trying to connect to CorNLP server. Try running:\n")
        print("\tcd stanford-corenlp-full-2018-10-05")
        print("\tjava -mx4g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer")
        exit()

    #create_features_file(args.inputdirtrain, features_files[0], features_megan_files[0])
    #create_features_file(args.inputdirdevel, features_files[1], features_megan_files[1])
    #if str(args.createtest) == "True":
    create_features_file(args.inputdirtest, features_files[2], features_megan_files[2])

    # get performance score
    # evaluate(args.inputdir, output_file)
