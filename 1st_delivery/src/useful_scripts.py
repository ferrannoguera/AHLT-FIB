import json
from nltk.tokenize import word_tokenize


def create_occurrences_file():
    f = open("entities/group.txt")
    l = f.readlines()
    f.close()
    l = [s[0:-1] for s in l]
    s = set(l)
    d = {}
    for e in s:
        d[e] = l.count(e)
    d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
    f = open("entities/group_occurrences.txt", 'w')
    for e in d:
        print(str(d[e]), e, file=f)
    f.close()

# create_occurrences_file()

def file2dict(label, dict):
    with open("entities/" + label + ".txt", 'r') as f:
        l = f.readlines()
    l = [s[0:-1] for s in l]
    l = set(l)
    #l = [word_tokenize(s) for s in l]
    for e in l:
        dict[e] = label
    return dict


# with open("entities/entity_dict_test.json", 'r') as f:
#     dict = json.load(f)
dict = {}
dict = file2dict("drug", dict)
dict = file2dict("drug_n", dict)
dict = file2dict("brand", dict)
dict = file2dict("group", dict)

with open("entities/entity_dict_2.json", 'w') as f:
    json.dump(dict, f)


def suff2dict(label,dict):
    with open("entities/" + label + ".txt", 'r') as f:
        l = f.readlines()

    l = [s[-5:-1] for s in l]
    l = set(l)
    l = [word_tokenize(s) for s in l]
    for e in l:
        dict[e[0]] = label
    return dict

# dict = {}
# dict = suff2dict("drug", dict)
#dict = suff2dict("drug_n", dict)
#dict = suff2dict("brand", dict)
#dict = suff2dict("group", dict)

# with open("entities/suffix_dict.json", 'w') as f:
#     json.dump(dict, f)
