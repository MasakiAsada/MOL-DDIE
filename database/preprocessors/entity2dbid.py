import sys
import pickle as pkl
from lxml import etree


# Convert corpus entity to drugbank id

if len(sys.argv) != 3:
    sys.stderr.write('Usage: python3 {} name_dict_path corpus_base'.format(sys.argv[0]))
    sys.exit(-1)

with open(sys.argv[1], 'rb') as f:
    name_dict = pkl.load(f)

def relaxed_match(entity, dic):
    matched = {} # key: drugbank_id, value: matching rate
    for k, v in dic.items():
        if entity.lower() in k.lower() or k.lower() in entity.lower():
            matched[v] = abs(len(k)-len(entity))
    if len(matched) == 0:
        return None
    else:
        return min(matched, key=matched.get)

def exact_match(entity, dic):
    for k, v in dic.items():
        if k.lower() == entity.lower():
            return v
    return None

base = sys.argv[2]
with open(base + '.ent', 'r') as f:
    entities = f.read().strip().split('\n')
entity_names = [x.split('\t')[:3:2] for x in entities]

entity_set = set()
for e1, e2 in entity_names:
    entity_set.add(e1.lower())
    entity_set.add(e2.lower())

# Match corpus entity with drugbank entry
matched_dict = {} # key: corpus_entity_name, value: drugbank_id
for e in entity_set:
    matched_dict[e.lower()] = relaxed_match(e.lower(), name_dict)

dbid_out = open(base + '.dbid', 'w')
for e1, e2 in entity_names:
    dbid_out.write('{}\t{}\n'.format(matched_dict[e1.lower()], matched_dict[e2.lower()]))
dbid_out.close()

