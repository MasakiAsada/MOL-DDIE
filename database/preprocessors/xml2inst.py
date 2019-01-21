import sys
import pickle as pkl
import itertools
from lxml import etree


if len(sys.argv) != 4:
    sys.stderr.write('Usage: python3 {} drugbank_xml smiles_dict_path corpus_test'.format(sys.argv[0]))
    sys.exit(-1)

root = etree.parse(sys.argv[1], parser=etree.XMLParser())

with open(sys.argv[2], 'rb') as f:
    smiles_dict = pkl.load(f)

all_drugs = set()
interacted_pairs = set()

# Get interacted drug pairs
for drug in root.xpath('./*[local-name()="drug"]'):
    drug_id = drug.xpath('./*[local-name()="drugbank-id"][@primary="true"]')[0].text
    if drug_id not in smiles_dict:
        continue
    all_drugs.add(drug_id)

    interacted_drugs = drug.xpath('./*[local-name()="drug-interactions"]')[0]
    for interacted_drug in interacted_drugs:
        interacted_drug_id = interacted_drug.xpath('./*[local-name()="drugbank-id"]')[0].text
        if interacted_drug_id not in smiles_dict:
            continue
        pair = sorted([drug_id, interacted_drug_id])
        interacted_pairs.add(':'.join(pair))


# Delete drug pairs in test corpus
with open(sys.argv[3] + '.dbid', 'r') as f:
    dbid = f.read().strip().split('\n')
for pair in dbid:
    pair = ':'.join(sorted(pair.split('\t')))
    if pair in interacted_pairs:
        interacted_pairs.remove(pair)

# Create pseudo negative pairs
all_pairs = set()
for x in itertools.combinations(all_drugs, 2):
    all_pairs.add(':'.join(sorted(list(x))))
not_interacted_pairs = all_pairs - interacted_pairs

n_pos = len(interacted_pairs)
n_neg = len(not_interacted_pairs)
n = min(n_pos, n_neg)
ratio = 4
train_out = open('../instances/train.txt', 'w')
test_out = open('../instances/test.txt', 'w')
for i, (pos, neg) in enumerate(zip(interacted_pairs, not_interacted_pairs)):
    if i < n//(ratio+1)*ratio:
        train_out.write('{}\t{}\n'.format(pos, 1))
        train_out.write('{}\t{}\n'.format(neg, 0))
    else:
        test_out.write('{}\t{}\n'.format(pos, 1))
        test_out.write('{}\t{}\n'.format(neg, 0))
train_out.close()
test_out.close()
