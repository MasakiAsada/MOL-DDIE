import sys
import pickle as pkl

if len(sys.argv) != 2:
    sys.stderr.write('Usage: python3 %s base ' % (sys.argv[0]))
    sys.exit(-1)

class Instance:
    def __init__(self, sent, label, entity):
        #print(sent, label, entity)
        self.words = [x.split('\t')[0] for x in sent.split('\n')]
        self.label = label
        self.entity = entity
        #print(self.words, label, entity)
        self.ep = (self.words.index('DRUG1'), self.words.index('DRUG2'))

    def word_to_indx(self, indx_dict, freq_dict, padding=True,
            max_sent_len=200, freq_threshold=1):
        self.indxs = []
        for w in self.words:
            w = w.lower()
            if w in freq_dict and freq_dict[w] <= freq_threshold:
                self.indxs.append(indx_dict['<UNK>'])
            elif w in indx_dict:
                self.indxs.append(indx_dict[w])
            else:
                self.indxs.append(indx_dict['<UNK>'])
        
sent_in = open(sys.argv[1] + '.split.sent', 'r')
entity_in = open(sys.argv[1] + '.ent', 'r')
label_in = open(sys.argv[1] + '.label', 'r')

sents = sent_in.read().strip().split('\n\n')
#sent_lines = sents.split('\n\n')
entities = entity_in.read().strip().split('\n')


def main():
    for sent,y,z in zip(sents, entities, labels):
        words = [x.split('\t')[0] for x in sent.split('\n')]
        print(words)
