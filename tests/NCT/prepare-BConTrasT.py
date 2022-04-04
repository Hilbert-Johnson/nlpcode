import json, os

os.mkdir('data')
datapath = '/home/zhaosh/'

os.chdir('/home/zhaosh/BConTrasT')

for pre in ['train','dev','test1_corpus-with_gold_references']:
    with open( pre + '.json', 'r') as f:
        data = json.load(f)

    with open( pre + '.src', 'w') as f1, open( pre + '.tgt', 'w') as f2:
        for key, conversation in data.items():
            for uttr in conversation:
                f1.write(uttr['source']+'\n')
                f2.write(uttr['target']+'\n')