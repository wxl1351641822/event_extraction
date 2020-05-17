def show():
    with open('data/webnlg/entity_end_position/train.json','r',encoding='utf-8') as f:
        dev=eval(f.read())
    print(len(dev))
    print(len(dev[0]))
    print(len(dev[1]))

    def get_id2(path):
        with open(path,'r',encoding='utf-8') as f:
            dic=eval(f.read())
        return [k for k,v in sorted(dic.items(), key=lambda d: d[1])]
    id2r=get_id2('data/webnlg/entity_end_position/relations2id.json')
    # print(id2r)
    id2w=get_id2('data/webnlg/entity_end_position/words2id.json')
    print(id2w)
    for i in range(10):
        print([id2w[w] for w in dev[0][i]])
        l=[]
        for j in range(len(dev[1][i])):
           if(j%3==0):
               l.append(id2r[dev[1][i][j]])
           else:
               l.append(id2w[dev[1][i][j]])
        print(l)
import json
import const
config_filename = './config.json'
cell_name = 'lstm'
decoder_type = 'one'
config = const.Config(config_filename=config_filename, cell_name=cell_name, decoder_type=decoder_type)
words_id2vec = json.load(open(config.words_id2vector_filename, 'r'))
words_vectors = [0] * (len(words_id2vec) + 1)
json
for i, key in enumerate(words_id2vec):
    if(key<)
    words_vectors[int(key)] = words_id2vec[key]