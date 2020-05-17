import pandas as pd
# data=pd.read_csv('./data/data34808/train_label.csv',sep='\t',header=None).fillna('')


def get_data():
    e = []
    idd = []
    count = data[1].value_counts()
    for k in count.keys():
        # print(k)
        triples = data[data[1] == k]
        id = []
        for i in triples[0]:
            id.append(i)
        dic = {}
        for event, role in triples[[2, 3]].values:
            if (event == ''):
                continue
            try:
                dic[event].append(role)
            except:
                dic[event] = [role]

        for k, v in dic.items():
            dic[k] = list(set(v))

        # print(dic)
        e.append(str(dic))

        idd.append(str(id))
    c = pd.DataFrame()
    c[0] = idd
    c[1] = count.keys()
    c[2] = count.tolist()
    c[3] = e
    c = c.sample(frac=1.0)
    c.to_csv('./data/event/data.csv', header=None, index=False)

def get_word2id_txt(lis):
    word=[]
    for d in lis:
        word.extend(list(d))
    word=set(word)
    word=['<PAD>','<UNK>']+list(word)
    word_dic={}
    for i,w in enumerate(word):
        word_dic[w]=i
    with open('./data/event/word2id.txt','w',encoding='utf-8') as f:
        f.write(str(word_dic))
# word_length=5566
# event_length=29
def get_event_dict(lis,beglist,path):
    e=[]
    for d in lis:
        if(d!=''):
            e.append(d)
    e=['<NA>']+list(set(e))
    dic={}
    for i,w in enumerate(e):
        dic[w]=i
    with open(path,'w',encoding='utf-8') as f:
        f.write(str(dic))

def get_entity2id(event2id):
    r=[]
    for k in event2id.keys():
        # print(k)
        if k!='<NA>':
            r.extend(['B-'+k,'M-'+k,'E-'+k,'S-'+k])
    r=['O']+list(set(r))
    entity_dic={}
    for i,w in enumerate(r):
        entity_dic[w]=i
    with open('./entity2id.txt','w',encoding='utf-8') as f:
        f.write(str(entity_dic))

def read_dic(path):
    with open(path,'r',encoding='utf-8') as f:
        dic=f.read()
        # print(dic)
        dic=eval(dic)
    return dic
def read_list(path):
    with open(path,'r',encoding='utf-8') as f:
        dic=f.read()
        # print(dic)
        dic=eval(dic)
    return list(dic.keys())


def get_data2id(c_data, word2id, event2id, entity2id, path):
    sent_length = []
    all_sent = []
    all_label = []
    for sent, dic in c_data[[1, 3]].values:
        sentence = [word2id[x] for x in list(sent)]
        sent_length.append(len(sentence))
        dic = eval(dic)
        label = []
        for k, v in dic.items():
            # print(k,v)
            label.append(event2id[k])
            entity_label = [entity2id['O']] * len(sentence)
            # print(entity2id)
            for entity in v:
                beg = sent.find(entity)
                end = sent.find(entity) + len(entity) - 1
                if (beg == end):
                    entity_label[beg] = entity2id['S-' + k]
                else:
                    entity_label[beg] = entity2id['B-' + k]
                    entity_label[end] = entity2id['E-' + k]
                    for i in range(beg + 1, end):
                        entity_label[i] = entity2id['M-' + k]
                # print(sent[beg:end+1],entity_label[beg:end+1])

                # print(sent,v,sent.find(entity),sent[sent.find(entity)],sent[sent.find(entity)+len(entity)-1])
                sent.find(entity)
            label.extend(entity_label)
        all_sent.append(sentence)
        all_label.append(label)
    data = [all_sent, all_label]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str(data))
    return data



# word2id=read_dic('./data/event/word2id.txt')
event2id=read_dic('./event2id.txt')
# entity2id=read_dic('./data/event/entity2id.txt')
get_entity2id(event2id)

# c=pd.read_csv('./work/data.csv',header=None)
# spl=int(c.shape[0]*0.8)
# train=c[:spl]
# dev=c[spl:]
# train1 = get_data2id(train, word2id, event2id, entity2id, './work/train.txt')
# dev1 = get_data2id(train, word2id, event2id, entity2id, './work/dev.txt')
# get_event_dict(lis=data[2][:],beglist=['<NA>'],path='./work/event2id.txt')
# get_word2id_txt(data[1][:])