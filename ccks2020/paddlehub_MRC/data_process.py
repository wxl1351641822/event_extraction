import pandas as pd

import json
def is_alphabet(uchar):
        """判断一个unicode是否是英文字母"""
        if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
                return True
        else:
                return False
def get_data2id(c_data,  path):
    sent_length = []
    data=[]
    max_seq_len=510
    for ids,sent, dic in c_data.values:
        paragraphs=[]
        ids=eval(ids)
        dic = eval(dic)
        for s in range(len(sent)//max_seq_len+1):
            context=sent[s*(max_seq_len):(s+1)*max_seq_len]
            qas=[]
            for i,(k, v) in enumerate(dic.items()):
                question="{}的主体是什么？".format(k)
                answers=[]
                for entity in v:
                    beg = context.find(entity)
                    for tag in ['，','.']:
                        if(tag == entity[0]):
                            entity=entity[1:]
                            # print(entity)
                        if(tag==entity[-1]):
                            if(not is_alphabet(entity[-2])):
                                entity=entity[:-1]
                            # print(entity)
                    # end = sent.find(entity) + len(entity)-1
                    if(beg!=-1):
                        answers.append({"text":entity,"answer_start":beg})
                if(len(answers)!=0):
                    qas.append({"id":ids[i%(len(ids))],"question":question,"answers":answers})
            paragraphs.append({"context":context,"qas":qas})
        data.append({"title":"","paragraphs":paragraphs})
        # if(len(paragraphs)>1):
        #     print(data[-1])
        # print(data[-1])
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps({"data":data}))
    return data


def read_dic(path):
    with open(path, 'r', encoding='utf-8') as f:
        dic = f.read()
        # print(dic)
        dic = eval(dic)
    return dic
def get_train_dev():
    # word2id = read_dic('./work/event/word2id.txt')
    c = pd.read_csv('./work/event/data.csv', header=None, sep='\t')
    spl = int(c.shape[0] * 0.8)
    train = c[:spl]
    dev = c[spl:]
    get_data2id(train,  './work/event/train.json')
    get_data2id(dev,  './work/event/dev.json')

get_train_dev()

# with open('./work/event/train.json', 'r', encoding='utf-8') as f:
#     data=json.load(f)
# print(data)