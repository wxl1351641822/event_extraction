import pandas as pd
import json
def is_alphabet(uchar):
        """判断一个unicode是否是英文字母"""
        if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
                return True
        else:
                return False
def get_data2id(c_data, max_seq_len, path,ispredict=False):
    sent_length = []
    data=[]
    question_num=0
    for ids,sent, dic in c_data.values:
        paragraphs=[]
        if(not ispredict):
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
                elif(ispredict):
                    qas.append({"id":str(ids)+'_'+str(s)+'_'+str(i),"question":question})
                    question_num+=1
            paragraphs.append({"context":context,"qas":qas})
        data.append({"title":"","paragraphs":paragraphs})
        # if(len(paragraphs)>1):
        #     print(data[-1])
        # print(data[-1])
    print(question_num)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps({"data":data}))
    return data


def read_dic(path):
    with open(path, 'r', encoding='utf-8') as f:
        dic = f.read()
        # print(dic)
        dic = eval(dic)
    return dic
def get_train_dev(max_seq_len):
    # word2id = read_dic('./work/event/word2id.txt')
    c = pd.read_csv('./work/event/data.csv', header=None, sep='\t')
    spl = int(c.shape[0] * 0.8)
    train = c[:spl]
    dev = c[spl:]
    get_data2id(train, max_seq_len, './work/event/train.json')
    get_data2id(dev, max_seq_len, './work/event/dev.json')
def get_predict(id,train_i,max_seq_len):
    predict = pd.read_csv('./work/result/{}event_predict.csv'.format(train_i), header=None, sep='\t')
    get_data2id(predict,max_seq_len,'./work/event/{}predict.json'.format(id),ispredict=True)


def get_result(id,train_i,max_seq_len):
    c_data = pd.read_csv('./work/result/{}event_predict.csv'.format(train_i), header=None, sep='\t')
    with open('./work/result/submit{}_{}.json'.format(train_i,id), "r") as reader:
        submit = json.load(reader)


    result = []
    for ids, sent, dic in c_data.values:
        dic = eval(dic)
        if (dic):
            for s in range(len(sent) // max_seq_len + 1):
                context = sent[s * (max_seq_len):(s + 1) * max_seq_len]
                for i, (k, v) in enumerate(dic.items()):
                    result.append('\t'.join([str(ids), k, submit[str(ids) + '_' + str(s) + '_' + str(i)]]))
                    # print('\t'.join([str(ids), context, k, submit[str(ids) + '_' + str(s) + '_' + str(i)]]))
        else:
            result.append('\t'.join([str(ids), '', '']))

    with open('./work/result/ucas_valid_result{}_{}.csv'.format(train_i,id), 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))



