import pandas as pd

def normal(s):
    s = s.lower()
    s = s.replace('(', ' ')
    s = s.replace('（', ' ')
    s = s.replace(')', ' ')
    s = s.replace('）', ' ')
    # s=s.replace(' ','')
    # s=s.replace(' ','')
    return s


def get_train_dev1():
    ################################################
    #       ####最长句子限制于100，多余的切前后，保证中间实体数目，复制句子与事件数目等次
    ########################################
    c = pd.read_csv('./work/data.csv', header=None, sep='\t')

    spl = int(c.shape[0] * 0.8)
    train = c[:spl]
    dev = c[spl:]

    def get_data2id(c_data, path):
        sent_length = []
        all_sent = []
        all_label = []
        s = ['text_a\tlabel']
        l=0
        for sent, dic in c_data[[1, 2]].values:

            sentence = list(sent)

            sent = normal(sent)
            sent_length.append(len(sentence))
            dic = eval(dic)
            label = []
            minbeg = 1000
            maxend = 0
            for k, v in dic.items():
                # print(k,v)
                entity_label = ['O'] * len(sentence)
                # print(entity2id)
                for entity in v:
                    entity = normal(entity)
                    beg = sent.find(entity)
                    end = sent.find(entity) + len(entity)
                    if (beg == -1):

                        # entity.strip()
                        for i in range(5):
                            sent = sent.replace(' ', '', i)
                            beg = sent.find(entity)
                            if (beg != -1):
                                end = beg + len(entity) + i
                                break
                        for i in range(5):
                            sent = sent.replace(' ', '', i)
                            if (sent.find(entity) != -1):
                                beg = sent.find(entity)
                                end = beg + len(entity) + i
                                break
                        # print(beg,end)
                        # if(beg==-1):
                        #     print(entity)
                        #     print(sent)
                    if (beg != -1):
                        if (beg < minbeg):
                            minbeg = beg
                        if (end > maxend):
                            maxend = end
                        entity_label[beg] = 'B-' + k
                        l+=1
                        for i in range(beg + 1, end):
                            entity_label[i] = 'I-' + k

                    # print(sent[beg:end+1],entity_label[beg:end+1])

                    # print(sent,v,sent.find(entity),sent[sent.find(entity)],sent[sent.find(entity)+len(entity)-1])
                    # sent.find(entity)
                label.extend(entity_label)

            if (len(label) == 0):
                if (len(sentence) > 100):
                    label = ["O"] * 100
                    sentence = sentence[:100]
                else:
                    label = ["O"] * len(sentence)
            else:

                if (len(sentence) > 100):
                    if (maxend - minbeg <= 100):
                        length = (100 - (maxend - minbeg)) // 2
                        # print(length,minbeg,maxend)
                        if (minbeg - length > 0):
                            beg = minbeg - length
                        else:
                            beg = 0
                    else:
                        beg = minbeg
                    end = beg + 100

                    if (end > len(sentence)):
                        end = len(label)
                        beg = end - 100

                    # print(sentence)
                    # print(len(label),len(sentence))
                    # print(len(label),len(sentence),beg,end,minbeg,maxend)
                    if (beg != -1):
                        ll = []

                        for i in range(len(label) // len(sentence)):
                            ll.extend(label[i * len(sentence) + beg:i * len(sentence) + end])
                        sentence = sentence[beg:end]
                        label = ll
                    else:
                        sentence = sentence[:100]
                        label = ["O"] * 100
                    # if(len(sentence)==0):
                    #     print(beg,end,len(ll))

                    # print(beg,end)
                    # print(len(ll),len(sentence))
                    # if(len(label)%len(sentence)!=0):
                    #     print(len(label),len(sentence),end-beg)
            try:
                ss = []
                for i in range(len(label) // len(sentence)):
                    ss.extend(sentence)
                # if(len(ss)!=len(label)):
                #     print(ss)
                #     print(label)
                #     print(len(sentence),len(ss),len(label))
                # print(' '.join(s) + '\t' + ' '.join(label))
                if (len(label) != len(ss)):
                    print('list:', len(label), len(ss))

                s.append("{}\t{}".format(str(ss), str(label)))

            except:
                pass
                # print(sent_length[-1])
                # print(sentence)
                # print(label)
                # print(len(sentence),len(label))
        print(l)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(s))
        return s

    train1 = get_data2id(train, './work/train.txt')
    dev1 = get_data2id(dev, './work/dev.txt')






########普通实体标注####################################
#{'B-主体': 0, 'I-主体': 1, 'O': 2}
# 36831
# 8925
def get_train_dev2():
    c = pd.read_csv('./work/data.csv', header=None, sep='\t')

    spl = int(c.shape[0] * 0.8)
    train = c[:spl]
    dev = c[spl:]

    def get_data2id(c_data, path):
        sent_length = []
        all_sent = []
        all_label = []
        s = ['text_a\tlabel']
        l = 0
        for sent, dic in c_data[[1, 2]].values:

            sentence = list(sent)

            sent = normal(sent)
            sent_length.append(len(sentence))
            dic = eval(dic)
            label = []
            minbeg = 1000
            maxend = 0
            entity_label = ['O'] * len(sentence)
            for k, v in dic.items():
                # print(k,v)
                # print(entity2id)
                for entity in v:
                    entity = normal(entity)
                    beg = sent.find(entity)
                    end = sent.find(entity) + len(entity)
                    if (beg == -1):
                        # entity.strip()
                        for i in range(5):
                            sent = sent.replace(' ', '', i)
                            beg = sent.find(entity)
                            if (beg != -1):
                                end = beg + len(entity) + i
                                break
                        for i in range(5):
                            sent = sent.replace(' ', '', i)
                            if (sent.find(entity) != -1):
                                beg = sent.find(entity)
                                end = beg + len(entity) + i
                                break
                        # print(beg,end)
                        # if(beg==-1):
                        #     print(entity)
                        #     print(sent)
                    if (beg != -1):
                        entity_label[beg] = 'B-主体'
                        for i in range(beg + 1, end):
                            entity_label[i] = 'I-主体'
                        # print(entity_label)
                        l += 1

            label = entity_label
            try:
                # print(len(sentence),sentence)
                # print(len(label),label)
                s.append("{}\t{}".format(str(sentence), str(label)))

            except:
                pass
                # print(sent_length[-1])
                # print(sentence)
                # print(label)
                # print(len(sentence),len(label))

        print(l)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(s))
        return s

    train1 = get_data2id(train, './work/train.txt')
    dev1 = get_data2id(dev, './work/dev.txt')


def get_train_dev3():
    c = pd.read_csv('./work/data.csv', header=None, sep='\t')

    spl = int(c.shape[0] * 0.8)
    train = c[:spl]
    dev = c[spl:]

    def get_data2id(c_data, path):
        sent_length = []
        all_sent = []
        all_label = []
        s = ['text_a\tlabel']
        l = 0
        for sent, dic in c_data[[1, 2]].values:

            sentence = list(sent)

            sent = normal(sent)
            sent_length.append(len(sentence))
            dic = eval(dic)
            label = []
            minbeg = 1000
            maxend = 0
            entity_label = ['O'] * len(sentence)
            for k, v in dic.items():
                # print(k,v)
                # print(entity2id)
                for entity in v:
                    entity = normal(entity)
                    beg = sent.find(entity)
                    end = sent.find(entity) + len(entity)
                    if (beg == -1):
                        # entity.strip()
                        for i in range(5):
                            sent = sent.replace(' ', '', i)
                            beg = sent.find(entity)
                            if (beg != -1):
                                end = beg + len(entity) + i
                                break
                        for i in range(5):
                            sent = sent.replace(' ', '', i)
                            if (sent.find(entity) != -1):
                                beg = sent.find(entity)
                                end = beg + len(entity) + i
                                break
                        # print(beg,end)
                        # if(beg==-1):
                        #     print(entity)
                        #     print(sent)
                    if (beg != -1):
                        entity_label[beg] = 'B-' + k
                        for i in range(beg + 1, end):
                            entity_label[i] = 'I-' + k
                        # print(entity_label)

            label = entity_label
            try:
                l += 1

                # else:
                # print(len(sentence),sentence)
                # print(len(label),label)
                s.append("{}\t{}".format(str(sentence), str(label)))

            except:
                pass
                # print(sent_length[-1])
                # print(sentence)
                # print(label)
                # print(len(sentence),len(label))

        print(l)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(s))
        return s

    train1 = get_data2id(train, './work/train.txt')
    dev1 = get_data2id(dev, './work/dev.txt')


# get_train_dev()

def count_dev():
    # get_train_dev2()
    id=7
    count_correct=0
    all=0
    path=['./result/2_maxlen倍数句子','./result/2_maxlen5025倍test','result/2_listext','./result/4_实体识别_1_3的O','./result/3_实体识别','result/5_事件标注无重叠',
          './result/5_dev'][id-1]
    result=list()

    with open(path,'r',encoding='utf-8') as f:

        for line in f:
            dic=eval(line.strip())
            # print(dic['id'],dic['text'],dic['labels'])
            entity=[]
            label=[]
            gold=[]
            for g,t,l in zip(dic['id'],dic['text'],dic['labels']):
                all+=1
                if(g==l):
                    count_correct+=1
                print(t,l,g)
                if(l[0]!='I'and len(entity)!=0):
                    label_event=label[0][2:]
                    result.append(''.join(dic['text'])+','+label_event+','+''.join(entity))
                    print(''.join(entity),label,gold)
                    if (l[0] == 'B'):
                        entity = [t]
                        label = [l]
                        gold=[g]
                    else:
                        entity=[]
                        label=[]
                        gold=[]
                else:
                    if(l[0]=='B'):
                        entity=[t]
                        label=[l]
                        gold=[g]
                    elif l[0]=='I':
                        entity.append(t)
                        label.append(l)
                        gold.append(g)
    print(path+'predict'+str(id))
    with open(path+'_predict_'+str(id)+'.csv','w',encoding='utf-8') as f:
        f.write('\n'.join(result))
    print(count_correct*1.0/all)

def see_predict(path,id):
    result = list()
    with open(path,'r',encoding='utf-8') as f:

        for line in f:
            dic=eval(line.strip())
            # print(dic['id'],dic['text'],dic['labels'])
            entity=[]
            label=[]

            for t,l in zip(dic['text'],dic['labels']):
                # print(t,l)
                if(l[0]!='I'and len(entity)!=0):
                    label_event=label[0][2:]
                    # result.append(str(dic['id'])+','+''.join(dic['text'])+','+label_event+','+''.join(entity))
                    result.append(
                        str(dic['id']) + '\t' + label_event + '\t' + ''.join(entity))
                    print(''.join(entity))
                    if (l[0] == 'B'):
                        entity = [t]
                        label = [l]
                    else:
                        entity=[]
                        label=[]
                else:
                    if(l[0]=='B'):
                        entity=[t]
                        label=[l]
                    elif l[0]=='I':
                        entity.append(t)
                        label.append(l)
    with open(path+'_predict_tj_'+str(id)+'.csv','w',encoding='utf-8') as f:
        f.write('\n'.join(result))

see_predict('./result/20',20)