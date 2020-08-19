#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""hello world"""
import os
import sys
import six
import json
import argparse
import collections
import pandas as pd
from collections import Counter

def read_by_lines(path, encoding="utf-8"):
    """read the data by line"""
    result = list()
    with open(path, "r",encoding=encoding) as infile:
        for line in infile:
            result.append(line.strip())
    return result


def write_by_lines(path, data, t_code="utf-8"):
    """write the data"""
    print(path,len(data))
    with open(path, "w",encoding='utf-8') as outfile:
        [outfile.write(d + "\n") for d in data]#.encode(t_code)


def data_process(path, model="trigger", is_predict=False):
    """data_process"""

    def label_data(data, start, l, _type):
        """label_data"""
        for i in range(start, start + l):
            suffix = u"B-" if i == start else u"I-"
            data[i] = u"{}{}".format(suffix, _type)
        return data

    sentences = []
    output = [u"text_a"] if is_predict else [u"text_a\tlabel"]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            d_json = json.loads(line.strip())#.decode("utf-8")
            _id = d_json["id"]
            text_a = [
                u"，" if t == u" " or t == u"\n" or t == u"\t" else t
                for t in list(d_json["text"].lower())
            ]
            if is_predict:
                sentences.append({"text": d_json["text"], "id": _id})
                output.append(u'\002'.join(text_a))
            else:
                if model == u"trigger":
                    labels = [u"O"] * len(text_a)
                    for event in d_json["event_list"]:
                        event_type = event["event_type"]
                        start = event["trigger_start_index"]
                        trigger = event["trigger"]
                        labels = label_data(labels, start,
                                            len(trigger), event_type)
                    output.append(u"{}\t{}".format(u'\002'.join(text_a),
                                                   u'\002'.join(labels)))
                elif model == u"role":
                    for event in d_json["event_list"]:
                        labels = [u"O"] * len(text_a)
                        for arg in event["arguments"]:
                            role_type = arg["role"]
                            argument = arg["argument"]
                            start = arg["argument_start_index"]
                            labels = label_data(labels, start,
                                                len(argument), role_type)
                        output.append(u"{}\t{}".format(u'\002'.join(text_a),
                                                       u'\002'.join(labels)))
    if is_predict:
        return sentences, output
    else:
        return output


def schema_process(path, model="trigger"):
    """schema_process"""

    def label_add(labels, _type):
        """label_add"""
        if u"B-{}".format(_type) not in labels:
            labels.extend([u"B-{}".format(_type), u"I-{}".format(_type)])
        return labels

    labels = []
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            d_json = json.loads(line.strip())
            if model == u"trigger":
                labels = label_add(labels, d_json["event_type"])
            elif model == u"role":
                for role in d_json["role_list"]:
                    labels = label_add(labels, role["role"])
    labels.append(u"O")
    return labels


def extract_result(text, labels):
    """extract_result"""
    ret, is_start, cur_type = [], False, None
    for i, label in enumerate(labels):
        if label != u"O":
            _type = label[2:]
            if label.startswith(u"B-"):
                is_start = True
                cur_type = _type
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif _type != cur_type:
                """
                # 如果是没有B-开头的，则不要这部分数据
                cur_type = None
                is_start = False
                """
                cur_type = _type
                is_start = True
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif is_start:
                ret[-1]["text"].append(text[i])
            else:
                cur_type = None
                is_start = False
        else:
            cur_type = None
            is_start = False
    return ret


def predict_data_process(trigger_file, role_file, schema_file, save_path):
    """predict_data_process"""
    pred_ret = []
    trigger_datas = read_by_lines(trigger_file)
    role_datas = read_by_lines(role_file)
    schema_datas = read_by_lines(schema_file)
    schema = {}
    for s in schema_datas:
        d_json = json.loads(s)
        schema[d_json["event_type"]] = [r["role"] for r in d_json["role_list"]]
    # 将role数据进行处理
    sent_role_mapping = {}
    for d in role_datas:
        d_json = json.loads(d)
        r_ret = extract_result(d_json["text"], d_json["labels"])
        role_ret = {}
        for r in r_ret:
            role_type = r["type"]
            if role_type not in role_ret:
                role_ret[role_type] = []
            role_ret[role_type].append(u"".join(r["text"]))
        sent_role_mapping[d_json["id"]] = role_ret

    for d in trigger_datas:
        d_json = json.loads(d)
        t_ret = extract_result(d_json["text"], d_json["labels"])
        pred_event_types = list(set([t["type"] for t in t_ret]))
        event_list = []
        for event_type in pred_event_types:
            role_list = schema[event_type]
            arguments = []
            for role_type, ags in sent_role_mapping[d_json["id"]].items():
                if role_type not in role_list:
                    continue
                for arg in ags:
                    if len(arg) == 1:
                        # 一点小trick
                        continue
                    arguments.append({"role": role_type, "argument": arg})
            event = {"event_type": event_type, "arguments": arguments}
            event_list.append(event)
        pred_ret.append({
            "id": d_json["id"],
            "text": d_json["text"],
            "event_list": event_list
        })
    pred_ret = [json.dumps(r, ensure_ascii=False) for r in pred_ret]
    write_by_lines(save_path, pred_ret)

def get_arguments(args):
    """print_arguments"""
    title=[]
    val=[]
    for arg, value in sorted(six.iteritems(vars(args))):
        title.append(arg)
        val.append(value)
    title=[str(x) for x in title]
    val=[str(x) for x in val]
    return ','.join(title),','.join(val)

def write_title(path,args,shiyan):
    title,_=get_arguments(args)
    s=shiyan+'\nid,f1,'
    with open(path,'a') as f:
        f.write(s+title+',score,备注\n')

def write_log(path,args,s):
    _,val=get_arguments(args)
    with open(path,'a') as f:
        f.write(s+','+val+'\n')

def read_result(path):
    result_dict = collections.defaultdict(list)
    with open(path,'r',encoding='utf-8') as f:
        result=f.readlines()
        # result=[r.strip().split('\t') for r in result]
        for r in result:
            # print(r)
            r=r.strip().split('\t')
            result_dict[r[0]].append(r[1:])
            # print(result_dict)
    return result_dict

def correct(orig,co):
    orig=read_result(orig)
    co=read_result(co)
    orig_length=len(orig)
    add=[]
    for k,v in orig.items():
        for i in range(len(v)):
            if(i>=len(co[k])):
                continue
            if(v[i][1]!=co[k][i][1]):
                flag=0
                for j in range(len(co[k])):
                    if(v[i][1]==co[k][j][1]):
                        flag=1
                if(flag==0):
                    print(v,co[k])
                    for ii in range(len(v)):
                        for cov in co[k]:
                            if(v[ii][1] in cov[1]):
                                print(v[ii][1],cov[1])
                                orig[k][ii][1]=cov[1]

def get_submit_correct(mcls=False):#开始边界为三vote,结束边界为最后一个--13.15.16--0.764
    output_predict_data_path='./work/test1_data/test1.json'
    output_path='./work/test1_data'
    orig_id=32
    correct_id=33
    correct2_id=34
    orig = read_by_lines("{}.{}.{}.pred".format(output_predict_data_path, 'role', orig_id))
    correct = read_by_lines("{}.{}.{}.pred".format(output_predict_data_path, 'role', correct_id))
    correct2 = read_by_lines("{}.{}.{}.pred".format(output_predict_data_path, 'role', correct2_id))
    submit = []
    count = 0
    for j in range(len(orig)):
        json_orig = json.loads(orig[j])
        json_correct=json.loads(correct[j])
        json_correct2 = json.loads(correct2[j])
        id=json_orig['id']
        print(id)
        # print(json_correct)
        correct_labels=json_correct['labels']
        correct2_labels=json_correct2['labels']
        orig_labels=json_orig['labels']
        text=json_orig['text']
        label=''
        entity_id=-1
        for i in range(len(orig_labels)):
            if(orig_labels[i]==correct_labels[i]=='O' and correct2_labels[i]=='<NA>'):
                if(label!=''):
                    if(mcls):
                        submit.append('\t'.join([str(id),text,text[entity_id:i],label]))
                    else:
                        submit.append('\t'.join([str(id), label, text[entity_id:i]]))
                    print(submit[-1])

                label=''
            else:
                # print(i, label, text[i], orig_labels[i], json_correct['text'][i], correct_labels[i],
                #       json_correct2['text'][i], correct2_labels[i])
                if(label==''):
                    #0.76->0.74，但可以加个规则？
                    # if (orig_labels[i][0] == 'B'):
                    #     entity_id=i
                    #     label=orig_labels[i][2:]
                    # elif(correct2_labels[i]!='<NA>'):
                    #     entity_id=i
                    #     label=correct2_labels[i]
                    if(orig_labels[i][0]=='B'):
                        if(correct_labels[i]=='B' or correct2_labels[i]!='<NA>'):
                            entity_id=i
                            label=orig_labels[i][2:]
                    else:
                        if(correct2_labels[i]!='<NA>' and correct_labels[i]=='B') :
                            entity_id=i
                            label=correct2_labels[i]

            # if (id == 2781379):
            #     print(i, label, text[i], orig_labels[i], json_correct['text'][i], correct_labels[i],
            #           json_correct2['text'][i], correct2_labels[i])
                    # if(label==''):
                    #     print(i,label,text[i],orig_labels[i],json_correct['text'][i],correct_labels[i],json_correct2['text'][i],correct2_labels[i])
    write_by_lines("{}/{}.{}.{}.ucas_valid_result.csv".format(output_path, orig_id,correct_id,correct2_id), submit)
        # if(len(correct_label)!=len(text) or len(text)!=len(orig_label)):
        #     print(len(text),len(orig_label),len(correct_label),len(json_correct['text']))
    #     text = json_result['text']
    #     label = json_result["labels"]
    #     now_label = ''
    #     now_entity = -1
    #     count = 0
    #     # print(len(text),len(label))
    #     for i, l in enumerate(label):
    #         # print(l,text[i])
    #         if (l == 'O' or l=='<NA>'):
    #             if (now_label != ''):
    #                 count += 1
    #                 if(check):
    #                     submit.append('\t'.join([str(json_result['id']), now_label, text[now_entity:i],
    #                                              str(json_result['input'][0][now_entity * 2:i * 2]),
    #                                              str(json_result['input'][0]), text, str(label)]))
    #                 else:
    #                     submit.append('\t'.join([str(json_result['id']), now_label,  text[now_entity:i]]))
    #                 now_label = ''
    #         else:
    #             if (l.startswith('B')):
    #                 if(args.change_event =='BIO_event'):
    #                     now_label = l[2:]
    #                 else:
    #                     now_label = l
    #                 now_entity = i
    #             elif (args.add_rule and now_label == ''):
    #                 if(label[i][2:]==label[now_entity][2:]):
    #                     now_label=label[i][2:]
    #                     submit.pop(-1)
    #     # if(count==0):
    #     #     submit.append('\t'.join([str(json_result['id']),'','',text,str(label)]))
    #     # print(submit)
    # if(check):
    #     write_by_lines("{}/{}ucas_valid_result_check.csv".format(output_path, id), submit)
    # else:
    #     write_by_lines("{}/{}ucas_valid_result.csv".format(output_path,id), submit)

def get_classify_correct(mcls=False):#开始边界为三vote,结束边界为最后一个--13.15.16--0.764
    output_predict_data_path='./work/test1_data/test1.json'
    output_path='./work/test1_data'
    orig_id='40-44'
    correct_id='45-49'
    correct2_id='50-54'
    orig = read_by_lines("{}.{}.{}.pred".format(output_predict_data_path, 'role', orig_id))
    correct = read_by_lines("{}.{}.{}.pred".format(output_predict_data_path, 'role', correct_id))
    correct2 = read_by_lines("{}.{}.{}.pred".format(output_predict_data_path, 'role', correct2_id))
    submit = []
    count = 0
    for j in range(len(orig)):
        json_orig = json.loads(orig[j])
        json_correct=json.loads(correct[j])
        json_correct2 = json.loads(correct2[j])
        id=json_orig['id']
        print(id)
        # print(json_correct)
        correct_labels=json_correct['labels']
        correct2_labels=json_correct2['labels']
        orig_labels=json_orig['labels']
        text=json_orig['text']
        label=''
        entity_id=-1
        for i in range(len(orig_labels)):
            if(orig_labels[i]==correct_labels[i]=='O' and correct2_labels[i]=='<NA>'):
                if(label!=''):
                    if (mcls):
                        submit.append('\t'.join([str(id), text, text[entity_id:i], label]))
                    else:
                        submit.append('\t'.join([str(id), label, text[entity_id:i]]))
                label=''
            else:
                print(i, label, text[i], orig_labels[i], json_correct['text'][i], correct_labels[i],
                      json_correct2['text'][i], correct2_labels[i])
                if(label==''):
                    #0.76->0.74，但可以加个规则？
                    if (orig_labels[i][0] == 'B'):
                        entity_id=i
                        label=orig_labels[i][2:]
                    elif(correct2_labels[i]!='<NA>'):
                        entity_id=i
                        label=correct2_labels[i]


            # if (id == 2781379):
            #     print(i, label, text[i], orig_labels[i], json_correct['text'][i], correct_labels[i],
            #           json_correct2['text'][i], correct2_labels[i])
                    # if(label==''):
                    #     print(i,label,text[i],orig_labels[i],json_correct['text'][i],correct_labels[i],json_correct2['text'][i],correct2_labels[i])
    write_by_lines("{}/{}.{}.{}.ucas_valid_result.csv".format(output_path, orig_id,correct_id,correct2_id), submit)
        # if(len(correct_label)!=len(text) or len(text)!=len(orig_label)):
        #     print(len(text),len(orig_label),len(correct_label),len(json_correct['text']))
    #     text = json_result['text']
    #     label = json_result["labels"]
    #     now_label = ''
    #     now_entity = -1
    #     count = 0
    #     # print(len(text),len(label))
    #     for i, l in enumerate(label):
    #         # print(l,text[i])
    #         if (l == 'O' or l=='<NA>'):
    #             if (now_label != ''):
    #                 count += 1
    #                 if(check):
    #                     submit.append('\t'.join([str(json_result['id']), now_label, text[now_entity:i],
    #                                              str(json_result['input'][0][now_entity * 2:i * 2]),
    #                                              str(json_result['input'][0]), text, str(label)]))
    #                 else:
    #                     submit.append('\t'.join([str(json_result['id']), now_label,  text[now_entity:i]]))
    #                 now_label = ''
    #         else:
    #             if (l.startswith('B')):
    #                 if(args.change_event =='BIO_event'):
    #                     now_label = l[2:]
    #                 else:
    #                     now_label = l
    #                 now_entity = i
    #             elif (args.add_rule and now_label == ''):
    #                 if(label[i][2:]==label[now_entity][2:]):
    #                     now_label=label[i][2:]
    #                     submit.pop(-1)
    #     # if(count==0):
    #     #     submit.append('\t'.join([str(json_result['id']),'','',text,str(label)]))
    #     # print(submit)
    # if(check):
    #     write_by_lines("{}/{}ucas_valid_result_check.csv".format(output_path, id), submit)
    # else:
    #     write_by_lines("{}/{}ucas_valid_result.csv".format(output_path,id), submit)

def get_submit_cross_validation_vote(cross_validation_num=5,begid=40,type='BIO_event'):#开始边界为三vote,结束边界为最后一个--13.15.16--0.764
    output_predict_data_path='./work/test1_data/test1.json'
    output_path='./work/test1_data'
    datalist=[]
    for id in range(begid,begid+cross_validation_num):
        datalist.append(read_by_lines("{}.{}.{}.pred".format(output_predict_data_path, 'role', id)))

    submit = []
    ret=[]
    for j in range(len(datalist[0])):
        json_data=[]
        labellist=[]
        for d in datalist:
            json_data.append(json.loads(d[j]))
            labellist.append(json_data[-1]['labels'])
        id=json_data[0]['id']
        text=json_data[0]['text']
        # print(id,labellist)
        now_entity=0
        now_label=''
        now_label_list=[]
        for label_index in range(len(labellist[0])):
            count=0
            O_flag=-1
            result_label=[]
            for index,label in enumerate(labellist):
                result_label.append(label[label_index])
            result_label=Counter(result_label).most_common(1)[0][0]
            now_label_list.append(result_label)
            if(result_label in ['O','<NA>']):
                if(now_label!=''):
                    submit.append('\t'.join([str(id),now_label,text[now_entity:label_index]]))
                    now_label=''
            else:
                if(now_label==''):
                    if(result_label[0]=='B'):
                        if(type=='BIO_event'):
                            now_label=result_label[2:]
                        elif(type=='BIO'):
                            now_label=result_label
                        now_entity=label_index
                    elif(type=='no'):
                        now_label=result_label
                        now_entity = label_index
        sent = {'id': id, 'text': text, 'labels':now_label_list}
        ret.append(json.dumps(sent, ensure_ascii=False))
    write_by_lines(
        "{}/test1.json.role.{}-{}.pred".format(output_path, begid, begid + cross_validation_num - 1), ret)

    write_by_lines("{}/{}-{}cross.ucas_valid_result.csv".format(output_path,begid,begid+cross_validation_num-1 ), submit)

def read_label_dict(path):
    with open(path, 'r', encoding='utf-8') as f:
        dic = f.read()
        # print(dic)
        dic = eval(dic)
    return dic
def read_label(path):
    dic=read_label_dict(path)
    return list(dic.keys())
def regrex_data(x):##去除空格啥的
    x=x.replace(' ','')
    x=x.replace(' ','')
    x=x.replace(' ','')
    x=x.replace('　','')
    x=x.replace('­','')
    x=x.replace('\xa0','')
    x=x.replace('\xad','')
    x=x.replace('\u3000','')
    x=x.replace('\u200b','')
    x=x.replace('</p>','')
    x=x.replace('<p>','')
    x = x.replace('<br>', '')
    x = x.replace('\uee0a', '')
    x = x.replace('\uec5e', '')
    x = x.replace('\uf06c', '')
    x = x.replace('\ued3c', '')
    x = x.replace('\ued22', '')
    x = x.replace('\x81', '')
    return x

def get_data():
    datapath='./work/all_with_neg.csv'
    c = pd.read_csv(datapath, header=None, sep='\t').fillna('').sort_values(by=0)
    label_dict=read_label_dict('./work/event2id.txt')
    labellist=list(label_dict.keys())[1:]
    s = ['text_a\tentity\t' + '\t'.join(labellist)]
    count = 0
    # print(c.head(3))
    oldid = 0
    data_dict={}
    multi_label_count=0
    for id, sent, label, entity in c.values:
        # if(id==oldid):
        #     print(id,label,entity,oldid)
        if(sent in data_dict.keys()):
            # print(id,data_dict[id])
            if(entity not in data_dict[sent].keys()):
                data_dict[sent][entity]=[0]*len(labellist)
                if (label != ''):
                    data_dict[sent][entity][label_dict[label]-1]=1
            else:
                if (label != ''):
                    data_dict[sent][entity][label_dict[label]-1]=1
                    multi_label_count+=1
        else:
            # oldid=id
            data_dict[sent]={entity:[0]*len(labellist)}
            if(label!=''):
                data_dict[sent][entity][label_dict[label]-1] = 1
        # print(data_dict)

    sents=[]
    entities=[]
    labels=[]
    multi_entity_count=0
    for sent,v in data_dict.items():

        flag=0
        for entity,label in v.items():
            flag+=1
            sent=regrex_data(sent)
            sents.append(sent)
            entities.append(entity)
            labels.append(label)
        if(flag>1):
            multi_entity_count+=1

    result = pd.DataFrame()
    # result['id']=idd
    result['sent']=sents
    result['entity']=entities
    result['label']=labels
    result=result.sample(frac=1.0)
    result.to_csv('./work/data_cls.csv', header=None, index=False,sep='\t')
    print(result)
    print(multi_entity_count/41600.0,multi_entity_count)
    print(multi_label_count / 41600.0, multi_label_count)


if __name__ == "__main__":
    get_classify_correct(True)
    # get_submit_cross_validation_vote(cross_validation_num=5, begid=40,type='BIO_event')
    # get_data()
    # get_submit_correct(True)
    # get_classify_correct(True)
    # data_path='./work/test1_data/'
    # orig=data_path+'13_1ucas_valid_result.csv'
    # co=data_path+'15ucas_valid_result.csv'
    # correct(orig, co)
    # parser = argparse.ArgumentParser(
    #     description="Official evaluation script for DuEE version 0.1.")
    # parser.add_argument(
    #     "--trigger_file",
    #     help="trigger model predict data path",
    #     default='./work/test1_data/')
    # parser.add_argument(
    #     "--role_file", help="role model predict data path", default='./work/test1_data/')
    # parser.add_argument(
    #     "--schema_file", help="schema file path", default='work/event_schema/event_schema.json')
    # parser.add_argument("--save_path", help="save file path", default='work/mix_predict/')
    # args = parser.parse_args()
    # trigger=pd.read_csv('./work/log/trigger.txt')
    # role=pd.read_csv('./work/log/role.txt')
    # pred_log=[]#'id,trigger_id,role_id,trigger_f1,role_f1,save_path']
    # id=1
    # for trigger_id,trigger_f1 in trigger[['id','f1']].values:
    #     for role_id,role_f1 in role[['id','f1']].values:
    #         if(id<103):
    #             id+=1
    #             continue
    #         ll = [id, int(trigger_id), trigger_f1, int(role_id), role_f1]
    #         ll=[str(x) for x in ll]
    #         print(ll)
    #         trigger_path=args.trigger_file+'test1.json.%s.0%s.pred'%('trigger',ll[1])
    #         role_path =args.role_file+ 'test1.json.%s.0%s.pred' % ('role', ll[3])
    #         save_path=args.save_path+'%s.trigger0%s.role0%s.json'%(ll[0],ll[1],ll[3])
    #         predict_data_process(trigger_path, role_path, args.schema_file,
    #                              save_path)
    #
    #         ll+=[save_path]
    #         pred_log.append(','.join(ll))
    #         id+=1
    # with open('./work/log.txt','a',encoding='utf-8') as f:
    #     f.write('\n'.join(pred_log))


