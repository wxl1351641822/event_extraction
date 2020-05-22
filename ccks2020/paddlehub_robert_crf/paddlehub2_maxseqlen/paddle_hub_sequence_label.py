#!/usr/bin/env python
# coding:utf-8
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
"""Finetuning on sequence labeling task."""

import argparse
import ast
import json
import shutil
import os
import pandas as pd
import numpy as np
import time
import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset
from paddlehub.common.logger import logger
from visualdl import LogWriter
from datetime import datetime
from Reader import SequenceLabelReader
from paddlehub_dataprocess import data_process
from paddlehub_dataprocess import schema_process
from paddlehub_dataprocess import write_by_lines
from paddlehub_dataprocess import write_title
from paddlehub_dataprocess import write_log
import io
import csv
from paddlehub.dataset import InputExample

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=7, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True,
                    help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--data_dir", type=str, default='work/', help="data save dir")
parser.add_argument("--schema_path", type=str, default='work/event_schema/event_schema.json', help="schema path")
parser.add_argument("--train_data", type=str, default='work/train_data/train.json', help="train data")
parser.add_argument("--dev_data", type=str, default='work/dev_data/dev.json', help="dev data")
parser.add_argument("--test_data", type=str, default='work/dev_data/dev.json', help="test data")
parser.add_argument("--predict_data", type=str, default='work/test1_data/test1.json', help="predict data")
parser.add_argument("--do_train", type=ast.literal_eval, default=True, help="do train")
parser.add_argument("--do_predict", type=ast.literal_eval, default=True, help="do predict")
parser.add_argument("--do_model", type=str, default="trigger", choices=["trigger", "role"], help="trigger or role")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=202, help="Number of words of the longest seqence.")
parser.add_argument("--eval_step", type=int, default=200, help="eval step")
parser.add_argument("--model_save_step", type=int, default=30000, help="model save step")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--add_crf", type=ast.literal_eval, default=True, help="add crf")
parser.add_argument("--checkpoint_dir", type=str, default='models/trigger', help="Directory to model checkpoint")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=True, help="Whether use data parallel.")
parser.add_argument("--random_seed", type=int, default=1666, help="seed")
# parser.add_argument("--dropout", type=float, default=0.0, help="dropout")
parser.add_argument(
    "--saved_params_dir",
    type=str,
    default="",
    help="Directory for saving model during ")


def is_path_valid(path):
    if path == "":
        return False
    path = os.path.abspath(path)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    return True


# import pandas as pd


def normal(s):
    s=s.lower()
    s=s.replace(' ','')
    s=s.replace('(','（')
    # s=s.replace('（',' ')
    s=s.replace(')','）')
    # s=s.replace(' ','')
    # s=s.replace(' ','')
    return s


def get_train_dev():
    c = pd.read_csv('./work/data.csv', header=None, sep='\t')

    spl = int(c.shape[0] * 0.8)
    train = c[:spl]
    dev = c[spl:]

    def get_data2id(c_data, path):
        sent_length = []
        all_sent = []
        all_label = []
        s = ['text_a\tlabel']
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

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(s))
        return s

    train1 = get_data2id(train, './work/train.txt')
    dev1 = get_data2id(dev, './work/dev.txt')


# get_train_dev()


# get_train_dev()


def get_predict():
    data = pd.read_csv('./data/data34808/test_unlabel.csv', sep='\t', header=None).fillna('')
    predict = pd.DataFrame()
    predict['text_a'] = [str(list(normal(x))) if len(x) <= 100 else str(list(x[:100] )) for x in data[1].values]
    predict.to_csv('./work/predict.txt', index=False)
    predict_sents = []
    data['text_a']=[str(list(x )) if len(x) <= 100 else str(list(x[:100] )) for x in data[1].values]
    for id, text,text_a in data.values:
        # print(id,text)
        predict_sents.append({'id': id,'text':text,'text_a': eval(text_a)})
    return list(predict['text_a'].values), predict_sents

def get_predict1():
    with open('./work/dev.txt', 'r', encoding='utf-8') as f:
        predict_sents = []
        predict_data=[str(['text_a'])]
        for i, line in enumerate(f.readlines()[1:]):
            # line=line[:-1]
            # print(line)
            line = line.split('\t')
            text = line[0]
            label=eval(line[1])
            predict_sents.append({'id': label, 'text': eval(text)})
            predict_data.append(text)
    return predict_data, predict_sents

def read_label(path):
    with open(path, 'r', encoding='utf-8') as f:
        dic = f.read()
        # print(dic)
        dic = eval(dic)
    return list(dic.keys())


# yapf: enable.
def process_data(args):
    # get_train_dev()
    predict_data, predict_sents = get_predict()

    # write_by_lines("{}/{}_train.tsv".format(args.data_dir, args.do_model), train_data)
    # write_by_lines("{}/{}_dev.tsv".format(args.data_dir, args.do_model), dev_data)
    # write_by_lines("{}/{}_test.tsv".format(args.data_dir, args.do_model), test_data)
    write_by_lines("{}/predict.txt".format(args.data_dir), predict_data)

    schema_labels = read_label('{}/entity2id.txt'.format(args.data_dir))
    return schema_labels, predict_data, predict_sents


class EEDataset(BaseNLPDataset):
    """EEDataset"""

    def __init__(self, data_dir, labels, model="trigger"):
        # 数据集存放位置
        super(EEDataset, self).__init__(
            base_path=data_dir,
            train_file="train.txt",
            dev_file="dev.txt",
            test_file="dev.txt",
            # 如果还有预测数据（不需要文本类别label），可以放在predict.tsv
            predict_file="predict.txt",
            train_file_with_header=True,
            dev_file_with_header=True,
            test_file_with_header=True,
            predict_file_with_header=True,
            # 数据集类别集合
            label_list=labels)

    def _read_file(self, input_file, phase=None):
        """Reads a tab separated value file."""
        has_warned = False
        with open(input_file, 'r', encoding='utf-8') as f:
            examples = []
            for i, line in enumerate(f.readlines()[1:]):
                # line=line[:-1]
                # print(line)
                line = line.split('\t')
                text1 = line[0]
                text = eval(text1)
                if len(line) < 2:
                    example = InputExample(
                        guid=i, text_a=text)
                else:
                    label1 = line[1]
                    label = eval(label1)
                    example = InputExample(
                        guid=i, text_a=text, label=label)
                examples.append(example)

        return examples


def add_hook(args, task, id):
    # 设定模式为 train，创建一个 scalar 组件
    # with log_writer.mode(args.do_model + "train%d" % id) as logger1:
    #     train_loss = logger1.scalar("loss")
    #     train = {}
    #     train["f1"] = logger1.scalar("f1")
    #     train["precision"] = logger1.scalar("precision")
    #     train["recall"] = logger1.scalar("recall")
    # # 设定模式为test，创建一个 image 组件
    # with log_writer.mode(args.do_model + "dev%d" % id) as shower:
    #     dev_loss = shower.scalar("loss")
    #     dev = {}
    #     dev["f1"] = shower.scalar("f1")
    #     dev["precision"] = shower.scalar("precision")
    #     dev["recall"] = shower.scalar("recall")
    def new_eval_end_event(self, run_states):
        """
        Paddlehub default handler for eval_end_event, it will complete visualization and metrics calculation
        Args:
            run_states (object): the results in eval phase
        """
        eval_scores, eval_loss, run_speed = self._calculate_metrics(run_states)

        if 'train' in self._envs:
            self.tb_writer.add_scalar(
                tag="Loss_{}".format(self.phase),
                scalar_value=eval_loss,
                global_step=self._envs['train'].current_step)

        log_scores = ""

        s = []

        for metric in eval_scores:
            if 'train' in self._envs:
                self.tb_writer.add_scalar(
                    tag="{}_{}".format(metric, self.phase),
                    scalar_value=eval_scores[metric],
                    global_step=self._envs['train'].current_step)
                # dev[metric].add_record(self._envs['train'].current_step, eval_scores[metric])
            log_scores += "%s=%.5f " % (metric, eval_scores[metric])
            s.append(eval_scores[metric])
            # dev[metric].add_record(self.current_step,eval_scores[metric])
        logger.eval(
            "[%s dataset evaluation result] loss=%.5f %s[step/sec: %.2f]" %
            (self.phase, eval_loss, log_scores, run_speed))
        s.append(eval_loss)
        if 'train' in self._envs:
            s = [self._envs['train'].current_step] + s
            # dev_loss.add_record(self._envs['train'].current_step,eval_loss)
        s = [str(x) for x in s]
        with open('./work/log/%s_dev%s.txt' % (args.do_model, id), 'a', encoding='utf-8') as f:
            f.write(','.join(s) + '\n')

        eval_scores_items = eval_scores.items()
        if len(eval_scores_items):
            # The first metric will be chose to eval
            main_metric, main_value = list(eval_scores_items)[0]
        else:
            logger.warning(
                "None of metrics has been implemented, loss will be used to evaluate."
            )
            # The larger, the better
            main_metric, main_value = "negative loss", -eval_loss
        if self.phase in ["dev", "val"] and main_value > self.best_score:
            self.best_score = main_value
            model_saved_dir = os.path.join(self.config.checkpoint_dir,
                                           "best_model")
            logger.eval("best model saved to %s [best %s=%.5f]" %
                        (model_saved_dir, main_metric, main_value))
            self.save_inference_model(dirname=model_saved_dir)

    def new_log_interval_event(self, run_states):
        """
        PaddleHub default handler for log_interval_event, it will complete visualization.
        Args:
            run_states (object): the results in train phase
        """
        scores, avg_loss, run_speed = self._calculate_metrics(run_states)
        self.tb_writer.add_scalar(
            tag="Loss_{}".format(self.phase),
            scalar_value=avg_loss,
            global_step=self._envs['train'].current_step)
        log_scores = ""

        s = [self.current_step]
        for metric in scores:
            self.tb_writer.add_scalar(
                tag="{}_{}".format(metric, self.phase),
                scalar_value=scores[metric],
                global_step=self._envs['train'].current_step)
            log_scores += "%s=%.5f " % (metric, scores[metric])
            s.append(scores[metric])
            # train[metric].add_record(self.current_step, scores[metric])
        logger.train("step %d / %d: loss=%.5f %s[step/sec: %.2f]" %
                     (self.current_step, self.max_train_steps, avg_loss,
                      log_scores, run_speed))
        s.append(avg_loss)
        # train_loss.add_record(self.current_step, avg_loss)
        s = [str(x) for x in s]
        with open('./work/log/%s_train%s.txt' % (args.do_model, id), 'a', encoding='utf-8') as f:
            f.write(','.join(s) + '\n')

    # # 利用Hook改写PaddleHub内置_log_interval_event实现，需要2步(假设task已经创建好)
    # # 1.删除PaddleHub内置_log_interval_event实现
    # # hook_type：你想要改写的事件hook类型
    # # name：hook名字，“default”表示PaddleHub内置_log_interval_event实现
    task.delete_hook(hook_type="log_interval_event", name="default")
    task.delete_hook(hook_type="eval_end_event", name="default")
    #
    # # 2.增加自定义_log_interval_event实现(new_log_interval_event)
    # # hook_type：你想要改写的事件hook类型
    # # name: hook名字
    # # func：自定义改写的方法
    task.add_hook(hook_type="log_interval_event", name="new_log_interval_event", func=new_log_interval_event)
    task.add_hook(hook_type="eval_end_event", name="new_eval_end_event", func=new_eval_end_event)
    #
    # # 输出hook信息
    # task.hook_info()


def get_task(args, schema_labels, id):
    # 加载PaddleHub 预训练模型ERNIE Tiny/RoBERTa large
    # 更多预训练模型 https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=SemanticModel
    # model_name = "ernie_tiny"
    model_name = "chinese-roberta-wwm-ext-large"
    module = hub.Module(name=model_name)
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # 加载数据并通过SequenceLabelReader读取数据
    dataset = EEDataset(args.data_dir, schema_labels, model=args.do_model)
    reader = SequenceLabelReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len,
        sp_model_path=module.get_spm_path(),
        word_dict_path=module.get_word_dict_path())

    # 构建序列标注任务迁移网络
    # 使用ERNIE模型字级别的输出sequence_output作为迁移网络的输入
    sequence_output = outputs["sequence_output"]
    # sequence_output  = fluid.layers.dropout(
    #     x=sequence_output ,
    #     dropout_prob=args.dropout,
    #     dropout_implementation="upscale_in_train")

    # 设置模型program需要输入的变量feed_list
    # 必须按照以下顺序设置
    feed_list = [
        inputs["input_ids"].name, inputs["position_ids"].name,
        inputs["segment_ids"].name, inputs["input_mask"].name
    ]

    # 选择优化策略
    strategy = hub.AdamWeightDecayStrategy(
        warmup_proportion=args.warmup_proportion,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate)

    # 配置运行设置
    config = hub.RunConfig(
        log_interval=100,
        eval_interval=args.eval_step,
        save_ckpt_interval=args.model_save_step,
        use_data_parallel=args.use_data_parallel,
        use_cuda=args.use_gpu,
        # enable_memory_optim=True,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=strategy)

    # 构建序列标注迁移任务
    seq_label_task = hub.SequenceLabelTask(
        data_reader=reader,
        feature=sequence_output,
        feed_list=feed_list,
        max_seq_len=args.max_seq_len,
        num_classes=dataset.num_labels,
        config=config,
        add_crf=args.add_crf)
    seq_label_task.main_program.random_seed = args.random_seed
    add_hook(args, seq_label_task, id)
    return seq_label_task, reader
    # PaddleHub Finetune API
    # 将自动训练、评测并保存模型
    # if args.do_train:
    #     print("start finetune and eval process")
    #     seq_label_task.finetune_and_eval()
    #     write_log('./work/log/'+args.do_model+'.txt',args,str(seq_label_task.best_score))
    # model_path='./result/round1/model-1'
    # id=''

def see_predict(path):
    result = list()
    with open(path, 'r', encoding='utf-8') as f:

        for line in f:
            dic = eval(line.strip())
            # print(dic['id'],dic['text'],dic['labels'])
            entity = []
            label = []

            for t, l in zip(dic['text'], dic['labels']):
                # print(t,l)
                if (l[0] != 'I' and len(entity) != 0):
                    label_event = label[0][2:]
                    result.append(
                        str(dic['id']) + ',' + ''.join(dic['text']) + ',' + label_event + ',' + ''.join(entity))
                    print(entity)
                    if (l[0] == 'B'):
                        entity = [t]
                        label = [l]
                    else:
                        entity = []
                        label = []
                else:
                    if (l[0] == 'B'):
                        entity = [t]
                        label = [l]
                    elif l[0] == 'I':
                        entity.append(t)
                        label.append(l)
    with open(path + '_predict_' +  '.csv', 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))
def predict_by_model_path(args, model_path,  id):
    schema_labels, predict_data, predict_sents = process_data(args)
    seq_label_task, reader = get_task(args, schema_labels, id)
    seq_label_task.init_if_necessary()
    seq_label_task.load_parameters(model_path)
    logger.info("PaddleHub has loaded model from %s" % model_path)
    if args.do_predict:
        print("start predict process")
        ret = []
        id2label = {val: key for key, val in reader.label_map.items()}
        input_data = [[eval(d)] for d in predict_data]
        run_states = seq_label_task.predict(data=input_data[1:])
        results = []
        for batch_states in run_states:
            batch_results = batch_states.run_results
            batch_infers = batch_results[0].reshape([-1]).astype(np.int32).tolist()
            seq_lens = batch_results[1].reshape([-1]).astype(np.int32).tolist()
            current_id = 0
            for length in seq_lens:
                seq_infers = batch_infers[current_id:current_id + length]
                seq_result = list(map(id2label.get, seq_infers[1: -1]))
                current_id += length if args.add_crf else args.max_seq_len
                results.append(seq_result)

        ret = []
        for sent, r_label in zip(predict_sents, results):
            # print(sent)
            sent["labels"] = r_label
            ret.append(json.dumps(sent, ensure_ascii=False))
        write_by_lines("./work/result/{}".format( id), ret)
        see_predict('./work/result/{}'.format(id))


def predict_model_path(id,args):
    # args = parser.parse_args()
    schema_labels, predict_data, predict_sents = process_data(args)
    # trigger = pd.read_csv('./work/log/role.txt', header=None)
    # # print(trigger)
    # # print(trigger[[0,17]])
    # for id, model_path in trigger[[0, 17]].values:
    id = 5
    model_path = './models/role'+str(id)
    print(id, model_path)
    args.do_model = 'role'
    args.checkpoint_dir=model_path
    schema_labels, predict_data, predict_sents = process_data(args)
    predict_by_model_path(args, model_path, schema_labels, predict_data, predict_sents, id)


def one(args, schema_labels, id):
    seq_label_task, reader = get_task(args, schema_labels, id)
    # 加载PaddleHub 预训练模型ERNIE Tiny/RoBERTa large
    # 更多预训练模型 https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=SemanticModel
    # model_name = "ernie_tiny"

    # PaddleHub Finetune API
    # 将自动训练、评测并保存模型
    if args.do_train:
        print("start finetune and eval process")
        seq_label_task.finetune_and_eval()
        write_log('./work/log/' + args.do_model + '.txt', args, id + ',' + str(seq_label_task.best_score))
def lrepochsearch():
    args = parser.parse_args()
    args.do_model = 'role'
    schema_labels = read_label('{}/entity2id.txt'.format(args.data_dir))

    # # 创建一个 LogWriter 对象 log_writer
    # log_writer = LogWriter("./log", sync_cycle=10)
    shiyan = """
######################################################################################################################################
                                202,不复制，不考虑重叠,lrepochsearch
######################################################################################################################################
    """
    write_title('./work/log/' + args.do_model + '.txt', args, shiyan)
    id = 6  # str(datetime.now().strftime('%m%d%H%M'))
    for lr in [5e-5,7e-5,1e-4,3e-4]:#[3e-5,1e-5,5e-6,1e-6]
        args.learning_rate=lr
        args.checkpoint_dir = 'models/' + args.do_model + str(id)
        for epoch in range(1,4):
            args.num_epoch=epoch
            one(args, schema_labels, str(id))
            predict_by_model_path(args,args.checkpoint_dir,  id)
            id+=1

def my():
    args = parser.parse_args()
    args.do_model = 'role'
    schema_labels = read_label('{}/entity2id.txt'.format(args.data_dir))

    # # 创建一个 LogWriter 对象 log_writer
    # log_writer = LogWriter("./log", sync_cycle=10)
    shiyan = """
######################################################################################################################################
                                202,不复制，不考虑重叠
######################################################################################################################################
    """
    write_title('./work/log/' + args.do_model + '.txt', args, shiyan)
    id = 5  # str(datetime.now().strftime('%m%d%H%M'))

    args.checkpoint_dir = 'models/' + args.do_model + str(id)
    one(args, schema_labels, str(id))
    schema_labels, predict_data, predict_sents = process_data(args)
    predict_by_model_path(args,args.checkpoint_dir, schema_labels, predict_data, predict_sents, id)
    # if args.do_predict:
    #     print("start predict process")
    #     ret = []
    #     id2label = {val: key for key, val in reader.label_map.items()}
    #     input_data = [[d] for d in predict_data]
    #     run_states = seq_label_task.predict(data=input_data[1:])
    #     results = []
    #     for batch_states in run_states:
    #         batch_results = batch_states.run_results
    #         batch_infers = batch_results[0].reshape([-1]).astype(np.int32).tolist()
    #         seq_lens = batch_results[1].reshape([-1]).astype(np.int32).tolist()
    #         current_id = 0
    #         for length in seq_lens:
    #             seq_infers = batch_infers[current_id:current_id + length]
    #             seq_result = list(map(id2label.get, seq_infers[1: -1]))
    #             current_id += length if args.add_crf else args.max_seq_len
    #             results.append(seq_result)
    #
    #     ret = []
    #     for sent, r_label in zip(predict_sents, results):
    #         sent["labels"] = r_label
    #         ret.append(json.dumps(sent, ensure_ascii=False))
    #     write_by_lines("{}.{}.{}.pred".format(args.predict_data, args.do_model, id), ret)


if __name__ == "__main__":
    lrepochsearch()

    #

    # args = parser.parse_args()

    # args.do_model = 'role'
    # schema_labels, predict_data, predict_sents = process_data(args)
    # # # 创建一个 LogWriter 对象 log_writer
    # # log_writer = LogWriter("./log", sync_cycle=10)
    #
    # id = str(datetime.now().strftime('%m%d%H%M%S'))
    # print(id)
    #
    # args.checkpoint_dir = 'models/trigger' + str(id)
    #
    # predict_by_model_path(args,model_path, schema_labels, predict_data, predict_sents, id)
#     shiyan="""
# ########################################################################################################################
#                                 实验1：lr
# ########################################################################################################################
#     """

#
# for do_model in ["trigger", "role"]:
#     write_title('./work/log/' + args.do_model + '.txt', args, shiyan)
# id = 1
#
# for lr in [2e-5,3e-5,5e-5,1e-4_实体识别_1_3的O,1e-5,1e-6]:
#     args.learning_rate = lr
#     for do_model in ["trigger", "role"]:
#         checkpoint_dir = 'models/' + do_model
#         args.do_model = do_model
#         args.checkpoint_dir=checkpoint_dir+str(id)
#         one(args,schema_labels,predict_data,predict_sents,id)
#         id+=1

