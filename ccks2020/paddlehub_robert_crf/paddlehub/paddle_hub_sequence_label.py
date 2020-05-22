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
import csv
import io
import pandas as pd
import numpy as np
import paddlehub as hub
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset
from paddlehub.common.logger import logger
from paddlehub.dataset import InputExample
from datetime import datetime

from Sequence_Reader import SequenceLabelReader
from sequence_task import SequenceLabelTask

from paddlehub_dataprocess import write_by_lines
from paddlehub_dataprocess import write_title
from paddlehub_dataprocess import write_log


# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=7, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False,
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
parser.add_argument("--max_seq_len", type=int, default=256, help="Number of words of the longest seqence.")
parser.add_argument("--eval_step", type=int, default=200, help="eval step")
parser.add_argument("--model_save_step", type=int, default=3000, help="model save step")
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


def get_train_dev():
    c = pd.read_csv('./work/data.csv', header=None,sep='\t')

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

            sent_length.append(len(sentence))
            dic = eval(dic)
            label = []
            for k, v in dic.items():
                # print(k,v)

                entity_label = ['O'] * len(sentence)
                # print(entity2id)
                for entity in v:
                    beg = sent.find(entity)
                    end = sent.find(entity) + len(entity) - 1
                    if (beg == end):
                        entity_label[beg] = 'S-' + k
                    else:
                        entity_label[beg] = 'B-' + k
                        entity_label[end] = 'E-' + k
                        for i in range(beg + 1, end):
                            entity_label[i] = 'I-' + k
                    # print(sent[beg:end+1],entity_label[beg:end+1])

                    # print(sent,v,sent.find(entity),sent[sent.find(entity)],sent[sent.find(entity)+len(entity)-1])
                    sent.find(entity)
                label.extend(entity_label)
            if(len(label)==0):
                label=['0']* len(sentence)
            s.append(' '.join(sentence) + '\t' + ' '.join(label))

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(s))
        return s

    train1 = get_data2id(train, './work/train.txt')
    dev1 = get_data2id(train, './work/dev.txt')
def get_predict():
    data = pd.read_csv('./data/data34808/test_unlabel.csv', sep='\t', header=None).fillna('')
    predict = pd.DataFrame()
    predict['text_a'] = [' '.join(list(x)) for x in data[1].values]
    predict.to_csv('./work/predict.txt', index=False)
    predict_sents = []
    for id, text in data.values:
        # print(id,text)
        predict_sents.append({'id': id, 'text': text})
    return list(predict['text_a'].values), predict_sents


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
        with io.open(input_file, "r", encoding="UTF-8") as file:
            reader = csv.reader(file, delimiter="\t", quotechar=None)
            examples = []
            for (i, line) in enumerate(reader):

                if i == 0:
                    ncol = len(line)
                    if self.if_file_with_header[phase]:
                        continue
                if(len(line)!=ncol):
                    print(line)
                if phase != "predict":
                    if ncol == 1:
                        raise Exception(
                            "the %s file: %s only has one column but it is not a predict file"
                            % (phase, input_file))
                    elif ncol == 2:
                        example = InputExample(
                            guid=i, text_a=line[0], label=line[1])
                    elif ncol == 3:
                        example = InputExample(
                            guid=i,
                            text_a=line[0],
                            text_b=line[1],
                            label=line[2])
                    else:
                        raise Exception(
                            "the %s file: %s has too many columns (should <=3_实体识别)"
                            % (phase, input_file))
                else:
                    if ncol == 1:
                        example = InputExample(guid=i, text_a=line[0])
                    elif ncol == 2:
                        if not has_warned:
                            logger.warning(
                                "the predict file: %s has 2 columns, as it is a predict file, the second one will be regarded as text_b"
                                % (input_file))
                            has_warned = True
                        example = InputExample(
                            guid=i, text_a=line[0], text_b=line[1])
                    else:
                        raise Exception(
                            "the predict file: %s has too many columns (should <=2)"
                            % (input_file))
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
    seq_label_task = SequenceLabelTask(
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


def predict_by_model_path(args, model_path, schema_labels, predict_data, predict_sents, id):
    seq_label_task, reader = get_task(args, schema_labels, predict_data, predict_sents, id)
    seq_label_task.init_if_necessary()
    seq_label_task.load_parameters(model_path)
    logger.info("PaddleHub has loaded model from %s" % model_path)
    if args.do_predict:
        print("start predict process")
        ret = []
        id2label = {val: key for key, val in reader.label_map.items()}
        input_data = [[d] for d in predict_data]
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
            sent["labels"] = r_label
            ret.append(json.dumps(sent, ensure_ascii=False))
        write_by_lines("{}.{}.{}.pred".format(args.predict_data, args.do_model, id), ret)


def predict_model_path():
    args = parser.parse_args()
    schema_labels, predict_data, predict_sents = process_data(args)
    trigger = pd.read_csv('./work/log/trigger.txt', header=None)
    # print(trigger)
    # print(trigger[[0,17]])
    for id, model_path in trigger[[0, 17]].values:
        print(id, model_path)
        predict_by_model_path(args, model_path, schema_labels, predict_data, predict_sents, id)


def one(args, schema_labels, predict_data, predict_sents, id):
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

    if args.do_predict:
        print("start predict process")
        ret = []
        id2label = {val: key for key, val in reader.label_map.items()}
        input_data = [[d] for d in predict_data]
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
            sent["labels"] = r_label
            ret.append(json.dumps(sent, ensure_ascii=False))
        write_by_lines("{}.{}.{}.pred".format(args.predict_data, args.do_model, id), ret)


def one_autofinetune(args, schema_labels, predict_data, predict_sents, id):
    seq_label_task, reader = get_task(args, schema_labels, id)
    # 加载PaddleHub 预训练模型ERNIE Tiny/RoBERTa large
    # 更多预训练模型 https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=SemanticModel
    # model_name = "ernie_tiny"

    # PaddleHub Finetune API
    # 将自动训练、评测并保存模型
    if args.do_train:
        print("start finetune and eval process")
        seq_label_task.finetune_and_eval()
        write_log('./work/log/' + args.do_model + '.txt', args, str(seq_label_task.best_score))

    if args.do_predict:
        print("start predict process")
        ret = []
        id2label = {val: key for key, val in reader.label_map.items()}
        input_data = [[d] for d in predict_data]
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
            sent["labels"] = r_label
            ret.append(json.dumps(sent, ensure_ascii=False))
        write_by_lines("{}.{}.{}.pred".format(args.predict_data, args.do_model, id), ret)
    # Load model from the defined model path or not
    #

    # seq_label_task.finetune_and_eval()
    # run_states = seq_label_task.eval()
    # eval_avg_score, eval_avg_loss, eval_run_speed =seq_label_task._calculate_metrics(
    #     run_states)
    # Move ckpt/best_model to the defined saved parameters directory
    best_model_dir = os.path.join(args.checkpoint_dir, "best_model")
    if is_path_valid(args.saved_params_dir) and os.path.exists(best_model_dir):
        shutil.copytree(best_model_dir, args.saved_params_dir)
        shutil.rmtree(args.checkpoint_dir)
    write_log('./work/log/' + args.do_model + '.txt', args, id + ',' + str(seq_label_task.best_score))
    print(seq_label_task.best_score)
    hub.report_final_result(seq_label_task.best_score)


def autofinetune():
    args = parser.parse_args()
    # args.do_model = 'role'
    schema_labels, predict_data, predict_sents = process_data(args)

    # schema_labels=read_label('./work/entity2id.txt')
    # # 创建一个 LogWriter 对象 log_writer
    # log_writer = LogWriter("./log", sync_cycle=10)

    id = str(datetime.now().strftime('%m%d%H%M'))
    print(id)

    args.checkpoint_dir = 'models/' + args.do_model + str(id)
    one_autofinetune(args, schema_labels, predict_data, predict_sents, id)


def lrsearch():
    args = parser.parse_args()
    # args.do_model = 'role'
    schema_labels, predict_data, predict_sents = process_data(args)
    # # 创建一个 LogWriter 对象 log_writer
    # log_writer = LogWriter("./log", sync_cycle=10)
    shiyan = """
######################################################################################################################################
                                trigger_lrgridsearch
######################################################################################################################################
    """
    write_title('./work/log/' + args.do_model + '.txt', args, shiyan)
    id = 1  # str(datetime.now().strftime('%m%d%H%M'))
    print(id)
    for lr in [3e-5, 1e-5, 1e-4]:
        args.learning_rate = lr
        args.checkpoint_dir = 'models/' + args.do_model + str(id)
        one(args, schema_labels, predict_data, predict_sents, str(id))
        id += 1


def bzsearch():
    args = parser.parse_args()
    # args.do_model = 'role'
    schema_labels, predict_data, predict_sents = process_data(args)
    # # 创建一个 LogWriter 对象 log_writer
    # log_writer = LogWriter("./log", sync_cycle=10)
    shiyan = """
######################################################################################################################################
                                trigger_batch_size gridsearch
######################################################################################################################################
    """
    write_title('./work/log/' + args.do_model + '.txt', args, shiyan)
    id = 4  # str(datetime.now().strftime('%m%d%H%M'))
    print(id)
    for bz in [32, 16, 8]:
        args.batch_size = bz
        args.checkpoint_dir = 'models/' + args.do_model + str(id)
        one(args, schema_labels, predict_data, predict_sents, str(id))
        id += 1


def lrbzsearch():
    args = parser.parse_args()
    # args.do_model = 'role'
    schema_labels, predict_data, predict_sents = process_data(args)
    # # 创建一个 LogWriter 对象 log_writer
    # log_writer = LogWriter("./log", sync_cycle=10)
    shiyan = """
######################################################################################################################################
                                trigger_batch_size lr gridsearch
######################################################################################################################################
    """
    write_title('./work/log/' + args.do_model + '.txt', args, shiyan)
    id = 7  # str(datetime.now().strftime('%m%d%H%M'))
    print(id)
    for bz in [32, 16, 8]:
        args.batch_size = bz
        for lr in [1e-6, 1e-5, 1e-4, 1e-3]:
            args.learning_rate = lr
            args.checkpoint_dir = 'models/' + args.do_model + str(id)
            one(args, schema_labels, predict_data, predict_sents, str(id))
            id += 1


def get_data():
    data={}
    with open('./data/data34808/train_label.csv','r',encoding='utf-8') as f:
        for line in f.readlines():
            line_list=line[:-1].split('	')
            if(len(line_list)!=4):
                line_list=line[:-1].split('\\t')
                if(len(line_list)!=4):
                    line_list=line_list[:1]+line_list[1].split('\t')
            # print(line_list[0])
            if (line_list[2] == '涉嫌欺诈'):
                print(line_list[2],line_list[3])
            if line_list[1] in data.keys():
                data[line_list[1]][0].append(line_list[0])
                # print(data[line_list[1]][0])
                # data[line_list[1]][0]
                if(line_list[2]=='NaN'):
                    continue
                if line_list[2] in data[line_list[1]][1].keys():
                    data[line_list[1]][1][line_list[2]].append(line_list[3])
                else:
                    data[line_list[1]][1][line_list[2]]=[line_list[3]]
            else:
                if (line_list[2] == 'NaN'):
                    data[line_list[1]] = [[line_list[0]], {}]
                else:
                    
                    data[line_list[1]]=[[line_list[0]],{line_list[2]:[line_list[3]]}]
    # print(data)
    idd=[]
    sentences=[]
    events=[]
    for k,v in data.items():
        sentences.append(k)
        idd.append(str(v[0]))
        events.append(str(v[1]))
    c = pd.DataFrame()
    c[0] = idd
    c[1] = sentences
    c[2] = events
    c = c.sample(frac=1.0)
    c.to_csv('./work/data.csv', header=None, index=False,sep='\t')


if __name__ == "__main__":
    # get_data()
    args = parser.parse_args()
    args.do_model = 'role'
    schema_labels, predict_data, predict_sents = process_data(args)
    shiyan = """
    ######################################################################################################################################
                                    trigger_batch_size lr gridsearch
    ######################################################################################################################################
        """
    write_title('./work/log/' + args.do_model + '.txt', args, shiyan)
    id = 0
    args.checkpoint_dir = 'models/' + args.do_model + str(id)
    one(args, schema_labels, predict_data, predict_sents, str(id))


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

