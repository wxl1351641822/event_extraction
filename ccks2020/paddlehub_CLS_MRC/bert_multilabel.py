from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
# os.system('pip install --upgrade paddlehub==1.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple')
import csv
import paddle
import paddle.fluid as fluid
import collections
try:
    from paddle.fluid.contrib.trainer import *
    from paddle.fluid.contrib.inferencer import *
except ImportError:
    from paddle.fluid.trainer import *
    from paddle.fluid.inferencer import *
# 自定义数据集
import argparse
import ast
import codecs
import csv
import io
import paddlehub as hub
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset
from collections import namedtuple
from paddlehub.reader import tokenization
import numpy as np
from tb_paddle import SummaryWriter
import time
from collections import OrderedDict
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from paddlehub.common.logger import logger
from paddlehub.finetune.checkpoint import save_checkpoint
from visualdl import LogWriter
from paddlehub.reader.batching import pad_batch_data
from paddlehub.common.utils import version_compare
import json
from paddlehub.dataset import InputExample

output_prefix = './work/event/'
checkpoint_dir = "model"

seed = 1666
np.random.seed(seed)
# ################启动时如果有checkpoint，是否使用上一次启动train的最好结果？#####################
test = None  # 不用
SPIECE_UNDERLINE = '▁'
log_interval = 50
eval_interval = 200

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--seed", type=int, default=0, help="Number of epoches for fine-tuning.")
parser.add_argument("--num_epoch", type=int, default=4, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--lr_scheduler", type=str, default="linear_decay", help="Directory to model checkpoint")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--checkpoint_dir", type=str, default='model_cls/', help="Directory to model checkpoint")
parser.add_argument("--model", type=str, default="bert_chinese_L-12_H-768_A-12", help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=203, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=True, help="Whether use data parallel.")


def get_diclist(path):
    with open(path,'r',encoding='utf-8') as f:
        dic=eval(f.read())
    result=[x for x in list(dic.keys())]
    return result
labellist=get_diclist(output_prefix+'event2id.txt')


# 新增prob--取代了text_b
class MyDataset(BaseNLPDataset):
    """DemoDataset"""

    def __init__(self):
        # 数据集存放位置
        self.dataset_dir = output_prefix

        super(MyDataset, self).__init__(
            base_path=self.dataset_dir,
            train_file="train.json",
            dev_file="dev.json",
            test_file=test,
            train_file_with_header=False,
            dev_file_with_header=False,
            test_file_with_header=False,
            # 数据集类别集合
            label_list=labellist)

    def _read_file(self, input_file, phase=False):
        """
        读入json格式数据集
        """
        examples = []
        drop = 0
        with open(input_file, "r") as reader:
            input_data = json.load(reader)["data"]
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                guid=[]
                labels=[0]*len(self.label_list)
                for qa in paragraph["qas"]:
                    guid.append(qa["id"])
                    labels[self.label_list.index(qa["question"].replace('的主体是什么？',''))]=1
                guid=str(list(set(guid)))
                example = InputExample(guid=guid, label=labels, text_a=paragraph_text)
                examples.append(example)
        logger.warning("%i bad examples has been dropped" % drop)
        return examples

# 重构train_log和eval_log时的事件，
# 1.增加visualdl
# 2.修改以eval_loss保存最好模型，
# 3.eval保存的模型不再是推断模型，而是与step时一样的训练模型
def change_task(task,train_i):
    def new_log_interval_event(self, run_states):
        scores, avg_loss, run_speed = self._calculate_metrics(run_states)
        self.tb_writer.add_scalar(
            tag="Loss_{}".format(self.phase),
            scalar_value=avg_loss,
            global_step=self._envs['train'].current_step)
        log_scores = ""
        log=[self._envs['train'].current_step,avg_loss]
        for metric in scores:
            self.tb_writer.add_scalar(
                tag="{}_{}".format(metric, self.phase),
                scalar_value=scores[metric],
                global_step=self._envs['train'].current_step)
            log_scores += "%s=%.5f " % (metric, scores[metric])
            log.append(scores[metric])
        logger.train("step %d / %d: loss=%.5f %s[step/sec: %.2f]" %
                     (self.current_step, self.max_train_steps, avg_loss,
                      log_scores, run_speed))
        log = [str(x) for x in log]
        with open('./work/log/cls_log_{}train.txt'.format(train_i), 'a', encoding='utf-8') as f:
            f.write(','.join(log) + '\n')

    # def new_run_step_event(self,run_states):
    def new_eval_end_event(self, run_states):
        """
        Paddlehub default handler for eval_end_event, it will complete visualization and metrics calculation
        Args:
            run_states (object): the results in eval phase
        """
        eval_scores, eval_loss, run_speed = self._calculate_metrics(run_states)
        log=[]
        if 'train' in self._envs:
            self.tb_writer.add_scalar(
                tag="Loss_{}".format(self.phase),
                scalar_value=eval_loss,
                global_step=self._envs['train'].current_step)
            log=[self._envs['train'].current_step]

        log_scores = ""

        log.append(eval_loss)
        for metric in eval_scores:
            if 'train' in self._envs:
                self.tb_writer.add_scalar(
                    tag="{}_{}".format(metric, self.phase),
                    scalar_value=eval_scores[metric],
                    global_step=self._envs['train'].current_step)
            log_scores += "%s=%.5f " % (metric, eval_scores[metric])
            log.append(eval_scores[metric])
        logger.eval(
            "[%s dataset evaluation result] loss=%.5f %s[step/sec: %.2f]" %
            (self.phase, eval_loss, log_scores, run_speed))
        log = [str(x) for x in log]
        with open('./work/log/cls_log_{}dev.txt'.format(train_i),'a',encoding='utf-8') as f:
            f.write(','.join(log)+'\n')

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



    # name：hook名字，“default”表示PaddleHub内置_log_interval_event实现
    task.delete_hook(hook_type="eval_end_event", name="default")
    task.delete_hook(hook_type="log_interval_event", name="default")
    task.add_hook(hook_type="eval_end_event", name="new_eval_end_event", func=new_eval_end_event)
    task.add_hook(hook_type="log_interval_event", name="new_log_interval_event", func=new_log_interval_event)
    return task


# 获取task和reader
def train(train_i,args):
    dataset = MyDataset()
    module = hub.Module(name=args.model)
    reader = hub.reader.MultiLabelClassifyReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len)

    strategy = hub.AdamWeightDecayStrategy(
        weight_decay=args.weight_decay,
        warmup_proportion=args.warmup_proportion,
        lr_scheduler=args.lr_scheduler,
        learning_rate=args.learning_rate)
    config = hub.RunConfig(
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        checkpoint_dir=args.checkpoint_dir+str(train_i),
        batch_size=args.batch_size,
        eval_interval=eval_interval,
        log_interval=log_interval,
        strategy=strategy)
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Use "pooled_output" for classification tasks on an entire sentence.
    pooled_output = outputs["pooled_output"]

    # feed_list的Tensor顺序不可以调整
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    cls_task = hub.MultiLabelClassifierTask(
        data_reader=reader,
        feature=pooled_output,
        feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config)
    cls_task.main_program.random_seed=args.seed
    change_task(cls_task, train_i)
    return cls_task, reader

# 最终预测结果
def predict(train_i, cls_task):
    test = pd.read_csv(output_prefix+'test_unlabel.csv', sep='\t', header=None)
    data= test[1].values.tolist()
    data = [[t] for t in data]
    run_states = cls_task.predict(data=data,return_result=True)
    all_events=[]
    for result in run_states:
        dic={}#collections.defaultdict(list)
        for l,r in zip(labellist,result):
            if(r[l]==1):
                dic[l]=[]
        all_events.append(dic)
    test[2]=all_events
    test.to_csv('./work/result/{}event_predict.csv'.format(train_i),sep='\t',header=False,index=False)
def one(train_i,args):
    # m=args.checkpoint_dir
    # args.checkpoint_dir = m + '_' + str(train_i)
    cls_task, reader = train(train_i,args)
    cls_task.finetune_and_eval()
    predict(train_i, cls_task)
    value = [train_i, cls_task.best_score] + list(args.__dict__.values())
    value = [str(x) for x in value]
    with open('./work/log/cls_log.txt', 'a', encoding='utf-8') as f:
        f.write(','.join(value)+',-\n')
    return cls_task.best_score,value[2:]

if __name__=='__main__':
    args = parser.parse_args()
    title = ['id', 'auc'] + list(args.__dict__.keys())
    with open('./work/log/cls_log.txt', 'a', encoding='utf-8') as f:
        f.write(','.join(title)+',备注\n')
    train_i=0
    one(train_i,args)
