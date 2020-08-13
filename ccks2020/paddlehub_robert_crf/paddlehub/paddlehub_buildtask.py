import json
import shutil
import os
import csv
import io
import json
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import paddlehub as hub
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset
from paddlehub.common.logger import logger
from paddlehub.dataset import InputExample
from datetime import datetime
# from paddlehub.dataset.base_nlp_dataset import MultiLabelDataset
# from Sequence_Reader import SequenceLabelReader
# from sequence_task import SequenceLabelTask
from paddlehub import MultiLabelClassifierTask
from paddlehub.reader import MultiLabelClassifyReader
from paddlehub import SequenceLabelTask
from paddlehub.reader import SequenceLabelReader

from paddlehub_dataprocess import write_by_lines,read_by_lines,regrex_data,read_label
from paddlehub_dataprocess import write_title
from paddlehub_dataprocess import write_log
from dataset import CCksDataset,EEDataset

from config import *
# parser.add_argument(
#     "--saved_params_dir",
#     type=str,
#     default="",
#     help="Directory for saving model during ")

def is_path_valid(path):
    if path == "":
        return False
    path = os.path.abspath(path)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    return True


def get_mcls_train_dev(args):
    c = pd.read_csv('./work/data_cls.csv', header=None, sep='\t').sort_values(by=0,ascending=True)
    spl = int(c.shape[0] * 0.8)
    train = c[:spl]
    dev = c[spl:]
    train.to_csv('./work/train_cls.csv',header=None, sep='\t',index=False)
    dev.to_csv('./work/dev_cls.csv', header=None, sep='\t', index=False)
    return train,dev


def get_mcls_predict(args):
    data = pd.read_csv(mcls_predict_path, sep='\t', header=None).fillna('')
    predict_sents = []
    predict_list=[]
    for id, sent,entity,label in data.values:
        # print(id,text)
        sent=regrex_data(sent)
        max_seq_len=len(sent)+3
        for pro in range(len(sent) // (max_seq_len - 2)+1):
            text = sent[pro * (max_seq_len - 2):(pro + 1) * (max_seq_len - 2)]
            predict_sents.append({'id': id, 'text': text,'entity':entity,'origlabel':label})
            predict_list.append([text,entity])

    return predict_list, predict_sents

def get_train_dev(args):
    c = pd.read_csv('./work/data.csv', header=None, sep='\t')
    # print(c.shape[0])
    spl = int(c.shape[0] * 0.8)
    train = c[:spl]
    dev = c[spl:]
    # test=[]

    def get_data2id(c_data, path):
        sent_length = []
        all_sent = []
        all_label = []
        s = ['text_a\tlabel']
        count=0
        count_n=0
        for sent, dic in c_data[[1, 2]].values:
            # print(sent)
            sent1=regrex_data(sent)
            sentence1 = list(sent1)
            dic = eval(dic)
            # print(count,len(s),len(sentence1)//(args.max_seq_len-2))
            max_seq_len=len(sentence1)+3
            for pro in range(len(sentence1)//(max_seq_len-2)+1):
                count+=1
                # print(pro)
                sentence=sentence1[pro*(max_seq_len-2):(pro+1)*(max_seq_len-2)]
                sent=sent1[pro*(max_seq_len-2):(pro+1)*(max_seq_len-2)]
                sent_length.append(len(sentence))
                # if (pro >=1):
                #     print(len(sentence1))
                    # print(sentence1,sentence,sent)
                if(args.change_event=='no'):
                    label = ['<NA>'] * len(sentence)
                else:
                    label = ['O'] * len(sentence)

                for k, v in dic.items():
                    # print(k,v)
                    # flag=1
                    entity_label = label
                    # print(entity2id)
                    for entity in v:
                        beg = sent.find(entity)
                        end = sent.find(entity) + len(entity) - 1
                        if(beg==-1):
                            continue
                        # print(sent,sentence)
                        # print(beg,end,entity,len(sentence),sent[beg:end+1],sentence[beg:end+1])
                        if(args.change_event=='no'):
                            for i in range(beg,end+1):

                                entity_label[i]=k
                        elif(args.change_event=='BIO_event'):
                            if (entity_label[beg] != 'O'):
                                count_n+=1
                                print(count_n,entity,dic)
                            entity_label[beg] = 'B-' + k

                            for i in range(beg + 1, end+1):
                                entity_label[i] = 'I-' + k
                        else:#BIO
                            entity_label[beg]='B'
                            for i in range(beg+1,end+1):
                                entity_label[i]='I'
                    label=entity_label

                # if (len(label) == 0):
                #     label = ['0'] * len(sentence)
                s.append('\002'.join(sentence) + '\t' + '\002'.join(label))


        # print(len(s))

        # with open(path, 'w', encoding='utf-8') as f:
        #     f.write('\n'.join(s))
        return s

    train1 = get_data2id(train, './work/train.txt')
    dev1 = get_data2id(dev, './work/dev.txt')
    return train1,dev1


def get_predict(args):
    data = pd.read_csv(predict_path, sep='\t', header=None).fillna('')
    # predict = pd.DataFrame()
    # predict['text_a'] = ['\002'.join(list(regrex_data(x))) for x in data[1].values]
    # predict.to_csv(predict_data_path, index=False)
    predict_sents = []
    predict_list=[]
    for id, sent in data.values:
        # print(id,text)
        sent=regrex_data(sent)
        max_seq_len=len(sent)+3
        for pro in range(len(sent) // (max_seq_len - 2)+1):
            text = sent[pro * (max_seq_len - 2):(pro + 1) * (max_seq_len - 2)]
            predict_sents.append({'id': id, 'text': text})
            predict_list.append('\002'.join(list(text)))

    return predict_list, predict_sents




# yapf: enable.
def process_data(args):
    if args.do_model=='mcls':
        train1,dev1=get_mcls_train_dev(args)
        predict_data, predict_sents = get_mcls_predict(args)
        # write_by_lines("{}/predict_mcls.txt".format(args.data_dir), predict_data)
        schema_labels = read_label('{}/event2id.txt'.format(args.data_dir))[1:]
    else:
        train1, dev1 = get_train_dev(args)
        predict_data, predict_sents = get_predict(args)

        write_by_lines("{}/train.txt".format(args.data_dir), train1)
        write_by_lines("{}/dev.txt".format(args.data_dir), dev1)
        # write_by_lines("{}/{}_test.tsv".format(args.data_dir, args.do_model), test_data)
        write_by_lines("{}/predict.txt".format(args.data_dir), predict_data)
        if (args.change_event == 'BIO_event'):
            schema_labels = read_label('{}/entity2id.txt'.format(args.data_dir))
        elif (args.change_event == 'no'):
            schema_labels = read_label('{}/event2id.txt'.format(args.data_dir))
        else:
            schema_labels = ['O', 'B', 'I']
    return schema_labels, predict_data, predict_sents




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
            self.vdl_writer.add_scalar(
                tag="Loss_{}".format(self.phase),
                value=eval_loss,
                step=self._envs['train'].current_step)

        log_scores = ""

        s = []

        for metric in eval_scores:
            if 'train' in self._envs:
                self.vdl_writer.add_scalar(
                    tag="{}_{}".format(metric, self.phase),
                    value=eval_scores[metric],
                    step=self._envs['train'].current_step)
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
            if(args.dev_goal=='f1'):
                main_metric, main_value = list(eval_scores_items)[0]
            else:#loss
                main_metric, main_value = "negative loss", -eval_loss
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
        self.vdl_writer.add_scalar(
            tag="Loss_{}".format(self.phase),
            value=avg_loss,
            step=self._envs['train'].current_step)
        log_scores = ""
        s = [self.current_step]
        for metric in scores:
            self.vdl_writer.add_scalar(
                tag="{}_{}".format(metric, self.phase),
                value=scores[metric],
                step=self._envs['train'].current_step)
            log_scores += "%s=%.5f " % (metric, scores[metric])
            s.append(scores[metric])
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
    model_name = args.model_name
    module = hub.Module(name=model_name)
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)
    # if args.model=='mcls':
    tokenizer = hub.BertTokenizer(vocab_file=module.get_vocab_path())  # 加载数据并通过SequenceLabelReader读取数据
    if(args.do_model=='mcls'):
        dataset = CCksDataset(args.data_dir, schema_labels, model=args.do_model,tokenizer=tokenizer,max_seq_len=args.max_seq_len)
        reader = MultiLabelClassifyReader(
            dataset=dataset,
            vocab_path=module.get_vocab_path(),
            max_seq_len=args.max_seq_len,
            sp_model_path=module.get_spm_path(),
            word_dict_path=module.get_word_dict_path())

        # 构建序列标注任务迁移网络
        # 使用ERNIE模型字级别的输出sequence_output作为迁移网络的输入
        output=outputs["pooled_output"]
    else:
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
        output = outputs["sequence_output"]
    # else:
    #     sequence_output = outputs["sequence_output"]
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
    if args.do_model=='mcls':
        task = MultiLabelClassifierTask(
            data_reader=reader,
            feature=output,
            feed_list=feed_list,
            num_classes=dataset.num_labels,
            config=config)
    else:
        task = SequenceLabelTask(
            data_reader=reader,
            feature=output,
            feed_list=feed_list,
            max_seq_len=args.max_seq_len,
            num_classes=dataset.num_labels,
            config=config,
            add_crf=args.add_crf)
    task.main_program.random_seed = args.random_seed
    add_hook(args, task, id)
    return task, reader
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
        write_by_lines("{}.{}.{}.pred".format(output_predict_data_path, args.do_model, id), ret)


def predict_model_path():
    args = parser.parse_args()
    schema_labels, predict_data, predict_sents = process_data(args)
    trigger = pd.read_csv('./work/log/trigger.txt', header=None)
    # print(trigger)
    # print(trigger[[0,17]])
    for id, model_path in trigger[[0, 17]].values:
        print(id, model_path)
        predict_by_model_path(args, model_path, schema_labels, predict_data, predict_sents, id)


def get_submit_postprocess(args,id,check=False,mcls=False):
    results = read_by_lines("{}.{}.{}.pred".format(output_predict_data_path, args.do_model, id))
    submit = []
    count = 0
    # print(results)
    for j in range(len(results)):
        json_result = json.loads(results[j])
        text = json_result['text']
        label = json_result["labels"]
        now_label = ''
        now_entity = -1
        count = 0
        # print(len(text),len(label))
        for i, l in enumerate(label):
            # print(l,text[i])
            if (l == 'O' or l=='<NA>'):
                if (now_label != ''):
                    count += 1
                    if(check):
                        submit.append('\t'.join([str(json_result['id']), now_label, text[now_entity:i],
                                                 str(json_result['input'][0][now_entity * 2:i * 2]),
                                                 str(json_result['input'][0]), text, str(label)]))
                    elif(mcls):
                        submit.append('\t'.join([str(json_result['id']), text,text[now_entity:i],now_label]))
                    else:
                        submit.append('\t'.join([str(json_result['id']), now_label,  text[now_entity:i]]))
                    now_label = ''
            else:
                print(l,text[i])
                if (l.startswith('B')):
                    if(args.change_event =='BIO_event'):
                        now_label = l[2:]
                    else:
                        now_label = l
                    now_entity = i
                    # print()
                elif (args.add_rule and now_label == ''):
                    # print(args.change_event)
                    if(args.change_event=='BIO_event' and label[i][2:]==label[now_entity][2:]):
                        now_label=label[i][2:]
                        submit.pop(-1)
        # if(count==0):
        #     submit.append('\t'.join([str(json_result['id']),'','',text,str(label)]))
        # print(submit)
    if(check):
        write_by_lines("{}/{}ucas_valid_result_check.csv".format(output_path, id), submit)
    else:
        write_by_lines("{}/{}ucas_valid_result.csv".format(output_path,id), submit)

