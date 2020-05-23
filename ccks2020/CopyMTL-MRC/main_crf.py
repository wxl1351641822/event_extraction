import os
import argparse
from typing import Tuple
from tqdm import tqdm
import torch
import torch.nn as nn

import const
import data_prepare
import evaluation
import torch.nn.functional as F



parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=str, default='0', help='gpu id')
parser.add_argument('--mode', '-m', type=str, default='train', help='train/valid/test')
parser.add_argument('--cell', '-c', type=str, default='lstm', help='gru/lstm')
parser.add_argument('--decoder_type', '-d', type=str, default='onecrf', help='one/multi/onecrf')

args = parser.parse_args()
mode = args.mode
cell_name = args.cell
decoder_type = args.decoder_type

torch.manual_seed(77)  # cpu
torch.cuda.manual_seed(77)  # gpu

if decoder_type=='onecrf':
    from model_crf import CrfSeq2seq as Seq2seq
else:
    from model import Seq2seq
class Evaluator(object):
    def __init__(self, config: const.Config, mode: str, device: torch.device) -> None:

        self.config = config

        self.device = device

        # self.seq2seq = Seq2seq(config, device=device)
        self.mode = mode
        self.max_sentence_length = config.max_sentence_length

    def load_model(self, seq2seq) -> None:
        id=1
        # model_path = os.path.join(self.config.runner_path, 'model.pkl')
        model_path = os.path.join('saved_model',
                                  self.config.dataset_name + '_' + self.config.cell_name + '_' + decoder_type+str(id) + '.pkl')
        seq2seq.load_state_dict(torch.load(model_path))
        return seq2seq

    def test_step(self, batch: data_prepare.InputData, seq2seq) -> Tuple[torch.Tensor, torch.Tensor]:

        sentence = batch.sentence_fw
        sentence_eos = batch.input_sentence_append_eos

        sentence = torch.from_numpy(sentence).to(self.device)
        sentence_eos = torch.from_numpy(sentence_eos).to(self.device)

        lengths = torch.Tensor(batch.input_sentence_length).int().tolist()
        all_events = batch.standard_outputs
        # all_triples = batch.all_triples
        all_events = torch.from_numpy(all_events).to(self.device).to(torch.long)
        if decoder_type=='onecrf':
            pred_action_list, loss = seq2seq(sentence, sentence_eos, lengths,all_events)
            # pred_action_list = torch.cat(list(map(lambda x: x.unsqueeze(1), pred_action_list)), dim=1)

            return pred_action_list, loss
        else:
            pred_action_list, pred_logits_list = seq2seq(sentence, sentence_eos, lengths)
            # pred_action_list = torch.cat(list(map(lambda x: x.unsqueeze(1), pred_action_list)), dim=1)

            return pred_action_list, pred_logits_list

    def test(self) -> Tuple[float, float, float]:
        predicts = []
        gold = []
        data = prepare.load_data(mode)
        if mode == 'test':
            data = prepare.test_process(data)
        else:
            data = prepare.process(data)
        data = data_prepare.Data(data, config.batch_size, config)
        for batch_i in range(data.batch_number):
            batch_data = data.next_batch(is_random=False)
            pred_action_list, pred_logits_list = self.test_step(batch_data)
            pred_action_list = pred_action_list.cpu().numpy()

            predicts.extend(pred_action_list)
            gold.extend(batch_data.all_triples)

        f1, precision, recall = evaluation.compare(predicts, gold, self.config, show_rate=None, simple=True)
        data.reset()
        return f1, precision, recall

    def rel_test(self) -> Tuple[Tuple[float, float, float]]:
        predicts = []
        gold = []
        data = prepare.load_data(mode)
        if mode == 'test':
            data = prepare.test_process(data)
        else:
            data = prepare.process(data)
        data = data_prepare.Data(data, config.batch_size, config)
        for batch_i in range(data.batch_number):
            batch_data = data.next_batch(is_random=False)
            pred_action_list, pred_logits_list = self.test_step(batch_data)
            pred_action_list = pred_action_list.cpu().numpy()

            predicts.extend(pred_action_list)
            gold.extend(batch_data.all_triples)

        (r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall) = evaluation.rel_entity_compare(predicts, gold,
                                                                                                     self.config)
        self.data.reset()
        return (r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall)

    def event_test(self, seq2seq) -> Tuple[Tuple[float, float, float]]:
        predicts = []
        gold = []
        data = prepare.load_data(self.mode)
        if mode == 'test':
            data = prepare.test_process(data)
        else:
            data = prepare.process(data)
        data = data_prepare.Data(data, config.batch_size, config)
        loss = 0.0
        # Loss=nn.NLLLoss()
        for batch_i in tqdm(range(data.batch_number)):
            batch_data = data.next_batch(is_random=False)
            pred_action_list, pred_logits_list = self.test_step(batch_data, seq2seq)
            pred_action_list = pred_action_list.cpu().numpy()
            gold.extend(batch_data.standard_outputs)
            # for i in range()
            predicts.extend([pred_action_list[:, i] for i in range(pred_action_list.shape[1])])

            if decoder_type=='onecrf':
                loss+=pred_logits_list#crf的时候输出的是loss
            else:
                if self.config.losstype == 1:
                    ##1.原来###################################
                    for t in range(seq2seq.decoder.decodelen):
                        # print(pred_logits_list[t])
                        loss = loss + F.nll_loss(pred_logits_list[t],
                                                 torch.from_numpy(batch_data.standard_outputs).to(self.device).to(
                                                     torch.long)[:, t])
                elif self.config.losstype == 2:
                    ##2.loss2,排列组合###################################
                    all_events = torch.from_numpy(batch_data.standard_outputs).to(self.device).to(torch.long)
                    all_triples = batch_data.all_triples
                    lengths = batch_data.input_sentence_length
                    # print(pred_logits_list)
                    # print(pred_action_list)
                    for i in range(all_events.shape[0]):
                        # print(all_triples[i])
                        now_loss = 0.
                        triple_num = min(len(all_triples[i]) // (lengths[i] + 1), self.config.triple_number)
                        # print(pred_action_list[:self.max_sentence_length*triple_num,i].shape)
                        # pred_logits_list_event[:1+triple_num,i]
                        # print(pred_logits_list_entity[:self.max_sentence_length*triple_num,i].shape)
                        for j in range(triple_num):
                            glob = all_events[i,
                                   j * (self.max_sentence_length + 1):(j + 1) * (self.max_sentence_length + 1)]
                            # print(glob.shape)
                            for k in range(triple_num):
                                # print(pred_logits_list.shape,pred_logits_list[k*(self.max_sentence_length+1):(k+1)*(self.max_sentence_length+1),i].shape)
                                now_loss += F.nll_loss(pred_logits_list[k * (self.max_sentence_length + 1):(k + 1) * (
                                        self.max_sentence_length + 1), i], glob)
                        if (triple_num != 0):
                            now_loss /= (triple_num * triple_num)
                        # print(pred_logits_list[triple_num*(self.max_sentence_length+1)+1,i],all_events[i,triple_num*(self.max_sentence_length+1)+1])
                        loss += now_loss + F.nll_loss(pred_logits_list[
                                                      triple_num * (self.max_sentence_length + 1):triple_num * (
                                                              self.max_sentence_length + 1) + 1, i],
                                                      all_events[i,
                                                      triple_num * (self.max_sentence_length + 1):triple_num * (
                                                              self.max_sentence_length + 1) + 1])

            # for t in range(seq2seq.decoder.decodelen):
            #     # print(pred_logits_list[t], batch_data.standard_outputs[:, t])
            #     loss += F.nll_loss(pred_logits_list[t],torch.from_numpy(batch_data.standard_outputs).to(self.device).to(torch.long)[:, t]).item()
            # print(loss)
        loss /= batch_i
        # for g in gold:
        #     for i in range(5):
        #         l=g[i*(config.max_sentence_length+1):(i+1)*(config.max_sentence_length+1)]
        #         if(l[0]>30):
        #             print(l)

        require_f1,require_precision,require_recall=evaluation.event_entity_yaoqiu_compare(predicts,gold,self.config)
        f1, precision, recall = evaluation.compare(predicts, gold, self.config, show_rate=None, simple=True)
        (r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall) = evaluation.rel_entity_compare(predicts, gold,
                                                                                                     self.config)
        (event_f1, event_precision, event_recall), (
            entity_f1, entity_precision, entity_recall) = evaluation.event_entity_compare(predicts, gold,self.config)
        data.reset()
        return loss.item(),(require_f1,require_precision,require_recall), (f1, precision, r_recall), (r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall), (
            event_f1, event_precision, event_recall), (entity_f1, entity_precision, entity_recall)

    def predict(self, seq2seq):
        predicts = []
        ids = []
        sentences = []
        lengths = []
        seq2seq = Seq2seq(self.config, device=self.device, load_emb=True)
        # gold = []
        data = prepare.load_data(self.mode)
        if mode == 'test':
            data = prepare.test_process(data)
        else:
            data = prepare.process(data)
        data = data_prepare.Data(data, config.batch_size, config)
        for batch_i in range(data.batch_number):
            batch_data = data.next_batch(is_random=False)
            pred_action_list, _ = self.test_step(batch_data, seq2seq)
            pred_action_list = pred_action_list.cpu().numpy()

            sentences.extend(batch_data.sentence_fw)
            predicts.extend([pred_action_list[:, i] for i in range(pred_action_list.shape[1])])
            # print(len(predicts))
            ids.extend(batch_data.standard_outputs)
            lengths.extend(batch_data.input_sentence_length)

        evaluation.get_result(ids, sentences, lengths, predicts, config)
        # gold.extend(batch_data.all_triples)

        # (r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall) = evaluation.rel_entity_compare(predicts, gold, self.config)
        data.reset()
        # return (r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall)


class SupervisedTrainer(object):
    def __init__(self, config: const.Config, device: torch.device) -> None:

        self.config = config

        self.device = device

        self.seq2seq = Seq2seq(config, device=device, load_emb=True)

        self.loss = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.seq2seq.parameters())
        config.name = 'train'
        data = prepare.load_data('train')
        data = prepare.process(data)
        self.data = data_prepare.Data(data, config.batch_size, config)
        self.max_sentence_length = config.max_sentence_length
        self.epoch_number = config.epoch_number + 1

    def train_step(self, batch: data_prepare.InputData) -> torch.Tensor:

        self.optimizer.zero_grad()

        sentence = batch.sentence_fw
        sentence_eos = batch.input_sentence_append_eos

        all_events = batch.standard_outputs
        all_triples = batch.all_triples
        all_events = torch.from_numpy(all_events).to(self.device).to(torch.long)
        sentence = torch.from_numpy(sentence).to(self.device)
        sentence_eos = torch.from_numpy(sentence_eos).to(self.device)

        lengths = torch.Tensor(batch.input_sentence_length).int().tolist()
        if decoder_type=='onecrf':
            pred_action_list, loss=self.seq2seq(sentence, sentence_eos, lengths,all_events)
        else:
            pred_action_list, pred_logits_list = self.seq2seq(sentence, sentence_eos, lengths)

            if self.config.losstype == 1:
                ##1.原来###################################
                loss = 0
                for t in range(self.seq2seq.decoder.decodelen):
                    # print(pred_logits_list[t])
                    loss = loss + self.loss(pred_logits_list[t], all_events[:, t])
                    print(loss)
                    # break
            elif self.config.losstype == 2:
                ##2.loss2,排列组合###################################
                loss = 0.
                # print(pred_logits_list)
                # print(pred_action_list)
                for i in range(all_events.shape[0]):
                    # print(all_triples[i])
                    now_loss = 0.
                    triple_num = min(len(all_triples[i]) // (lengths[i] + 1), self.config.triple_number)
                    # print(pred_action_list[:self.max_sentence_length*triple_num,i].shape)
                    # pred_logits_list_event[:1+triple_num,i]
                    # print(pred_logits_list_entity[:self.max_sentence_length*triple_num,i].shape)
                    for j in range(triple_num):
                        glob = all_events[i, j * (self.max_sentence_length + 1):(j + 1) * (self.max_sentence_length + 1)]
                        # print(glob.shape)
                        for k in range(triple_num):
                            # print(pred_logits_list_entity[k*(self.max_sentence_length+1):(k+1)*(self.max_sentence_length+1),i].shape)
                            now_loss += self.loss(pred_logits_list[k * (self.max_sentence_length + 1):(k + 1) * (
                                        self.max_sentence_length + 1), i], glob)
                    if triple_num!=0:
                        now_loss /= (triple_num * triple_num)
                    # print(pred_logits_list[triple_num*(self.max_sentence_length+1)+1,i],all_events[i,triple_num*(self.max_sentence_length+1)+1])
                    loss += now_loss + self.loss(pred_logits_list[triple_num * (self.max_sentence_length + 1):triple_num * (
                                self.max_sentence_length + 1) + 1, i], all_events[i, triple_num * (
                                self.max_sentence_length + 1):triple_num * (self.max_sentence_length + 1) + 1])

        require_f1, require_precision, require_recall = evaluation.event_entity_yaoqiu_compare(
            [pred_action_list[:, i] for i in range(pred_action_list.shape[1])], batch.standard_outputs,
            self.config)
        loss.backward()
        self.optimizer.step()
        return loss,require_f1, require_precision, require_recall

    def train(self, id,evaluator: Evaluator = None) -> None:
        # self.seq2seq = evaluator.load_model(self.seq2seq)
        train_loss = 0.
        best_event_f1 = 0.0
        best_entity_f1 = 0.0
        best_f1 = 0.0
        ll = ['epoch', 'step', 'train_loss', 'eval_loss', 'f1', 'precision', 'recall', 'r_f1', 'r_precision', 'r_recall', 'e_f1', 'e_precision',
              'e_recall', 'event_f1', 'event_precision', 'event_recall', 'entity_f1', 'entity_precision', 'entity_recall']
        train_f1,train_precision,train_recall=0.,0.,0.

        with open('./log/' + self.config.dataset_name + '_' + self.config.cell_name + '_' + decoder_type + str(
                id) + '.txt', 'w', encoding='utf-8') as f:
            f.write('\t'.join(ll)+'\n')
        for epoch in range(1, self.epoch_number + 1):

            for step in range(self.data.batch_number):
                batch = self.data.next_batch(is_random=True)
                loss,require_f1, require_precision, require_recall = self.train_step(batch)

                # for i in range()


                train_loss += loss.item()
                train_f1+=require_f1
                train_precision+=require_precision
                train_recall+=require_recall
                # print(loss)
                # print('train:epoch: %d\t step: %d \t loss:%f' % (epoch, step, loss))
                if(step%10==0):
                    if (step != 0):
                        train_loss /= 10.
                        train_f1 /= 10.
                        train_precision /= 10.
                        train_recall /= 10.
                    print("train \t epoch %d\t step %d \t trainloss: %f \t f1: %f  \t precision: %f  \t recall: %f  \t" % (
                        epoch, step, train_loss,train_f1,train_precision,train_recall))
                    ll = [epoch, step, train_loss,train_f1,train_precision,train_recall]
                    ll = [str(x) for x in ll]
                    with open(
                            './log/train' + self.config.dataset_name + '_' + self.config.cell_name + '_' + decoder_type + str(
                                    id) + '.txt', 'a', encoding='utf-8') as f:
                        f.write('\t'.join(ll) + '\n')
                    train_loss,train_f1, train_precision, train_recall = 0.,0., 0., 0.
                if (step % 200 == 0):

                    model_path = os.path.join('saved_model',
                                              self.config.dataset_name + '_' + self.config.cell_name + '_' + decoder_type +str(id)+ '.pkl')
                    torch.save(self.seq2seq.state_dict(), model_path)

                    if evaluator:
                        with torch.no_grad():
                            # evaluator.load_model()
                            # f1, precision, recall = evaluator.test()
                            eval_loss,(require_f1,require_precision,require_recall),(f1, precision, recall), (r_f1, r_precision, r_recall), (
                            e_f1, e_precision, e_recall), (
                                event_f1, event_precision, event_recall), (
                                entity_f1, entity_precision, entity_recall) = evaluator.event_test(self.seq2seq)

                            print('_' * 60)
                            print("epoch %d\t step %d \t evalloss: %f  \t" % (
                                epoch, step, eval_loss))
                            print("require_event_entity \t F1: %f \t P: %f \t R: %f \t" % (require_f1,require_precision,require_recall))
                            print("total \t F1: %f \t P: %f \t R: %f \t" % (f1, precision, recall))
                            print("event \t F1: %f \t P: %f \t R: %f \t" % (r_f1, r_precision, r_recall))
                            print("entity \t F1: %f \t P: %f \t R: %f \t" % (e_f1, e_precision, e_recall))
                            print("macro event \t F1: %f \t P: %f \t R: %f \t" % (
                            event_f1, event_precision, event_recall))
                            print("macro entity \t F1: %f \t P: %f \t R: %f \t" % (
                                entity_f1, entity_precision, entity_recall))
                            print('_' * 60)

                            ll=[epoch,step,require_f1,require_precision,require_recall,eval_loss,f1, precision, recall,r_f1, r_precision, r_recall,e_f1, e_precision, e_recall,event_f1, event_precision, event_recall,entity_f1, entity_precision, entity_recall]
                            ll=[str(x) for x in ll]
                            with open('./log/dev'+self.config.dataset_name + '_' + self.config.cell_name + '_' + decoder_type +str(id)+ '.txt','a',encoding='utf-8') as f:
                                f.write('\t'.join(ll)+'\n')
                            train_loss = 0.0
                            if (best_f1 < f1):
                                best_f1 = f1
                                model_path = os.path.join('saved_model',
                                                          'best_'+self.config.dataset_name + '_' + self.config.cell_name + '_' + decoder_type + str(id)+'.pkl')
                                torch.save(self.seq2seq.state_dict(), model_path)
                            if (best_event_f1 < event_f1):
                                best_event_f1 = event_f1
                                model_path = os.path.join('saved_model',
                                                          'event_best_'+self.config.dataset_name + '_' + self.config.cell_name + '_' + decoder_type + str(id)+'.pkl')
                                torch.save(self.seq2seq.state_dict(), model_path)
                            if (best_entity_f1 < entity_f1):
                                best_entity_f1 = entity_f1
                                model_path = os.path.join('saved_model',
                                                          'entity_best_'+self.config.dataset_name + '_' + self.config.cell_name + '_' + decoder_type + str(id)+'.pkl')
                                torch.save(self.seq2seq.state_dict(), model_path)


if __name__ == '__main__':

    config_filename = './config.json'
    config = const.Config(config_filename=config_filename, cell_name=cell_name, decoder_type=decoder_type)
    config.name = mode
    assert cell_name in ['lstm', 'gru']
    assert decoder_type in ['one', 'multi','onecrf']
    # config.dataset_name = const.DataSet.WEBNLG
    if config.dataset_name == const.DataSet.NYT:
        prepare = data_prepare.NYTPrepare(config)
    elif config.dataset_name == const.DataSet.WEBNLG:
        prepare = data_prepare.WebNLGPrepare(config)
    elif config.dataset_name == const.DataSet.CCKS2020EVENT:
        prepare = data_prepare.CCKSPrepare(config)
    else:
        print('illegal dataset name: %s' % config.dataset_name)
        exit()

    device = torch.device('cuda:' + args.gpu)

    train = True if mode == 'train' else False
    # evaluator = Evaluator(config, 'valid', device)
    # evaluator.data.reset()
    # # evaluator.load_model()
    # # f1, precision, recall = evaluator.test()
    # (f1, precision, recall), (r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall), (
    #     event_f1, event_precision, event_recall), (
    #     entity_f1, entity_precision, entity_recall) = evaluator.event_test()
    id=1
    if train:
        trainer = SupervisedTrainer(config, device)
        evaluator = Evaluator(config, 'valid', device)
        trainer.train(id,evaluator)
    else:
        tester = Evaluator(config, mode, device)
        seq2seq = Seq2seq(config, device=device, load_emb=True)
        seq2seq = tester.load_model(seq2seq)

        tester.predict(seq2seq)

# 非常幸运和大家分到一个班级，交到许多朋友，也非常荣幸能够成为704班的生活委员。
# 学习方面，专业课加权平均分85左右，思政、英语、体育课外加权平均87左右。
# 职务方面，参加过一次院系组织的卫生检查、负责保管班级邮箱钥匙、统计办理学生公交卡等。
# 比赛方面，参加了北京市经济和信息化局 & CCF大数据专家委员会 & 中国中文信息学会信息检索专业委员会主办的 《疫情期间网民情绪识别》，A榜36、B榜63。参加了2019CCF主办的互联网新闻情感分析但并未进入复赛。
# 参军的经历保证了我思想上的端正态度，目前是一名积极分子。
#
# 1.篮球赛女生投篮
# 2.10.11班级团建
# 4.10.26班级上山送水
# 5.参观两弹一星纪念馆
# 6.进行宿舍卫生检查工作（职责）


