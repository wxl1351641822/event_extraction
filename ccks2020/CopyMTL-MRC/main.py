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

from model import Seq2seq

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=str, default='0', help='gpu id')
parser.add_argument('--mode', '-m', type=str, default='train', help='train/valid/test')
parser.add_argument('--cell', '-c', type=str, default='lstm', help='gru/lstm')
parser.add_argument('--decoder_type', '-d', type=str, default='one', help='one/multi/onecrf')

args = parser.parse_args()
mode = args.mode
cell_name = args.cell
decoder_type = args.decoder_type

torch.manual_seed(77)  # cpu
torch.cuda.manual_seed(77)  # gpu


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

    def rel_test(self,seq2seq) -> Tuple[Tuple[float, float, float]]:
        predicts = []
        gold = []
        loss=0.0
        data = prepare.load_data(self.mode)
        if mode == 'test':
            data = prepare.test_process(data)
        else:
            data = prepare.process(data)
        data = data_prepare.Data(data, config.batch_size, config)
        for batch_i in tqdm(range(data.batch_number)):

            batch_data = data.next_batch(is_random=False)

            pred_action_list, pred_logits_list = self.test_step(batch_data,seq2seq)

            predicts.extend(pred_action_list)
            gold.extend(batch_data.all_triples)
            mean_loss=0.0
            if self.config.losstype == 1:
                ##1.原来###################################
                for t in range(seq2seq.decoder.decodelen):
                    # print(pred_logits_list[t])
                    mean_loss = mean_loss + F.nll_loss(pred_logits_list[t],
                                             torch.from_numpy(batch_data.standard_outputs).to(self.device).to(
                                                 torch.long)[:, t])
                    # print(pred_logits_list[t],
                    #                          torch.from_numpy(batch_data.standard_outputs).to(self.device).to(
                    #                              torch.long)[:, t])
                    # print(torch.from_numpy(batch_data.standard_outputs).to(self.device).to(
                    #                              torch.long)[:, t],pred_logits_list[t].shape,F.nll_loss(pred_logits_list[t],
                    #                          torch.from_numpy(batch_data.standard_outputs).to(self.device).to(
                    #                              torch.long)[:, t]),loss)
            mean_loss/=pred_logits_list[0].shape[0]
            if(batch_i<1000):
                loss+=mean_loss




        loss /= 1000
        f1, precision, recall = evaluation.compare(predicts, gold, self.config, show_rate=None, simple=True)
        (r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall) = evaluation.rel_entity_compare(predicts, gold,
                                                                                                     self.config)

        return loss.item(),(f1, precision, recall),(r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall)

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
        require_f1, require_precision, require_recall = evaluation.event_entity_yaoqiu_compare(predicts, gold,
                                                                                               self.config)
        f1, precision, recall = evaluation.compare(predicts, gold, self.config, show_rate=None, simple=True)
        (r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall) = evaluation.rel_entity_compare(predicts, gold,
                                                                                                     self.config)
        (event_f1, event_precision, event_recall), (
            entity_f1, entity_precision, entity_recall) = evaluation.event_entity_compare(predicts, gold, self.config)
        data.reset()
        return loss.item(), (require_f1, require_precision, require_recall), (f1, precision, r_recall), (
        r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall), (
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
            pred_action_list, pred_logits_list = self.test_step(batch_data, seq2seq)
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

        pred_action_list, pred_logits_list = self.seq2seq(sentence, sentence_eos, lengths)

        if self.config.losstype == 1:
            ##1.原来###################################
            loss = 0
            for t in range(self.seq2seq.decoder.decodelen):
                # print(pred_logits_list[t])
                loss = loss + self.loss(pred_logits_list[t], all_events[:, t])
                # break
        (r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall) = evaluation.rel_entity_compare(pred_action_list, batch.all_triples,
                                                                                                     self.config)
        f1, precision, recall = evaluation.compare(pred_action_list, batch.all_triples,self.config, show_rate=None, simple=True)
        loss.backward()
        self.optimizer.step()
        return loss,(f1, precision, recall),(r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall)

    def train(self, id,evaluator: Evaluator = None) -> None:
        # self.seq2seq = evaluator.load_model(self.seq2seq)
        train_loss = 0.
        train_f1, train_precision, train_recall = 0., 0., 0.
        train_rf1, train_rprecision, train_rrecall = 0., 0., 0.
        train_ef1, train_eprecision, train_erecall = 0., 0., 0.
        ll = ['epoch','step','train_loss','train_f1','train_precision','train_recall','train_rf1','train_rprecision','train_rrecall','train_ef1','train_eprecision','train_erecall']
        with open('./log/train' + self.config.dataset_name + '_' + self.config.cell_name + '_' + decoder_type + str(
                id) + '.txt', 'w', encoding='utf-8') as f:
            f.write('\t'.join(ll)+'\n')
        ll = ['epoch',' step',' f1',' precision',' recall',' r_f1',' r_precision',' r_recall',' e_f1',' e_precision',' e_recall']
        with open('./log/dev' + self.config.dataset_name + '_' + self.config.cell_name + '_' + decoder_type + str(
                id) + '.txt', 'w', encoding='utf-8') as f:
            f.write('\t'.join(ll) + '\n')
        for epoch in range(1, self.epoch_number + 1):
            for step in range(self.data.batch_number):
                batch = self.data.next_batch(is_random=True)
                loss, (f1, precision, recall),(r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall) = self.train_step(batch)

                train_loss += loss.item()
                train_f1 += f1
                train_precision += precision
                train_recall += recall
                train_rf1+=r_f1
                train_rprecision+=r_precision
                train_rrecall+=r_recall
                train_ef1 += e_f1
                train_eprecision += e_precision
                train_erecall += e_recall
                if (step % 10 == 0):#train log
                    if (step != 0):
                        train_loss /= 10.
                        train_rf1 /= 10.
                        train_rprecision /= 10.
                        train_rrecall/= 10.
                        train_ef1/= 10.
                        train_eprecision/= 10.
                        train_erecall /= 10.
                    print(
                        "train \t epoch %d\t step %d \t trainloss: %f \t f1: %f  \t precision: %f  \t recall: %f  \t r_f1: %f  \t r_precision: %f  \t r_recall: %f  \t e_f1: %f  \t e_precision: %f  \t e_recall: %f  \t" % (
                            epoch, step, train_loss, train_f1, train_precision, train_recall,train_rf1, train_rprecision, train_rrecall, train_ef1, train_eprecision, train_erecall))
                    ll = [epoch, step, train_loss,train_f1, train_precision, train_recall,train_rf1, train_rprecision, train_rrecall, train_ef1, train_eprecision, train_erecall]
                    ll = [str(x) for x in ll]
                    with open(
                            './log/train' + self.config.dataset_name + '_' + self.config.cell_name + '_' + decoder_type + str(
                                id) + '.txt', 'a', encoding='utf-8') as f:
                        f.write('\t'.join(ll) + '\n')
                    train_loss = 0.
                    train_f1, train_precision, train_recall = 0., 0., 0.
                    train_rf1, train_rprecision, train_rrecall = 0., 0., 0.
                    train_ef1, train_eprecision, train_erecall = 0., 0., 0.

                if (step % 200 == 0):#eval log

                    model_path = os.path.join('saved_model',
                                              self.config.dataset_name + '_' + self.config.cell_name + '_' + decoder_type + str(
                                                  id) + '.pkl')
                    torch.save(self.seq2seq.state_dict(), model_path)

                    if evaluator:
                        with torch.no_grad():

                            dev_loss,(f1, precision, recall), (r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall)=evaluator.rel_test(self.seq2seq)
                            print('_' * 60)
                            print("epoch %d\t step %d \t dev_loss: %f \t F1: %f \t P: %f \t R: %f \t" %(
                                epoch, step,dev_loss,f1, precision, recall))
                            print("event_mrc \t F1: %f \t P: %f \t R: %f \t" % (r_f1, r_precision, r_recall))
                            print("entity \t F1: %f \t P: %f \t R: %f \t" % (e_f1, e_precision, e_recall))
                            print('_' * 60)

                            ll = [epoch, step,dev_loss,f1, precision, recall, r_f1, r_precision, r_recall, e_f1, e_precision, e_recall]
                            ll = [str(x) for x in ll]
                            with open('./log/dev' + self.config.dataset_name + '_' + self.config.cell_name + '_' + decoder_type + str(
                                            id) + '.txt', 'a', encoding='utf-8') as f:
                                f.write('\t'.join(ll) + '\n')
                            train_loss = 0.0
def read_dic(path):
    with open(path,'r',encoding='utf-8') as f:
        dic=f.read()
        # print(dic)
        dic=eval(dic)
    return dic
########改成阅理解的train/dev####################################
def normal(s):
    s = s.lower()
    s = s.replace('(', ' ')
    s = s.replace('（', ' ')
    s = s.replace(')', ' ')
    s = s.replace('）', ' ')
    # s=s.replace(' ','')
    # s=s.replace(' ','')
    return s
def get_train_dev(path,max_seq_len=80):
    import pandas as pd
    ###############################################################################
    #           限制最长长度后，将label转为[event1,entity1_beg,entity2_beg]的三元组序列
    ###############################################################################
    c = pd.read_csv(path+'data.csv', header=None, sep='\t')
    event2id = read_dic(path+'event2id.txt')
    word2id=read_dic(path+'word2id.txt')
    spl = int(c.shape[0] * 0.8)
    train = c[:spl]
    dev = c[spl:]

    def get_data2id(c_data, path):
        sent_length = []
        entity_list = []
        # s = ['text_a\tlabel']
        count={}
        all_label=[]
        all_sent=[]
        for sent, dic in c_data[[1, 2]].values:
            # print(sent,dic)
            sent=sent[:max_seq_len]
            sentence = list(sent[:max_seq_len])
            # print(sentence)
            sentence=[word2id[s] if s in word2id.keys() else word2id['[UNK]'] for s in sentence]
            sent = normal(sent)
            sent_length.append(len(sentence))
            dic = eval(dic)
            label = []
            for k, v in dic.items():
                # print(k,v)
                # print(entity2id)
                for entity in v:
                    entity_list.append(entity)
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
                        label.extend([event2id[k], beg, end])
                        break
                        # entity_label[beg] = 'B-'+k
                        # for i in range(beg + 1, end):
                        #     entity_label[i] = 'I-'+k
                        # print(entity,sentence[i])
                        # print(entity_label)
            all_label.append(label)
            all_sent.append(sentence)
            if(len(label)//3 in count.keys()):
                count[len(label)//3]+=1
            else:
                count[len(label) // 3]=1

        with open(path, 'w', encoding='utf-8') as f:
            f.write(str([all_sent,all_label]))
        return list(set(entity_list)),count

    train1,train_count = get_data2id(train,path+'train.txt')
    dev1,dev_count = get_data2id(dev, path+'dev.txt')
    count = 0.
    for e in dev1:
        if e in train1:
            count += 1.
    print(count / len(train1), count / len(dev1), len(train1), len(dev1), count)
    print(train_count)
    print(dev_count)
if __name__ == '__main__':

    config_filename = './config.json'
    config = const.Config(config_filename=config_filename, cell_name=cell_name, decoder_type=decoder_type)
    config.name = mode
    get_train_dev(config.data_home,max_seq_len=config.max_sentence_length)
    assert cell_name in ['lstm', 'gru']
    assert decoder_type in ['one', 'multi','onecrf']
    # config.dataset_name = const.DataSet.WEBNLG
    if config.dataset_name == const.DataSet.NYT:
        prepare = data_prepare.NYTPrepare(config)
    elif config.dataset_name == const.DataSet.WEBNLG:
        prepare = data_prepare.WebNLGPrepare(config)
    elif config.dataset_name == const.DataSet.CCKS2020EVENT:
        prepare = data_prepare.CCKSPrepare(config)
    elif config.dataset_name == const.DataSet.CCKSMRC:
        prepare = data_prepare.CCKSMrcPrepare(config)
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

