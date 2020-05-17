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
parser.add_argument('--decoder_type', '-d', type=str, default='one', help='one/multi')

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
        self.mode=mode


    def load_model(self,seq2seq) -> None:

        # model_path = os.path.join(self.config.runner_path, 'model.pkl')
        model_path = os.path.join('saved_model',
                                  self.config.dataset_name + '_' + self.config.cell_name + '_' + decoder_type + '.pkl')
        seq2seq.load_state_dict(torch.load(model_path))
        return seq2seq

    def test_step(self, batch: data_prepare.InputData,seq2seq) -> Tuple[torch.Tensor, torch.Tensor]:

        sentence = batch.sentence_fw
        sentence_eos = batch.input_sentence_append_eos

        sentence = torch.from_numpy(sentence).to(self.device)
        sentence_eos = torch.from_numpy(sentence_eos).to(self.device)

        lengths = torch.Tensor(batch.input_sentence_length).int().tolist()


        pred_action_list, pred_logits_list = seq2seq(sentence, sentence_eos, lengths)
        pred_action_list = torch.cat(list(map(lambda x: x.unsqueeze(1), pred_action_list)), dim=1)

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

    def event_test(self,seq2seq) -> Tuple[Tuple[float, float, float]]:
        predicts = []
        gold = []
        data = prepare.load_data(self.mode)
        if mode == 'test':
            data = prepare.test_process(data)
        else:
            data = prepare.process(data)
        data = data_prepare.Data(data, config.batch_size, config)
        loss=0.0
        # Loss=nn.NLLLoss()
        for batch_i in tqdm(range(data.batch_number)):
            batch_data = data.next_batch(is_random=False)
            pred_action_list, pred_logits_list = self.test_step(batch_data,seq2seq)
            pred_action_list = pred_action_list.cpu().numpy()

            predicts.extend(pred_action_list)
            gold.extend(batch_data.standard_outputs)
            for t in range(seq2seq.decoder.decodelen):
                # print(pred_logits_list[t], batch_data.standard_outputs[:, t])
                loss += F.nll_loss(pred_logits_list[t],torch.from_numpy(batch_data.standard_outputs).to(self.device).to(torch.long)[:, t]).item()
            # print(loss)
        loss/=batch_i
        # for g in gold:
        #     for i in range(5):
        #         l=g[i*(config.max_sentence_length+1):(i+1)*(config.max_sentence_length+1)]
        #         if(l[0]>30):
        #             print(l)
        f1, precision, recall = evaluation.compare(predicts, gold, self.config, show_rate=None, simple=True)
        (r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall) = evaluation.rel_entity_compare(predicts, gold,
                                                                                                     self.config)
        (event_f1, event_precision, event_recall), (
        entity_f1, entity_precision, entity_recall) = evaluation.event_entity_compare(predicts, gold,
                                                                                      self.config)
        data.reset()
        return loss,(f1, precision, r_recall), (r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall), (
        event_f1, event_precision, event_recall), (entity_f1, entity_precision, entity_recall)

    def predict(self,seq2seq):
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
            pred_action_list, pred_logits_list = self.test_step(batch_data,seq2seq)
            pred_action_list = pred_action_list.cpu().numpy()
            sentences.extend(batch_data.sentence_fw)
            predicts.extend(pred_action_list)
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

        self.epoch_number = config.epoch_number + 1

    def train_step(self, batch: data_prepare.InputData) -> torch.Tensor:

        self.optimizer.zero_grad()

        sentence = batch.sentence_fw
        sentence_eos = batch.input_sentence_append_eos

        all_events = batch.standard_outputs

        all_events = torch.from_numpy(all_events).to(self.device).to(torch.long)
        sentence = torch.from_numpy(sentence).to(self.device)
        sentence_eos = torch.from_numpy(sentence_eos).to(self.device)

        lengths = torch.Tensor(batch.input_sentence_length).int().tolist()

        pred_action_list, pred_logits_list = self.seq2seq(sentence, sentence_eos, lengths)

        loss = 0
        for t in range(self.seq2seq.decoder.decodelen):

            loss = loss + self.loss(pred_logits_list[t], all_events[:, t])

        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, evaluator: Evaluator = None) -> None:
        train_loss=0.
        for epoch in range(1, self.epoch_number + 1):

            for step in range(self.data.batch_number):
                batch = self.data.next_batch(is_random=True)
                loss = self.train_step(batch)
                train_loss+=loss.item()
                # print(loss)
                # print('train:epoch: %d\t step: %d \t loss:%f' % (epoch, step, loss))
                if (step % 100 == 0):

                    model_path = os.path.join('saved_model',
                                              self.config.dataset_name + '_' + self.config.cell_name + '_' + decoder_type + '.pkl')
                    torch.save(self.seq2seq.state_dict(), model_path)
                    if(step!=0):
                        train_loss/=step
                    if evaluator:
                        with torch.no_grad():
                            # evaluator.load_model()
                            # f1, precision, recall = evaluator.test()
                            eval_loss,(f1, precision, recall), (r_f1, r_precision, r_recall), (e_f1, e_precision, e_recall), (
                            event_f1, event_precision, event_recall), (
                            entity_f1, entity_precision, entity_recall) = evaluator.event_test(self.seq2seq)

                            print('_' * 60)
                            print("epoch %d\t step %d \t trainloss: %f \t evalloss: %f  \t" % (
                            epoch, step,train_loss, eval_loss))
                            print("total \t F1: %f \t P: %f \t R: %f \t" % (f1, precision, recall))
                            print("event \t F1: %f \t P: %f \t R: %f \t" % (r_f1, r_precision, r_recall))
                            print("entity \t F1: %f \t P: %f \t R: %f \t" % (e_f1, e_precision, e_recall))
                            print("macro event \t F1: %f \t P: %f \t R: %f \t" % (event_f1, event_precision, event_recall))
                            print("macro entity \t F1: %f \t P: %f \t R: %f \t" % (
                            entity_f1, entity_precision, entity_recall))
                            print('_' * 60)
                            train_loss=0.0


if __name__ == '__main__':

    config_filename = './config.json'
    config = const.Config(config_filename=config_filename, cell_name=cell_name, decoder_type=decoder_type)
    config.name = mode
    assert cell_name in ['lstm', 'gru']
    assert decoder_type in ['one', 'multi']
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

    if train:
        trainer = SupervisedTrainer(config, device)
        evaluator = Evaluator(config, 'valid', device)
        trainer.train(evaluator)
    else:
        tester = Evaluator(config, mode, device)
        tester.load_model()

        f1, precision, recall = tester.predict()

        # rel_f1, rel_precision, rel_recall = tester.rel_test()
        # print('_' * 60)
        print("triplet \t F1: %f \t P: %f \t R: %f \t" % (f1, precision, recall))
        #
        # print('.' * 60)
        # print("relation \t F1: %f \t P: %f \t R: %f \t" % (rel_f1, rel_precision, rel_recall))
        #
