import numpy as np
from typing import List, Tuple
import os
import const
import json

import torch.nn as nn
import torch
import torch.nn.functional as F

# torch_bool 
try:
    torch_bool = torch.bool
except:
    torch_bool = torch.uint8


class Encoder(nn.Module):
    def __init__(self, config: const.Config, embedding: nn.modules.sparse.Embedding) -> None:
        super(Encoder, self).__init__()
        self.config = config

        self.hidden_size = config.encoder_num_units
        self.emb_size = config.embedding_dim
        self.words_number = config.words_number
        self.maxlen = config.max_sentence_length

        self.dropout = nn.Dropout(0.1)

        self.embedding = embedding
        self.cell_name = config.cell_name
        if config.cell_name == 'gru':
            self.rnn = nn.GRU(self.emb_size, self.hidden_size, bidirectional=True, batch_first=True)
        elif config.cell_name == 'lstm':
            self.rnn = nn.LSTM(self.emb_size, self.hidden_size, bidirectional=True, batch_first=True)
        else:
            raise ValueError('cell name should be gru/lstm!')

    def forward(self, sentence: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        # sentence.to(torch.long)
        embedded = self.embedding(sentence.to(torch.long))

        # embedded = self.dropout(embedded)

        if lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths=lengths, batch_first=True)

        output, hidden = self.rnn(embedded)

        if lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, total_length=self.maxlen, batch_first=True)

        output = (lambda a: sum(a) / 2)(torch.split(output, self.hidden_size, dim=2))
        if self.cell_name == 'gru':
            hidden = (lambda a: sum(a) / 2)(torch.split(hidden, 1, dim=0))
        elif self.cell_name == 'lstm':
            hidden = tuple(map(lambda state: sum(torch.split(state, 1, dim=0)) / 2, hidden))
        # hidden = (lambda a: sum(a)/2)(torch.split(hidden, 1, dim=0))

        return output, hidden




class CrfCCKSDecoder(nn.Module):
    def __init__(self, config: const.Config, embedding: nn.modules.sparse.Embedding, device) -> None:
        super(CrfCCKSDecoder, self).__init__()

        self.device = device

        self.cell_name = config.cell_name
        self.decoder_type = config.decoder_type
        self.batch_size=config.batch_size
        self.hidden_size = config.decoder_num_units
        self.emb_size = config.embedding_dim
        self.words_number = config.words_number
        self.maxlen = config.max_sentence_length
        self.decodelen = config.decoder_output_max_length
        self.eventlen = config.event_length  ##实体数量+
        self.word_embedding = embedding
        # if config.dataset_name==const.DataSet.CCKS2020EVENT:
        self.event_eos = config.event_number
        self.event_number = config.event_number
        self.event_embedding = nn.Embedding(config.event_number + 1, config.embedding_dim)
        self.entity_embedding=nn.Embedding(config.entity_size,config.embedding_dim)
        self.do_predict = nn.Linear(self.hidden_size, self.event_number)
        # else:
        # self.relation_eos = config.relation_number
        # self.relation_number = config.relation_number
        # self.relation_embedding = nn.Embedding(config.relation_number + 1, config.embedding_dim)
        # self.do_predict = nn.Linear(self.hidden_size, self.relation_number)

        self.sos_embedding = nn.Embedding(1, config.embedding_dim)

        self.combine_inputs = nn.Linear(self.hidden_size + self.emb_size, self.emb_size)
        self.attn = nn.Linear(self.hidden_size * 2, 1)

        if self.cell_name == 'gru':
            self.rnn = nn.GRU(self.emb_size, self.hidden_size, batch_first=True)
        elif self.cell_name == 'lstm':
            self.rnn = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)

        self.do_eos = nn.Linear(self.hidden_size, 1)
        self.entity_size = config.entity_size
        self.do_entity_tag = nn.Linear(self.hidden_size, self.entity_size)

        self.fuse = nn.Linear(self.hidden_size * 2, 100)
        self.do_copy_linear = nn.Linear(100,1 )#预测其位置
        ###crf##############################################################
        self.transitions = nn.Parameter(
            torch.randn(self.entity_size, self.entity_size)).to(device)

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag

        self.transitions.data[0, :] = -10000
        self.transitions.data[:, -1] = -10000


    def calc_context(self, decoder_state: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:

        # decoder_state.size() == torch.Size([1, 100, 1000])
        # -> torch.Size([100, 1, 1000]) -> torch.Size([100, 80, 1000]) -cat-> torch.Size([100, 80, 2000])
        attn_weight = torch.cat((decoder_state.permute(1, 0, 2).expand_as(encoder_outputs), encoder_outputs), dim=2)
        attn_weight = F.softmax((self.attn(attn_weight)), dim=1)
        attn_applied = torch.bmm(attn_weight.permute(0, 2, 1), encoder_outputs).squeeze(1)

        return attn_applied

    def do_copy(self, output: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:

        out = torch.cat((output.unsqueeze(1).expand_as(encoder_outputs), encoder_outputs),
                        dim=2)  # [batch_size,maxlen,hidden*2]

        out = F.selu(self.fuse(F.selu(out)))  # [batch_size,maxlen,100]

        out = self.do_copy_linear(out).squeeze(2)#[batch_size,maxlen,entity_size]
        # print(out.shape)
        # out = (self.do_copy_linear(out).squeeze(2))
        return out

    def _decode_step(self, rnn_cell: nn.modules,
                     emb: torch.Tensor,
                     decoder_state: torch.Tensor,
                     encoder_outputs: torch.Tensor) \
            -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # 这里输入了一个时间步
        if self.cell_name == 'gru':
            decoder_state_h = decoder_state
        elif self.cell_name == 'lstm':
            decoder_state_h = decoder_state[0]
        else:
            raise ValueError('cell name should be lstm or gru')

        context = self.calc_context(decoder_state_h, encoder_outputs)

        output = self.combine_inputs(torch.cat((emb, context), dim=1))

        output, decoder_state = rnn_cell(output.unsqueeze(1), decoder_state)  ##[16,1，1000][batch_size,1，hidden_size]

        output = output.squeeze()  # [16,1000][batch_size,hidden_size]

        # eos_logits = F.selu(self.do_eos(output))
        # predict_logits = F.selu(self.do_predict(output))
        eos_logits = (self.do_eos(output))  # [16,1]
        predict_logits = (self.do_predict(output))  # [16,event_number]

        predict_logits = F.log_softmax(torch.cat((predict_logits, eos_logits), dim=1), dim=1) # 预测其是哪一个事件或者是eos
        # print(eos_logits.shape,predict_logits.shape)
        # print(output.shape[16,1000],encoder_outputs.shape)[16,80,1000]

        copy_logits = self.do_copy(output, encoder_outputs)

        # assert copy_logits.size() == first_entity_mask.size()
        # original
        # copy_logits = copy_logits * first_entity_mask
        # copy_logits = copy_logits

        copy_logits = torch.cat((copy_logits, eos_logits), dim=1)#[batch_size,maxlen+1]每个位置被拷贝的概率
        copy_logits = F.log_softmax(copy_logits, dim=1)#[batch_size,maxlen+1]每个位置被拷贝的概率

        # # bug fix
        # copy_logits = torch.cat((copy_logits, eos_logits), dim=1)
        # first_entity_mask = torch.cat((first_entity_mask, torch.ones_like(eos_logits)), dim=1)
        #
        # copy_logits = F.softmax(copy_logits, dim=1)
        # copy_logits = copy_logits * first_entity_mask
        # copy_logits = torch.clamp(copy_logits, 1e-10, 1.)
        # copy_logits = torch.log(copy_logits)
        predict_entity_tag_logits=self.do_entity_tag(output) #[batch_size,1]
        # print(predict_logits.shape,predict_entity_tag_logits.shape)
        # print(predict_logits.shape,predict_entity_tag_logits.shape)
        return (predict_logits, copy_logits,predict_entity_tag_logits), decoder_state

    def forward(self, *input):
        raise NotImplementedError('abstract method!')

class OneCrfCCKSDecoder(CrfCCKSDecoder):

    def __init__(self, config: const.Config, embedding: nn.modules.sparse.Embedding, device) \
            -> None:
        super(OneCrfCCKSDecoder, self).__init__(config=config, embedding=embedding, device=device)
        ##crf#######################################################################################
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.


    # def _viterbi_decode(self, feats):
    #     backpointers = []
    #     # Initialize the viterbi variables in log space
    #     init_vvars = torch.full((1, self.tagset_size), -10000.)
    #     init_vvars[0][self.tag_to_ix[START_TAG]] = 0
    #
    #     # forward_var at step i holds the viterbi variables for step i-1
    #     forward_var_list = []
    #     forward_var_list.append(init_vvars)
    #
    #     for feat_index in range(feats.shape[0]):
    #         gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
    #         gamar_r_l = torch.squeeze(gamar_r_l)
    #         next_tag_var = gamar_r_l + self.transitions
    #         viterbivars_t,bptrs_t = torch.max(next_tag_var,dim=1)
    #
    #         t_r1_k = torch.unsqueeze(feats[feat_index], 0)
    #         forward_var_new = torch.unsqueeze(viterbivars_t,0) + t_r1_k
    #
    #         forward_var_list.append(forward_var_new)
    #         backpointers.append(bptrs_t.tolist())
    #
    #     # Transition to STOP_TAG
    #     terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
    #     best_tag_id = torch.argmax(terminal_var).tolist()
    #     path_score = terminal_var[0][best_tag_id]
    #
    #
    #     assert start == self.tag_to_ix[START_TAG]  # Sanity check
    #     best_path.reverse()
    #     return path_score, best_path
    def _forward_alg(self, feats):
        init_alphas = torch.full([self.batch_size, self.entity_size], -10000.).to(self.device)
        # START_TAG has all of the score.
        init_alphas[:][0] = 0.
        a_forward_var_list = init_alphas  ##计算loss的初始alpha
        alpha=torch.zeros(1).to(self.device)
        for feat_index in range(feats.shape[0]):
            if(feat_index%self.eventlen==0):
                if(feat_index!=0):
                    terminal_var = a_forward_var_list + self.transitions[-1]#[2,5]
                    alpha+=torch.mean(torch.logsumexp(terminal_var, dim=1))

            else:
                ##前向后向算法--学习--loss计算##################################################################################
                a_gammar_r_l = torch.stack([a_forward_var_list] * feats.shape[-1])  # [5,2,5]
                # print(a_gammar_r_l.shape)
                t_r1_k = torch.unsqueeze(feats[feat_index], 0)  # [1,2,5]
                aa = a_gammar_r_l + t_r1_k + self.transitions.unsqueeze(1)  # [5,2,5]
                a_forward_var_list = torch.logsumexp(aa, dim=-1).transpose(0, 1)
        terminal_var = a_forward_var_list + self.transitions[-1]  # [2,5]
        alpha += torch.mean(torch.logsumexp(terminal_var, dim=1))
        return alpha/(self.decodelen//self.eventlen)

    def _score_sentence(self, feats,pred_event_logits_list, labels):
        # feats#decodelen,batch,5
        # Gives the score of a provided tag sequence
        # print(tags)
        score = torch.zeros(1).to(self.device)
        #
        start = torch.tensor([0], dtype=torch.long).to(self.device)
        for j in range(self.batch_size):
            tags=labels[j]
            # tags = torch.cat([torch.tensor([0], dtype=torch.long).to(self.device), tags])
            # print(tags)
            for i, feat in enumerate(feats[:,j]):
                if(i%self.eventlen==0):
                    score=score+pred_event_logits_list[i//self.decodelen,j,tags[i+1]]
                    score=score+self.transitions[tags[i+1], start] + feat[tags[i+1]]
                elif(i%self.eventlen==self.eventlen-1):
                    score=score+self.transitions[-1,tags[i]]
                else:
                    score = score + \
                            self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]

            # score = score + self.transitions[-1, tags[-1]]
        return score/self.batch_size/(self.decodelen//self.eventlen)

    def get_loss(self,sentence: torch.Tensor, decoder_state: torch.Tensor, encoder_outputs: torch.Tensor,labels):
        pred_action_list,pred_logits_list,pred_event_logits_list=self.forward(sentence, decoder_state, encoder_outputs)
        alpha=self._forward_alg(pred_logits_list)
        sentence_score=self._score_sentence(pred_logits_list,pred_event_logits_list, labels)
        return pred_action_list,alpha-sentence_score



    def forward(self, sentence: torch.Tensor, decoder_state: torch.Tensor, encoder_outputs: torch.Tensor) \
            -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        # sos = go = 0
        ##1.decode init#########################################################
        pred_action_list = torch.zeros(self.decodelen, self.batch_size).to(self.device)
        pred_logits_list = torch.zeros(self.decodelen, self.batch_size, self.entity_size).to(self.device)
        pred_event_logits_list=torch.zeros(self.decodelen//self.eventlen,self.batch_size,self.event_number+1).to(self.device)
        # pred_logits_list_event=torch.zeros(self.decodelen//(self.maxlen+1),self.batch_size,self.event_number+1).to(self.device)
        go = torch.zeros(sentence.size()[0], dtype=torch.int64).to(self.device)
        output = self.sos_embedding(go)

        # first_entity_mask = torch.ones(go.size()[0], self.maxlen).to(self.device)


        ##2.crf init####################################################
        backpointers = []
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1,self.batch_size, self.entity_size), -10000.).to(self.device)
        init_vvars[0][:][0] = 0.

        # forward_var at step i holds the viterbi variables for step i-1
        # forward_var_list
        forward_var_list=init_vvars##解码的初始状态

        path_score_list=torch.zeros(self.batch_size)
        # path_list=torch.zeros(())#[[]*self.batch_size]
        # print(path_list)
        for t in range(self.decodelen):

            bag, decoder_state = self._decode_step(self.rnn, output, decoder_state, encoder_outputs)
            predict_logits, copy_logits,predict_entity_tag_logits = bag

            if t % self.eventlen == 0:
                ##1.logits#############################################
                action_logits = predict_logits
                # pred_logits_list[t]=action_logits
                pred_event_logits_list[t//self.eventlen]=action_logits
                max_action = torch.argmax(action_logits, dim=1).detach()

                pred_action_list[t] = max_action
                output = max_action
                output = self.event_embedding(output)
                if(t!=0):
                    ##2.crf########################################################
                    # Transition to STOP_TAG
                    # print(forward_var_list[-1].shape,self.transitions[-1].shape)
                    terminal_var = (forward_var_list + self.transitions[-1]).squeeze()
                    # print(terminal_var.shape)
                    best_tag_id = torch.argmax(terminal_var,axis=-1).tolist()

                    # a_terminal_var = a_forward_var_list[-1] + self.transitions[-1]
                    # terminal_var = torch.unsqueeze(terminal_var, 0)
                    # alpha = torch.logsumexp(terminal_var, dim=1)[0]
                    for i,best_id in enumerate(best_tag_id):#batch_size
                        # print(best_id)
                        path_score = terminal_var[0][best_id]
                        ###路径################################################
                        # Follow the back pointers to decode the best path.
                        best_path = [best_id]
                        # print(bptrs_t[:,i])
                        k = 1
                        pred_action_list[t-k,i]=best_id

                        for bptrs_t in reversed(backpointers):
                            # print(bptrs_t)
                            best_id= bptrs_t[best_id][i]
                            # best_path.append(best_id)
                            if(best_id==0):
                                break
                            k+=1
                            pred_action_list[t - k, i] = best_id
                            # print(t-k)

                        # best_path.reverse()
                        path_score_list[i]+=path_score

                    ##3.crf init####################################################
                    forward_var_list=init_vvars
                    backpointers = []
            else:
                action_logits = predict_entity_tag_logits#copy_logits

                pred_logits_list[t]=action_logits

                #crf viterbi#####################################################
                feat_index=t % self.eventlen - 1
                gamar_r_l = torch.stack([forward_var_list] * action_logits.shape[1])#[5,1,2,5]
                gamar_r_l = torch.squeeze(gamar_r_l)#[5,2,5]
                # print(gamar_r_l.shape)
                next_tag_var = gamar_r_l+ self.transitions.unsqueeze(1)#[5,2,5]
                viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=-1)#[5,2]
                t_r1_k = torch.unsqueeze(action_logits, 0)#[1,2,5]
                # print(t_r1_k.shape,next_tag_var.shape)
                # t_r1_k=t_r1_k.transpose(1,2)
                # print(viterbivars_t.shape, bptrs_t.shape, torch.unsqueeze(viterbivars_t, 0).shape,t_r1_k.shape)
                forward_var_new = torch.unsqueeze(viterbivars_t, 0).transpose(1, 2) + t_r1_k  # [1,2,5]
                # print(forward_var_new.shape)
                forward_var_list = forward_var_new
                backpointers.append(bptrs_t.tolist())


        return pred_action_list,pred_logits_list,pred_event_logits_list


class CrfSeq2seq(nn.Module):
    def __init__(self, config: const.Config, device, load_emb=False, update_emb=True):
        super(CrfSeq2seq, self).__init__()

        self.device = device
        self.config = config

        self.emb_size = config.embedding_dim
        self.words_number = config.words_number
        self.maxlen = config.max_sentence_length

        self.word_embedding = nn.Embedding(self.words_number + 1, self.emb_size)
        # if load_emb:
        #     self.load_pretrain_emb(config)
        # self.word_embedding.weight.requires_grad = update_emb

        self.encoder = Encoder(config, embedding=self.word_embedding)
        if config.dataset_name == 'event':
            # if config.decoder_type == 'one':
            #     self.decoder = OneCCKSDecoder(config, embedding=self.word_embedding, device=device)
            # elif config.decoder_type == 'multi':
            #     self.decoder = MultiCCKSDecoder(config, embedding=self.word_embedding, device=device)
            if config.decoder_type == 'onecrf':
                self.decoder = OneCrfCCKSDecoder(config, embedding=self.word_embedding, device=device)
            else:
                raise ValueError('decoder type one/multi!!')

        self.to(self.device)

    def load_pretrain_emb(self, config: const.Config) -> None:
        if os.path.isfile(config.words_id2vector_filename):
            # logger.info('Word Embedding init from %s' % config.words_id2vector_filename)
            print('load_embedding!')
            words_id2vec = json.load(open(config.words_id2vector_filename, 'r'))
            words_vectors = [0] * (config.words_number + 1)

            for i, key in enumerate(words_id2vec):
                if (i >= config.words_number):
                    break
                words_vectors[int(key)] = words_id2vec[key]

            # words_vectors[len(words_id2vec) + 1] = [0] * len(words_id2vec[key])
            words_vectors[config.words_number] = [0] * len(words_id2vec[key])

            print(len(words_vectors))
            self.word_embedding.weight.data.copy_(torch.from_numpy(np.array(words_vectors)))

    def forward(self, sentence: torch.Tensor, sentence_eos: torch.Tensor, lengths: List[int],labels=None) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        o, h = self.encoder(sentence, lengths)
        if labels==None:
            pred_action_list,_,_ = self.decoder(sentence=sentence_eos.to(torch.long), decoder_state=h,
                                                              encoder_outputs=o)
            return pred_action_list,_
        pred_action_list, loss=self.decoder.get_loss(sentence=sentence_eos.to(torch.long), decoder_state=h,
                                                              encoder_outputs=o,labels=labels)



        return pred_action_list, loss




