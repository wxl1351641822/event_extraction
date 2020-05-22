
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import six
from collections import namedtuple

import paddle.fluid as fluid

from paddlehub.reader import tokenization
from paddlehub.common.logger import logger
from paddlehub.common.utils import sys_stdout_encoding
from paddlehub.dataset.dataset import InputExample
from paddlehub.reader.batching import pad_batch_data

from paddlehub.reader.nlp_reader import BaseNLPReader
class SequenceLabelReader(BaseNLPReader):
    def __init__(self,
                 vocab_path,
                 dataset=None,
                 label_map_config=None,
                 max_seq_len=512,
                 max_events_len=5,
                 do_lower_case=True,
                 random_seed=None,
                 use_task_id=False,
                 sp_model_path=None,
                 word_dict_path=None,
                 in_tokens=False):
        self.max_events_len=max_events_len
        super(SequenceLabelReader, self).__init__(
            vocab_path=vocab_path,
            dataset=dataset,
            label_map_config=label_map_config,
            max_seq_len=max_seq_len,
            do_lower_case=do_lower_case,
            random_seed=random_seed,
            use_task_id=use_task_id,
            sp_model_path=sp_model_path,
            word_dict_path=word_dict_path,
            in_tokens=in_tokens)
        if sp_model_path and word_dict_path:
            self.tokenizer = tokenization.FullTokenizer(
                vocab_file=vocab_path,
                do_lower_case=do_lower_case,
                use_sentence_piece_vocab=True)

    def _pad_batch_records(self, batch_records, phase=None):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]

        # padding
        padded_token_ids, input_mask, batch_seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            max_seq_len=self.max_seq_len,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids,
            max_seq_len=self.max_seq_len,
            pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids,
            max_seq_len=self.max_seq_len,
            pad_idx=self.pad_id)

        if phase != "predict":
            batch_label_ids = [record.label_id for record in batch_records]
            padded_label_ids = pad_batch_data(
                batch_label_ids,
                max_seq_len=self.max_seq_len*self.max_events_len,
                pad_idx=len(self.label_map) - 1)

            return_list = [
                padded_token_ids, padded_position_ids, padded_text_type_ids,
                input_mask, padded_label_ids, batch_seq_lens
            ]

            if self.use_task_id:
                padded_task_ids = np.ones_like(
                    padded_token_ids, dtype="int64") * self.task_id
                return_list = [
                    padded_token_ids, padded_position_ids, padded_text_type_ids,
                    input_mask, padded_task_ids, padded_label_ids,
                    batch_seq_lens
                ]

        else:
            return_list = [
                padded_token_ids, padded_position_ids, padded_text_type_ids,
                input_mask, batch_seq_lens
            ]

            if self.use_task_id:
                padded_task_ids = np.ones_like(
                    padded_token_ids, dtype="int64") * self.task_id
                return_list = [
                    padded_token_ids, padded_position_ids, padded_text_type_ids,
                    input_mask, padded_task_ids, batch_seq_lens
                ]

        return return_list

    def _reseg_token_label(self, tokens, tokenizer, phase, labels=None):

        if phase != "predict":

            if  len(labels)!=len(tokens):
                raise ValueError(
                    "The length of tokens must be same with labels")
            ret_tokens = []
            ret_labels = []

            for token, label in zip(tokens, labels):
                for i in range(len(label)//len(token)):
                    one_label=label[i * len(token): (i + 1) * len(token)]
                    sub_token = tokenizer.tokenize(token)
                    if len(sub_token) == 0:
                        continue
                    ret_tokens.extend(sub_token)
                    ret_labels.append(one_label)
                    if len(sub_token) < 2:
                        continue
                    sub_label = one_label
                    if one_label.startswith("B-"):
                        sub_label = "I-" + one_label[2:]
                    ret_labels.extend([sub_label] * (len(sub_token) - 1))
            # print(labels,ret_labels)
            if len(labels)!=len(tokens):
                raise ValueError(
                    "The length of ret_tokens can't match with labels")
            return ret_tokens, ret_labels
        else:
            ret_tokens = []
            for token in tokens:
                sub_token = tokenizer.tokenize(token)
                if len(sub_token) == 0:
                    continue
                ret_tokens.extend(sub_token)
                if len(sub_token) < 2:
                    continue

            return ret_tokens

    def _convert_example_to_record(self,
                                   example,
                                   max_seq_length,
                                   tokenizer,
                                   phase=None):

        tokens = tokenization.convert_to_unicode(example.text_a).split(u"")

        if phase != "predict":
            labels = tokenization.convert_to_unicode(example.label).split(u"")
            tokens, labels = self._reseg_token_label(
                tokens=tokens, labels=labels, tokenizer=tokenizer, phase=phase)

            if len(tokens) > max_seq_length - 2:
                tokens = tokens[0:(max_seq_length - 2)]
                ll=[]
                for i in range((len(labels)//len(tokens))):
                    ll.extend(labels[i*(max_seq_length-2):(i+1)*(max_seq_length-2)])
                labels=ll
            no_entity_id = len(self.label_map) - 1
            # print(self.label_map)
            ll = []
            for i in range((len(labels) // len(tokens))):
                ll.extend([no_entity_id
                         ] +[self.label_map[label] for label in labels[i * (len(tokens)):(i + 1) * (len(tokens))]] +[no_entity_id
                         ] )
            label_ids = ll
            # label_ids = [no_entity_id
            #              ] + [self.label_map[label]
            #                   for label in labels] + [no_entity_id]

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            position_ids = list(range(len(token_ids)))
            text_type_ids = [0] * len(token_ids)

            record = self.Record_With_Label_Id(
                token_ids=token_ids,
                text_type_ids=text_type_ids,
                position_ids=position_ids,
                label_id=label_ids)
        else:
            tokens = self._reseg_token_label(
                tokens=tokens, tokenizer=tokenizer, phase=phase)

            if len(tokens) > max_seq_length - 2:
                tokens = tokens[0:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            position_ids = list(range(len(token_ids)))
            text_type_ids = [0] * len(token_ids)

            record = self.Record_Wo_Label_Id(
                token_ids=token_ids,
                text_type_ids=text_type_ids,
                position_ids=position_ids,
            )

        return record