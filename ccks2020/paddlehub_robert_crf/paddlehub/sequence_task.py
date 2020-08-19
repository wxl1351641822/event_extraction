from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from collections import OrderedDict

import numpy as np
import paddle
import paddle.fluid as fluid
from paddlehub.finetune.evaluate import chunk_eval, calculate_f1
from paddlehub.common.utils import version_compare
from paddlehub.finetune.task.base_task import BaseTask
from paddlehub import SequenceLabelTask

class MySequenceLabelTask(SequenceLabelTask):
    def __init__(self,
                 feature,
                 max_seq_len,
                 num_classes,
                 dataset=None,
                 feed_list=None,
                 data_reader=None,
                 startup_program=None,
                 config=None,
                 metrics_choices="default",
                 add_crf=False,
                 return_logits=False):
        self.return_logits=return_logits
        super(MySequenceLabelTask, self).__init__(
            feature=feature,
            max_seq_len=max_seq_len,
            dataset=dataset,
            num_classes=num_classes,
            feed_list=feed_list,
            data_reader=data_reader,
            startup_program=startup_program,
            config=config,
            metrics_choices=metrics_choices,
            add_crf=add_crf)
    @property
    def fetch_list(self):
        if self.is_train_phase or self.is_test_phase:
            return [metric.name for metric in self.metrics] + [self.loss.name]
        elif self.is_predict_phase:
            if(self.return_logits):
                return [self.ret_infers.name] + [self.seq_len_used.name]+[output.name for output in self.outputs]
            else:
                return [self.ret_infers.name] + [self.seq_len_used.name]
        return [output.name for output in self.outputs]

