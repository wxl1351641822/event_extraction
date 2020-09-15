
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
