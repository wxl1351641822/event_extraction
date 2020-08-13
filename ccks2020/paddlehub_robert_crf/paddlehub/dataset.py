
import csv
import io
from tqdm import tqdm
import pandas as pd
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset
from paddlehub.common.logger import logger
from paddlehub.dataset import InputExample


class CCksDataset(BaseNLPDataset):
    """EEDataset"""

    def __init__(self, data_dir, labels, model="trigger",tokenizer = None,max_seq_len=128):
        # 数据集存放位置
        # if tokenizer is None:
        #     tokenizer=
        super(CCksDataset, self).__init__(
            base_path=data_dir,
            train_file="train_cls.csv",
            dev_file="dev_cls.csv",
            test_file="dev_cls.csv",
            # 如果还有预测数据（不需要文本类别label），可以放在predict.tsv
            predict_file="dev_cls.csv",
            train_file_with_header=False,
            dev_file_with_header=False,
            test_file_with_header=False,
            predict_file_with_header=False,
            # 数据集类别集合
            label_list=labels,
            tokenizer=tokenizer,
            max_seq_len=128
            )
        # print(labels)

    def _convert_examples_to_records(self, examples, phase):
        """
        Returns a list[dict] including all the input information what the model need.
        Args:
            examples (list): the data examples, returned by _read_file.
            phase (str): the processing phase, can be "train" "dev" "test" or "predict".
        Returns:
            a list with all the examples record.
        """
        records = []
        with tqdm(total=len(examples)) as process_bar:
            for example in examples:
                record = self.tokenizer.encode(
                    text=example.text_a,
                    text_pair=example.text_b,
                    max_seq_len=self.max_seq_len)

                # CustomTokenizer will tokenize the text firstly and then lookup words in the vocab
                # When all words are not found in the vocab, the text will be dropped.
                if not record:
                    logger.info(
                        "The text %s has been dropped as it has no words in the vocab after tokenization."
                        % example.text_a)
                    continue

                if example.label:
                    record["label"] = [int(label) for label in example.label]
                records.append(record)
                process_bar.update(1)
        return records
    def _read_file(self, input_file, phase=None):
        """Reads a tab separated value file."""
        data=pd.read_csv(input_file,sep='\t',header=None)
        examples=[]
        i=0
        if(phase!='predict'):
            for sent,entity,label in data.values:
                # print(type(label))
                examples.append(InputExample(guid=i,text_a=sent,text_b=str(entity),label=eval(label)))
                i += 1
        else:
            for sent, entity, label in data.values:
                examples.append(InputExample(guid=i,text_a=sent, text_b=str(entity), label=None))
                i+=1
        return examples


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
        # print(labels)

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
                if (len(line) != ncol):
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
                # print(example)
            return examples
