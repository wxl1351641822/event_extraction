import argparse
import ast
import os
def is_path_valid(path):
    if path == "":
        return False
    path = os.path.abspath(path)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        return False
    return True
predict_path='./data/data37221/test_unlabel.csv'
if(not is_path_valid(predict_path)):
    predict_path='./data/data34808/test_unlabel.csv'
predict_data_path='./work/predict.txt'
mcls_predict_path='work/test1_data/32.33.34.ucas_valid_result.csv'
mcls_predict_data_path='./work/predict_mcls.txt'
output_path='work/test1_data/'
output_predict_data_path=output_path+'test1.json'



# yapf: disable
id=6
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True,
                    help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--data_dir", type=str, default='work/', help="data save dir")
# parser.add_argument("--schema_path", type=str, default='work/event_schema/event_schema.json', help="schema path")
# parser.add_argument("--train_data", type=str, default='work/train_data/train.json', help="train data")
# parser.add_argument("--dev_data", type=str, default='work/dev_data/dev.json', help="dev data")
# parser.add_argument("--test_data", type=str, default='work/dev_data/dev.json', help="test data")
# parser.add_argument("--predict_data", type=str, default='work/test1_data/test1.json', help="predict data")
parser.add_argument("--do_train", type=ast.literal_eval, default=True, help="do train")
parser.add_argument("--do_predict", type=ast.literal_eval, default=True, help="do predict")
parser.add_argument("--do_model", type=str, default="mcls", choices=["mcls", "role"], help="trigger or role")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=256, help="Number of words of the longest seqence.")
parser.add_argument("--eval_step", type=int, default=200, help="eval step")
parser.add_argument("--model_save_step", type=int, default=10000, help="model save step")
parser.add_argument("--batch_size", type=int, default=16, help="Total examples' number in batch for training.")
parser.add_argument("--add_crf", type=ast.literal_eval, default=True, help="add crf")
parser.add_argument("--checkpoint_dir", type=str, default='models/trigger', help="Directory to model checkpoint")
parser.add_argument("--model_name", type=str, default='chinese-roberta-wwm-ext-large', help="Directory to model checkpoint")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=True, help="Whether use data parallel.")
parser.add_argument("--random_seed", type=int, default=1666, help="seed")
parser.add_argument("--dev_goal", type=str, default='f1',choices=['f1','loss'], help="choose model on the value of dev_goal")
parser.add_argument("--add_rule", type=ast.literal_eval, default=True, help="add rule---post process result")
parser.add_argument("--change_event",type=str,default='BIO_event',choices=['BIO_event','no','BIO'],help="BIO-event:B-event I-event O;BIO;no:event O")
