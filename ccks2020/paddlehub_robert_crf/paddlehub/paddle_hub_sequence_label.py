
import json
import random
import numpy as np

from paddlehub_dataprocess import write_by_lines,read_by_lines,regrex_data,read_label
from paddlehub_dataprocess import write_title
from paddlehub_dataprocess import write_log,get_submit_cross_validation_vote


from config import *

from paddlehub_buildtask import *


def one(args, schema_labels, predict_data, predict_sents, id):
    seq_label_task, reader = get_task(args, schema_labels, id)
    # 加载PaddleHub 预训练模型ERNIE Tiny/RoBERTa large
    # 更多预训练模型 https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=SemanticModel
    # model_name = "ernie_tiny"

    # PaddleHub Finetune API
    # 将自动训练、评测并保存模型
    if args.do_train:
        print("start finetune and eval process")
        seq_label_task.finetune_and_eval()
        write_log('./work/log/' + args.do_model + '.txt', args, id + ',' + str(seq_label_task.best_score))

    if args.do_predict:
        print("start predict process")
        ret = []
        id2label = {val: key for key, val in reader.label_map.items()}
        input_data = [[d] for d in predict_data]
        # print(input_data[:10])
        run_states = seq_label_task.predict(data=input_data)
        results = []
        for batch_states in run_states:
            batch_results = batch_states.run_results
            # print('batch_infers',batch_results )
            batch_infers = batch_results[0].reshape([-1]).astype(np.int32).tolist()
            seq_lens = batch_results[1].reshape([-1]).astype(np.int32).tolist()
            current_id = 0
            for length in seq_lens:
                seq_infers = batch_infers[current_id:current_id + length]
                seq_result = list(map(id2label.get, seq_infers[1: -1]))
                current_id += length if args.add_crf else args.max_seq_len
                results.append(seq_result)

        ret = []
        for sent,input, r_label in zip(predict_sents,input_data,results):
            sent["input"]=input
            sent["labels"] = r_label
            ret.append(json.dumps(sent, ensure_ascii=False))
        write_by_lines("{}.{}.{}.pred".format(output_predict_data_path, args.do_model, id), ret)
        get_submit_postprocess(args, id)
        get_submit_postprocess(args, id,check=True)

def cross_validation(args,id):
    begid=id
    for i in range(args.cross_validation_num):
        if (id < 40):
            id += 1
            continue
        schema_labels, predict_data, predict_sents = process_data(args, i)
        args.checkpoint_dir = 'models/' + args.do_model + str(id)
        one(args, schema_labels, predict_data, predict_sents, str(id))
        get_submit_postprocess(args, id, check=True)
        get_submit_postprocess(args, id, check=False)

        id += 1
    get_submit_cross_validation_vote(cross_validation_num=args.cross_validation_num, begid=begid,
                                     type=args.change_event)
    return id

def testone():##按默认执行
    # get_data()
    args = parser.parse_args()
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    args.do_model = 'mcls'
    schema_labels, predict_data, predict_sents = process_data(args)
    shiyan = """
    mcls
        """
    write_title('./work/log/' + args.do_model + '.txt', args, shiyan)
    args.checkpoint_dir = 'models/' + args.do_model + str(id)
    one(args, schema_labels, predict_data, predict_sents, str(id))

def findlr():
    id=9
    args = parser.parse_args()
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    args.do_model = 'role'
    schema_labels, predict_data, predict_sents = process_data(args)

    for lr in [1e-5,5e-5,3e-4,3e-6]:
        if(id<10):
            continue
        args.learning_rate=lr
        args.checkpoint_dir = 'models/' + args.do_model + str(id)
        one(args, schema_labels, predict_data, predict_sents, str(id))
        id+=1

def change_event_label():
    id=40
    args = parser.parse_args()
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    args.do_model = 'role'
    # shiyan = """
    # 增加一列：change_event
    #     """
    # write_title('./work/log/' + args.do_model + '.txt', args, shiyan)
    for event_label in ['BIO_event','BIO','no']:
        if(event_label=='BIO_event'):
            args.add_rule=True
        else:
            args.add_rule=False
        args.change_event=event_label
        if(args.use_cross_validation):
            id=cross_validation(args,id)
        else:
            schema_labels, predict_data, predict_sents = process_data(args, i)
            args.checkpoint_dir = 'models/' + args.do_model + str(id)
            one(args, schema_labels, predict_data, predict_sents, str(id))
            get_submit_postprocess(args, id, check=True)
            get_submit_postprocess(args, id, check=False)
            id += 1
if __name__ == "__main__":
    # findlr()
    # id=17
    # testone()
    change_event_label()


