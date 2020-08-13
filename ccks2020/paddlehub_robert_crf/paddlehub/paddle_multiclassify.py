
import json
import random
import numpy as np

from paddlehub_dataprocess import write_by_lines,read_by_lines,regrex_data,read_label
from paddlehub_dataprocess import write_title
from paddlehub_dataprocess import write_log


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

    if args.do_model=='role' and args.do_predict:
        print("start predict process")
        ret = []
        id2label = {val: key for key, val in reader.label_map.items()}

        input_data = [[d] for d in predict_data]
        # print(input_data[:10])
        run_states = seq_label_task.predict(data=input_data)
        results = []
        for batch_states in run_states:
            batch_results = batch_states.run_results
            batch_infers = batch_results[0].reshape([-1]).astype(np.int32).tolist()
            # print(batch_results)
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
        get_submit_postprocess(args, id, check=True)

    if args.do_model == 'mcls' and args.do_predict:
        input_data = predict_data
        result=seq_label_task.predict(data=input_data, return_result=True)
        ret = []
        submit=[]
        for s,r in zip(predict_sents,result):
            s['labels'] = []
            # print(r)
            for r0 in r:
                print(r0)
                for k,v in r0.items():
                    print(k,v)
                    if(v==1):
                        s['labels'].append(k)
                        submit.append('\t'.join([str(s["id"]),k,s["entity"]]))
            ret.append(json.dumps(s,ensure_ascii=False))
        write_by_lines("{}.{}.{}.pred".format(output_predict_data_path, args.do_model, id), ret)
        write_by_lines("{}{}.{}.ucas_valid_result.csv".format(output_path, args.do_model, id), submit)
        # for r,data in zip(result,input_data):
        #     sent,entity=data
        #     for k,v in r.items():


def testone():##按默认执行
    # get_data()
    args = parser.parse_args()
    # get_submit_postprocess(args, id)
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
    args.do_model = 'mcls'
    schema_labels, predict_data, predict_sents = process_data(args)

    for lr in [1e-5,5e-5,3e-4,3e-6]:
        if(id<10):
            continue
        args.learning_rate=lr
        args.checkpoint_dir = 'models/' + args.do_model + str(id)
        one(args, schema_labels, predict_data, predict_sents, str(id))
        id+=1


if __name__ == "__main__":
    # findlr()
    id=1
    testone()
    # change_event_label()
    # args = parser.parse_args()
    # args.do_model='role'
    # args.change_event = 'BIO'
    # id=33
    # get_submit_postprocess(args, id,mcls=True)


