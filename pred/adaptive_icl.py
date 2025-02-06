import json
import logging
import math
import os.path
import sys
from tqdm import tqdm
import torch

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_path not in sys.path:
    sys.path.append(project_path)
import argparse
from common.config import read_config
from zpd.dataset import read_ice_mapping, GSM8KProcessor, EZStanceProcessor
# from zpd.zones import read_zones
from transformers import AutoTokenizer
from zpd.eval import MathEvaluator, EZStanceEvaluator
from common.utils import to_md5, set_logging
from irt_model import MultiIRT

model2_id = {'meta-llama/Llama-2-7b-hf': 0, 'meta-llama/Llama-2-7b-chat-hf': 1, 'meta-llama/Llama-2-13b-hf': 2, 'meta-llama/Llama-2-13b-chat-hf': 3, 'meta-llama/Meta-Llama-3-8B-Instruct': 4, 'meta-llama/Meta-Llama-3-8B': 5, 'meta-llama/Meta-Llama-3-70B-Instruct': 6, 'meta-llama/Meta-Llama-3-70B': 7}


def get_adaptive_icl(model_base_output, model_icl_output, ice_mapping, model_name, tokenizer,  evaluator, processor, train_data, test_data, separator, ice_template, query_template, ckpt_path, embeddings, use_answer, filter_id=None):
    examples_info = {}
    logging.info('Loading IRT model from {}'.format(ckpt_path))
    irt_model = MultiIRT(num_testers=len(model2_id), item_ftr_dim=768, enable_gamma=True, trait_dim=32)
    irt_model.load_state_dict(torch.load(ckpt_path))

    logging.info('Reading base output {}'.format(model_base_output))
    with open(model_base_output, 'r') as fp:
        for line in fp.readlines():
            data = json.loads(line.strip())
            if 'id' not in data:
                data_id = 'gsm8k#test#{}'.format(to_md5(data['question']))
            else:
                data_id = data['id']
            if filter_id and data_id not in filter_id:
                continue
            if data_id not in examples_info:
                examples_info[data_id] = {}
            examples_info[data_id]['base_performance'] = int(evaluator.eval_single(data, extract_n=False))

    logging.info('Reading icl output {}'.format(model_icl_output))
    with open(model_icl_output, 'r') as fp:
        for line in fp.readlines():
            data = json.loads(line.strip())
            if 'id' not in data:
                data_id = 'gsm8k#test#{}'.format(to_md5(data['question']))
            else:
                data_id = data['id']
            if filter_id and data_id not in filter_id:
                continue
            examples_info[data_id]['icl_performance'] = int(evaluator.eval_single(data, extract_n=True))

    logging.info('Calculating # of tokens')
    for test_id in ice_mapping:
        if filter_id and test_id not in filter_id:
            continue
        ice_text = separator.join([processor.build_ice(ice_template, train_data[ice_id]) for ice_id in ice_mapping[test_id]])
        query_text = processor.build_query(query_template, test_data[test_id])
        examples_info[test_id]['ice_tokens'] = len(tokenizer(ice_text+separator+query_text)['input_ids']) + len(separator)
        examples_info[test_id]['query_tokens'] = len(tokenizer(query_text)['input_ids'])

    logging.info('Calculating Correct Prob')
    for example_id in tqdm(examples_info):
        emb = embeddings[example_id]['query_answer_embedding'] if use_answer else embeddings[example_id]['query_embedding']

        base_correct_prob = irt_model(
            tester_id=torch.tensor(model2_id[model_name], dtype=torch.int64).unsqueeze(0),
            item_repr=torch.tensor(emb, dtype=torch.float32).unsqueeze(0),
            icl_flag=torch.tensor([[0, ]], dtype=torch.float64)
        )['logits'].detach().numpy().tolist()[0][0]
        icl_correct_prob = irt_model(
            tester_id=torch.tensor(model2_id[model_name], dtype=torch.int64).unsqueeze(0),
            item_repr=torch.tensor(emb, dtype=torch.float32).unsqueeze(0),
            icl_flag=torch.tensor([[1, ]], dtype=torch.float64)
        )['logits'].detach().numpy().tolist()[0][0]
        # print(icl_correct_prob)
        # print(base_correct_prob)
        examples_info[example_id]['icl_correct_prob'] = 1 / (1 + math.exp(-icl_correct_prob))
        examples_info[example_id]['base_correct_prob'] = 1 / (1 + math.exp(-base_correct_prob))

    base_performance = 0
    base_tokens = 0
    icl_performance = 0
    icl_tokens = 0

    logging.info('Calculating base and full icl performance')
    for example_id in examples_info:
        base_performance += examples_info[example_id]['base_performance']
        icl_performance += examples_info[example_id]['icl_performance']
        base_tokens += examples_info[example_id]['query_tokens']
        icl_tokens += examples_info[example_id]['ice_tokens']

    # print(x, len(examples_info))
    # exit(1)
    base_performance /= len(examples_info)
    icl_performance /= len(examples_info)

    # exit(1)
    result = {
        'base': {'tokens': base_tokens, 'performance': base_performance},
        'full_icl': {'tokens': icl_tokens, 'performance': icl_performance},
        'adaptive_performance_all': [],
        'adaptive_performance_default': []
    }

    def get_performance(base_threshold, icl_threshold):
        adaptive_performance = 0
        adaptive_tokens = 0
        for example_id in examples_info:
            # print('his', examples_info[example_id])
            if examples_info[example_id]['base_correct_prob'] < base_threshold and examples_info[example_id]['icl_correct_prob'] > icl_threshold:
                adaptive_performance += examples_info[example_id]['icl_performance']
                adaptive_tokens += examples_info[example_id]['ice_tokens']
            else:
                adaptive_performance += examples_info[example_id]['base_performance']
                adaptive_tokens += examples_info[example_id]['query_tokens']
        return {'base_threshold': base_threshold, 'icl_threshold': icl_threshold, 'performance': adaptive_performance/len(examples_info), 'tokens': adaptive_tokens}

    result['adaptive_performance_default'].append(get_performance(0.5, 0.5))  # default_threshold

    logging.info('Grid searching thresholds')
    all_base_correct = [value['base_correct_prob'] for value in examples_info.values()]
    all_icl_correct = [value['icl_correct_prob'] for value in examples_info.values()]
    for i in range(0, 100):
        for j in range(0, 100):
            bt = i * (max(all_base_correct) - min(all_base_correct)) / 50 + min(all_base_correct)
            it = j * (max(all_icl_correct) - min(all_icl_correct)) / 50 + min(all_icl_correct)
            res = get_performance(bt, it)
            result['adaptive_performance_all'].append(res)

    return result


def analyze_model_behavior(ckpt_path, all_data, model_name, use_answer=True):
    irt_model = MultiIRT(num_testers=len(model2_id), item_ftr_dim=768, enable_gamma=True, trait_dim=32)
    irt_model.load_state_dict(torch.load(ckpt_path))
    # fp = open('{}_{}_IRT_behavior.jsonl'.format(args.dataset, model_name.replace('/', '')), 'w')
    ana_result = []
    for example_id in tqdm(all_data):
        emb = embeddings[example_id]['query_answer_embedding'] if use_answer else embeddings[example_id]['query_embedding']

        # base_correct_prob = irt_model(
        #     tester_id=torch.tensor(model2_id[model_name], dtype=torch.int64).unsqueeze(0),
        #     item_repr=torch.tensor(emb, dtype=torch.float32).unsqueeze(0),
        #     icl_flag=torch.tensor([[0, ]], dtype=torch.float64)
        # )['logits'].detach().numpy().tolist()[0][0]
        output = irt_model(
            tester_id=torch.tensor(model2_id[model_name], dtype=torch.int64).unsqueeze(0),
            item_repr=torch.tensor(emb, dtype=torch.float32).unsqueeze(0),
            icl_flag=torch.tensor([[1, ]], dtype=torch.float64)
        )
        ana_result.append({
            'model': model_name,
            'id': example_id,
            'logits': output['logits'].detach().numpy().tolist()[0][0],
            'difficulty': output['difficulty'].detach().numpy().tolist()[0][0],
            'match': output['match'].detach().numpy().tolist()[0][0],
            'learnability': output['learnability'].detach().numpy().tolist()[0][0],
        })

        # print({
        #     'id': example_id,
        #     'logits': output['logits'].detach().numpy().tolist()[0][0],
        #     'difficulty': output['difficulty'].detach().numpy().tolist()[0][0],
        #     'match': output['match'].detach().numpy().tolist()[0][0],
        #     'learnability': output['learnability'].detach().numpy().tolist()[0][0],
        # })
        # exit(1)
    return ana_result


if __name__ == '__main__':
    set_logging(log_file=None)

    model_names = [
        'meta-llama/Llama-2-7b-hf',
        'meta-llama/Llama-2-7b-chat-hf',
        'meta-llama/Llama-2-13b-hf',
        'meta-llama/Llama-2-13b-chat-hf',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Meta-Llama-3-8B',
        'meta-llama/Meta-Llama-3-70B-Instruct',
        'meta-llama/Meta-Llama-3-70B'
    ]

    conf = read_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ez_stance')
    parser.add_argument('--icl_strategy', type=str, default='similarity')
    # parser.add_argument('--model_name', type=str, default=model_names[0])
    parser.add_argument('--use_answer', type=str, default=False)
    parser.add_argument('--use_icl', type=str, default=True)
    args = parser.parse_args()

    processor = None
    evaluator = None
    train_data = None
    test_data = None

    if args.dataset == 'gsm8k':
        train_data = GSM8KProcessor.read_gsm8k(data_dir=conf['dataset'][args.dataset], split='train')
        test_data = GSM8KProcessor.read_gsm8k(data_dir=conf['dataset'][args.dataset], split='test')
        all_answer = {}
        for key in train_data:
            all_answer[key] = train_data[key]['answer']
        for key in test_data:
            all_answer[key] = test_data[key]['answer']
        evaluator = MathEvaluator(num_ice=8, prob_prefix=['Math Problem:', 'Question:'], ans_prefix=['Solution:', 'Answer:'], all_answer=all_answer)
        processor = GSM8KProcessor
    elif args.dataset == 'ez_stance':
        train_data = EZStanceProcessor.read_ez_stance(data_dir=conf['dataset'][args.dataset], splits=['train'], targets=['mixed'], domains=[], sub_tasks=['subtaskA'])
        test_data = EZStanceProcessor.read_ez_stance(data_dir=conf['dataset'][args.dataset], splits=['val'], targets=['mixed'], domains=[], sub_tasks=['subtaskA'])
        evaluator = EZStanceEvaluator()
        processor = EZStanceProcessor
    else:
        logging.error('Unknown dataset')
        exit(1)

    # irt_model_name = 'irt'
    # if args.use_icl:
    #     irt_model_name += '_icl'
    # if args.use_answer:
    #     irt_model_name += 'answer'

    logging.info('On [{}] dataset!'.format(args.dataset))
    conf = read_config()

    embeddings = {}
    embedding_file = conf['irt']['embeddings'][args.dataset]
    with open(embedding_file, 'r') as fp:
        for line in fp.readlines():
            data = json.loads(line.strip())
            embeddings[data['id']] = data

    test_data_ids = set()
    val_data_ids = set()
    test_irt_file = os.path.join(conf['irt']['data_dir'][args.dataset], 'test.jsonl')
    val_irt_file = os.path.join(conf['irt']['data_dir'][args.dataset], 'val.jsonl')
    with open(test_irt_file, 'r') as fp:
        for line in fp.readlines():
            test_data_ids.add(json.loads(line.strip())['query_id'])
    # with open(val_irt_file, 'r') as fp:
    #     for line in fp.readlines():
    #         val_data_ids.add(json.loads(line.strip())['query_id'])

    # read_zones(conf['zones'][args.dataset])
    with open('{}_irt_behavior.jsonl'.format(args.dataset), 'w') as fp:
        for model_name in model_names:
            print(model_name)
            result = analyze_model_behavior(
                ckpt_path=os.path.join(conf['irt']['model_save_dir'][args.dataset], 'mirt_irt_icl_answer_best.ckpt'),
                all_data=test_data,
                use_answer=True,
                model_name=model_name
            )
            for r in result:
                fp.write(json.dumps(r)+'\n')
    #
    # fp = open('{}_adaptive_res.jsonl'.format(args.dataset), 'w')
    #
    # for model_name in model_names:
    #     logging.info('Start evaluating {}'.format(model_name))
    #     adp_icl = get_adaptive_icl(
    #         model_base_output=os.path.join(conf['icl']['model_output_dir'][args.dataset], '{}.base.output.jsonl'.format(model_name.replace('/', '_'))),
    #         model_icl_output=os.path.join(conf['icl']['model_output_dir'][args.dataset], '{}.{}.output.jsonl'.format(model_name.replace('/', '_'), args.icl_strategy)),
    #         ice_mapping=read_ice_mapping(os.path.join(conf['icl']['ice'][args.icl_strategy][args.dataset])),
    #         tokenizer=AutoTokenizer.from_pretrained(model_name),
    #         evaluator=evaluator,
    #         processor=processor,
    #         train_data=train_data,
    #         test_data=test_data,
    #         separator=conf['templates'][args.dataset]['separator'],
    #         ice_template=conf['templates'][args.dataset]['ice_template'],
    #         query_template=conf['templates'][args.dataset]['query_template'],
    #         ckpt_path=os.path.join(conf['irt']['model_save_dir'][args.dataset], 'mirt_irt_icl_answer_best.ckpt'),
    #         embeddings=embeddings,
    #         use_answer=True,
    #         model_name=model_name,
    #         filter_id=test_data_ids | val_data_ids
    #     )
    #
    #     # display results
    #     adp_icl['adaptive_performance_all'].sort(key=lambda x: x['performance'])
    #     fp.write(json.dumps(
    #         {'model_name': model_name, 'adaptive_performance': adp_icl['adaptive_performance_all'], 'base_performance': adp_icl['base'], 'full_icl_performance': adp_icl['full_icl']}
    #     )+'\n')
    #     for key in adp_icl:
    #         if key == 'adaptive_performance_all':
    #             print('='*100)
    #             for item in enumerate(adp_icl[key][-100:]):
    #                 print(item)
    #         else:
    #             print(key, adp_icl[key])
    #
    # fp.close()
    #


