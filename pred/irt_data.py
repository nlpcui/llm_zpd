import logging
import os
import sys
sys.path.append('/cluster/home/pencui/Projects/AdaptiveInstruct')
import json
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from common.utils import to_md5, set_logging, get_gpu_info
from common.config import read_config
from zpd.dataset import GSM8KProcessor, EZStanceProcessor
from zpd.zones import get_zone_dist, read_zones
# from zpd.eval import MathEvaluator
import random
import torch
import argparse


class IRTDataset(Dataset):
    def __init__(self, data_dir, embedding_file, split, enable_gamma, target_models=None):
        super(IRTDataset, self).__init__()
        self.spilt = split
        self.enable_gamma = enable_gamma  #
        self.model2id = {}  # {model_name: id}

        self.model_ids = []          # query_id
        self.query_ids = []        # model_id
        self.query_reprs = []       # Encoder(question)
        self.answer_reprs = []      # Encoder(answer)
        self.query_answer_reprs = []  # Encoder(question+answer)
        self.labels = []            # 0/1
        self.icl_flags = []         # bool (0/1), w/ or w/o ICL

        self.embeddings = {}
        with open(embedding_file, 'r') as fp:
            for line in fp.readlines():
                data = json.loads(line.strip())
                self.embeddings[data['id']] = data

        data_file = os.path.join(data_dir, '{}.jsonl'.format(split))
        with open(data_file, 'r') as fp:
            for line in fp.readlines():
                data = json.loads(line.strip())

                if target_models and data['model_name'] not in target_models:
                    continue

                if data['model_name'] not in self.model2id:
                    self.model2id[data['model_name']] = len(self.model2id)
                # print('here', self.model2id)
                if not self.enable_gamma and data['icl_flag'] == 1:
                    continue
                self.model_ids.append(self.model2id[data['model_name']])
                self.query_ids.append(data['query_id'])
                self.query_reprs.append(self.embeddings[data['query_id']]['query_embedding'])
                self.answer_reprs.append(self.embeddings[data['query_id']]['answer_embedding'])
                self.query_answer_reprs.append(self.embeddings[data['query_id']]['query_answer_embedding'])
                self.labels.append(data['label'])
                self.icl_flags.append(data['icl_flag'])

        self.model_ids = torch.tensor(self.model_ids, dtype=torch.int64)
        self.query_reprs = torch.tensor(self.query_reprs, dtype=torch.float32)
        self.answer_reprs = torch.tensor(self.answer_reprs, dtype=torch.float32)
        self.query_answer_reprs = torch.tensor(self.query_answer_reprs, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(-1)
        self.icl_flags = torch.tensor(self.icl_flags, dtype=torch.float32).unsqueeze(-1)

        # print('hh', self.model2id)
        self.num_models = len(self.model2id)
        self.num_queries = len(set(self.query_ids))
        self.repr_dim = len(self.query_reprs[0])

        self.id2model = {self.model2id[model_name]: model_name for model_name in self.model2id}

    @staticmethod
    def prepare_irt_data(model_zones, save_dir, split_ratio=0.1):
        response_data = []  # [{model_name, query_id, label, icl_flag}   TODO: split的数量不对，train只有603个example
        example_ids = set([])
        for model_name, zones in model_zones:
            for example_id in zones['z_0_0']:
                example_ids.add(example_id)
                # z_0_0 case 1: label=0, icl_flag=0
                response_data.append({
                    'model_name': model_name, 'query_id': example_id, 'label': 0, 'icl_flag': 0
                })
                # z_0_0 case 2: label=0, icl_flag=1
                response_data.append({
                    'model_name': model_name, 'query_id': example_id, 'label': 0, 'icl_flag': 1
                })
            for example_id in zones['z_0_1']:
                example_ids.add(example_id)
                # z_0_1 case 1: label=0, icl_flag=0
                response_data.append({
                    'model_name': model_name, 'query_id': example_id, 'label': 0, 'icl_flag': 0
                })
                # z_0_1 case 2: label=0, icl_flag=1
                response_data.append({
                    'model_name': model_name, 'query_id': example_id, 'label': 1, 'icl_flag': 1
                })
            for example_id in zones['z_1_0'] + zones['z_1_1']:
                example_ids.add(example_id)
                # z_1_0/z_1_1: label=1, icl_flag=0
                response_data.append({
                    'model_name': model_name, 'query_id': example_id, 'label': 1, 'icl_flag': 0
                })
                response_data.append({
                    'model_name': model_name, 'query_id': example_id, 'label': 1, 'icl_flag': 1
                })

        train_fp = open(os.path.join(save_dir, 'train.jsonl'), 'w')
        test_fp = open(os.path.join(save_dir, 'test.jsonl'), 'w')
        val_fp = open(os.path.join(save_dir, 'val.jsonl'), 'w')

        # split datasets
        example_ids = list(example_ids)
        val_size = test_size = int(len(example_ids) * split_ratio)
        train_size = len(example_ids) - test_size - val_size
        val_test_example_ids = random.sample(example_ids, val_size+test_size)

        test_example_ids = set(random.sample(val_test_example_ids, test_size))
        val_example_ids = set(val_test_example_ids) - test_example_ids
        train_example_ids = set(example_ids) - test_example_ids - val_example_ids

        logging.info('{} items for train, {} items for val, and {} items for test!'.format(train_size, val_size, test_size))
        for response in response_data:
            if response['query_id'] in train_example_ids:
                train_fp.write(json.dumps(response)+'\n')
            elif response['query_id'] in test_example_ids:
                test_fp.write(json.dumps(response)+'\n')
            elif response['query_id'] in val_example_ids:
                val_fp.write(json.dumps(response)+'\n')

        train_fp.close()
        val_fp.close()
        test_fp.close()

        logging.info('Finish IRT data generation, {} train examples, {} val examples, {} test examples.'.format(train_size, val_size, test_size))

    @staticmethod
    def gen_embeddings(raw_data, embedding_model, save_path, device, processor):
        encoder = SentenceTransformer(embedding_model).to(device)

        data_ids = []
        queries = []
        answers = []
        query_answers = []

        for data_id, data in raw_data.items():
            data_ids.append(data_id)
            queries.append(processor.extract_query(data))
            answers.append(processor.extract_answer(data))
            query_answers.append('{} {}'.format(processor.extract_query(data), processor.extract_answer(data)))

        query_embeddings = encoder.encode(queries, device=device).tolist()
        answer_embeddings = encoder.encode(answers, device=device).tolist()
        query_answer_embeddings = encoder.encode(query_answers, device=device).tolist()

        with open(save_path, 'w') as fp:
            for i, data_id in enumerate(data_ids):
                fp.write(json.dumps({
                    'id': data_ids[i], 'query_embedding': query_embeddings[i], 'answer_embedding': answer_embeddings[i], 'query_answer_embedding': query_answer_embeddings[i]
                })+'\n')

    def __len__(self):
        return len(self.model_ids)

    def __getitem__(self, idx):
        return {
            'query_id': self.query_ids[idx],
            'model_id': self.model_ids[idx],
            'query_repr': self.query_reprs[idx],
            'answer_repr': self.answer_reprs[idx],
            'query_answer_repr': self.query_answer_reprs[idx],
            'label': self.labels[idx],
            'icl_flag': self.icl_flags[idx]
        }


if __name__ == '__main__':
    conf = read_config()
    set_logging(log_file=None)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ez_stance')
    parser.add_argument('--job', type=str, default='gen_splits')
    args = parser.parse_args()

    if args.dataset == 'ez_stance':
        processor = EZStanceProcessor()

    if args.job == 'gen_embeddings':
        num_gpu, local_rank, device = get_gpu_info()
        # print(num_gpu, device)
        # exit(1)
        IRTDataset.gen_embeddings(
            raw_data=EZStanceProcessor.read_ez_stance(
                conf['dataset']['ez_stance'],
                sub_tasks=['subtaskA'],
                splits=['val'],
                domains=['consumption_and_entertainment_domain', 'covid19_domain', 'education_and_culture_domain', 'environmental_protection_domain', 'politic', 'rights_domain', 'sports_domain',
                     'world_event_domain'],
                targets=['mixed']
            ),
            embedding_model='paraphrase-mpnet-base-v2',
            save_path=conf['irt']['embeddings']['ez_stance'],
            device=device,
            processor=processor
        )
    elif args.job == 'gen_splits':
        model_zones = read_zones(os.path.join(conf['zones'][args.dataset], 'zones.jsonl'))
        IRTDataset.prepare_irt_data(
            model_zones=[(item[0], item[1]['zone_examples']['overall']) for item in model_zones.items()],
            save_dir=conf['irt']['data_dir'][args.dataset],
            split_ratio=0.1
        )
    # model_zones = []
    # icl_strategies = ['oracle', 'random1', 'random2', 'random3', 'similarity']
    # for model_name in conf['models']:
    #     # print(model_name)
    #     fmt_model_name = model_name.replace('/', '_')
    #     zone = get_zone_dist(
    #         base_outputs=('gsm8k', os.path.join(conf['data']['model_output_dir'], 'gsm8k', '{}.base.output.jsonl'.format(fmt_model_name))),
    #         icl_outputs=[(
    #             model_name, 'gsm8k', icl_strategy, os.path.join(conf['data']['model_output_dir'], 'gsm8k', '{}.{}.output.jsonl'.format(fmt_model_name, icl_strategy))
    #         ) for icl_strategy in icl_strategies],
    #         evaluator=MathEvaluator(num_ice=8, prob_prefix=['Math Problem:', 'Question:'], ans_prefix=['Solution:', 'Answer:'])
    #     )
    #     # print('='*200)
    #     # print('model_name', model_name)
    #     model_zones.append((model_name, zone))

