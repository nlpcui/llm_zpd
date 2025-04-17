import logging
import random
import sys
sys.path.append('../')
import os
import json
from torch.utils.data import Dataset
from common.utils import to_md5
import pandas as pd

base_prompt = '''Let's solve the problem step by step.

Math Problem: {problem_body}
Solution:'''

icl_prompt = '''{examples}

Math Problem: {problem_body}
Solution:'''


prompts = {
    'base': {'template': base_prompt, 'suffix': 'Therefore, the answer is'},
    'icl': {'template': icl_prompt, 'suffix': 'Therefore, the answer is'},
}


class EZStanceProcessor:
    @staticmethod
    def read_ez_stance(data_dir, sub_tasks, splits, domains, targets, max_examples=-1):
        data = {}

        all_files = []

        for sb in sub_tasks:
            for target in targets:
                target_dir = os.path.join(data_dir, sb, target)
                if sb == 'subtaskA':
                    for split in splits:
                        filename = 'raw_{}_all_onecol.csv'.format(split)
                        all_files.append({
                            'path': os.path.join(target_dir, filename),
                            'subtask': sb,
                            'domain': None,
                            'target': target
                        })
                else:  # subtaskB
                    for domain in domains:
                        target_dir = os.path.join(data_dir, sb, target, domain)
                        for split in splits:
                            filename = 'raw_{}_all_onecol.csv'.format(split)
                            all_files.append({
                                'path': os.path.join(target_dir, filename),
                                'subtask': sb,
                                'domain': domain,
                                'target': target
                            })

        for filepath in all_files:
            df = pd.read_csv(filepath['path'])
            for lid, (row_id, row) in enumerate(df.iterrows()):
                if lid >= max_examples > 0:
                    break
                example_id = to_md5(row['Text']+row['Target 1']+row['Stance 1'])
                data[example_id] = {
                    'text': row['Text'],
                    'target': row['Target 1'],
                    'stance': 'neutral' if row['Stance 1'] == 'NONE' else row['Stance 1'].lower(),
                    'subtask': filepath['subtask'],
                    'domain': filepath['domain'],
                    'target_type': filepath['target']
                }
        return data

    @staticmethod
    def build_ice(template, example):
        return template.format(text=example['text'], stance=example['stance'], target=example['target'])

    @staticmethod
    def build_query(template, example):
        return template.format(text=example['text'], target=example['target'])

    @staticmethod
    def extract_query(example):
        return example['text']

    @staticmethod
    def extract_answer(example):
        return example['stance']

    @staticmethod
    def extract_label(example):
        return example['stance']


class GSM8KProcessor:
    def __init__(self):
        pass

    @staticmethod
    def read_gsm8k(data_dir, split, enable_socratic=True, max_examples=-1):
        data = {}
        if enable_socratic:
            data_file = os.path.join(data_dir, '{}_socratic.jsonl'.format(split))
        else:
            data_file = os.path.join(data_dir, '{}.jsonl'.format(split))

        with open(data_file, 'r') as fp:
            for lid, line in enumerate(fp.readlines()):
                if 0 < max_examples <= lid:
                    break
                example = json.loads(line.strip())
                answer = parse_gsm8k_answer(example['answer'], enable_socratic)
                data_id = 'gsm8k#{}#{}'.format('train' if split == 'train' else 'test', to_md5(example['question']))   # TODO: unify the name
                data[data_id] = {
                    'question': example['question'],
                    'answer': answer['answer'],
                    'cot_answer': answer['cot_answer'],
                    'equations': answer['equations'],
                    'socratic': answer['socratic_questions'],
                    'num_steps': answer['steps']
                }

        return data

    @staticmethod
    def build_query(template, example):
        return template.format(query=example['question'])

    @staticmethod
    def build_ice(template, example):
        return template.format(query=example['question'], answer=example['cot_answer'])

    @staticmethod
    def extract_answer(example):
        return example['cot_answer']

    @staticmethod
    def extract_label(example):
        return example['answer']

    @staticmethod
    def extract_query(example):
        return example['question']

    @staticmethod
    def re_split(data_file, ratio, output_dir):
        all_data = []
        with open(data_file, 'r') as fp:
            for line in fp.readlines():
                data = json.loads(line.strip())
                all_data.append(data)
        if ratio < 1:
            cnt = int(len(all_data) * ratio)
        else:
            cnt = ratio
        sampled = random.sample([i for i in range(len(all_data))], cnt)

        fp_new_test = open(os.path.join(output_dir, 'new_test_socratic.jsonl'), 'w')
        fp_new_val = open(os.path.join(output_dir, 'new_val_socratic.jsonl'), 'w')

        for i in range(len(all_data)):
            if i in sampled:
                fp_new_test.write(json.dumps(all_data[i])+'\n')
            else:
                fp_new_val.write(json.dumps(all_data[i])+'\n')

        fp_new_test.close()
        fp_new_val.close()


def parse_gsm8k_answer(answer, enable_socratic):
    parsed = {
            'answer': None,
            'cot_answer': [],
            'equations': [],
            'socratic_questions': [],
    }
    lines = answer.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('####'):
            parsed['answer'] = line[4:].strip()
        else:
            if enable_socratic:
                socratic_question, solution = line.split('**')
                parsed['socratic_questions'].append(socratic_question)
            else:
                solution = line
            equation = find_between(solution, '<<', '>>')
            # solution = solution.replace('<<{}>>'.format(equation), '')
            parsed['equations'].append(equation)
            parsed['cot_answer'].append(solution.strip())
    parsed['steps'] = len(parsed['cot_answer'])
    parsed['cot_answer'] = '\n'.join(parsed['cot_answer']).strip() + '\n{} {}'.format(prompts['icl']['suffix'], parsed['answer'])
    return parsed


def collate_fn_wrap(tokenizer, truncation, padding):
    def collate_fn(data):

        texts = [datum['prompt'] for datum in data]
        encoded = tokenizer(texts, return_tensors='pt', truncation=truncation, padding=padding)
        batch_data = {
                'ids': [datum['id'] for datum in data],
                'input_ids': encoded['input_ids'],
                'attention_masks': encoded['attention_mask'],
                'labels': encoded['input_ids'],
                'answer': [datum['answer'] for datum in data],
                'question': [datum['question'] for datum in data]
        }

        return batch_data

    return collate_fn


def find_between(sequence, prefix, suffix):
    start = sequence.find(prefix)
    end = sequence.find(suffix)
    if not start or not suffix:
        return None
    return sequence[start+len(prefix):end]




def read_ice_mapping(mapping_file):
    ice_mapping = {}
    with open(mapping_file, 'r') as fp:
        for line in fp.readlines():
            data = json.loads(line.strip())
            try:
                ice_mapping[data['test_id']] = data['oracle_ice']   # read oracle
            except:
                for key in data:
                    ice_mapping[key] = data[key]   # read non-oracle
    return ice_mapping


class ICLDataset(Dataset):
    def __init__(self, train_data, test_data, ice_mapping, ice_template, query_template, separator, tokenizer, processor):
        super(ICLDataset, self).__init__()

        self.ice_mapping = ice_mapping

        self.ids = []
        self.inputs = []   # ICE + question
        self.labels = []
        self.tokenizer = tokenizer

        for test_id, test_data in test_data.items():
            ice_ids = self.ice_mapping[test_id] if self.ice_mapping else []
            ices = [processor.build_ice(ice_template, train_data[ice_id]) for ice_id in ice_ids]   # demonstrations
            query = [processor.build_query(query_template, test_data)]

            input_ = separator.join(ices + query)
            self.ids.append(test_id)
            self.inputs.append(input_)
            self.labels.append(processor.extract_answer(test_data))

    def __getitem__(self, idx):
        return {
            'id': self.ids[idx],
            'input': self.inputs[idx],
            'answer': self.labels[idx],
        }

    def __len__(self):
        return len(self.ids)

    @classmethod
    def build_collate_fn(cls, tokenizer, padding='longest', return_tensors='pt'):
        def collate_fn(batch_data):
            encoded = tokenizer([data['input'] for data in batch_data], padding=padding, return_tensors=return_tensors)
            return {
                'input_ids': encoded['input_ids'],
                'attention_masks': encoded['attention_mask'],
                'ids': [data['id'] for data in batch_data],
                'answers': [data['answer'] for data in batch_data],
            }
        return collate_fn


if __name__ == '__main__':
    pass
    # GSM8KProcessor.re_split(data_file='/cluster/work/sachan/pencui/datasets/GSM8K/test_socratic.jsonl', ratio=319, output_dir='/cluster/work/sachan/pencui/datasets/GSM8K')