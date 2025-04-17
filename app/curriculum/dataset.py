import logging
import random
import sys
import torch
from torch.utils.data import Dataset, Subset
import json
import os
from tqdm import tqdm
import torch.nn.functional as F
import math

'''
Curriculum:
1. None curriculum 
2. zpd-based curriculum (distance-based)
'''
model2_id = {'meta-llama/Llama-2-7b-hf': 0, 'meta-llama/Llama-2-7b-chat-hf': 1, 'meta-llama/Llama-2-13b-hf': 2, 'meta-llama/Llama-2-13b-chat-hf': 3, 'meta-llama/Meta-Llama-3-8B-Instruct': 4, 'meta-llama/Meta-Llama-3-8B': 5, 'meta-llama/Meta-Llama-3-70B-Instruct': 6, 'meta-llama/Meta-Llama-3-70B': 7}


class MultiIRT(torch.nn.Module):
    def __init__(self, num_testers, item_ftr_dim, enable_gamma, trait_dim=32):
        super(MultiIRT, self).__init__()
        self.enable_gamma = enable_gamma
        self.theta = torch.nn.Embedding(num_testers, trait_dim)
        self.theta_icl = torch.nn.Embedding(num_testers, trait_dim)

        self.layer_item_traits = torch.nn.Linear(item_ftr_dim, trait_dim)
        self.layer_item_traits_icl = torch.nn.Linear(item_ftr_dim, trait_dim)
        self.layer_item_overall_dif_0 = torch.nn.Linear(item_ftr_dim, item_ftr_dim//2)
        self.layer_item_overall_dif_1 = torch.nn.Linear(item_ftr_dim//2, 1)

    def forward(self, tester_id, item_repr, icl_flag):
        theta = self.theta(tester_id)
        item_trait = F.tanh(self.layer_item_traits(item_repr))
        overall_difficulty = self.layer_item_overall_dif_1(F.tanh(self.layer_item_overall_dif_0(item_repr)))

        item_trait_icl = F.tanh(self.layer_item_traits_icl(item_repr))
        theta_icl = self.theta_icl(tester_id)

        match = torch.sum(theta * item_trait, dim=-1, keepdim=True)
        icl_learnability = torch.sum(item_trait_icl * theta_icl, dim=-1, keepdim=True)
        # s = theta * item_trait
        # print('theta', theta.shape)
        # print('item_trait', item_trait.shape)
        # print('s', s.shape)
        # print('match', match.shape, match)
        # print('learnability', icl_learnability.shape, icl_learnability)
        # print('overall_difficulty', overall_difficulty.shape, overall_difficulty)
        # print('icl_flag', icl_flag.shape, icl_flag)
        # exit(1)
        return {
            'logits': match - overall_difficulty + icl_flag * icl_learnability,
            'difficulty': overall_difficulty,
            'learnability': icl_learnability,
            'match': match
        }


class CLDataset(Dataset):
    def __init__(self, train_data, test_data, irt_model_path, curriculum, processor, tokenizer, embeddings, model_name, demonstrations=None, query_template=None, ice_template=None, separator=None, use_answer=True):
        super(CLDataset, self).__init__()
        assert curriculum in ['zpd', 'random', 'anti_zpd']
        self.curriculum = curriculum
        self.tokenizer = tokenizer
        self.embeddings = embeddings
        self.model_name = model_name
        # create curriculum
        cl_data_test = list(test_data.items())
        # print(difficulties.keys())

        self.irt_model = MultiIRT(num_testers=len(model2_id), item_ftr_dim=768, enable_gamma=True, trait_dim=32)
        self.irt_model.load_state_dict(torch.load(irt_model_path))

        values = {}

        for item in tqdm(cl_data_test):
            id_ = item[0]
            emb = self.embeddings[id_]['query_answer_embedding'] if use_answer else embeddings[id_]['query_embedding']
            base_correct_prob = self.irt_model(
                tester_id=torch.tensor(model2_id[model_name], dtype=torch.int64).unsqueeze(0),
                item_repr=torch.tensor(emb, dtype=torch.float32).unsqueeze(0),
                icl_flag=torch.tensor([[0, ]], dtype=torch.float64)
            )['logits'].detach().numpy().tolist()[0][0]
            base_correct_prob = 1/(1+(math.exp((-base_correct_prob))))

            icl_correct_prob = self.irt_model(
                tester_id=torch.tensor(model2_id[model_name], dtype=torch.int64).unsqueeze(0),
                item_repr=torch.tensor(emb, dtype=torch.float32).unsqueeze(0),
                icl_flag=torch.tensor([[1, ]], dtype=torch.float64)
            )['logits'].detach().numpy().tolist()[0][0]
            icl_correct_prob = 1/(1+(math.exp((-icl_correct_prob))))

            values[id_] = icl_correct_prob - base_correct_prob

        if self.curriculum == 'zpd':
            cl_data_test.sort(key=lambda x: values[x[0]], reverse=True)
        elif self.curriculum == 'anti-zpd':
            cl_data_test.sort(key=lambda x: values[x[0]], reverse=False)
        # elif self.curriculum == 'easy-to-hard':
        #     cl_data_test.sort(key=lambda x: values[x], reverse=False)
        # elif self.curriculum == 'hard-to-easy':
        #     cl_data_test.sort(key=lambda x: values[x], reverse=True)
        elif self.curriculum == 'random':
            random.shuffle(cl_data_test)
        else:
            logging.error('Unknown curriculum')
            exit(1)

        self.data_processor = processor

        self.demonstrations = demonstrations
        self.query_template = query_template
        self.ice_template = ice_template
        self.separator = separator
        self.test_data = test_data
        self.train_data = train_data

        self.ids = []
        self.inputs = []
        self.labels = []
        self.label_lengths = []
        self.difficulties = []

        for data_id, data in cl_data_test:
            self.ids.append(data_id)
            self.inputs.append(self.format_input(data_id))
            # labels = self.format_input(data_id)
            labels_len = len(self.tokenizer(self.data_processor.extract_answer(data))['input_ids'])
            self.labels.append(self.format_input(data_id))
            self.label_lengths.append(labels_len)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            'id': self.ids[idx],
            'input': self.inputs[idx],
            'label': self.labels[idx],
            'label_len': self.label_lengths[idx],
        }

    def format_input(self, example_id):
        if not self.demonstrations:
            return self.data_processor.build_ice(self.ice_template, self.test_data[example_id])
        input_ = []
        for ice_id in self.demonstrations[example_id]:
            input_.append(self.data_processor.build_ice(self.ice_template, self.train_data[ice_id]))
        input_.append(self.data_processor.build_ice(self.ice_template, self.test_data[example_id]))
        return self.separator.join(input_)

    @staticmethod
    def build_collate_fn(tokenizer, return_tensors='pt', padding='longest'):
        def collate_fn(batch_data):
            encoded = tokenizer([data['input'] for data in batch_data], return_tensors=return_tensors, padding=padding)
            # mask query part with -100
            label = encoded['input_ids'].clone()
            for i, l in enumerate(label):
                label[i, :-batch_data[i]['label_len']] = -100

            return {
                'id': [data['id'] for data in batch_data],
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'label': label,
            }
        return collate_fn


def create_schedule(dataset, num_chunks=3):
    chunk_size = int(len(dataset) / num_chunks)
    subsets = []
    indices = list([i for i in range(len(dataset))])

    for i in range(num_chunks):
        end = (i+1) * chunk_size if i != num_chunks -1 else len(dataset)
        subset = Subset(dataset, indices[0:end])
        subsets.append(subset)

    return subsets
