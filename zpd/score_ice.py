# score examples using LLM
import logging
import sys
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch


def read_candidates(cand_file):
    candidates = {}
    with open(cand_file, 'r') as fp:
        for line in fp.readlines():
            all_candidates = json.loads(line.strip())
            for channel in all_candidates:
                for test_id in all_candidates[channel]:
                    if test_id not in candidates:
                        candidates[test_id] = {}
                    for train_id in all_candidates[channel][test_id]:
                        if train_id not in candidates[test_id]:
                            candidates[test_id][train_id] = {'src': [], 'loss': float('inf')}
                        candidates[test_id][train_id]['src'].append(channel)
    return candidates


class ScoreDataset(Dataset):
    def __init__(self, train_data, test_data, selected_ices, template, query_field, cot_answer_field, answer_field, separator, candidates, tokenizer):
        super(ScoreDataset, self).__init__()
        self.ids = []  # to_be_scored_test_id + already_selected_train_id + target_test_id
        self.inputs = []  # concatenated ICEs + target
        self.answer_lengths = []
        self.template = template
        self.separator = separator

        # tokenizer
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = tokenizer.eos_token
        for test_example_id, test_example in test_data.items():
            # ICE_to_be_scored + ICE_selected + query_answer
            for candidate_id in candidates[test_example_id]:
                # ICE_to_be_scored
                ids = [candidate_id]
                ice_query = train_data[candidate_id][query_field]
                ice_answer = train_data[candidate_id][cot_answer_field if cot_answer_field else answer_field]
                inputs = [self.template.format(query=ice_query, answer=ice_answer)]
                # ICE_already_selected
                for ice_id in selected_ices[test_example_id]:
                    ids.append(ice_id)
                    ice_query = train_data[candidate_id][query_field]
                    ice_answer = train_data[candidate_id][cot_answer_field if cot_answer_field else answer_field]
                    inputs.append(self.template.format(query=ice_query, answer=ice_answer))
                # target_query_answer
                ids.append(test_example_id)
                target_query = test_example[query_field]
                target_answer = test_example[cot_answer_field if cot_answer_field else answer_field]
                inputs.append(self.template.format(query=target_query, answer=target_answer))

                ids = '$'.join(ids)
                inputs = self.separator.join(inputs)
                self.answer_lengths.append(len(self.tokenizer(target_answer)['input_ids']))
                self.inputs.append(inputs)
                self.ids.append(ids)

    def __getitem__(self, idx):
        return {
            'input': self.inputs[idx],
            'answer_length': self.answer_lengths[idx],
            'id': self.ids[idx]
        }

    def __len__(self):
        return len(self.inputs)

    def build_collate_fn(self):
        def collate_fn(batch_data):
            encoded = self.tokenizer([data['input'] for data in batch_data], return_tensors='pt', padding='longest')
            labels = encoded['input_ids'].clone()
            for i, data in enumerate(batch_data):
                labels[i, :-data['answer_length']] = -100
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': labels,
                'ids': [data['id'] for data in batch_data]
            }
        return collate_fn


class ICEScorer:
    def __init__(self, train_data, test_data, model_name, candidates, batch_size, template, prune_size, device, local_rank, num_ice=8):
        logging.info('Scoring candidates using {}'.format(model_name))
        self.num_ice = num_ice
        self.train_data = train_data
        self.test_data = test_data

        self.candidates = candidates
        self.prune_size = prune_size

        self.model_name = model_name
        logging.info('Loading model {} ...'.format(model_name))
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.template = template
        self.selected_ices = {test_id: [] for test_id in self.test_data}  # test_id: [train_ids] descending_oder

        # inference setting
        self.batch_size = batch_size
        self.candidates = candidates  # {test_id: {train_id: {score, source(list)})}
        self.device = device
        self.local_rank = local_rank

        self.model.to(device)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def __score(self, dataset):
        logging.info('Scoring dataset size {}'.format(len(dataset)))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=dataset.build_collate_fn(), shuffle=False)
        # score candidates
        logging.info('Scoring ICEs ...')
        with torch.no_grad():
            for batch_data in tqdm(dataloader):
                model_outputs = self.model(
                    input_ids=batch_data['input_ids'].to(self.device),
                    attention_mask=batch_data['attention_mask'].to(self.device),
                    labels=batch_data['labels'].to(self.device)
                )
                # print('official loss', model_outputs.loss)
                # print('labels', batch_data['labels'])
                shift_labels = batch_data['labels'][:, 1:].contiguous()
                shift_logits = model_outputs.logits[:, :-1, :].contiguous()
                batch_loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(self.device))
                batch_loss = batch_loss.view(batch_data['input_ids'].size(0), batch_data['input_ids'].size(1)-1).sum(dim=1).detach().cpu().numpy().tolist()

                for b_id in range(self.batch_size):
                    example_ids = batch_data['ids'][b_id].split('$')
                    train_example_id, test_example_id = example_ids[0], example_ids[-1]
                    self.candidates[test_example_id][train_example_id]['loss'] = batch_loss[b_id]

    def iterate_score(self):
        pruned_candidates = {k: list(v.keys()) for k, v in self.candidates.items()}
        for i in range(self.num_ice):
            dataset = ScoreDataset(
                train_data=self.train_data,
                test_data=self.test_data,
                selected_ices=self.selected_ices,
                template=self.template,
                query_field='question',
                cot_answer_field='cot_answer',
                answer_field='answer',
                separator='\n\n',
                candidates=pruned_candidates,
                tokenizer=self.tokenizer,
            )
            self.__score(dataset)
            for test_example_id in self.candidates:
                sorted_candidates = sorted(self.candidates[test_example_id].items(), key=lambda x: x[1]['loss'])
                best_train_example_id = sorted_candidates[0][0]
                self.selected_ices[test_example_id].insert(0, best_train_example_id)
                self.candidates[test_example_id].pop(best_train_example_id)

                if self.prune_size > 0:
                    pruned_candidates[test_example_id] = [item[0] for item in sorted_candidates[1:self.prune_size+1]]   # exclude the best selected candidate
                else:
                    pruned_candidates[test_example_id] = [item[0] for item in sorted_candidates[1:]]

    def single_score(self):
        pruned_candidates = {k: list(v.keys()) for k, v in self.candidates.items()}
        dataset = ScoreDataset(
            train_data=self.train_data,
            test_data=self.test_data,
            selected_ices=self.selected_ices,
            template=self.template,
            query_field='question',
            answer_field='answer',
            cot_answer_field='cot_answer',
            separator='\n\n',
            candidates=pruned_candidates,
            tokenizer=self.tokenizer,
        )
        self.__score(dataset)
        for test_id in self.candidates:
            # print('here', list(self.candidates[test_id].items()))
            sorted_candidates = sorted(list(self.candidates[test_id].items()), key=lambda x: x[1]['loss'])
            self.selected_ices[test_id] = [item[0] for item in sorted_candidates[:self.num_ice]]

    def save(self, output_file):
        with open(output_file, 'w') as fp:
            for test_id in self.selected_ices:
                fp.write(json.dumps({'test_id': test_id, 'oracle_ice': self.selected_ices[test_id]})+'\n')