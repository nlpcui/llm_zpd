import json
import sys
sys.path.append('/cluster/home/pencui/Projects/AdaptiveInstruct')
import os
from collections import OrderedDict
from zpd.eval import MathEvaluator, EZStanceEvaluator
from zpd.dataset import GSM8KProcessor
from common.config import read_config
from common.utils import to_md5
import argparse
from pprint import pprint

import matplotlib.pyplot as plt


def read_zones(zone_file):
    zones = {}
    with open(zone_file, 'r') as fp:
        for line in fp.readlines():
            data = json.loads(line.strip())
            zones[data['model_name']] = data['zones']
    return zones


def get_zone_inters(zones, target_zone='z_0_1'):
    inters_dist = {}  # {example_id: in which model's target zone are the example}
    for model_name, z in zones:
        for example_id in z[target_zone]:
            if example_id not in inters_dist:
                inters_dist[example_id] = set()
            inters_dist[example_id].add(model_name)
    num_shared_zone_dist = {}  # {number of models: number of examples shared by these models' target zones}
    for example_id in inters_dist:
        if len(inters_dist[example_id]) not in num_shared_zone_dist:
            num_shared_zone_dist[len(inters_dist[example_id])] = 0
        num_shared_zone_dist[len(inters_dist[example_id])] += 1
    return num_shared_zone_dist


def get_zone_dist(base_outputs, icl_outputs, evaluator):
    # desired output: {strategy: {z_0_0: {ids} z_0_1: {ids}, z_1_1: {}, z_1_0: {}}}   zdp: z_1_1
    zones = OrderedDict({})
    zones['base'] = {'z_0_0': [], 'z_0_1': [], 'z_1_1': [], 'z_1_0': []}
    for item in icl_outputs:
        zones[item[2]] = {'z_0_0': [], 'z_0_1': [], 'z_1_1': [], 'z_1_0': []}
    zones['overall'] = {'z_0_0': [], 'z_0_1': [], 'z_1_1': [], 'z_1_0': []}

    base_performance = {}
    print('reading {}'.format(base_outputs[1]))
    with open(base_outputs[1], 'r') as fp:
        for line in fp.readlines():
            data = json.loads(line.strip())
            if 'id' not in data:
                data_id = 'gsm8k#test#{}'.format(to_md5(data['question']))
            else:
                data_id = data['id']
            base_performance[data_id] = int(evaluator.eval_single(data, extract_n=False))

    icl_performance = {}
    for model_name, task, icl_strategy, file_path in icl_outputs:
        print('reading {}'.format(file_path))
        if not os.path.exists(file_path):
            print('missing {} output'.format(icl_strategy))
            continue
        icl_performance[icl_strategy] = {}
        with open(file_path, 'r') as fp:
            for line in fp.readlines():
                data = json.loads(line.strip())
                if 'id' not in data:
                    data_id = 'gsm8k#test#{}'.format(to_md5(data['question']))
                else:
                    data_id = data['id']
                icl_performance[icl_strategy][data_id] = int(evaluator.eval_single(data, extract_n=True))

    for id_ in base_performance:
        overall_performance = []
        for icl_strategy in icl_performance:
            if icl_strategy == 'base' or icl_strategy == 'overall':
                continue
            # print('base performance', base_performance[id_])
            # print('icl performance', icl_performance[icl_strategy][id_])
            if id_ not in icl_performance[icl_strategy]:
                continue
            zone = 'z_{}_{}'.format(base_performance[id_], icl_performance[icl_strategy][id_])
            zones[icl_strategy][zone].append(id_)
            overall_performance.append(int(icl_performance[icl_strategy][id_]))

        zones['overall']['z_{}_{}'.format(base_performance[id_], max(base_performance[id_], max(overall_performance)))].append(id_)

        # for strategy in zones:
        #     assert sum([len(zone) for zone in zones[strategy]]) == len(base_performance)

    zone_prop = OrderedDict({})
    for icl_strategy in zones:
        zone_prop[icl_strategy] = {}
        for z in zones[icl_strategy]:
            zone_prop[icl_strategy][z] = len(zones[icl_strategy][z]) / len(base_performance)

    zone_info = {
        'base_result': sum(base_performance.values()) / len(base_performance),
        'zone_examples': zones,
        'zone_prop': zone_prop
    }
    return zone_info


if __name__ == '__main__':
    conf = read_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gsm8k')

    args = parser.parse_args()

    if args.dataset == 'ez_stance':
        evaluator = EZStanceEvaluator()
    elif args.dataset == 'gsm8k':
        train_data = GSM8KProcessor.read_gsm8k(data_dir=conf['dataset']['gsm8k'], split='train')
        test_data = GSM8KProcessor.read_gsm8k(data_dir=conf['dataset']['gsm8k'], split='test')
        all_answer = {}
        for key in train_data:
            all_answer[key] = train_data[key]['answer']
        for key in test_data:
            all_answer[key] = test_data[key]['answer']
        evaluator = MathEvaluator(num_ice=8, prob_prefix=['Math Problem:', 'Question:'], ans_prefix=['Solution:', 'Answer:'], all_answer=all_answer)

    icl_strategies = [
        'oracle',
        'random1',
        'random2',
        'random3',
        'similarity'
    ]
    model_zones = []

    models = [
        'meta-llama/Llama-2-7b-hf',
        'meta-llama/Llama-2-7b-chat-hf',
        'meta-llama/Llama-2-13b-hf',
        'meta-llama/Llama-2-13b-chat-hf',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Meta-Llama-3-8B',
        'meta-llama/Meta-Llama-3-70B-Instruct',
        'meta-llama/Meta-Llama-3-70B'
    ]
    for model_name in models:
        print(model_name)
        fmt_model_name = model_name.replace('/', '_')
        zone = get_zone_dist(
            base_outputs=(args.dataset, os.path.join(conf['icl']['model_output_dir'][args.dataset], '{}.base.output.jsonl'.format(fmt_model_name))),
            icl_outputs=[(
                model_name, args.dataset, icl_strategy, os.path.join(conf['icl']['model_output_dir'][args.dataset], '{}.{}.output.jsonl'.format(fmt_model_name, icl_strategy))
            ) for icl_strategy in icl_strategies],
            evaluator=evaluator
        )
        print('='*200)
        model_zones.append((model_name, zone))

    with open(os.path.join(conf['zones'][args.dataset], 'zones.jsonl'), 'w') as fp:
        for model_name, zones in model_zones:
            fp.write(json.dumps({
                'model_name': model_name,
                'zones': zones
            })+'\n')

    # intersections = get_zone_inters([(item[0], item[1]['zone_examples']['overall']) for item in model_zones])
    # print(intersections)