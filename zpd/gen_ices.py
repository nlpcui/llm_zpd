import json
import logging
import os.path
import sys
sys.path.append('../')
from retrieve_ice import CandidateRetriever
from score_ice import ICEScorer, read_candidates
from zpd.dataset import GSM8KProcessor, EZStanceProcessor
from common.config import read_config
from common.utils import set_logging, get_gpu_info
import argparse
import random
from tqdm import tqdm


if __name__ == '__main__':
    set_logging(log_file=None)
    conf = read_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,)
    parser.add_argument('--ice_type', type=str, )  # [random1/2/3, similarity, oracle]
    parser.add_argument('--num_candidates', type=int, default=10)
    parser.add_argument('--num_ices', type=int, default=8)

    args = parser.parse_args()

    save_file = conf['icl']['ice'][args.ice_type][args.dataset]

    logging.info('Generate {} demonstrations for {} dataset saved at {}'.format(args.ice_type, args.dataset, save_file))

    if args.dataset == 'gsm8k':
        train_data = GSM8KProcessor.read_gsm8k(data_dir=conf['dataset'][args.dataset], split='train')
        test_data = GSM8KProcessor.read_gsm8k(data_dir=conf['dataset'][args.dataset], split='test')
        processor = GSM8KProcessor
    elif args.dataset == 'ez_stance':
        train_data = EZStanceProcessor.read_ez_stance(
            data_dir=conf['dataset']['ez_stance'],
            sub_tasks=['subtaskA'],
            splits=['train'],
            domains=['consumption_and_entertainment_domain', 'covid19_domain', 'education_and_culture_domain', 'environmental_protection_domain', 'politic', 'rights_domain', 'sports_domain',
                     'world_event_domain'],
            targets=['mixed']
        )
        test_data = EZStanceProcessor.read_ez_stance(
            data_dir=conf['dataset']['ez_stance'],
            sub_tasks=['subtaskA'],
            splits=['val'],
            domains=['consumption_and_entertainment_domain', 'covid19_domain', 'education_and_culture_domain', 'environmental_protection_domain', 'politic', 'rights_domain', 'sports_domain',
                     'world_event_domain'],
            targets=['mixed']
        )
        processor = EZStanceProcessor

    else:
        logging.error('unknown dataset')
        exit(1)

    if args.ice_type.startswith('random'):
        ices = {}
        train_ids = list(train_data.keys())
        for test_id in tqdm(test_data):
            ices[test_id] = random.sample(train_ids, args.num_ices)

        with open(save_file, 'w') as fp:
            fp.write(json.dumps(ices)+'\n')

    else:
        # retrieve candidates
        # print(list(test_data.items())[0])
        # exit(1)
        if not os.path.exists(conf['icl']['ice']['candidates'][args.dataset]):
            logging.info('Retrieve similar candidates ...')
            candidate_retriever = CandidateRetriever(
                top_k=10,
                embedding_model_name='paraphrase-mpnet-base-v2',
                train_data=train_data,
                test_data=test_data,
                processor=processor,
                ice_template=conf['templates'][args.dataset]['ice_template']
            )
            candidate_retriever.retrieve_surface_sim()
            candidate_retriever.retrieve_semantic_sim()
            logging.info('Saving similar candidates to {}'.format(conf['icl']['ice']['candidates'][args.dataset]))
            candidate_retriever.save(output_file=conf['icl']['ice']['candidates'][args.dataset])

        if args.ice_type == 'similarity':
            ices = {}
            with open(conf['icl']['ice']['candidates'][args.dataset], 'r') as fp:
                data = json.loads(fp.readlines()[0])
                for test_id in data['semantic_similarity_query_answer']:
                    ices[test_id] = data['semantic_similarity_query_answer'][test_id][:args.num_ices]

            with open(save_file, 'w') as fp:
                fp.write(json.dumps(ices)+'\n')

        elif args.ice_type == 'oracle':
            # score and organize candidates
            num_gpu, local_rank, device = get_gpu_info()
            ice_candidates = read_candidates(conf['icl']['ice']['candidates'][args.dataset])

            for model_name in conf['models']:
                logging.info('Scoring candidates for model {}'.format(model_name))
                scorer = ICEScorer(
                    train_data=train_data,
                    test_data=test_data,
                    model_name=model_name,
                    candidates=ice_candidates,
                    batch_size=1,
                    template='Question: {query}\nAnswer: {answer}',
                    prune_size=5,
                    num_ice=8,
                    device=device,
                    local_rank=local_rank
                )
                scorer.single_score()
                scorer.save(os.path.join('./oracle_ices/gsm8k', model_name.replace('/', '_')))
        else:
            logging.error('Unknown ICE Type!')
            exit(1)


