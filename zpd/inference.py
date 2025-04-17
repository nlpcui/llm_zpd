import os
import sys
sys.path.append('..')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import logging
import torch
import json
import argparse
from torch.utils.data import DataLoader
from common.config import read_config
from transformers import AutoModelForCausalLM, AutoTokenizer
from common.utils import set_logging, get_gpu_info
from tqdm import tqdm
from dataset import ICLDataset, GSM8KProcessor, read_ice_mapping, EZStanceProcessor
from eval import MathEvaluator
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, dispatch_model
from huggingface_hub import snapshot_download
from transformers import AutoConfig


torch.manual_seed(41)


def postprocess_llama_icl(output, n=5, separator='Math Problem:'):
    target = output.split(separator)[n+1]
    return target


def llm_generate(model_name, dataset, batch_size, output_file, max_new_tokens, num_gpu, half_precision=True, model_ckpt=None):

    if num_gpu > 1:
        config = AutoConfig.from_pretrained(model_name)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
            # model = AutoModelForCausalLM.from_pretrained(model_ckpt, map_location='cpu')
        if model_ckpt:
            logging.info('Loading model from {} on {} GPUs'.format(model_ckpt, num_gpu))
            model.load_state_dict(torch.load(model_ckpt,  map_location='cpu'))
            # model.load_state_dict(torch.load(model_ckpt))
        print('start load and dispatch')
        model = load_checkpoint_and_dispatch(
            model,
            model_ckpt if model_ckpt else snapshot_download(model_name),
            device_map='auto',
            offload_folder=None,
            offload_state_dict=True,
            dtype=torch.bfloat16 if half_precision else torch.float32,
            no_split_module_classes=["LlamaDecoderLayer"],
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if half_precision else torch.float32).to(device)
        if model_ckpt:
            model.load_state_dict(torch.load(model_ckpt))
            logging.info('Loading model from {} on single GPU'.format(model_ckpt))

    logging.info('Local rank {}, loading model {}'.format(local_rank, model_name))
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # padding='left')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = conf['models'][model_name]['model_max_length']
    tokenizer.padding_side = 'left'

    logging.info('Local rank {}, loading dataset {} examples, batch_size {}'.format(local_rank, len(dataset), batch_size))

    collate_fn = ICLDataset.build_collate_fn(tokenizer, padding='longest', return_tensors='pt')
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )
    output_fp = open(output_file, 'w')
    with torch.no_grad():
        for batch_id, batch_data in enumerate(tqdm(data_loader)):
            output = model.generate(
                input_ids=batch_data['input_ids'].to('cuda'),
                attention_mask=batch_data['attention_masks'].to('cuda'),
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
            )
            predictions = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
            if batch_id == 0 and local_rank == 0:
                logging.info('=' * 100)
                inputs = tokenizer.batch_decode(batch_data['input_ids'])
                for i in range(batch_data['input_ids'].size(0)):
                    logging.info('Input: {}'.format(inputs[i]))
                    logging.info('Output: {}'.format(predictions[i]))
                logging.info('='*100)
            for i in range(len(predictions)):
                output_fp.write(json.dumps({
                    'id': batch_data['ids'][i],
                    'raw_prediction': predictions[i],
                    'answer': batch_data['answers'][i],
                })+'\n')
                output_fp.flush()
    output_fp.close()


if __name__ == '__main__':
    conf = read_config()
    set_logging(log_file=None)

    num_gpu, local_rank, device = get_gpu_info()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--icl_strategy', type=str)
    parser.add_argument('--model_ckpt', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--split', type=str)    # [train, test, new_test, new_val]
    parser.add_argument('--curriculum', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_examples', type=int, default=-1)

    args = parser.parse_args()

    if args.dataset == 'gsm8k':
        logging.info('Reading [GSM8K] dataset ...')
        train_data = GSM8KProcessor.read_gsm8k(data_dir=conf['dataset']['gsm8k'], split='train', max_examples=args.max_examples)
        test_data = GSM8KProcessor.read_gsm8k(data_dir=conf['dataset']['gsm8k'], split=args.split, max_examples=args.max_examples)
        processor = GSM8KProcessor

    elif args.dataset == 'ez_stance':
        logging.info('Reading [EZ Stance] dataset ...')
        train_data = EZStanceProcessor.read_ez_stance(
            data_dir=conf['dataset']['ez_stance'],
            sub_tasks=['subtaskA'],
            splits=['train'],
            domains=['consumption_and_entertainment_domain', 'covid19_domain', 'education_and_culture_domain', 'environmental_protection_domain', 'politic', 'rights_domain', 'sports_domain', 'world_event_domain'],
            targets=['mixed'],
            max_examples=args.max_examples
        )
        test_data = EZStanceProcessor.read_ez_stance(
            conf['dataset']['ez_stance'],
            sub_tasks=['subtaskA'],
            splits=[args.split],
            domains=['consumption_and_entertainment_domain', 'covid19_domain', 'education_and_culture_domain', 'environmental_protection_domain', 'politic', 'rights_domain', 'sports_domain',
                     'world_event_domain'],
            targets=['mixed'],
            max_examples=args.max_examples
        )
        processor = EZStanceProcessor
    else:
        logging.error('unknown dataset')
        exit(1)

    logging.info('Inference on dataset [{}], {} train examples, {} test examples!'.format(args.dataset, len(train_data), len(test_data)))
    logging.info('Run {} inference on model {}, using {} GPU!'.format(args.icl_strategy, args.model_name, num_gpu))
    model_name_clean = args.model_name.replace('/', '_')

    if args.icl_strategy == 'base':
        ice_mapping = None
    elif args.icl_strategy == 'oracle':
        ice_mapping = read_ice_mapping(mapping_file=conf['icl']['ice'][args.icl_strategy][args.dataset][args.model_name])
    else:
        ice_mapping = read_ice_mapping(mapping_file=conf['icl']['ice'][args.icl_strategy][args.dataset])

    icl_dataset = ICLDataset(
            train_data=train_data,
            test_data=test_data,
            ice_mapping=ice_mapping,
            ice_template=conf['templates'][args.dataset]['ice_template'],
            query_template=conf['templates'][args.dataset]['query_template'],
            separator=conf['templates'][args.dataset]['separator'],
            tokenizer=AutoTokenizer.from_pretrained(args.model_name),
            processor=processor
    )

    if args.model_ckpt:
        output_dir = conf['finetune'][args.dataset]['model_output_dir']
    else:
        output_dir = conf['icl']['model_output_dir'][args.dataset]

    output_save_name = '{model_name}.{ice}'.format(model_name=model_name_clean, ice=args.icl_strategy)
    if args.curriculum:
        output_save_name += '.{}'.format(args.curriculum)
    if args.epoch:
        output_save_name += '.ep{}'.format(args.epoch)
    output_save_name += '.output.jsonl'

    logging.info('{} examples'.format(len(icl_dataset)))
    llm_generate(
        model_name=args.model_name,
        batch_size=args.batch_size,
        output_file=os.path.join(output_dir, output_save_name),
        max_new_tokens=conf['max_generate_tokens'][args.dataset],
        dataset=icl_dataset,
        model_ckpt=args.model_ckpt,
        num_gpu=num_gpu,
    )

