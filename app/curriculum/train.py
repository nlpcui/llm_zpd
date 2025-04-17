import json
import logging
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_path not in sys.path:
    sys.path.append(project_path)
import torch
import argparse
from torch.utils.data import DataLoader
from dataset import CLDataset, create_schedule
from huggingface_hub import snapshot_download
from common.utils import set_logging, get_gpu_info
from common.config import read_config
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
# import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
# from peft import LoraConfig, PeftModel
from transformers import AdamW
from zpd.dataset import GSM8KProcessor, EZStanceProcessor, read_ice_mapping
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map



def train_cl(model_name, train_data, val_data, curriculum, device, batch_size, lr, warmup_rate, num_epoch, model_save_dir, ices, query_template, ice_template, separator, processor, num_gpu, half_precision, irt_model_path, embeddings, epoch_per_bucket=2):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = conf['models'][model_name]['model_max_length']
    tokenizer.padding_side = 'left'

    logging.info('Local rank {}, constructing dataset'.format(local_rank))

    cl_dataset_train = CLDataset(
        train_data=train_data,
        test_data=val_data,   # validation set for train, and test for evaluation
        irt_model_path=irt_model_path,
        curriculum=curriculum,
        demonstrations=ices,
        query_template=query_template,
        ice_template=ice_template,
        separator=separator,
        processor=processor,
        tokenizer=tokenizer,
        embeddings=embeddings,
        model_name=model_name
    )
    collate_fn = CLDataset.build_collate_fn(tokenizer=tokenizer)

    sub_datasets = create_schedule(cl_dataset_train, num_chunks=3)

    # sub_dataloaders = [DataLoader(subset, collate_fn=collate_fn, shuffle=True, batch_size=1) for subset in sub_datasets]

    # if local_rank == 0:
    #     for data in sub_dataloaders[0]:
    #         logging.info('local_rank {}, input: {}'.format(local_rank, tokenizer.batch_decode(data['input_ids'])))
    #         labels = torch.where(data['label'] == -100, tokenizer.pad_token_id, data['label'])
    #         logging.info('local_rank {}, labels: {}'.format(local_rank, tokenizer.batch_decode(labels)))
    #         break

    if local_rank == 0:
        logging.info('Local rank {}, dataset bucket size: {}'.format(local_rank, [len(subset) for subset in sub_datasets]))

    if num_gpu > 1:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if half_precision else torch.float32)
        model = load_checkpoint_and_dispatch(
            model,
            snapshot_download(model_name),
            device_map='auto',
            dtype=torch.bfloat16 if half_precision else torch.float32,
            no_split_module_classes=["LlamaDecoderLayer"],
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if half_precision else torch.float32).to(device)

    model.train()
    logging.info('loading model [{}].'.format(model_name))

    step = 0
    total_examples = 0
    for epoch_id in range(num_epoch):
        total_examples += len(sub_datasets[int(epoch_id/epoch_per_bucket)])
    total_steps = int(total_examples / batch_size)
    warmup_steps = int(total_steps * warmup_rate)

    optimizer = AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    loss_record = {epoch_id: {} for epoch_id in range(num_epoch)}
    print('111')
    logging.info('Start training {}, total_steps {}, warmup rate {}, warmup steps {}, lr {}, batch_size {} ...'.format(model_name, total_steps, warmup_rate, warmup_steps, lr, batch_size))
    for epoch_id in range(num_epoch):
        epoch_loss = 0
        data_loader = DataLoader(sub_datasets[epoch_id // epoch_per_bucket], batch_size=1, shuffle=True, collate_fn=collate_fn)
        for batch_id, batch_data in enumerate(data_loader):
            # logging.info('Local rank {}, input length {}'.format(local_rank, batch_data['input_ids'].shape))
            outputs = model(
                input_ids=batch_data['input_ids'].to(device),
                attention_mask=batch_data['attention_mask'].to(device),
                labels=batch_data['label'].to(device),
            )
            # print('id', batch_data['id'])
            if batch_data['id'][0] not in loss_record[epoch_id]:
                loss_record[epoch_id][batch_data['id'][0]] = {}

            flip_labels = torch.where(batch_data['label'] == -100, batch_data['input_ids'], -100)
            flip_outputs = model(
                input_ids=batch_data['input_ids'].to(device),
                attention_mask=batch_data['attention_mask'].to(device),
                labels=flip_labels.to(device),
            )

            # record loss for analysis
            loss_record[epoch_id][batch_data['id'][0]]['flip_loss'] = flip_outputs.loss.detach().cpu().numpy().tolist()
            loss_record[epoch_id][batch_data['id'][0]]['loss'] = outputs.loss.detach().cpu().numpy().tolist()

            # num_label = torch.sum(torch.where(batch_data['label'] == -100, 0, 1))
            logging.info('Epoch {}/{}, batch {}/{}, loss is {}.'.format(epoch_id+1, num_epoch, batch_id+1, len(data_loader), outputs.loss))
            # logging.info('Epoch {}/{}, batch {}/{}, seq_len {}, label_len {}.'.format(epoch_id+1, num_epoch, batch_id+1, len(data_loader), batch_data['input_ids'].shape, num_label))
            outputs.loss.backward()
            epoch_loss += outputs.loss.detach().cpu().numpy().tolist()

            # print(torch.cuda.memory_summary(device=local_rank))
            # if local_rank == 0 and batch_id == 0:
            #     logging.info('='*200)
            #     inputs = tokenizer.batch_decode(batch_data['input_ids'])
            #     predictions = tokenizer.batch_decode(outputs.logits.argmax(dim=-1))
            #     logging.info('Input: {}'.format(inputs[0]))
            #     logging.info('Output: {}'.format(predictions[0]))
            #     logging.info('Loss: {}'.format(outputs.loss))
            #     logging.info('='*200)

            # gradient acc
            if (batch_id+1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

        logging.info('Epoch {}, avg loss is {}'.format(epoch_id+1, epoch_loss/len(data_loader)))
        # model.save_pretrained('/cluster/project/sachan/pencui/ProjectsData/ICLIRT/{}/ckpts/{}_cl_{}_ep{}'.format(args.dataset, curriculum, model_name.replace('/', '_'), epoch_id+1))
        model_save_path = os.path.join(model_save_dir, '{}_cl_{}_ep{}.ckpt'.format(curriculum, model_name.replace('/', '_'), epoch_id+1))
        logging.info('Saving {}th epoch model to {}'.format(epoch_id+1, model_save_path))
        torch.save(model.state_dict(), model_save_path)

        # loss save
        log_save_path = os.path.join(
            '/cluster/project/sachan/pencui/ProjectsData/ICLIRT/gsm8k/log',
            '{}_{}_{}_log.jsonl'.format(model_name.replace('/', '_'), args.dataset, curriculum)
        )
        with open(log_save_path, 'w') as fp:
            fp.write(json.dumps(loss_record) + '\n')


def eval_cl(model_name, model_ckpt, dataset, evaluator, num_gpu, half_precision, max_new_tokens=256):
    if num_gpu > 1:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if half_precision else torch.float32)
            model.load_state_dict(torch.load(model_ckpt))
            model = load_checkpoint_and_dispatch(
                model,
                snapshot_download(model_name),
                device_map='auto',
                # offload_folder="offload",
                # offload_state_dict=True,
                dtype=torch.bfloat16 if half_precision else torch.float32,
                no_split_module_classes=["LlamaDecoderLayer"],
            )
        # model.gradient_checkpointing_enable()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if half_precision else torch.float32).to(device)
        model.load_state_dict(torch.load(model_ckpt))
    model.eval()

    score = 0

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=CLDataset.build_collate_fn(tokenizer))
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            output = model.generate(
                input_ids=batch_data['input_ids'].to(device),
                attention_mask=batch_data['attention_masks'].to(device),
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens
            )
            predictions = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
            for prediction in predictions:
                score += evaluator.eval_single(prediction)


if __name__ == '__main__':
    model_names = [
        # 'meta-llama/Llama-2-7b-hf',
        # 'meta-llama/Llama-2-7b-chat-hf',
        # 'meta-llama/Llama-2-13b-hf',
        # 'meta-llama/Llama-2-13b-chat-hf',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        # 'meta-llama/Meta-Llama-3-8B',
        # 'meta-llama/Llama-2-70b-chat-hf',
        # 'meta-llama/Meta-Llama-3-70B-Instruct',
        # 'meta-llama/Meta-Llama-3-70B'
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--curriculum', type=str)  # [zpd, random]
    parser.add_argument('--icl_strategy', type=str, default='similarity')
    parser.add_argument('--train_paradigm', type=str, default='sft')   # ['icl_warmup', 'sft']
    parser.add_argument('--use_answer', type=bool, default=True)   # base_irt, icl_irt, answer_aware_irt
    parser.add_argument('--use_icl', type=bool, default=True)
    parser.add_argument('--half_precision', type=bool, default=False)
    args = parser.parse_args()

    log_file = '{}_{}_{}.log'.format(args.model_name.replace('/', '_'), args.dataset, args.curriculum)

    set_logging(log_file=log_file)
    conf = read_config()

    num_gpu, local_rank, device = get_gpu_info()
    logging.info('{} cards'.format(num_gpu))

    # irt_model_name = 'irt'
    # if args.use_icl:
    #     irt_model_name += '_icl'
    # if args.use_answer:
    #     irt_model_name += 'answer'
    # irt_model_name = '{}_params.jsonl'.format(irt_model_name)

    train_data = None
    test_data = None
    val_data = None
    processor = None

    ices = None
    if 'train_paradigm' == 'icl_warmup':
        ices = read_ice_mapping(conf['icl']['ices'][args.icl_strategy][args.dataset])

    if args.dataset == 'gsm8k':
        train_data = GSM8KProcessor.read_gsm8k(conf['dataset'][args.dataset], split='train')
        test_data = GSM8KProcessor.read_gsm8k(conf['dataset'][args.dataset], split='new_test')
        val_data = GSM8KProcessor.read_gsm8k(conf['dataset'][args.dataset], split='new_val')
        processor = GSM8KProcessor
    elif args.dataset == 'ez_stance':
        train_data = EZStanceProcessor.read_ez_stance(conf['dataset'][args.dataset], splits=['train'], domains=[], targets=['mixed'], sub_tasks=['subtaskA'])
        test_data = EZStanceProcessor.read_ez_stance(conf['dataset'][args.dataset], splits=['test'], domains=[], targets=['mixed'], sub_tasks=['subtaskA'])
        val_data = EZStanceProcessor.read_ez_stance(conf['dataset'][args.dataset], splits=['val'], domains=[], targets=['mixed'], sub_tasks=['subtaskA'])
        processor = EZStanceProcessor

    logging.info('Do [{}] training on the [{}] dataset, {} train examples, {} val examples, {} test examples!'.format(
        args.train_paradigm, args.dataset, len(train_data), len(val_data), len(test_data))
    )
    logging.info('Local rank {}, using [{}] curriculum!'.format(local_rank, args.curriculum))

    irt_model_path = os.path.join(conf['irt']['model_save_dir'][args.dataset], 'mirt_irt_icl_answer_best.ckpt')
    embeddings = {}
    embedding_file = conf['irt']['embeddings'][args.dataset]
    with open(embedding_file, 'r') as fp:
        for line in fp.readlines():
            data = json.loads(line.strip())
            embeddings[data['id']] = data

    train_cl(
        model_name=args.model_name,
        train_data=train_data,
        val_data=val_data,
        curriculum=args.curriculum,
        device=device,
        batch_size=4,
        lr=1e-5,
        warmup_rate=0.05,
        num_epoch=6,
        model_save_dir='/cluster/project/sachan/pencui/ProjectsData/ICLIRT/{}/ckpts'.format(args.dataset),
        ice_template=conf['templates'][args.dataset]['ice_template'],
        query_template=conf['templates'][args.dataset]['query_template'],
        separator=conf['templates'][args.dataset]['separator'],
        ices=ices,
        processor=processor,
        num_gpu=num_gpu,
        half_precision=args.half_precision,
        irt_model_path=irt_model_path,
        embeddings=embeddings
    )

