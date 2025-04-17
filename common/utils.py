import hashlib
import logging
import torch.distributed as dist
import os
import torch


def extract_answer(content, suffix):
    if suffix:
        start = content.find(suffix)
        if start == -1:
            return None
        answer = content[start + len(suffix):]
    else:
        answer = content
    answer = to_number(answer).strip()
    return answer


def to_number(string):
    number = []
    for c_id, char in enumerate(string):
        if char.isdigit():
            number.append(char)
        elif char == '.' and c_id != 0 and c_id != len(string)-1 and string[c_id-1].isdigit() and string[c_id+1].isdigit():
            number.append(char)
    return ''.join(number)


def get_gpu_info():

    num_gpu = torch.cuda.device_count()
    if num_gpu == 0:
        device = torch.device('cpu')
        local_rank = 0
        logging.info('Run on CPU!')
    elif num_gpu == 1:
        device = torch.device('cuda:0')
        local_rank = 0
        logging.info('Run on single GPU!')
    else:
        local_rank = 0
        device = torch.device('cuda:{}'.format(local_rank))
        logging.info('Run on {} GPU!'.format(num_gpu))

    return num_gpu, local_rank, device


def to_md5(data):
    return hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()

def set_logging(log_file):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        level=logging.INFO,
        filemode='w'
    )