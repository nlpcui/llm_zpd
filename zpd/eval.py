import json
import os
import argparse
import sys
sys.path.append('..')
from common.config import read_config
from common.utils import to_md5
from dataset import GSM8KProcessor, EZStanceProcessor


class MathEvaluator:
    def __init__(self, prob_prefix, ans_prefix, all_answer, num_ice=8, ):
        self.num_ice = num_ice
        self.prob_prefix = prob_prefix  # 'Math Problem:'
        self.ans_prefix = ans_prefix   # 'Solution:'
        self.all_answers = all_answer

    def eval_single(self, item, extract_n=True):
        if extract_n:
            prediction = self.extract_n_solution(item['raw_prediction'])
            prediction = self.extract_last_number(prediction)
        else:
            prediction = self.extract_last_number(item['raw_prediction'])
        if 'id' not in item:
            item_id = 'gsm8k#test#{}'.format(to_md5(item['question']))
        else:
            item_id = item['id']
        if float(self.extract_last_number(self.all_answers[item_id])) == float(prediction.strip()):
            return True
        else:
            return False

    def eval_batch(self, result_file, extract_n):
        total_cnt = 0
        correct_cnt = 0

        with open(result_file, 'r') as fp:
            for line in fp.readlines():
                data = json.loads(line.strip())
                total_cnt += 1
                label = self.eval_single(item=data, extract_n=extract_n)
                if label:
                    correct_cnt += 1
        return {
            'correct': correct_cnt,
            'total': total_cnt,
            'accuracy': correct_cnt / total_cnt
        }

    @staticmethod
    def extract_last_number(string, default='0'):
        all_numbers = []
        number = []

        for char in string:
            if char in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                number.append(char)
            elif char in ['.', ','] and number:
                number.append(char)
            elif number:
                all_numbers.append(''.join(number))
                number.clear()
        if number:
            all_numbers.append(''.join(number))
        if all_numbers:
            return all_numbers[-1].strip('.').replace(',', '')
        else:
            return default

    def extract_n_solution(self, answer):
        # print('answer', answer)
        segments = answer.split(self.prob_prefix[0])
        if len(segments) < self.num_ice+1:
            prob_prefix_true = self.prob_prefix[1]
            ans_prefix_true = self.ans_prefix[1]
        else:
            prob_prefix_true = self.prob_prefix[0]
            ans_prefix_true = self.ans_prefix[0]
        # print(prob_prefix_true, ans_prefix_true)
        segments = answer.split(prob_prefix_true)
        # print('segments', len(segments), segments)
        # exit(1)
        target_seg = segments[self.num_ice+1]
        target_solution = target_seg.split(ans_prefix_true)[1]
        # print('target_seg', target_seg,)
        # print('target_solution', target_solution)
        # exit(1)
        return target_solution


class EZStanceEvaluator:
    def __init__(self, text_prefix='Text:', ans_prefix='Answer', num_ice=8):
        self.ans_prefix = ans_prefix
        self.text_prefix = text_prefix
        self.num_ice = num_ice
        pass

    def eval_single(self, data, extract_n):
        ground_truth = data['answer'].lower()
        if extract_n:
            prediction = data['raw_prediction'].split(self.text_prefix)
            # print(prediction)
            # print(self.num_ice)
            prediction = prediction[self.num_ice+1]
            prediction = prediction.split(self.ans_prefix)[1]
        else:
            prediction = data['raw_prediction'].split(self.ans_prefix)[1]
        if ground_truth == 'none':
            ground_truth = 'neutral'
        if ground_truth in prediction:
            return 1
        else:
            return 0

    def eval_batch(self, output_file, extract_n, max_examples=-1):
        score = []
        with open(output_file, 'r') as fp:
            for lid, line in enumerate(fp.readlines()):
                if lid >= max_examples > 0:
                    break
                result = self.eval_single(json.loads(line), extract_n=extract_n)
                score.append(result)
        return sum(score) / len(score)


class TRECEvaluator:
    def __init__(self):
        pass

    def eval_single(self):
        pass

    def eval_batch(self):
        pass


class BreakEvaluator:
    def __init__(self):
        pass


if __name__ == '__main__':
    # acc_gpt_35_base = evaluate_answer_accuracy(result_file='results/base_sbs/gpt-3.5-turbo-0125.output.jsonl')
    # print(acc_gpt_35_base)
    # acc_gpt_4_base = evaluate_answer_accuracy(result_file='results/base/gpt-4-turbo-2024-04-09.output.jsonl')
    # print(acc_gpt_4_base)
    # math_evaluator = MathEvaluator(num_ice=8)
    # acc_llama2_7b_base = math_evaluator.eval_batch(result_file='results/base/meta-llama_Llama-2-7b-chat-hf.output.jsonl')
    # acc_llama2_13b_base = math_evaluator.eval_batch(result_file='results/base/meta-llama_Llama-2-13b-chat-hf.output.jsonl')
    # acc_llama3_8b_base = math_evaluator.eval_batch(result_file='results/base/meta-llama_Meta-Llama-3-8B.output.jsonl')
    # acc_llama3_8b_ins_base = math_evaluator.eval_batch(result_file='results/base/meta-llama_Meta-Llama-3-8B-Instruct.output.jsonl')
    # acc_llama2_7b_rdm_1 = math_evaluator.eval_batch(result_file='results/icl/random_1_meta-llama_Llama-2-7b-chat-hf.output.jsonl')
    # print(acc_llama2_7b_base, acc_llama2_13b_base, acc_llama3_8b_base, acc_llama3_8b_ins_base)
    # print(acc_llama2_7b_rdm_1)
    # acc_gpt_35_icl = evaluate_answer_accuracy(result_file='results/icl/gpt-3.5-turbo-0125.n5.output.jsonl')
    # acc_gpt_4_icl = evaluate_answer_accuracy(result_file='results/icl/gpt-4-turbo-2024-04-09.n5.output.jsonl')
    # print(acc_gpt_35_icl, acc_gpt_4_icl)

    # acc_gpt_35_cot = evaluate_answer_accuracy(result_file='results/cot/gpt-3.5-turbo-0125.output.jsonl')
    # print(acc_gpt_35_cot)

    # experiment_model_output_files = [
    #     ('gpt-3.5-turbo', 'results/cot/gpt-3.5-turbo-0125.output.jsonl'),
    #     ('gpt-4-turbo', 'results/cot/gpt-4-turbo-2024-04-09.output.jsonl'),
    #     ('llama-2-7b-chat', 'results/cot/meta-llama_Llama-2-7b-chat-hf.output.jsonl'),
    #     ('llama-3-8b-instruct', 'results/cot/meta-llama_Meta-Llama-3-8B-Instruct.output.jsonl'),
    #     ('llama-2-13b-chat', 'results/cot/meta-llama_Llama-2-13b-chat-hf.output.jsonl'),
    #     ('llama-3-8B', 'results/cot/meta-llama_Meta-Llama-3-8B.output.jsonl')
    # ]
    #
    # for model_name, model_fie in experiment_model_output_files:
    #     print(model_name, evaluate_answer_accuracy(model_fie))
    # ez_stance_evaluaror = EZStanceEvaluator()
    # res = ez_stance_evaluaror.eval_batch('/cluster/home/pencui/Projects/AdaptiveInstruct/data/model_output/ez_stance/meta-llama_Llama-2-7b-hf.base.output.jsonl')
    # print(res)

    # math_evaluator = MathEvaluator(num_ice=8, prob_prefix=['Question:', 'Math Problem:'], ans_prefix=['Answer:', 'Solution:'])
    # data_dir = '/cluster/project/sachan/pencui/ProjectsData/ICLIRT/gsm8k/model_output'
    # for file in os.listdir(data_dir):
    #     print(file)
    #     file_path = os.path.join(data_dir, file)
    #     res = math_evaluator.eval_batch(file_path, extract_n=False)
    #     print(res)
    #     print('='*100)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ez_stance')

    args = parser.parse_args()
    conf = read_config()

    if args.dataset == 'gsm8k':
        train_data = GSM8KProcessor.read_gsm8k(data_dir=conf['dataset']['gsm8k'], split='train')
        test_data = GSM8KProcessor.read_gsm8k(data_dir=conf['dataset']['gsm8k'], split='test')
        all_answer = {}
        for key in train_data:
            all_answer[key] = train_data[key]['answer']
        for key in test_data:
            all_answer[key] = test_data[key]['answer']
        evaluator = MathEvaluator(num_ice=8, prob_prefix=['Question:', 'Math Problem:'], ans_prefix=['Answer:', 'Solution'], all_answer=all_answer)

    elif args.dataset == 'ez_stance':
        evaluator = EZStanceEvaluator()

    for filename in os.listdir(conf['icl']['model_output_dir'][args.dataset]):
        filepath = os.path.join(conf['icl']['model_output_dir'][args.dataset], filename)
        print(filename)
        if 'base' in filename:
            score = evaluator.eval_batch(filepath, extract_n=False)
        else:
            score = evaluator.eval_batch(filepath, extract_n=True)
        print(score)
        print('='*100)

    # for filename in os.listdir(conf['finetune'][args.dataset]['model_output_dir']):
    #     filepath = os.path.join(conf['finetune'][args.dataset]['model_output_dir'], filename)
    #     print(filename)
    #     if 'base' in filename:
    #         score = evaluator.eval_batch(filepath, extract_n=False, max_examples=-1)
    #     else:
    #         score = evaluator.eval_batch(filepath, extract_n=True, max_examples=-1)
    #     print(score)
    #     print('='*100)