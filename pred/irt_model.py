import json
import logging
import os.path
import random
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from irt_data import IRTDataset, get_gpu_info
from common.utils import set_logging
from common.config import read_config
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

'''
ITR Implementation
1. Prepare IRT inputs with each line: {'question_id', 'model_name', 'question_vector', 'question_answer_vector', 'label', 'icl_flag', }
2. train model
3. evaluate performance
'''


class IRT1PL(torch.nn.Module):
    def __init__(self, item_ftr_dim, num_testers, ):
        super(IRT1PL, self).__init__()

        self.theta = torch.nn.Embedding(num_testers, 1)
        self.layer_item_difficulty_0 = torch.nn.Linear(item_ftr_dim, item_ftr_dim//2)
        self.layer_item_difficulty_1 = torch.nn.Linear(item_ftr_dim//2, 1)

    def forward(self, item_repr, tester_id):
        theta = self.theta(tester_id)
        d = self.layer_item_difficulty_1(F.tanh(self.layer_item_difficulty_0(item_repr)))
        return {
            'logits': theta - d,
            'difficulty': d,
            'ability': d
        }


class IRT2PL(torch.nn.Module):
    def __init__(self, item_ftr_dim, num_testers):
        super(IRT2PL, self).__init__()
        self.theta = torch.nn.Embedding(num_testers, 1)
        self.layer_item_difficulty_0 = torch.nn.Linear(item_ftr_dim, item_ftr_dim // 2)
        self.layer_item_difficulty_1 = torch.nn.Linear(item_ftr_dim // 2, 1)

        self.layer_item_discriminator_0 = torch.nn.Linear(item_ftr_dim, item_ftr_dim //2)
        self.layer_item_discriminator_1 = torch.nn.Linear(item_ftr_dim//2, 1)

    def forward(self, tester_id, item_repr):
        theta = self.theta(tester_id)
        difficulty = self.layer_item_difficulty_1(F.tanh(self.layer_item_difficulty_0(item_repr)))
        discriminator = self.layer_item_discriminator_1(F.tanh(self.layer_item_discriminator_0(item_repr)))
        return {
            'logits': discriminator * (theta - difficulty),
            'difficulty': difficulty,
            'discriminator': discriminator
        }


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


class LatentFactorIRT(torch.nn.Module):
    def __init__(self, num_testers, num_item_ftr, enable_gamma):
        super(LatentFactorIRT, self).__init__()

        self.testers = torch.nn.Embedding(num_testers, 1)
        self.fc0 = torch.nn.Linear(num_item_ftr, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.gamma = torch.nn.Embedding(num_testers, 1)
        self.enable_gamma = enable_gamma

    def forward(self, tester_id, item_ftr, icl_flag):
        '''
        :param tester_id: int
        :param item_ftr: vector
        :param icl_flag: bool (0/1)
        :return: logits
        '''
        a = self.testers(tester_id)
        gamma = self.gamma(tester_id)
        d = F.tanh(self.fc0(item_ftr))
        d = F.tanh(self.fc1(d))
        d = F.relu(self.fc2(d))
        d = self.fc3(d)
        if self.enable_gamma:
            return a + icl_flag * gamma - d, d
        else:
            return a - d, d


def train_irt(irt_data_dir, embedding_file, batch_size, num_epoch, enable_gamma, model_save_dir, param_save_dir, use_answer, irt_model_type, trait_dim, target_models, device, val_metric='roc_auc'):

    model_name = '{}_irt'.format(irt_model_type)
    if enable_gamma:
        model_name += '_icl'
    if use_answer:
        model_name += '_answer'

    train_dataset = IRTDataset(data_dir=irt_data_dir, split='train', embedding_file=embedding_file, enable_gamma=enable_gamma, target_models=target_models)
    print('model2id', train_dataset.model2id)
    print('repr_dim', train_dataset.repr_dim)
    logging.info('{} train examples, {} with icl tag!'.format(len(train_dataset), sum(train_dataset.icl_flags)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if irt_model_type == '1pl':
        model = IRT1PL(num_testers=train_dataset.num_models, item_ftr_dim=train_dataset.repr_dim).to(device)
    elif irt_model_type == '2pl':
        model = IRT2PL(num_testers=train_dataset.num_models, item_ftr_dim=train_dataset.repr_dim).to(device)
    else:
        model = MultiIRT(num_testers=train_dataset.num_models, item_ftr_dim=train_dataset.repr_dim, enable_gamma=enable_gamma, trait_dim=trait_dim).to(device)
    # model = LatentFactorIRT(num_testers=train_dataset.num_models, num_item_ftr=train_dataset.repr_dim, enable_gamma=enable_gamma)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    best_dev_performance = None

    num_batches = len(train_loader)
    # print(train_dataset.num_models)
    # print(model.testers.weight)
    # exit(1)

    # model_info = {
    #     'a': {key: 0 for key in list(train_dataset.model2id.keys())},
    #     'd': {},
    #     'gamma': {},
    #     'performance': {},
    # }

    for epoch_id in range(num_epoch):
        model.train()
        epoch_loss = 0
        for batch_id, batch_data in enumerate(train_loader):
            # print('there', train_dataset.model2id)
            # print(batch_data['model_id'])
            # exit(1)
            if irt_model_type == '1pl' or irt_model_type == '2pl':
                output = model(
                    tester_id=batch_data['model_id'].to(device),
                    item_repr=batch_data['query_answer_repr'].to(device) if use_answer else batch_data['query_repr'].to(device),
                )
            else:
                output = model(
                    tester_id=batch_data['model_id'].to(device),
                    item_repr=batch_data['query_answer_repr'].to(device) if use_answer else batch_data['query_repr'].to(device),
                    icl_flag=batch_data['icl_flag'].to(device)
                )

            label = batch_data['label']
            # print(output['logits'].shape, label.shape)
            # exit(1)
            loss = criterion(output['logits'], label.to(device))

            # for idx, d in enumerate(difficulties):
            #     model_info['d'][batch_data['query_id'][idx]] = float(d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('here')
            # logging.info('Training IRT: {}/{} epoch, {}/{} batch, loss is {}'.format(epoch_id+1, num_epoch, batch_id+1, num_batches, loss))
            epoch_loss += loss

        epoch_loss /= len(train_loader)
        logging.info('Training IRT: {}/{} epoch, average loss {}'.format(epoch_id+1, num_epoch, epoch_loss))

        # train_performance = eval_irt(data_dir=irt_data_dir, split='train', embedding_file=embedding_file, ckpt=model, num_item_ftr=train_dataset.repr_dim, enable_gamma=enable_gamma, irt_model_type=irt_model_type, target_models=target_models, use_answer=use_answer, device=device)
        dev_performance = eval_irt(data_dir=irt_data_dir, split='val', embedding_file=embedding_file, ckpt=model, num_item_ftr=train_dataset.repr_dim, enable_gamma=enable_gamma, irt_model_type=irt_model_type, target_models=target_models, use_answer=use_answer, device=device)
        test_performance = eval_irt(data_dir=irt_data_dir, split='test', embedding_file=embedding_file, ckpt=model, num_item_ftr=train_dataset.repr_dim, enable_gamma=enable_gamma, irt_model_type=irt_model_type, target_models=target_models, use_answer=use_answer, device=device)

        logging.info('Training IRT: {}/{} epoch, dev_performance: {}, '.format(epoch_id+1, num_epoch, dev_performance))
        logging.info('Training IRT: {}/{} epoch, test_performance: {}, '.format(epoch_id + 1, num_epoch, test_performance))
        # logging.info('Training IRT: {}/{} epoch, train_performance: {}, '.format(epoch_id + 1, num_epoch, train_performance))

        # for idx, var in enumerate(model.testers.weight):
        #     model_info['a'][train_dataset.id2model[idx]] = float(var)
        # for idx, var in enumerate(model.gamma.weight):
        #     model_info['gamma'][train_dataset.id2model[idx]] = float(var)
        # model_info['performance'] = test_performance
        #
        # param_save_path = os.path.join(param_save_dir, '{}_params.jsonl'.format(model_name))
        # with open(param_save_path, 'w') as fp:
        #     fp.write(json.dumps(model_info)+'\n')

        if not best_dev_performance or dev_performance['overall'][val_metric] > best_dev_performance['overall'][val_metric]:
            best_dev_performance = dev_performance
            model_save_path = os.path.join(model_save_dir,  '{}_best.ckpt'.format(model_name))
            torch.save(model.state_dict(), model_save_path)
            logging.info('Model saved to {}!'.format(model_save_path))

    logging.info('Best performance {}'.format(best_dev_performance))
    # with open(output_path, 'w') as fp:
    #     fp.write()


def eval_irt(data_dir, split, embedding_file, ckpt, num_item_ftr, enable_gamma, irt_model_type, target_models, use_answer, device):
    eval_dataset = IRTDataset(data_dir=data_dir, split=split, embedding_file=embedding_file, enable_gamma=True, target_models=target_models)
    logging.info('{} eval examples, {} with icl tag!'.format(len(eval_dataset), sum(eval_dataset.icl_flags)))
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    if type(ckpt) == str:
        if irt_model_type == '1pl':
            model = IRT1PL(num_testers=eval_dataset.num_models, item_ftr_dim=eval_dataset.repr_dim).to(device)
        elif irt_model_type == '2pl':
            model = IRT2PL(num_testers=eval_dataset.num_models, item_ftr_dim=eval_dataset.repr_dim).to(device)
        else:
            model = MultiIRT(num_testers=eval_dataset.num_models, item_ftr_dim=eval_dataset.repr_dim, enable_gamma=enable_gamma, trait_dim=32).to(device)
        model.load_state_dict(torch.load(ckpt))
    else:
        model = ckpt

    model.eval()

    all_pred_base = []
    all_true_base = []
    all_pred_proba_base = []

    all_random_base = []
    all_random_proba_base = []

    all_pred_icl = []
    all_true_icl = []
    all_pred_proba_icl = []
    all_random_icl = []
    all_random_proba_icl = []

    all_pred_overall = []
    all_true_overall = []
    all_pred_proba_overall = []
    all_random_overall = []
    all_random_proba_overall = []

    with torch.no_grad():
        for batch_id, batch_data in enumerate(eval_dataloader):
            if irt_model_type == '1pl' or irt_model_type == '2pl':
                output = model(tester_id=batch_data['model_id'].to(device), item_repr=batch_data['query_answer_repr'].to(device) if use_answer else batch_data['query_repr'].to(device))
            else:
                output = model(tester_id=batch_data['model_id'].to(device), item_repr=batch_data['query_answer_repr'].to(device) if use_answer else batch_data['query_repr'].to(device), icl_flag=batch_data['icl_flag'].to(device))
            pred_proba = torch.sigmoid(output['logits']).detach().cpu()
            random_proba = random.random()
            pred = torch.where(pred_proba > 0.5, 1, 0).numpy().tolist()
            random_label = 1 if random_proba > 0.5 else 0
            pred_proba = pred_proba.numpy().tolist()
            label = batch_data['label'].to(torch.int32).numpy().tolist()
            # print(pred)
            # print(label)
            # exit(1)
            for i in range(len(pred)):
                # print('here', batch_data['icl_flag'].numpy().tolist()[i])
                if batch_data['icl_flag'].numpy().tolist()[i][0] == 0:
                    all_pred_base.append(pred[i][0])
                    all_true_base.append(label[i][0])
                    all_pred_proba_base.append(pred_proba[i][0])

                    all_random_base.append(random_label)
                    all_random_proba_base.append(random_proba)
                else:
                    all_pred_icl.append(pred[i][0])
                    all_true_icl.append(label[i][0])
                    all_pred_proba_icl.append(pred_proba[i][0])
                    all_random_icl.append(random_label)
                    all_random_proba_icl.append(random_label)

                all_pred_overall.append(pred[i][0])
                all_true_overall.append(label[i][0])
                all_pred_proba_overall.append(pred_proba[i][0])
                all_random_overall.append(random_label)
                all_random_proba_overall.append(random_proba)
    # print('overall len', len(all_true_overall))
    # print('base len', len(all_true_base))
    # print('icl len', len(all_true_icl))
    # # exit(1)
    # print('true_icl', all_true_icl)
    # print('true_base', all_true_base)
    # print('pred_icl', all_pred_icl)
    # print('pred_base', all_pred_base)
    # exit(1)
    # performance for predicting base
    accuracy_base = accuracy_score(all_true_base, all_pred_base)
    f1_score_base = f1_score(all_true_base, all_pred_base)
    roc_auc_base = roc_auc_score(all_true_base, all_pred_proba_base)
    accuracy_random_base = accuracy_score(all_true_base, all_random_base)
    f1_score_random_base = f1_score(all_true_base, all_random_base)
    roc_auc_random_base = roc_auc_score(all_true_base, all_random_proba_base)

    # performance for predicting icl
    accuracy_icl = accuracy_score(all_true_icl, all_pred_icl)
    f1_score_icl = f1_score(all_true_icl, all_pred_icl)
    roc_auc_icl = roc_auc_score(all_true_icl, all_pred_proba_icl)

    accuracy_random_icl = accuracy_score(all_true_icl, all_random_icl)
    f1_score_random_icl = f1_score(all_true_icl, all_random_icl)
    roc_auc_random_icl = roc_auc_score(all_true_icl, all_random_proba_icl)

    # overall
    accuracy_overall = accuracy_score(all_true_overall, all_pred_overall)
    f1_score_overall = f1_score(all_true_overall, all_pred_overall)
    roc_auc_overall = roc_auc_score(all_true_overall, all_pred_proba_overall)

    accuracy_random_overall = accuracy_score(all_true_overall, all_random_overall)
    f1_score_random_overall = f1_score(all_true_overall, all_random_overall)
    roc_auc_random_overall = roc_auc_score(all_true_overall, all_random_proba_overall)

    result = {
        'base': {'accuracy': accuracy_base, 'f1_score': f1_score_base, 'roc_auc': roc_auc_base},
        'icl': {'accuracy': accuracy_icl, 'f1_score': f1_score_icl, 'roc_auc': roc_auc_icl},
        'overall': {'accuracy': accuracy_overall, 'f1_score': f1_score_overall, 'roc_auc': roc_auc_overall},
        'random_base': {'accuracy': accuracy_random_base, 'f1_score': f1_score_random_base, 'roc_auc': roc_auc_random_base},
        'random_icl': {'accuracy': accuracy_random_icl, 'f1_score': f1_score_random_icl, 'roc_auc': roc_auc_random_icl},
        'random_overall': {'accuracy': accuracy_random_overall, 'f1_score': f1_score_random_overall, 'roc_auc': roc_auc_random_overall},
    }
    return result


if __name__ == '__main__':
    set_logging(log_file=None)
    num_gpu, local_rank, device = get_gpu_info()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gsm8k')
    parser.add_argument('--use_answer', type=bool, default=True)
    parser.add_argument('--irt_model_type', type=str)  # [1pl, 2pl, mirt]
    parser.add_argument('--enable_gamma', type=bool, default=True)   # [icl_irt, base_irt]
    parser.add_argument('--eval_metrics', type=str, default='roc_auc')  # [roc_auc, f1_score, accuracy]
    parser.add_argument('--trait_dim', type=int, default=32)  # only for multi-dimension IRT
    parser.add_argument('--target_models', type=list, default=None)
    parser.add_argument('--job', type=str)  # [train, eval]

    args = parser.parse_args()

    conf = read_config()
    if args.job == 'train':
        train_irt(
            irt_data_dir=conf['irt']['data_dir'][args.dataset],
            embedding_file=conf['irt']['embeddings'][args.dataset],
            batch_size=16,
            num_epoch=20,
            enable_gamma=args.enable_gamma,
            model_save_dir=conf['irt']['model_save_dir'][args.dataset],
            param_save_dir=conf['irt']['params'][args.dataset],
            val_metric='roc_auc',
            use_answer=args.use_answer,
            irt_model_type=args.irt_model_type,
            trait_dim=args.trait_dim,
            target_models=args.target_models,
            device=device,
        )
    else:
        model_name = '{}_irt'.format(args.irt_model_type)
        if args.enable_gamma:
            model_name += '_icl'
        if args.use_answer:
            model_name += '_answer'

        ckpt_path = os.path.join(conf['irt']['model_save_dir'][args.dataset], '{}_best.ckpt'.format(model_name))
        print(ckpt_path)
        eval_result = eval_irt(conf['irt']['data_dir'][args.dataset], split='val',
                 embedding_file=conf['irt']['embeddings'][args.dataset],
                 ckpt=ckpt_path,
                 num_item_ftr=0,
                 enable_gamma=args.enable_gamma,
                 irt_model_type=args.irt_model_type,
                 target_models=args.target_models,
                 use_answer=args.use_answer,
                 device=device)
        print(eval_result)
        print('='*100)
