"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

import utils
from model.position_emb import prepare_graph_variables

from sklearn.metrics import roc_auc_score

import numpy as np

# def instance_bce_with_logits(logits, labels, reduction='mean'):
#     assert logits.dim() == 2
#     loss = F.binary_cross_entropy_with_logits(
#                                 logits, labels, reduction=reduction)
#     if reduction == "mean":
#         loss *= labels.size(1)
#     return loss


# def compute_score_with_logits(logits, labels, device):
#     # argmax
#     logits = torch.max(logits, 1)[1].data
#     logits = logits.view(-1, 1)
#     one_hots = torch.zeros(*labels.size()).to(device)
#     one_hots.scatter_(1, logits, 1)
#     scores = (one_hots * labels)
#     return scores

# def compute_auroc_with_logits(logits, target_labels, device):
#     logits = torch.max(logits, 1)[1].data
#     pred_labels = logits.cpu().numpy()
#     target_labels = target_labels.cpu().numpy()
#     return roc_auc_score(target_labels, pred_labels)


def train_test(model, train_loader, eval_loader, test_loader, args, device=torch.device("cuda")):
    N = len(train_loader.dataset)
    lr_default = args.base_lr
    num_epochs = args.epochs
    lr_decay_epochs = range(args.lr_decay_start, num_epochs,
                            args.lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default,
                            1.5 * lr_default, 2.0 * lr_default]

    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr=lr_default, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss()

    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    best_eval_score = 0

    utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f,'
                 % (lr_default, args.lr_decay_step,
                    args.lr_decay_rate) + 'grad_clip=%.2f' % args.grad_clip)
    logger.write('LR decay epochs: '+','.join(
                                        [str(i) for i in lr_decay_epochs]))
    last_eval_score, eval_score = 0, 0
    relation_type = train_loader.dataset.relation_type
    best_score = -1 # maximize au_roc
    remaining_steps = args.early_stop_steps # If auc_roc decreases these many times, it'll do early stop
    best_model = None

    for epoch in range(0, num_epochs):
        pbar = tqdm(total=len(train_loader))
        total_norm, count_norm = 0, 0
        total_loss = 0
        count, average_loss, att_entropy = 0, 0, 0
        t = time.time()
        if epoch < len(gradual_warmup_steps):
            for i in range(len(optim.param_groups)):
                optim.param_groups[i]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %.4f' %
                         optim.param_groups[-1]['lr'])
        elif (epoch in lr_decay_epochs or
              eval_score < last_eval_score and args.lr_decay_based_on_val):
            for i in range(len(optim.param_groups)):
                optim.param_groups[i]['lr'] *= args.lr_decay_rate
            logger.write('decreased lr: %.4f' % optim.param_groups[-1]['lr'])
        else:
            logger.write('lr: %.4f' % optim.param_groups[-1]['lr'])
        last_eval_score = eval_score

        mini_batch_count = 0
        batch_multiplier = args.grad_accu_steps
        for i, (v, norm_bb, q, target, _, _, bb, spa_adj_matrix,
                sem_adj_matrix) in enumerate(train_loader):
            batch_size = v.size(0)
            num_objects = v.size(1)
            if mini_batch_count == 0:
                optim.step()
                optim.zero_grad()
                mini_batch_count = batch_multiplier

            v = Variable(v).to(device)
            norm_bb = Variable(norm_bb).to(device)
            q = Variable(q).to(device)
            target = Variable(target).to(device)
            pos_emb, sem_adj_matrix, spa_adj_matrix = prepare_graph_variables(
                relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
                args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
                args.sem_label_num, device)
            pred, att = model(v, norm_bb, q, pos_emb, sem_adj_matrix,
                              spa_adj_matrix, target)
            # loss = instance_bce_with_logits(pred, target)
            loss = criterion(pred, target)

            loss /= batch_multiplier
            loss.backward()
            mini_batch_count -= 1
            total_norm += nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.grad_clip)
            count_norm += 1
            total_loss += loss.data.item() * batch_multiplier * v.size(0)
            pbar.update(1)

            if args.log_interval > 0:
                average_loss += loss.data.item() * batch_multiplier
                if model.module.fusion == "ban":
                    current_att_entropy = torch.sum(calc_entropy(att.data))
                    att_entropy += current_att_entropy / batch_size / att.size(1)
                count += 1
                if i % args.log_interval == 0:
                    att_entropy /= count
                    average_loss /= count
                    print("step {} / {} (epoch {}), ave_loss {:.3f},".format(
                            i, len(train_loader), epoch,
                            average_loss),
                          "att_entropy {:.3f}".format(att_entropy))
                    average_loss = 0
                    count = 0
                    att_entropy = 0

        total_loss /= N
        if eval_loader is not None:
            eval_score, entropy = evaluate(
                model, eval_loader, device, args)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f'
                     % (total_loss, total_norm / count_norm))
        if eval_loader is not None:
            logger.write('\tval AUROC: %.4f'
                         % (eval_score))

            if entropy is not None:
                info = ''
                for i in range(entropy.size(0)):
                    info = info + ' %.2f' % entropy[i]
                logger.write('\tentropy: ' + info)
        if (eval_loader is not None):
            if (eval_score > best_score):
                best_score = eval_score
                remaining_steps = args.early_stop_steps
                best_model = model
                # logger.write("saving current model weights to folder")
                # model_path = os.path.join(args.output, 'best_model.pth')
                # opt = optim if args.save_optim else None
                # utils.save_model(model_path, model, epoch, opt)
            else:
                remaining_steps = remaining_steps - 1
                if remaining_steps == 0:
                    logger.write(f"stopped at epoch {epoch}")
                    break

    test_score, _ = evaluate(best_model, test_loader, device, args)
    logger.write('\ntest AUROC: %.4f' % (test_score))


@torch.no_grad()
def evaluate(model, dataloader, device, args):
    model.eval()
    relation_type = dataloader.dataset.relation_type
    num_data = 0
    N = len(dataloader.dataset)
    entropy = None
    if model.module.fusion == "ban":
        entropy = torch.Tensor(model.module.glimpse).zero_().to(device)
    pbar = tqdm(total=len(dataloader))
    pred_labels = np.array([])
    target_labels = np.array([])

    for i, (v, norm_bb, q, target, _, _, bb, spa_adj_matrix,
            sem_adj_matrix) in enumerate(dataloader):
        batch_size = v.size(0)
        num_objects = v.size(1)
        v = Variable(v).to(device)
        norm_bb = Variable(norm_bb).to(device)
        q = Variable(q).to(device)
        target = Variable(target).to(device)

        pos_emb, sem_adj_matrix, spa_adj_matrix = prepare_graph_variables(
            relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
            args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
            args.sem_label_num, device)
        pred, att = model(v, norm_bb, q, pos_emb, sem_adj_matrix,
                          spa_adj_matrix, target)
        # batch_score = compute_auroc_with_logits(
        #                 pred, target, device) * batch_size
        pred_batch_labels = torch.max(pred, 1)[1].data
        pred_labels = np.concatenate([pred_labels, pred_batch_labels.cpu().numpy()])
        target_labels = np.concatenate([target_labels, target.cpu().numpy()])

        # score += batch_score
        num_data += pred.size(0)
        if att is not None and 0 < model.module.glimpse\
                and entropy is not None:
            entropy += calc_entropy(att.data)[:model.module.glimpse]
        pbar.update(1)

    score = roc_auc_score(target_labels, pred_labels)

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)
    model.train()
    return score, entropy


def calc_entropy(att):
    # size(att) = [b x g x v x q]
    sizes = att.size()
    eps = 1e-8
    p = att.view(-1, sizes[1], sizes[2] * sizes[3])
    return (-p * (p + eps).log()).sum(2).sum(0)  # g
