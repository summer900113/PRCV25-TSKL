import numpy as np
import time
import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist

import utils
from tqdm import tqdm
import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from clip import clip
from sentence_transformers import util

@torch.no_grad()
def evaluation(model, data_loader, device, args):
    # test
    model.eval()

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_embeds = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        try:
            text_input = data_loader.dataset.preprocess_text(text).to(device)
        except:
            print(text)
            raise
        text_embed = model.encode_text(text_input)
        text_embeds.append(text_embed)

    text_embeds = torch.cat(text_embeds, dim=0)

    image_embeds = []
    for image, img_id in tqdm(data_loader):
        image = image.to(device)
        image_embed = model.encode_image(image)
        image_embeds.append(image_embed)

    image_embeds = torch.cat(image_embeds, dim=0)

    score_matrix_i2t, score_matrix_t2i = model.get_similarity(
        image_embeds, text_embeds)
    score_matrix_i2i, score_matrix_t2t = model.get_similarity(
        image_embeds, text_embeds, False)
    score_matrix_i2t = score_matrix_i2t.contiguous()
    score_matrix_t2i = score_matrix_t2i.contiguous()
    score_matrix_i2i = score_matrix_i2i.contiguous()
    score_matrix_t2t = score_matrix_t2t.contiguous()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy(), score_matrix_i2i.cpu().numpy(), score_matrix_t2t.cpu().numpy()



from scipy.special import softmax
@torch.no_grad()
def KL(a, b, tau_a=1, tau_b=1):
    a = softmax(a / tau_a, axis=1)
    b = softmax(b / tau_b, axis=1)
    # a = np.log(a) 
    kl_divs = []
    for p_row, q_row in zip(a, b):
        kl = np.sum(p_row * np.log(p_row / q_row))
        kl_divs.append(kl)
    return np.array(kl_divs)

def KLContrastiveSimLoss(logits, softlabel, tau=1.0, softlabel_tau=1.0):
        
        sim_targets = F.softmax(softlabel / softlabel_tau, dim=1)

        logit_inputs = F.log_softmax(logits / tau, dim=1)

        loss = F.kl_div(logit_inputs, sim_targets, reduction='none')
        loss = loss.sum(dim=1)
        return loss

def itm_eval(config, scores_i2t_, scores_t2i_, scores_i2i_, scores_t2t_, txt2img, img2txt, test_sl_rate=1.0, tau_i2t=1.0, tau_t2i=1.0, tau_i2i=1.0, tau_t2t=1.0, device='cuda', flag=False):
    scores_i2t = torch.tensor(scores_i2t_).to(device)
    scores_t2i = torch.tensor(scores_t2i_).to(device)
    scores_i2i = torch.tensor(scores_i2i_).to(device)
    scores_t2t = torch.tensor(scores_t2t_).to(device)
    scores_i2t_sc = torch.zeros(scores_i2t.shape).to(device)
    scores_t2i_sc = torch.zeros(scores_t2i.shape).to(device)
    top_k = 256
    if flag:
        batch_size = 256
        for i in range(0, scores_i2t.shape[0], batch_size):
            a = scores_i2t[i:min(i + batch_size, scores_i2t.shape[0])]
            arg = torch.topk(a, k=top_k * 5, dim=1)[1]
            a = torch.gather(a, 1, arg)
            a = a.unsqueeze(1).expand(-1, a.shape[-1], -1).reshape(-1, a.shape[-1])
            b = []
            for j in range(arg.size(0)):
                try:
                    b.append(scores_t2t[arg[j].unsqueeze(-1), arg[j].unsqueeze(0)])
                except:
                    print(scores_t2t.shape)
                    print(arg.shape)
                    raise
            b = torch.stack(b)
            b = b.reshape(-1, b.shape[-1])
            tmp1 = KLContrastiveSimLoss(a, b, tau_i2t, tau_t2t).reshape(min(batch_size, scores_i2t.shape[0] - i), -1)
            for j in range(tmp1.shape[0]):
                scores_i2t_sc[i + j, arg] = scores_i2t[i + j, arg] - tmp1[j] * test_sl_rate + 1
        for i in range(0, scores_t2i.shape[0], batch_size):
            a = scores_t2i[i:min(i + batch_size, scores_t2i.shape[0])]
            arg = torch.topk(a, k=min(top_k, a.shape[1]), dim=1)[1]
            a = torch.gather(a, 1, arg)
            a = a.unsqueeze(1).expand(-1, a.shape[-1], -1).reshape(-1, a.shape[-1])
            b = []
            for j in range(arg.size(0)):
                b.append(scores_i2i[arg[j].unsqueeze(-1), arg[j].unsqueeze(0)])
            b = torch.stack(b)
            b = b.reshape(-1, b.shape[-1])
            tmp1 = KLContrastiveSimLoss(a, b, tau_t2i, tau_i2i).reshape(min(batch_size, scores_t2i.shape[0] - i), -1)
            for j in range(tmp1.shape[0]):
                scores_t2i_sc[i + j, arg] = scores_t2i[i + j, arg] - tmp1[j] * test_sl_rate + 1
    scores_i2t_sc = scores_i2t_sc.cpu().numpy()
    if not flag:
        scores_i2t_sc = scores_i2t_
    ranks = np.zeros(scores_i2t_sc.shape[0])
    for index, score in enumerate(scores_i2t_sc):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        if 'CUB' not in config['data_path']:
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
        else:
            for i in range(len(inds)):
                if img2txt[index] == txt2img[inds[i]]:
                    ranks[index] = i
                    break

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    scores_t2i_sc = scores_t2i_sc.cpu().numpy()
    if not flag:
        scores_t2i_sc = scores_t2i_
    ranks = np.zeros(scores_t2i_sc.shape[0])
    for index, score in enumerate(scores_t2i_sc):
        inds = np.argsort(score)[::-1]
        if 'CUB' not in config['data_path']:
            ranks[index] = np.where(inds == txt2img[index])[0][0]
        else:
            for i in range(len(inds)):
                if txt2img[index] == img2txt[inds[i]]:
                    ranks[index] = i
                    break

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    reserveNumber = 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean,
                   'r_sum': r_mean * 6}
    return eval_result, scores_i2t_sc, scores_t2i_sc
