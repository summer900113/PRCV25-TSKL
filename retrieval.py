import argparse
import datetime
import json
import logging
import os
import random
import time
import numpy as np
import yaml
import torch
from torch import distributed, optim
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Subset
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torchvision import transforms
from dataset import create_loader, create_sampler, get_dataset
from evaluation import evaluation, itm_eval
from unire.model import unire
import sys
from tqdm import tqdm
import utils
from scheduler import create_scheduler
from optim import create_optimizer
import shutil
from sentence_transformers import SentenceTransformer
import math

def main(args, config):
    config['schedular']['warmup_lr'] = float(config['schedular']['warmup_lr'])
    config['schedular']['min_lr'] = float(config['schedular']['min_lr'])
    # if not math.isclose(config['schedular']['warmup_lr'] / config['schedular']['min_lr'], 10.0, rel_tol=1e-6, abs_tol=1e-8):
    #     print(config['schedular']['warmup_lr'] / config['schedular']['min_lr'])
    #     raise
    device = torch.device(args.gpu)
    if config['do_uni_ab']:
        config['loss_config']['uni_softlabel']['is_on'] = False
    if config['do_cross_ab']:
        config['loss_config']['cross_softlabel']['is_on'] = False

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    if args.resume or args.eval:
        # try to continue training
        print('load checkpoint from %s' % args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        start_epoch = checkpoint['epoch'] + 1
        best = checkpoint['best']
        best_epoch = checkpoint['best_epoch']
        # config = checkpoint['config']
    else:
        start_epoch = 0
        best = 0
        best_epoch = 0
        state_dict = None

    print("args: ", args)
    print("config: ", config)
    print("config prefix: ", json.dumps(config, indent=4))

    # for training
    # get model
    # when resume, state_dict is not None, so we can load model from state_dict
    print("Creating model")
    model = unire(args, config)
    msg = model.load_state_dict(state_dict)
    print(msg)
    model.to(device)

    # get dataset
    print("Creating dataset")
    if args.experiment:
        train_dataset, val_dataset, test_dataset = [get_dataset(config['dataset_name'], config['data_path'], split, model.preprocess) for split in [
            'experiment', 'val', "test"]]
    else:
        if config['backbone'] == 'CLIP':
            train_dataset, val_dataset, test_dataset = [get_dataset(
                config['dataset_name'], config['data_path'], split, model.preprocess, config) for split in ['train', 'val', "test"]]
        elif config['backbone'] == 'X2VLM':
            from torchvision.transforms import InterpolationMode
            from X2VLM.dataset.randaugment import RandomAugment
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(model.c['image_res'], scale=(0.5, 1.0),
                                            interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize((model.c['image_res'], model.c['image_res']), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
            train_dataset, val_dataset, test_dataset = [get_dataset(
                config['dataset_name'], config['data_path'], split, transform, config) for split, transform in [('train', train_transform), ('val', test_transform), ('test', test_transform)]]
    # if 'coco' in config['data_path']:
    #     test_dataset_split = get_dataset(config['dataset_name'], config['data_path'], 'test_split', model.preprocess)
    # get sampler
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(
            [train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    # get loader
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers, batch_size=[config['batch_size_train'], config[
        'batch_size_test'], config['batch_size_testall']], num_workers=[16, 16, 16], is_trains=[True, False, False], collate_fns=[None, None, None])

    # assisant model
    # use sentence transformer to get text softlabel
    txt_enc_assisant = SentenceTransformer('/home/jncsnlp3/SSD2/syy/huggingface/all-mpnet-base-v2',cache_folder=r"/home/jncsnlp3/SSD2/syy/huggingface/all-mpnet-base-v2").to(device=device)
    if args.distributed:
        txt_enc_assisant = torch.nn.parallel.DistributedDataParallel(txt_enc_assisant, device_ids=[args.gpu])

    # train setting
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    # optimizer
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)

    # scheduler
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    from X2VLM.dataset import build_tokenizer
    if config['backbone'] == 'CLIP':
        tokenizer = None
    elif config['backbone'] == 'X2VLM':
        tokenizer = build_tokenizer(model.c['text_encoder'])

    #  train
    print("Start training")
    start_time = time.time()
    if args.eval:
        model.eval()
        print("Start eval")
        if config['backbone'] == 'CLIP':
            score_val_i2t, score_val_t2i, score_val_i2i, score_val_t2t = evaluation(model, val_loader, device, args)
            score_test_i2t, score_test_t2i, score_test_i2i, score_test_t2t = evaluation(model, test_loader, device, args)
        elif config['backbone'] == 'X2VLM':
            from X2VLM.Retrieval import evaluation as tmp_ev
            score_val_i2t, score_val_t2i, score_val_i2i, score_val_t2t = tmp_ev(model.x2_model, val_loader, tokenizer, device, model.c)
            score_test_i2t, score_test_t2i, score_test_i2i, score_test_t2t = tmp_ev(model.x2_model, test_loader, tokenizer, device, model.c)

        def do_45(x):
            reserveNumber = 2
            for key in x.keys():
                x[key] = round(x[key], reserveNumber)
            return x
        print('none')
        val_result, _, _ = itm_eval(config, score_val_i2t, score_val_t2i, score_val_i2i, score_val_t2t,
                              val_loader.dataset.txt2img, val_loader.dataset.img2txt, 
                              device=device, flag=False)
        print(do_45(val_result))
        start_time = time.time()
        for _ in range(10):
            test_result, scores_i2t_base, scores_t2i_base = itm_eval(config, score_test_i2t, score_test_t2i, score_test_i2i, score_test_t2t,
                                test_loader.dataset.txt2img, test_loader.dataset.img2txt, 
                                device=device, flag=False)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"代码运行时间：{run_time:.6f} 秒")
        print(do_45(test_result))
        print('1.0 1.0 1.0 1.0 1.0')
        val_result, _, _ = itm_eval(config, score_val_i2t, score_val_t2i, score_val_i2i, score_val_t2t,
                              val_loader.dataset.txt2img, val_loader.dataset.img2txt, 
                              device=device, test_sl_rate=1.0, tau_i2t=1.0, tau_t2t=1.0, 
                              tau_t2i=1.0, tau_i2i=1.0, flag=True)
        print(do_45(val_result))

        # from line_profiler import LineProfiler
        # # 创建分析器并运行函数
        # lp = LineProfiler()
        # lp.add_function(itm_eval)  # 监听目标函数
        # lp_wrapper = lp(itm_eval)  # 包装函数
        # test_result, _, _ = lp_wrapper(config, score_test_i2t, score_test_t2i, score_test_i2i, score_test_t2t,
        #                        test_loader.dataset.txt2img, test_loader.dataset.img2txt,
        #                        device=device, test_sl_rate=1.0, tau_i2t=1.0, tau_t2t=1.0,
        #                        tau_t2i=1.0, tau_i2i=1.0, flag=True)  # 执行函数
        # print(do_45(test_result))
        # lp.print_stats()
        # raise
        start_time = time.time()
        for _ in range(10):
            test_result, scores_i2t_10, scores_t2i_10 = itm_eval(config, score_test_i2t, score_test_t2i, score_test_i2i, score_test_t2t,
                                test_loader.dataset.txt2img, test_loader.dataset.img2txt, 
                                device=device, test_sl_rate=1.0, tau_i2t=1.0, tau_t2t=1.0, 
                                tau_t2i=1.0, tau_i2i=1.0, flag=True)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"代码运行时间：{run_time:.6f} 秒")
        print(do_45(test_result))
        raise
        if False:
            arg = []
            num = 0
            for i in range(scores_t2i_base.shape[0]):
                if scores_t2i_base[i].argmax() != i // 5 and scores_t2i_10[i].argmax() == i // 5:
                    num +=1
                    arg.append(i)
            print(num, len(scores_i2t_base))
            print(arg)
            indices = np.argsort(-scores_t2i_base[arg])
            print(indices.shape)
            text = test_loader.dataset.text
            image_path = []
            for tmp in tqdm(test_loader.dataset.image):
                image_path.append(tmp)
            texts = [text[arg[i]] + '//' + str(arg[i]) + '//' + ','.join(map(str, indices[i,:5].tolist())) for i in range(len(arg))]
            image_paths = [[image_path[indices[i][j]] for j in range(5)] for i in range(len(arg))]
            print(len(texts))
            print(len(image_paths))
            print(len(image_paths[0]))

            from PIL import Image, ImageDraw, ImageFont

            # ---------------------- 配置参数 ----------------------
            img_width, img_height = 256, 256  # 单张图片尺寸
            text_height = 25  # 单行文本高度（根据字体大小调整）
            # text_font = ImageFont.truetype("Arial.ttf", 18)  # 英文字体（如 Arial）
            text_font = ImageFont.load_default()
            margin = 5  # 边距和间隔
            max_text_width = img_width  # 文本宽度等于单张图片宽度
            save_path = "/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_06/img.jpg"  # 保存路径

            # ---------------------- 英文文本分行函数 ----------------------
            def split_english_text_into_lines(text, font, max_width):
                lines = []
                current_line = []
                for word in text.split():
                    word_width = font.getbbox(word + ' ')[2] - font.getbbox(word)[0]  # 单词宽度+空格
                    if sum(font.getbbox(w)[2] - font.getbbox(w)[0] for w in current_line) + word_width <= max_width:
                        current_line.append(word)
                    else:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                if current_line:
                    lines.append(' '.join(current_line))
                return lines

            # ---------------------- 计算总尺寸 ----------------------
            n = len(texts)
            m = max(len(row) for row in image_paths) if image_paths else 0
            text_lines_list = [split_english_text_into_lines(text, text_font, max_text_width) for text in texts]
            max_lines_per_text = max(len(lines) for lines in text_lines_list) if text_lines_list else 1

            total_width = (text_height + 2 * margin) + img_width * m  # 文本宽度 + 图片总宽度
            total_height = (max_lines_per_text * text_height + img_height + margin * 2) * n + margin

            # ---------------------- 创建画布并拼接 ----------------------
            result_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))
            draw = ImageDraw.Draw(result_img)

            for i in range(n):
                lines = text_lines_list[i]
                num_text_lines = len(lines)

                # 绘制文本（英文自动换行）
                text_x = margin
                text_y = i * (max_lines_per_text * text_height + img_height + margin) + margin
                for line_idx, line in enumerate(lines):
                    draw.text((text_x, text_y + line_idx * text_height), line, font=text_font, fill=(0, 0, 0))

                # 绘制图片
                img_x = text_height + 2 * margin
                img_y = text_y + num_text_lines * text_height + margin
                for j in range(m):
                    path = image_paths[i][j] if j < len(image_paths[i]) else None
                    try:
                        with Image.open(path) as img:
                            img = img.resize((img_width, img_height))
                            result_img.paste(img, (img_x + j * img_width, img_y))
                    except:
                        # 填充空白图
                        blank_img = Image.new("RGB", (img_width, img_height), (220, 220, 220))
                        result_img.paste(blank_img, (img_x + j * img_width, img_y))

            # ---------------------- 保存图片 ----------------------
            result_img.save(save_path)
            print(f"英文图文拼接完成，已保存至：{save_path}")
            raise

        print('1.0 0.6 0.4 0.6 0.4')
        val_result, _, _ = itm_eval(config, score_val_i2t, score_val_t2i, score_val_i2i, score_val_t2t,
                              val_loader.dataset.txt2img, val_loader.dataset.img2txt,
                              device=device, test_sl_rate=1.0, tau_i2t=0.6, tau_t2t=0.4,
                              tau_t2i=0.6, tau_i2i=0.4, flag=True)
        print(do_45(val_result))
        test_result, _, _ = itm_eval(config, score_test_i2t, score_test_t2i, score_test_i2i, score_test_t2t,
                               test_loader.dataset.txt2img, test_loader.dataset.img2txt, 
                               device=device, test_sl_rate=1.0, tau_i2t=0.6, tau_t2t=0.4, 
                               tau_t2i=0.6, tau_i2i=0.4, flag=True)
        print(do_45(test_result))
        # raise
        # 计算总迭代次数
        total_iterations = 10 * 10 * 10 + 10 * 10 * 10
        arr1 = np.empty((10, 10, 10), dtype=object)
        arr2 = np.empty((10, 10, 10), dtype=object)
        # 创建 tqdm 进度条
        pbar = tqdm(total=total_iterations)
        best_test = None
        best_test_sl_rate = 1.0
        best_tau_i2t = 1.0
        best_tau_t2t = 1.0
        best_test_sl_rate = 1.0
        best_test_sl_rate = 1.0
        step_len = 10
        score_test_i2t = score_val_i2t
        score_test_t2i = score_val_t2i
        score_test_i2i = score_val_i2i
        score_test_t2t = score_val_t2t
        test_loader = val_loader

        for test_sl_rate_ in range(0, 10):
            test_sl_rate = test_sl_rate_ / step_len
            test_sl_rate = 10 / step_len - test_sl_rate
            tmp = None
            tmp_i2t = 1.0
            tmp_t2t = 1.0
            for tau_i2t_ in range(0, 10):
                tau_i2t = tau_i2t_ / step_len
                tau_i2t = 10 / step_len - tau_i2t
                for tau_t2t_ in range(0, 10):
                    tau_t2t = tau_t2t_ / step_len
                    tau_t2t = 10 / step_len - tau_t2t
                    if False:
                        for i in range(5):
                            textid = 5000 * i
                            imgid = 1000 * i
                            test_result, _, _ = itm_eval(config, score_test_i2t[imgid:imgid + 1000, textid:textid + 5000], score_test_t2i[textid:textid + 5000, imgid:imgid + 1000], score_test_i2i[imgid:imgid + 1000, imgid:imgid + 1000],
                                                        score_test_t2t[textid:textid + 5000, textid:textid + 5000], test_loader.dataset.txt2img, test_loader.dataset.img2txt, 
                                                        device=device, test_sl_rate=test_sl_rate, tau_i2t=tau_i2t, tau_t2t=tau_t2t, flag=True)
                            if i == 0:
                                test_results = test_result
                            else:
                                for key, value in test_result.items():
                                    test_results[key] += value
                        for key, value in test_results.items():
                            test_results[key] = value / 5
                    else:
                        test_result, _, _ = itm_eval(config, score_test_i2t, score_test_t2i, score_test_i2i,
                                                    score_test_t2t, test_loader.dataset.txt2img, test_loader.dataset.img2txt, 
                                                    device=device, test_sl_rate=test_sl_rate, tau_i2t=tau_i2t, tau_t2t=tau_t2t, flag=True)
                    arr = {
                        'result': test_result,
                        'test_sl_rate': test_sl_rate,
                        'tau_i2t': tau_i2t,
                        'tau_t2i': 10 / step_len,
                        'tau_t2t': tau_t2t,
                        'tau_i2i': 10 / step_len,
                    }
                    arr1[test_sl_rate_, tau_i2t_, tau_t2t_] = arr
                    if tmp is None or test_result['r_sum'] > tmp['r_sum']:
                        tmp = test_result
                        tmp_i2t = tau_i2t
                        tmp_t2t = tau_t2t
                        # print()
                        # print('tmp:')
                        # print(do_45(tmp))
                        # print(test_sl_rate, tmp_i2t, tmp_t2t, 10 / step_len, 10 / step_len)
                        # print()
                        # sys.stdout.flush()
                    pbar.update(1)
            for tau_t2i_ in range(0, 10):
                tau_t2i = tau_t2i_ / step_len
                tau_t2i = 10 / step_len - tau_t2i
                for tau_i2i_ in range(0, 10):
                    tau_i2i = tau_i2i_ / step_len
                    tau_i2i = 10 / step_len - tau_i2i
                    if False:
                        for i in range(5):
                            textid = 5000 * i
                            imgid = 1000 * i
                            test_result, _, _ = itm_eval(config, score_test_i2t[imgid:imgid + 1000, textid:textid + 5000], score_test_t2i[textid:textid + 5000, imgid:imgid + 1000], score_test_i2i[imgid:imgid + 1000, imgid:imgid + 1000],
                                                        score_test_t2t[textid:textid + 5000, textid:textid + 5000], test_loader.dataset.txt2img, test_loader.dataset.img2txt, 
                                                        device=device, test_sl_rate=test_sl_rate, tau_i2t=tmp_i2t, tau_t2t=tmp_t2t, 
                                                        tau_t2i=tau_t2i, tau_i2i=tau_i2i, flag=True)
                            if i == 0:
                                test_results = test_result
                            else:
                                for key, value in test_result.items():
                                    test_results[key] += value
                        for key, value in test_results.items():
                            test_results[key] = value / 5
                    else:
                        test_result, _, _ = itm_eval(config, score_test_i2t, score_test_t2i, score_test_i2i,
                                                    score_test_t2t, test_loader.dataset.txt2img, test_loader.dataset.img2txt, 
                                                    device=device, test_sl_rate=test_sl_rate, tau_i2t=tmp_i2t, tau_t2t=tmp_t2t, 
                                                    tau_t2i=tau_t2i, tau_i2i=tau_i2i, flag=True)
                    arr = {
                        'result': test_result,
                        'test_sl_rate': test_sl_rate,
                        'tau_i2t': tmp_i2t,
                        'tau_t2i': tau_t2i,
                        'tau_t2t': tmp_t2t,
                        'tau_i2i': tau_i2i,
                    }
                    arr2[test_sl_rate_, tau_t2i_, tau_i2i_] = arr
                    if best_test is None or test_result['r_sum'] > best_test['r_sum']:
                        best_test = test_result
                        best_test_sl_rate =test_sl_rate
                        best_tau_i2t = tmp_i2t
                        best_tau_t2t = tmp_t2t
                        best_tau_t2i = tau_t2i
                        best_tau_i2i = tau_i2i
                    pbar.update(1)
                print()
                print('best:')
                print(do_45(best_test))
                print(best_test_sl_rate, best_tau_i2t, best_tau_t2t, best_tau_t2i, best_tau_i2i)
                print()
                sys.stdout.flush()
                np.save('arr1.npy', arr1)
                np.save('arr2.npy', arr2)
                # print(do_45(tmp))
                # print(test_sl_rate, tmp_i2t, tmp_t2t, 1.0, 1.0)
                # print(do_45(best_test))
                # print(best_test_sl_rate, best_tau_i2t, best_tau_t2t, best_tau_t2i, best_tau_i2i)
                # sys.stdout.flush()
        print()
        print('best:')
        print(do_45(best_test))
        print(best_test_sl_rate, best_tau_i2t, best_tau_t2t, best_tau_t2i, best_tau_i2i)
        print()
        sys.stdout.flush()
        np.save('arr1.npy', arr1)
        np.save('arr2.npy', arr2)
        return

    for epoch in range(start_epoch, max_epoch):
        copy_file("/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_06/train.log", config['logger_name'])
        lr_scheduler.step(epoch)
        # set epoch
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_stats = {}
        # train
        train_stats = train(model, train_loader, optimizer, lr_scheduler, epoch, warmup_steps, device, config, txt_enc_assisant, tokenizer=tokenizer)

        # eval
        if config['backbone'] == 'CLIP':
            score_val_i2t, score_val_t2i, score_val_i2i, score_val_t2t = evaluation(model, val_loader, device, args)
            score_test_i2t, score_test_t2i, score_test_i2i, score_test_t2t = evaluation(model, test_loader, device, args)
        elif config['backbone'] == 'X2VLM':
            from X2VLM.Retrieval import evaluation as tmp_ev
            score_val_i2t, score_val_t2i, score_val_i2i, score_val_t2t = tmp_ev(model.x2_model, val_loader, tokenizer, device, model.c)
            score_test_i2t, score_test_t2i, score_test_i2i, score_test_t2t = tmp_ev(model.x2_model, test_loader, tokenizer, device, model.c)


        # save model and log
        if utils.is_main_process():
            def do_45(x):
                reserveNumber = 2
                for key in x.keys():
                    x[key] = round(x[key], reserveNumber)
                return x
            print('none')
            val_result, _, _ = itm_eval(config, score_val_i2t, score_val_t2i, score_val_i2i, score_val_t2t,
                                val_loader.dataset.txt2img, val_loader.dataset.img2txt, 
                                device=device, flag=False)
            print(do_45(val_result))
            test_result, _, _ = itm_eval(config, score_test_i2t, score_test_t2i, score_test_i2i, score_test_t2t,
                                test_loader.dataset.txt2img, test_loader.dataset.img2txt, 
                                device=device, flag=False)
            print(do_45(test_result))
            if 'coco' in config['data_path']:
                test_result_split = None
                for i in range(5):
                    tmp, _, _ = itm_eval(config, score_test_i2t[i * 1000:(i + 1) * 1000, i * 5000:(i + 1) * 5000],
                                   score_test_t2i[i * 5000:(i + 1) * 5000, i * 1000:(i + 1) * 1000], 
                                   score_test_i2i[i * 1000:(i + 1) * 1000, i * 1000:(i + 1) * 1000], 
                                   score_test_t2t[i * 5000:(i + 1) * 5000, i * 5000:(i + 1) * 5000], 
                                   test_loader.dataset.txt2img, test_loader.dataset.img2txt, flag=False)
                    if test_result_split is None:
                        test_result_split = tmp
                    else:
                        for key in test_result_split.keys():
                            test_result_split[key] += tmp[key]

                for key in test_result_split.keys():
                    test_result_split[key] /= 5
                print(do_45(test_result_split))
            if config['do_test_sl']:
                print('1.0 1.0 1.0 1.0 1.0')
                val_result, _, _ = itm_eval(config, score_val_i2t, score_val_t2i, score_val_i2i, score_val_t2t,
                                    val_loader.dataset.txt2img, val_loader.dataset.img2txt, 
                                    device=device, test_sl_rate=1.0, tau_i2t=1.0, tau_t2t=1.0, 
                                    tau_t2i=1.0, tau_i2i=1.0, flag=True)
                print(do_45(val_result))
                test_result, _, _ = itm_eval(config, score_test_i2t, score_test_t2i, score_test_i2i, score_test_t2t,
                                    test_loader.dataset.txt2img, test_loader.dataset.img2txt, 
                                    device=device, test_sl_rate=1.0, tau_i2t=1.0, tau_t2t=1.0, 
                                    tau_t2i=1.0, tau_i2i=1.0, flag=True)
                print(do_45(test_result))
                if 'coco' in config['data_path']:
                    test_result_split = None
                    for i in range(5):
                        tmp, _, _ = itm_eval(config, score_test_i2t[i * 1000:(i + 1) * 1000, i * 5000:(i + 1) * 5000],
                                    score_test_t2i[i * 5000:(i + 1) * 5000, i * 1000:(i + 1) * 1000], 
                                    score_test_i2i[i * 1000:(i + 1) * 1000, i * 1000:(i + 1) * 1000], 
                                    score_test_t2t[i * 5000:(i + 1) * 5000, i * 5000:(i + 1) * 5000], 
                                    test_loader.dataset.txt2img, test_loader.dataset.img2txt, 
                                    device=device, test_sl_rate=1.0, tau_i2t=1.0, tau_t2t=1.0, 
                                    tau_t2i=1.0, tau_i2i=1.0, flag=True)
                        if test_result_split is None:
                            test_result_split = tmp
                        else:
                            for key in test_result_split.keys():
                                test_result_split[key] += tmp[key]

                    for key in test_result_split.keys():
                        test_result_split[key] /= 5
                    print(do_45(test_result_split))

            print("Train stats:", train_stats)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_result.items()},
                         **{f'test_{k}': v for k, v in test_result.items()},
                         'epoch': epoch,
                         }
            with open(os.path.join(config['logger_name'], "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if test_result['r_mean'] > best:
                save_obj = {
                    'model': model.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'best': best,
                    'best_epoch': best_epoch,
                }
                torch.save(save_obj, os.path.join(config['model_name'], 'checkpoint_best.pth'))
                best = test_result['r_mean']
                best_epoch = epoch
                np.save('tmp/score_val_i2t.npy', score_val_i2t)
                np.save('tmp/score_val_t2i.npy', score_val_t2i)
                np.save('tmp/score_val_i2i.npy', score_val_i2i)
                np.save('tmp/score_val_t2t.npy', score_val_t2t)
                np.save('tmp/score_test_i2t.npy', score_test_i2t)
                np.save('tmp/score_test_t2i.npy', score_test_t2i)
                np.save('tmp/score_test_i2i.npy', score_test_i2i)
                np.save('tmp/score_test_t2t.npy', score_test_t2t)

            save_obj = {
                'model': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'best': best,
                'best_epoch': best_epoch,
            }
            pri_save_obj = {
                'epoch': epoch,
                'best_mean': best,
                'best_rsum': best * 6,
                'best_epoch': best_epoch,
            }
            print(pri_save_obj)
            # torch.save(save_obj, os.path.join(
            #     config['model_name'], 'checkpoint_{}.pth'.format(str(epoch).zfill(2))))
        # synchronize()
        # dist.barrier()
        # # release gpu memory
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print('Training time {}'.format(total_time_str))

    if utils.is_main_process():
        with open(os.path.join(config['logger_name'], "log.txt"), "a") as f:
            f.write("best epoch: %d\n\n" % best_epoch)
    copy_file("/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_06/train.log", config['logger_name'])


def train(model, train_loader, optimizer, lr_scheduler, epoch, warmup_steps, device, config, txt_enc_assisant, tokenizer=None):
    model.train()

    # set metric logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss_contrastive', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_cross_modal', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_uni_modal', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metrics = [
        "tau",
        "cross_tau", "cross_tau_image", "cross_tau_text", "cross_the_softlabel_tau", "cross_the_softlabel_tau_image", "cross_the_softlabel_tau_text",
        "uni_tau", "uni_tau_image", "uni_tau_text", "uni_the_softlabel_tau", "uni_the_softlabel_tau_image", "uni_the_softlabel_tau_text"
    ]
    for val in metrics:
        if hasattr(model, val):
            metric_logger.add_meter(val, utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    for i, (image, caption, image_features, raw_captions, idx, raw_captions_r) in enumerate(train_loader):
        image = image.to(device, non_blocking=True)
        caption = caption.to(device, non_blocking=True)

        # softlabel feature for cross-modal retrieval and uni-modal retrieval
        with torch.no_grad():
            image_features = image_features.to(device, non_blocking=True)
            caption_features = txt_enc_assisant.encode(
                raw_captions, device=device, show_progress_bar=False, convert_to_tensor=True).to(device, non_blocking=True)
            caption_features_r = txt_enc_assisant.encode(
                raw_captions_r, device=device, show_progress_bar=False, convert_to_tensor=True).to(device, non_blocking=True)
            # raw_captions_r = [raw_captions_r[x][y] for y in range(128) for x in range(5)]
            # caption_features_r = txt_enc_assisant.encode(
            #     raw_captions_r, device=device, show_progress_bar=False, convert_to_tensor=True).to(device, non_blocking=True)
            # caption_features_r = caption_features_r.view(128, 5, 768).mean(dim=1)
        if config['backbone'] == 'X2VLM':
            caption = tokenizer(raw_captions, padding='longest', max_length=model.c['max_tokens'], return_tensors="pt").to(device)
        # get loss
        cross_modal_loss, uni_modal_loss, contrastive_loss, sl_loss = model(image, caption, image_features, caption_features, epoch, idx, caption_features_r)

        loss = cross_modal_loss + uni_modal_loss + contrastive_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update metric logger
        for val in metrics:
            if hasattr(model, val):
                metric_logger.update(**{val: getattr(model, val).item()})
        metric_logger.update(loss_cross_modal=cross_modal_loss.item())
        metric_logger.update(loss_uni_modal=uni_modal_loss.item())
        metric_logger.update(loss_contrastive=contrastive_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if i % print_freq == 0 or i == len(train_loader) - 1:
            log = ''
            for val in metrics:
                if hasattr(model, val):
                    log += f'{val}: {getattr(model, val).item():.6f} '
            log = f'loss: {loss.item():.6f} ' + log
            log = f'loss_cross_modal: {cross_modal_loss.item():.6f} ' + log
            log = f'loss_uni_modal: {uni_modal_loss.item():.6f} ' + log
            log = f'loss_contrastive: {contrastive_loss.item():.6f} ' + log
            log = f'lr: {optimizer.param_groups[0]["lr"]:.6f} ' + log
            print(f'epoch: {epoch} [{i}/{len(train_loader)}]', log)
            print('sl_loss:', sl_loss)
            print()
            sys.stdout.flush()
            # lr_tmp = []
            # for i, param_group in enumerate(optimizer.param_groups):
            #     for param in param_group['params']:
            #         for name, p in model.named_parameters():
            #             if p is param:
            #                 if param_group['lr'] in lr_tmp:
            #                     continue
            #                 else:
            #                     lr_tmp.append(param_group['lr'])
            #                 print(f"epoch: {epoch} Parameter name: {name} lr = {param_group['lr']:.6f}")
            #                 break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def copy_file(source_path, destination_path):
    """
    复制文件到指定位置
    
    参数:
        source_path: 源文件路径
        destination_path: 目标路径，可以是文件或目录
        
    返回:
        成功返回目标路径，失败抛出异常
    """
    try:
        # 确保目标目录存在
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        # 如果目标路径是目录，则在目录下创建同名文件
        if os.path.isdir(destination_path):
            destination_path = os.path.join(destination_path, os.path.basename(source_path))
        
        # 复制文件（保留元数据）
        shutil.copy2(source_path, destination_path)
        return destination_path
    except Exception as e:
        print(f"文件复制失败: {e}")
        raise

def parser_args():
    parser = argparse.ArgumentParser(description="PyTorch Image Retrieval Training")
    parser.add_argument('--config', type=str, default='', help='The config file.')
    parser.add_argument('--eval', action='store_true', help='Is eval?')
    parser.add_argument('--experiment', action='store_true', help='Is experiment?')
    parser.add_argument('--resume', action='store_true', help='Is resume?')
    parser.add_argument('--seed', default=23, type=int, help='Seed for initializing training.')
    parser.add_argument("--num_workers", default=8, type=int, help="The number of workers to use for data loading.")
    parser.add_argument('--distributed', default=True, type=bool, help='Is distributed?')
    parser.add_argument('--checkpoint', type=str, default='', help='The checkpoint file to resume from.')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # set env
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # set args
    args = parser_args()
    # set distributed
    utils.init_distributed_mode(args)

    assert not (args.config == '' and args.checkpoint == ''), "config and checkpoint cannot be empty at the same time"
    config = None
    if args.config != '':
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
            config['save_path'] = config['save_path'] + "_seed" + str(args.seed)
            config['logger_name'] = os.path.join(config['save_path'], "log")
            config['model_name'] = os.path.join(config['save_path'], "checkpoints")

    if args.resume and args.checkpoint == '':
        modelList = os.listdir(config['model_name'])
        modelList.sort()
        modelPath = modelList[-2]
        args.checkpoint = os.path.join(config['model_name'], modelPath)

    if utils.is_main_process():
        if not os.path.exists(config['save_path']):
            os.makedirs(config['save_path'])
        # Copy the configuration file to storage
        try:
            # If the file exists
            if os.path.exists(args.config):
                os.system("cp -f %s %s" % (args.config, os.path.join(config['save_path'])+"/"))
        except:
            pass
        for i in range(10000):
            tmp1 = config['model_name']+'_'+str(i)
            tmp2 = config['logger_name']+'_'+str(i)
            if not os.path.exists(tmp1) and not os.path.exists(tmp2):
                os.makedirs(tmp1)
                os.makedirs(tmp2)
                config['model_name'] = tmp1
                config['logger_name'] = tmp2
                print(config['logger_name'])
                with open(config['logger_name'] + '/config.yaml', 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                break
            if i >= 999:
                raise
    args.gpu = "cuda:1"
    import pynvml

    def get_gpu_remaining_memory(device_id=0):
        """使用 pynvml 获取指定 GPU 的真实剩余显存（MB）"""
        if not torch.cuda.is_available():
            return 0

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

        # 获取总显存和已用显存
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = info.total / 1024**2  # MB
        used = info.used / 1024**2
        free = info.free / 1024**2

        pynvml.nvmlShutdown()

        return {
            'total': total,
            'used': used,
            'free': free,
            'usage_percent': used / total * 100
        }

    # 示例：获取 GPU 0 的真实剩余显存
    itr = 0
    while True:
        gpu_info = get_gpu_remaining_memory(0)
        print('epoch: ', itr)
        print(f"GPU 0: 总显存 {gpu_info['total']:.2f} MB, "
              f"已使用 {gpu_info['used']:.2f} MB, "
              f"剩余 {gpu_info['free']:.2f} MB ({gpu_info['usage_percent']:.2f}%)")
        itr += 1
        sys.stdout.flush()
        if gpu_info['free'] >= 18000:
            break
        time.sleep(60)
    main(args, config)
