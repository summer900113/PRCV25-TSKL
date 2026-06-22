import json
import os
import traceback

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from clip import clip
from .utils import pre_caption
import pandas as pd
import random


class cross_coco_dataset(Dataset):
    def __init__(self, root, transform=None, split="train", max_words=64, config=None):
        self.root = root
        self.transform = transform
        self.split = split
        self.max_words = max_words
        self.config = config

        if 'CUB' not in root:
            self.dataPath = os.path.join(self.root, "new_{}.json".format(self.split))
            with open(self.dataPath, "r", encoding="utf8") as f:
                self.dataList = json.load(f)
        else:
            print(torch.load(root + '/metadata.pth').keys())
            data_imgs = torch.load(root + '/imgs_train_val_256x256.pth')
            data_meta = torch.load(root + '/metadata.pth')
            data_ids_train_val = torch.load(root + '/metadata.pth')['train_val_img_ids']
            self.dataList = []
            annotations = pd.read_csv(root + '/res.json', sep='\t', header=None)
            caps = np.array(annotations)
            for i in range(len(data_ids_train_val)):
                for k in range(len(data_meta['img_id_to_encoded_caps'][data_ids_train_val[i]])):
                    data_cap = [data_meta['word_id_to_word'][j] for j in data_meta['img_id_to_encoded_caps'][data_ids_train_val[i]][k] if j != 717]
                    tmp = {
                        'image_id': data_ids_train_val[i],
                        'caption': ' '.join(data_cap),
                        'caption_r': caps[i][0],
                        'class_id': data_meta['img_id_to_class_id'][data_ids_train_val[i]],
                        'img': data_imgs[i]
                    }
                    self.dataList.append(tmp)
            arg = [(i + t) * 10 + u for i in range(0, len(self.dataList) // 10 , 10) for t in range(9) for u in range(10)]
            self.dataList = [self.dataList[t] for t in arg if t < len(self.dataList)]
            
        self.img_ids = {}
        n = 0
        for ann in self.dataList:
            if 'CUB' not in root:
                img_id = ann["image_id"]
            else:
                img_id = ann["class_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        if self.split == "experiment":
            self.split = "train"
        try:
            self.unicom_fea = np.load(os.path.join(self.root, "{}_unicom.npy".format(self.split)), allow_pickle=True).item()
        except:
            self.unicom_fea = None

    def __len__(self):
        # return 200
        return len(self.dataList)

    def __getitem__(self, index):
        tmpData = self.dataList[index].copy()
        if 'VL-check' in self.root:
            tmpData["caption"] = self.dataList[index]["caption"][random.randint(0, len(self.dataList[index]) - 1)]
            tmpData["caption_r"] = self.dataList[index]["caption"][random.randint(0, len(self.dataList[index]) - 1)]
        
        caption = pre_caption(tmpData["caption"], self.max_words)

        raw_caption = caption
        if self.config['do_LLMs_ab']:
            # raw_caption_r = (self.dataList[index // 5 * 5]["caption"] +
            #                  self.dataList[index // 5 * 5 + 1]["caption"] +
            #                  self.dataList[index // 5 * 5 + 2]["caption"] +
            #                  self.dataList[index // 5 * 5 + 3]["caption"] +
            #                  self.dataList[index // 5 * 5 + 4]["caption"])
            # raw_caption_r = [pre_caption(self.dataList[index // 5 * 5 + i]["caption"], self.max_words) for i in range(5)]
            raw_caption_r = pre_caption(tmpData["caption"], self.max_words)
        else:
            raw_caption_r = pre_caption(tmpData["caption_r"], self.max_words)
        image_feature = torch.tensor([0.0])
        if self.unicom_fea is not None:
            image_feature = self.unicom_fea.get(tmpData["image_id"])

        caption = clip.tokenize(caption)[0]
        if 'CUB' not in self.root:
            im = Image.open(os.path.join(self.root, tmpData["image_path"])).convert('RGB')
        else:
            np_array = tmpData['img'].permute(1, 2, 0).numpy()
            im = Image.fromarray(np_array)
        im = self.transform(im)
        if 'CUB' not in self.root:
            return im, caption, image_feature, raw_caption, self.img_ids[tmpData["image_id"]], raw_caption_r
        else:
            return im, caption, image_feature, raw_caption, self.img_ids[tmpData["class_id"]], raw_caption
            # return im, caption, image_feature, raw_caption, self.img_ids[tmpData["class_id"]], raw_caption_r


class cross_coco_test_dataset(Dataset):
    def __init__(self, root, transform=None, split="test", max_words=64, i=None):
        self.root = root
        self.transform = transform
        self.split = split
        self.max_words = max_words
        self.dataPath = os.path.join(self.root, "new_{}.json".format(self.split))
        # self.dataPath = '/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_06/dataset/test/new_test.json'

        if 'CUB' not in root:
            with open(self.dataPath, "r", encoding="utf8") as f:
                """
                [{
                    "image_path": "COCO_val2014_000000184613.jpg",
                    "image_id": "184613",
                    "caption": "A young man holding an umbrella next to a herd of cattle ."
                }, ...]
                """
                if i is None:
                    self.dataList = json.load(f)
                else:
                    self.dataList = json.load(f)[i * 5000:i * 5000 + 5000]
            """
            {
                "<image_id>":{
                    "image_path": "COCO_val2014_000000184613.jpg",
                    "caption":[//5 captions]
                }
            }
            """
        else:
            print(torch.load(root + '/metadata.pth').keys())
            if split == 'val':
                data_imgs = torch.load(root + '/imgs_train_val_256x256.pth')
                data_ids_train_val = torch.load(root + '/metadata.pth')['train_val_img_ids']
            elif split == 'test':
                data_imgs = torch.load(root + '/imgs_test_256x256.pth')
                data_ids_train_val = torch.load(root + '/metadata.pth')['test_img_ids']
            else:
                raise

            data_meta = torch.load(root + '/metadata.pth')
            self.dataList = []
            for i in range(len(data_ids_train_val)):
                for k in range(len(data_meta['img_id_to_encoded_caps'][data_ids_train_val[i]])):
                    data_cap = [data_meta['word_id_to_word'][j] for j in data_meta['img_id_to_encoded_caps'][data_ids_train_val[i]][k] if j != 717]
                    tmp = {
                        'image_id': data_ids_train_val[i],
                        'caption': ' '.join(data_cap),
                        'class_id': data_meta['img_id_to_class_id'][data_ids_train_val[i]],
                        'image_path': data_imgs[i]
                    }
                    self.dataList.append(tmp)
            if split == 'val':
                arg = [(i + 9) * 10 + u for i in range(0, len(self.dataList) // 10 , 10) for u in range(10)]
                # arg = [(i + t) * 10 + u for i in range(0, len(self.dataList) // 10 , 10) for t in range(9) for u in range(10)]
                self.dataList = [self.dataList[t] for t in arg if t < len(self.dataList)]
        tmpData = {}
        if 'CUB' not in root:
            for val in self.dataList:
                if val.get("image_id") not in tmpData:
                    tmpData[val.get("image_id")] = {
                        "image_path": val.get("image_path"), "caption": [pre_caption(val.get("caption"), self.max_words)]}
                else:
                    tmpData[val.get("image_id")]["caption"].append(pre_caption(val.get("caption"), self.max_words))
            imgIdKeys = sorted(list(tmpData.keys()))
            self.text = []
            self.image = []
            self.img2txt = {}
            self.txt2img = {}
            txt_id = 0
            for id, key in enumerate(imgIdKeys):
                self.image.append(tmpData[key]["image_path"])
                self.img2txt[id] = []
                for tid, caption in enumerate(tmpData[key]["caption"]):
                    self.text.append(caption)
                    self.img2txt[id].append(txt_id)
                    self.txt2img[txt_id] = id
                    txt_id += 1
        else:
            import pandas as pd

            df = pd.DataFrame(self.dataList)
            self.text = df["caption"].tolist()
            self.image = df["image_path"].iloc[::10].tolist()
            self.img2txt = df["class_id"].iloc[::10].tolist()
            self.txt2img = df["class_id"].tolist()
        # print(tmpData[list(tmpData.keys())[1]])
        # np_array = tmpData[list(tmpData.keys())[1]]['image_path'].permute(1, 2, 0).numpy()
        # # 创建 PIL Image 并保存
        # image = Image.fromarray(np_array)
        # image.save("original_image.png")  # 保存为 PNG 格式
        # raise
        # sort image_id keys to keep the order of images

    def preprocess_text(self, textList):
        preCaptionList = clip.tokenize(textList, truncate=True)
        return preCaptionList

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        
        if 'CUB' not in self.root:
            im = Image.open(os.path.join(self.root, self.image[index])).convert('RGB')
        else:
            np_array = self.image[index].permute(1, 2, 0).numpy()
            im = Image.fromarray(np_array)
        
        im = self.transform(im)
        return im, index
