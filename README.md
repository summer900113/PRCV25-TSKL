# False Negatives Do Matter: A Novel Soft Label and Reranking Based Plug-in Method for Image-Text Retrieval
PRCV25 Accepted Paper：

**[False Negatives Do Matter: A Novel Soft Label and Reranking Based Plug-in Method for Image-Text Retrieval](https://link.springer.com/chapter/10.1007/978-981-95-5679-3_11)**

Heng-yang Lu & Yiyang Sung 

Abstract: *False negatives, where semantically relevant image-text pairs are incorrectly labeled as unmatched, present a significant challenge in image-text retrieval (ITR). However, existing approaches addressing this issue face two major limitations. First, the generated soft labels are often of poor quality. Second, reranking methods typically rely on access to the full set of queries and targets, which limits their applicability in various scenarios. To tackle the limitations of existing approaches, we propose Text Soft label and KL similarity (TSKL) method consists of two independent plug-in modules. In the training phase, the Text Soft Label (TSL) module generates soft similarity labels from image captions. In the testing phase, the KL similarity (KLsim) module. KLsim only requires the current query and all targets to compute similarity scores. Both plug-in modules can be easily integrated into existing ITR models. We evaluate TSKL on three widely used datasets, Flickr30K, MSCOCO, and CUB Captions. Experimental results show our method consistently improves the performance of base models by an average of 12.3% in terms of Rsum, and effectively mitigates the impact of false negatives by providing more accurate similarity rankings.*

![False Negative](false_negative.png)

## DataSets

Dataset source can be downloaded here.

- [Flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k). The data partitioning we used is this [this file](dataset/my_f30k). (unzip new_train.zip)
- [MS COCO](https://huggingface.co/datasets/shunk031/MSCOCO). The data partitioning we used is this [this file](dataset/my_coco). (unzip new_train.zip)
- [CUB200](https://huggingface.co/datasets/cassiekang/cub200_dataset). The data partitioning is in [the code](dataset/cross_coco_dataset.py).

## Prepare

Our code run on environment, the required list in [requirements.txt](requirements.txt). Please install the list before running.

## Fine-tunning

The Fine-tunning code is in [retrieval.py](retrieval.py). Run the code for fine-tunning:
```bash
python retrieval.py --config "./configs/vitb32/flickr/tskl.yaml"
```

## Evaluating

The Evaluating code is in [retrieval.py](retrieval.py). Run the code for evaluating:
```bash
python retrieval.py --eval --checkpoint ".pth" --config "./configs/vitb32/flickr/tskl.yaml"
```
## Citation

```bash
@inproceedings{lu2025false,
  title={False Negatives Do Matter: A Novel Soft Label and Reranking Based Plug-in Method for Image-Text Retrieval},
  author={Lu, Heng-yang and Sung, Yiyang},
  booktitle={Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  pages={149--163},
  year={2025},
  organization={Springer}
}
```
