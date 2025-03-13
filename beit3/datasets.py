# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
import json
import random
import torch
import glob
from collections import defaultdict, Counter
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data import create_transform

import utils
from glossary import normalize_word
from randaug import RandomAugment
import pandas as pd

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path, split, transform, 
        tokenizer, num_max_bpe_tokens, task=None,
    ):
        index_files = self.get_index_files(split, task=task)
        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.data_path = data_path
        items = []
        self.index_files = index_files

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(data_path, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                print("Load %d image-text pairs from %s. " % (len(items) - offset, index_file))
                offset = len(items)
        self.items = items
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.loader = default_loader
        self.transform = transform
        self.split = split

    @staticmethod
    def get_index_files(split):
        raise NotImplementedError()

    def _get_image(self, image_path: str):
        image_path = os.path.join(self.data_path, image_path)
        image = self.loader(image_path)
        return self.transform(image)

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens

    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text_segment"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = '{' + "\n  Number of items: %s," % self.__len__()
        body += "\n  data root = %s," % self.data_path
        body += "\n  split = %s," % self.split
        body += "\n  dataset index files = %s" % str(self.index_files)
        body += "\n  num max bpe tokens = %s" % self.num_max_bpe_tokens
        body += "\n  transforms = ["
        for t in self.transform.transforms:
            body += "\n    %s" % str(t)
        body += "\n  ]"
        body += "\n}"

        return head + body


def _write_data_into_jsonl(items, jsonl_file):
    with open(jsonl_file, mode="w", encoding="utf-8") as writer:
        for data in items:
            writer.write(json.dumps(data, indent=None))
            writer.write('\n')
    print("Write %s with %d items !" % (jsonl_file, len(items)))


def _make_retrieval_coco_karpathy_dataset_index(
        data_path, 
        tokenizer, 
        split=("train", "restval"), 
        split_name="train", 
):
    coco_karpathy_split_json_file = os.path.join(data_path, "dataset_coco.json")
    items = []
    image_counter = set()
    print("read %s" % coco_karpathy_split_json_file)
    with open(coco_karpathy_split_json_file, mode="r", encoding="utf-8") as reader:
        data = json.loads(reader.read())
        for item in data["images"]:
            if item["split"] in split:
                image_path = os.path.join(item["filepath"], item["filename"])
                for sent in item["sentences"]:
                    tokens = tokenizer.tokenize(sent["raw"])
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    items.append({
                            "image_path": image_path, 
                            "text_segment": token_ids, 
                            "image_id": len(image_counter), 
                    })
                if image_path not in image_counter:
                    image_counter.add(image_path)
    print("Find %d images and %d image-text pairs for karpathy dataset %s split !" % \
        (len(image_counter), len(items), split_name))
    index_file = os.path.join(data_path, "coco_retrieval.%s.jsonl" % split_name)
    _write_data_into_jsonl(items, index_file)
    pass


def _make_captioning_coco_karpathy_dataset_index(
        data_path, 
        tokenizer, 
        split=("train", "restval"), 
        split_name="train", 
):
    coco_karpathy_split_json_file = os.path.join(data_path, "dataset_coco.json")
    items = []
    image_counter = set()
    print("read %s" % coco_karpathy_split_json_file)
    with open(coco_karpathy_split_json_file, mode="r", encoding="utf-8") as reader:
        data = json.loads(reader.read())
        for item in data["images"]:
            if item["split"] in split:
                image_path = os.path.join(item["filepath"], item["filename"])
                if item["split"] in ["train", "restval"]:
                    for sent in item["sentences"]:
                        tokens = tokenizer.tokenize(sent["raw"])
                        token_ids = tokenizer.convert_tokens_to_ids(tokens)
                        items.append({
                                "image_path": image_path, 
                                "text_segment": token_ids, 
                                "image_id": item["cocoid"], 
                        })
                else:
                    items.append({
                                "image_path": image_path, 
                                "text_segment": None, 
                                "image_id": item["cocoid"], 
                    })
                if image_path not in image_counter:
                    image_counter.add(image_path)
    print("Find %d images and %d image-text pairs for karpathy dataset %s split !" % \
        (len(image_counter), len(items), split_name))
    index_file = os.path.join(data_path, "coco_captioning.%s.jsonl" % split_name)
    _write_data_into_jsonl(items, index_file)
    pass


def _make_nocaps_dataset_index(
        data_path,  
        split="val", 
):
    if split == "val":
        json_file = "nocaps_val_4500_captions.json"
    elif split == "test":
        json_file = "nocaps_test_image_info.json"
    nocaps_split_json_file = os.path.join(data_path, json_file)
    items = []
    image_counter = set()
    print("read %s" % nocaps_split_json_file)
    with open(nocaps_split_json_file, mode="r", encoding="utf-8") as reader:
        data = json.loads(reader.read())
        for item in data["images"]:
            image_path = os.path.join(split, item["file_name"])
            items.append({
                "image_path": image_path, 
                "text_segment": None, 
                "image_id": item["id"], 
            })

            if image_path not in image_counter:
                image_counter.add(image_path)

    print("Find %d images and %d image-text pairs for nocaps dataset %s split !" % \
        (len(image_counter), len(items), split))
    index_file = os.path.join(data_path, "nocaps.%s.jsonl" % split)
    _write_data_into_jsonl(items, index_file)


class NLVR2Dataset(BaseDataset):
    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("nlvr2.train.index.jsonl", )
        elif split == "val":
            return ("nlvr2.dev.index.jsonl", )
        elif split == "test":
            return ("nlvr2.test-P.index.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        item = self.items[index]
        img_path = item["image2_path"]
        img = self._get_image(img_path)
        data["image2"] = img
        data["label"] = self.items[index]["label"]
        return data

    @staticmethod
    def __preprocess_json(preifx, json_file, tokenizer, index_file):
        items = []
        with open(json_file, mode="r", encoding="utf-8") as reader:
            for line in reader:
                data = json.loads(line)
                path = os.path.join(preifx, str(data["directory"])) if "directory" in data else preifx
                path = os.path.join(path, "-".join(data["identifier"].split("-")[:-1]))
                tokens = tokenizer.tokenize(data["sentence"])
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                items.append({
                    "image_path": path + "-img0.png",
                    "image2_path": path + "-img1.png",
                    "text_segment": token_ids,
                    "label": 1 if data["label"] == "True" else 0,
                    "identifier": data["identifier"], 
                })
        _write_data_into_jsonl(items, index_file)

    @classmethod
    def make_dataset_index(cls, data_path, tokenizer, nlvr_repo_path):
        cls.__preprocess_json(
            preifx="images/train", json_file=os.path.join(nlvr_repo_path, "nlvr2/data/train.json"), 
            tokenizer=tokenizer, index_file=os.path.join(data_path, cls.get_index_files("train")[0]), 
        )
        cls.__preprocess_json(
            preifx="dev", json_file=os.path.join(nlvr_repo_path, "nlvr2/data/dev.json"), 
            tokenizer=tokenizer, index_file=os.path.join(data_path, cls.get_index_files("val")[0]), 
        )
        cls.__preprocess_json(
            preifx="test1", json_file=os.path.join(nlvr_repo_path, "nlvr2/data/test1.json"), 
            tokenizer=tokenizer, index_file=os.path.join(data_path, cls.get_index_files("test")[0]), 
        )


class ImageNetDataset(BaseDataset):
    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("imagenet.train.index.jsonl", )
        elif split == "val":
            return ("imagenet.val.index.jsonl", )
        elif split == "test":
            return ("imagenet.val.index.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["label"] = item["label"]
        return data
    
    @staticmethod
    def _find_classes(dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    @staticmethod
    def _make_imagenet_index(data_path, index_path, data_path_prefix, class_to_idx, split):
        items = []
        index_file = os.path.join(index_path, f"imagenet.{split}.index.jsonl")
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(data_path, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    path = path.replace(data_path_prefix, "")
                    items.append({
                        "image_path": path,
                        "label": class_index,
                    })

        _write_data_into_jsonl(items, index_file)

    @classmethod
    def make_dataset_index(cls, train_data_path, val_data_path, index_path):
        data_path_prefix = train_data_path[:[x[0]==x[1] for x in zip(train_data_path, val_data_path)].index(0)]
        classes, class_to_idx = cls._find_classes(train_data_path)
        cls._make_imagenet_index(
             data_path=train_data_path, index_path=index_path, data_path_prefix=data_path_prefix,
             class_to_idx=class_to_idx, split="train",
        )
        cls._make_imagenet_index(
             data_path=val_data_path, index_path=index_path, data_path_prefix=data_path_prefix,
             class_to_idx=class_to_idx, split="val",
        )

class VQADataset(BaseDataset):
    def __init__(self, data_path, split, transform, root_folder, tokenizer, num_max_bpe_tokens):
        num_sample = -1
        if split == 'train':
            num_sample = -1
        elif split == 'val':
            num_sample = 2000
        else:
            num_sample = 1000
        self.dataframe = pd.read_csv(os.path.join(data_path, f"{split}.csv"))[:num_sample]
        self.dataframe.dropna(inplace=True)
        if split == 'train':
            df = pd.read_csv(os.path.join(data_path, f"data.csv"))
            unique_answers = set(df["answer"].tolist())
            self.answer2id = {ans: i for i, ans in enumerate(unique_answers)}
            self.id2answer = {i: ans for i, ans in enumerate(unique_answers)}
            with open("answer2label.json", mode="w", encoding="utf-8") as writer:
                writer.write(json.dumps(self.answer2id))
            with open("label2lanswer.json", mode="w", encoding="utf-8") as writer:
                writer.write(json.dumps(self.id2answer))
        else:
            with open("answer2label.json", mode="r", encoding="utf-8") as f:
                self.answer2id = json.load(f)
            with open("label2lanswer.json", mode="r", encoding="utf-8") as f:
                self.id2answer = json.load(f)
        self.root_folder = root_folder
        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.loader = default_loader
        self.transform = transform

    def _get_image(self, image_path: str):
        image_path = os.path.join(self.root_folder, image_path)
        image = self.loader(image_path)
        return self.transform(image)

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens

    def _get_image_text_example(self, index: int, data: dict):
        item = self.dataframe.iloc[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        question_text = item["question"]
        if type(question_text) is not str:
            print(question_text)
        tokens = self.tokenizer.tokenize(question_text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        language_tokens, padding_mask, _ = self._get_text_segment(token_ids)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        item = self.dataframe.iloc[index]
        answer = item["answer"]
        labels = [0.] * len(self.answer2id)
        labels[self.answer2id[answer]] = 1
        data["labels"] = torch.FloatTensor(labels)
        return data

class RetrievalDataset(BaseDataset):
    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return (f"{task}.train.jsonl", )
        elif split == "val":
            return (f"{task}.val.jsonl", )
        elif split == "test":
            return (f"{task}.test.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        data["image_id"] = self.items[index]["image_id"]
        return data

    @staticmethod
    def make_flickr30k_dataset_index(data_path, tokenizer, karpathy_path):

        with open(os.path.join(karpathy_path, "dataset_flickr30k.json"), "r") as reader:
            captions = json.loads(reader.read())

        captions = captions["images"]
        split2items = defaultdict(list)
        split2images = defaultdict(set)

        for each_item in captions:
            image_path = os.path.join("flickr30k-images", each_item["filename"])
            split = each_item["split"]

            for text_segment in each_item["sentences"]:
                tokens = tokenizer.tokenize(text_segment["raw"])
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                split2items[split].append({
                    "image_path": image_path, 
                    "text_segment": token_ids, 
                    "image_id": len(split2images[split]), 
                })

            assert each_item["filename"] not in split2images[split]
            split2images[split].add(each_item["filename"])

        for split in split2items:
            print("%d images and %d image-text pairs!" % (len(split2images[split]), len(split2items[split])))
            _write_data_into_jsonl(split2items[split], os.path.join(data_path, "flickr30k.%s.jsonl" % split))

    @staticmethod
    def make_coco_dataset_index(data_path, tokenizer):
        _make_retrieval_coco_karpathy_dataset_index(data_path, tokenizer, split=("train", "restval"), split_name="train")
        _make_retrieval_coco_karpathy_dataset_index(data_path, tokenizer, split=("val", ), split_name="val")
        _make_retrieval_coco_karpathy_dataset_index(data_path, tokenizer, split=("test", ), split_name="test")


class CaptioningDataset(BaseDataset):

    def __init__(self, data_path, split, transform, 
                tokenizer, num_max_bpe_tokens, task, mask_prob):
        super().__init__(
            data_path=data_path, split=split, 
            transform=transform, tokenizer=tokenizer, 
            num_max_bpe_tokens=num_max_bpe_tokens, task=task, 
        )
        self.mask_token_id = tokenizer.mask_token_id
        self.language_vocab_size = tokenizer.vocab_size
        self.mask_prob = mask_prob

    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("coco_captioning.train.jsonl", )
        elif split == "val":
            return (f"{task}.val.jsonl", )
        elif split == "test":
            return (f"{task}.test.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def _get_mask_token(self, token):
        p = random.random()
        if p < 0.8:
            return self.mask_token_id
        elif p < 0.9:
            return token
        else:
            return random.randint(3, self.language_vocab_size - 1)

    def _masking_on_text_tokens(self, tokens, num_tokens, mask_prob):
        bool_masked_pos = [0] * len(tokens)
        to_mask = min(int(num_tokens * mask_prob + 0.5), num_tokens - 1)
        to_mask = max(to_mask, 1)
        num_masked_tokens = 0
        while num_masked_tokens < to_mask:
            i = random.randint(1, num_tokens - 1)
            if bool_masked_pos[i] == 0:
                bool_masked_pos[i] = 1
                tokens[i] = self._get_mask_token(tokens[i])
                num_masked_tokens += 1

        return tokens, bool_masked_pos

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["image_id"] = item["image_id"]

        text_segment = item["text_segment"]
        if text_segment is not None:
            language_tokens, padding_mask, num_tokens = self._get_text_segment(text_segment)
            masked_tokens = language_tokens[:]
            masked_tokens, language_masked_pos = \
                self._masking_on_text_tokens(masked_tokens, num_tokens, self.mask_prob)
            data["language_tokens"] = language_tokens
            data["masked_tokens"] = masked_tokens
            data["language_masked_pos"] = language_masked_pos
            data["padding_mask"] = padding_mask
        return data

    @staticmethod
    def make_coco_captioning_dataset_index(data_path, tokenizer):
        _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("train", "restval"), split_name="train")
        _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("val", ), split_name="val")
        _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("test", ), split_name="test")

    @staticmethod
    def make_nocaps_captioning_dataset_index(data_path):
        _make_nocaps_dataset_index(data_path, split="val")
        _make_nocaps_dataset_index(data_path, split="test")


task2dataset = {
    "nlvr2": NLVR2Dataset, 
    "vqav2": VQADataset, 
    "flickr30k": RetrievalDataset, 
    "coco_retrieval": RetrievalDataset,  
    "coco_captioning": CaptioningDataset,
    "nocaps": CaptioningDataset,
    "imagenet": ImageNetDataset,
}


def create_dataloader(dataset, is_train, batch_size, num_workers, pin_mem, dist_eval=False):
    if is_train or dist_eval:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        if not is_train and dist_eval and len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')

        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        collate_fn=utils.merge_batch_tensors_by_dict_key,
    )


def build_transform(is_train, args):
    if args.task in ["imagenet"]:
        return build_imagenet_transform(is_train, args)

    if is_train:
        t = [
            RandomResizedCropAndInterpolation(args.input_size, scale=(0.5, 1.0), interpolation=args.train_interpolation), 
            transforms.RandomHorizontalFlip(),
        ]
        if args.randaug:
            t.append(
                RandomAugment(
                    2, 7, isPIL=True, 
                    augs=[
                        'Identity','AutoContrast','Equalize','Brightness','Sharpness', 
                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 
                    ]))
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD), 
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size), interpolation=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    return t


def build_imagenet_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def get_sentencepiece_model_for_beit3(args):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(args.sentencepiece_model)


def create_dataset_by_split(args, split, is_train=True):
    transform = build_transform(is_train=is_train, args=args)
    dataset_class = task2dataset[args.task]
    tokenizer = get_sentencepiece_model_for_beit3(args)

    opt_kwargs = {}
    if args.task in ["coco_captioning", "nocaps"]:
        opt_kwargs["mask_prob"] = args.captioning_mask_prob

    dataset = dataset_class(
        data_path=args.data_path, split=split, 
        transform=transform, root_folder=args.root_folder, tokenizer=tokenizer, 
        num_max_bpe_tokens=args.num_max_bpe_tokens, **opt_kwargs, 
    )
    if is_train:
        batch_size = args.batch_size
        if split == 'val':
            batch_size *= 2
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.5)

    return create_dataloader(
        dataset, is_train=is_train, batch_size=batch_size, 
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval, 
    )


def create_downstream_dataset(args, is_eval=False):
    if is_eval:
        return create_dataset_by_split(args, split="test", is_train=False)
    else:
        return \
            create_dataset_by_split(args, split="train", is_train=True), \
            create_dataset_by_split(args, split="val", is_train=True)