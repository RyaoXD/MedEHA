# from MGCA
import os
import pickle
import re
import numpy as np
import pandas as pd
import json
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from medeha.constants import *
from medeha.datasets.utils import get_imgs
from tqdm import tqdm
from transformers import BertTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 entity_extraction=False,
                 imsize=256, max_words=112, sent_num=3,
                 sentence_token: bool = False,
                 max_report_sents: int = 30,
                 max_words_sent: int = 30,
                 **kwargs):
        super().__init__()
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")

        self.transform = transform
        self.imsize = imsize
        self.entity_extraction = entity_extraction
        task = ''
        self.tokenizer = BertTokenizer.from_pretrained(str(BERT_DIR))
        self.max_words = max_words
        self.sentence_token = bool(sentence_token and entity_extraction)
        self.max_report_sents = int(max_report_sents)
        self.max_words_sent = int(max_words_sent)
        if entity_extraction:
            entity_csv = str(MIMIC_CXR_MASTER_ENTITY_CSV)
            if not os.path.isfile(entity_csv):
                raise RuntimeError(
                    f"Entity pretrain requires {entity_csv} "
                    f"(impression, findings, and Label_index)."
                )
            self.df = pd.read_csv(entity_csv)
            self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
            self.rad_graph_results = np.load(
                ENTITY_DIR / "landmark_observation_adj_mtx.npy").astype(int)
            self.obs_book = json.load(open(
                ENTITY_DIR / "observation_explanation.json", "r"))
            self.ana_book = json.load(open(
                ENTITY_DIR / "anaomy_explanation.json", "r"))
            
            obs_names = list(self.obs_book.keys())
            ana_names = list(self.ana_book.keys())
            self.obs_to_idx = {name: idx for idx, name in enumerate(obs_names)}
            self.ana_to_idx = {name: idx for idx, name in enumerate(ana_names)}
            task='_entity'
        else:
            self.rad_graph_results = None
            self.df = pd.read_csv(MIMIC_CXR_MASTER_CSV)
            self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
            # self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            #     lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:])))

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split, task=task)
        
        self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)

    def load_text_data(self, split, task=''):
        # get study to captions mapping
        filepath = os.path.join(
            BASE_DIR, f"../../data/captions{task}.pickle")
        if not os.path.isfile(filepath):
            print(
                f"Caption file {filepath} does not exit. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        for row in self.df.itertuples():
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
            path = getattr(row, MIMIC_CXR_PATH_COL)
            if cur_split == split and path in path2sent:
                filenames.append(path)

        return filenames, path2sent

    def create_path_2_sent_mapping(self):
        path2sent = self._create_path2sent_regex_csv()
        if self.entity_extraction:
            return self._attach_entity_labels(path2sent)
        return path2sent

    def _create_path2sent_regex_csv(self):
        """Legacy: impression+findings from CSV, split by numbering / periods + NLTK."""
        path2sent = {}
        sent_lens, num_sents = [], []
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            captions = ""
            captions += row["impression"]
            captions += " "
            captions += row["findings"]
            captions = captions.replace("\n", " ")
            splitter = re.compile("[0-9]+\\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            for cap in captions:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())
                if len(tokens) <= 1:
                    continue
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))
                cnt += len(included_tokens)

            if cnt >= 3:
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[row[MIMIC_CXR_PATH_COL]] = {"text": study_sent}

        sent_lens = np.array(sent_lens) if sent_lens else np.array([0])
        num_sents = np.array(num_sents) if num_sents else np.array([0])
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} "
            f"[{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} "
            f"[{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent

    def _attach_entity_labels(self, path2sent):
        num_entities = []
        out = {}
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            obs, ana, sta = [], [], []
            entity_matrix = self.rad_graph_results[row["Label_index"]]
            for x, y in np.argwhere(entity_matrix != -1):
                obs.append(y)
                ana.append(x)
                sta.append(entity_matrix[x, y])
            if len(obs) >= 1:
                num_entities.append(len(obs))
                p = row[MIMIC_CXR_PATH_COL]
                if p not in path2sent:
                    continue
                out[p] = {
                    "text": path2sent[p]["text"],
                    "entity": [obs, ana, sta],
                }
        num_entities = np.array(num_entities) if num_entities else np.array([], dtype=int)
        if num_entities.size > 0:
            print(
                f"entity count: {num_entities.min()},{num_entities.mean()},{num_entities.max()} "
                f"[{np.percentile(num_entities, 5)}, {np.percentile(num_entities, 95)}]"
            )
        else:
            print("entity count: (no rows after report + entity filter)")
        return out

    def __len__(self):
        return len(self.filenames)

    def get_caption(self, path):
        series_sents = self.path2sent[path]['text']
        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))

        if self.entity_extraction and self.sentence_token:
            # CoMA-style: one BERT row per short sentence (truncated separately)
            series_sents = series_sents[: self.max_report_sents]
            tokens = self.tokenizer(
                series_sents,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_words_sent,
            )
            x_len = len(series_sents)
        else:
            sent = " ".join(series_sents)
            tokens = self.tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_words,
            )
            x_len = [len([t for t in tokens["input_ids"][0] if t != 0])]

        if self.entity_extraction:
            series_entities = self.path2sent[path]['entity']
            entitys = torch.tensor(series_entities)
            if not self.sentence_token:
                x_len = entitys.shape[1]
        else:
            entitys = []
            
        return tokens, entitys, x_len

    def __getitem__(self, index):
        key = self.filenames[index]
        caps, entitys, cap_len = self.get_caption(key)
        path = str(MIMIC_CXR_DATA_DIR / key.replace("2.0.0/", ""))
        imgs = get_imgs(path, self.imsize, self.transform, multiscale=False)
        return imgs, caps, entitys, cap_len, key, self.entity_extraction, self.sentence_token


def multimodal_collate_fn(batch):
    """Stack batch tensors. In entity mode, ``entindex`` is one batch-study id per entity
    (length = total entities); ``reportindex`` is one batch-study id per BERT caption row
    (length = B if one encoding per study, else total short sentences when ``sentence_token``)."""
    isentity = batch[0][5]
    sentence_token = batch[0][6] if len(batch[0]) > 6 else False
    imgs, ids, tokens, attention, entids = [], [], [], [], []
    entindex_sub = []
    for i, b in enumerate(batch):
        img, cap, ent, cap_l, _, isentity_i, sentence_token_i = b
        imgs.append(img)
        # if len(cap)==3 and not entity:
        #     entity = True
        ids.append(cap["input_ids"])
        tokens.append(torch.zeros_like(cap["token_type_ids"]))
        attention.append(cap["attention_mask"])
        if isentity:
            entids.append(ent)
            n_ent = int(ent.shape[1])
            entindex_sub.append([i for _ in range(n_ent)])

    # stack
    imgs = torch.stack(imgs)

    if sentence_token and isentity:
        ids = torch.cat(ids, dim=0)
        tokens = torch.cat(tokens, dim=0)
        attention = torch.cat(attention, dim=0)
    else:
        ids = torch.vstack(ids).squeeze()
        tokens = torch.vstack(tokens).squeeze()
        attention = torch.vstack(attention).squeeze()
    if isentity:
        entids = torch.cat(entids, dim=1)
        entindex = torch.tensor(
            [j for sub in entindex_sub for j in sub], dtype=torch.long
        )

    return_dict = {
        "caption_ids": ids,
        "token_type_ids": tokens,
        "attention_mask": attention,
        "entity_ids": entids,
        "imgs": imgs,
    }
    if isentity:
        return_dict["entindex"] = entindex
        if sentence_token:
            report_rows = []
            for i, b in enumerate(batch):
                cap_l = b[3]
                report_rows.append([i for _ in range(cap_l)])
            return_dict["reportindex"] = torch.tensor(
                [j for sub in report_rows for j in sub], dtype=torch.long
            )
            return_dict["caption_is_sentence"] = True
        else:
            return_dict["reportindex"] = torch.arange(len(batch), dtype=torch.long)
            return_dict["caption_is_sentence"] = False
    else:
        return_dict["reportindex"] = []
        return_dict["caption_is_sentence"] = False
    return return_dict


if __name__ == "__main__":
    from .transforms import DataTransforms
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="train", transform=transform)
    data = dataset[0]
    print(data)
