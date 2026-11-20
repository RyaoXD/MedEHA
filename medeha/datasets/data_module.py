# from MGCA
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, collate_fn, transforms, data_pct, batch_size, num_workers, valid_mode="valid",
                 entity_extraction=False, crop_size=224, random_state=42,
                 sentence_token=False, max_report_sents=30, max_words_sent=30):
        super().__init__()

        self.dataset = dataset
        self.collate_fn = collate_fn
        self.transforms = transforms
        self.data_pct = data_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_mode = valid_mode
        self.entity_extraction = entity_extraction
        self.crop_size = crop_size
        self.random_state = random_state
        self.sentence_token = bool(sentence_token and entity_extraction)
        self.max_report_sents = max_report_sents
        self.max_words_sent = max_words_sent

    def train_dataloader(self):
        if self.transforms:
            transform = self.transforms(True, self.crop_size)
        else:
            transform = None
        
        dataset = self.dataset(
            split="train", transform=transform,
            data_pct=self.data_pct, entity_extraction=self.entity_extraction,
            random_state=self.random_state,
            sentence_token=self.sentence_token,
            max_report_sents=self.max_report_sents,
            max_words_sent=self.max_words_sent,
        )

        print("Train dataset size:", len(dataset))
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            split=self.valid_mode, transform=transform,
            data_pct=self.data_pct, entity_extraction=self.entity_extraction,
            random_state=self.random_state,
            sentence_token=self.sentence_token,
            max_report_sents=self.max_report_sents,
            max_words_sent=self.max_words_sent,
        )
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            split="test", transform=transform,
            data_pct=self.data_pct, entity_extraction=self.entity_extraction,
            random_state=self.random_state,
            sentence_token=self.sentence_token,
            max_report_sents=self.max_report_sents,
            max_words_sent=self.max_words_sent,
        )
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )