import re
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from pathlib import Path
from datasets import load_dataset, Dataset as HFDataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import lightning as L


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def process_dataset(ds: HFDataset, config):
    cleaned_data = []
    for item in ds:
        source = item["translation"][config["lang_src"]]
        target = item["translation"][config["lang_tgt"]]
        source_tokens = re.findall(r"\w+|[^\w\s]+", source)
        target_tokens = re.findall(r"\w+|[^\w\s]+", target)
        if len(source_tokens) > 150:
            continue
        if len(target_tokens) > len(source_tokens) + 10:
            continue
        cleaned_data.append({"id": item["id"], "translation": item["translation"]})
    return cleaned_data


class BilingualDataset(Dataset):
    def __init__(
        self,
        ds,
        tokenizer_src,
        tokenizer_tgt,
        src_lang,
        tgt_lang,
        src_seq_len,
        tgt_seq_len,
    ) -> None:
        super().__init__()
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.src_seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.tgt_seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.src_seq_len
        assert decoder_input.size(0) == self.tgt_seq_len
        assert label.size(0) == self.tgt_seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int()
            & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


class TranslationDataModule(L.LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        self.ds_raw = load_dataset(
            "opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train"
        )
        cleaned_data = process_dataset(self.ds_raw, config)
        self.ds_raw = HFDataset.from_list(mapping=cleaned_data)

        self.tokenizer_src = self.get_or_build_tokenizer(
            config=config, ds=self.ds_raw, lang=config["lang_src"]
        )
        self.tokenizer_tgt = self.get_or_build_tokenizer(
            config=config, ds=self.ds_raw, lang=config["lang_tgt"]
        )

        self.train_ds_size = int(0.9 * len(self.ds_raw))
        self.val_ds_size = len(self.ds_raw) - self.train_ds_size

    def get_all_sentences(self, ds, lang):
        for item in ds:
            yield item["translation"][lang]

    def get_or_build_tokenizer(self, config, ds, lang):
        tokenizer_path = Path(config["tokenizer_file"].format(lang))
        if not Path.exists(tokenizer_path):
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(
                special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
            )
            tokenizer.train_from_iterator(
                self.get_all_sentences(ds, lang), trainer=trainer
            )
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer

    def prepare_data(self):
        pass

    def get_seq_lens(self):
        max_len_src = 0
        max_len_tgt = 0
        for item in self.ds_raw:
            src_ids = self.tokenizer_src.encode(
                item["translation"][self.config["lang_src"]]
            ).ids
            tgt_ids = self.tokenizer_tgt.encode(
                item["translation"][self.config["lang_tgt"]]
            ).ids

            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))
        self.src_seq_len = max_len_src + 2
        self.tgt_seq_len = max_len_tgt + 1
        print(f"Max length of source sentence: {max_len_src}")
        print(f"Max length of target sentence: {max_len_tgt}")
        return max_len_src, max_len_tgt

    def setup(self, stage=None):
        train_ds_raw, val_ds_raw = random_split(
            self.ds_raw, [self.train_ds_size, self.val_ds_size]
        )

        self.train_dataset = BilingualDataset(
            ds=train_ds_raw,
            tokenizer_src=self.tokenizer_src,
            tokenizer_tgt=self.tokenizer_tgt,
            src_lang=self.config["lang_src"],
            tgt_lang=self.config["lang_tgt"],
            src_seq_len=self.src_seq_len,
            tgt_seq_len=self.tgt_seq_len,
        )

        self.val_dataset = BilingualDataset(
            ds=val_ds_raw,
            tokenizer_src=self.tokenizer_src,
            tokenizer_tgt=self.tokenizer_tgt,
            src_lang=self.config["lang_src"],
            tgt_lang=self.config["lang_tgt"],
            src_seq_len=self.src_seq_len,
            tgt_seq_len=self.tgt_seq_len,
        )

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.config["batch_size"], shuffle=True
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset, batch_size=1, shuffle=True)
        return val_dataloader
