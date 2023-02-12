import torch
from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    def __init__(self, source_path, target_path, tokenizer, source_max_length, target_max_length, prefix, device):
        super().__init__()
        with open(source_path) as source_file, open(target_path) as target_file:
            self.source_data = [prefix+line.strip() for line in source_file.readlines()]
            self.target_data = [line.strip() for line in target_file.readlines()]
        self.source_max_len = source_max_length
        self.target_max_len = target_max_length
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, item):
        source_text = self.source_data[item]
        target_text = self.target_data[item]
        source = self.tokenizer.batch_encode_plus(
            [source_text],
            padding="max_length",
            return_length=False,
            return_tensors="pt",
            truncation=True,
            max_length=self.source_max_len
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            padding="max_length",
            return_length=False,
            return_tensors="pt",
            truncation=True,
            max_length=self.target_max_len
        )
        target_ids = target["input_ids"][:, :-1]
        lm_labels = target["input_ids"][:, 1:].clone().detach()
        lm_labels[target["input_ids"][:, 1:] == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": source["input_ids"].squeeze().to(self.device, dtype=torch.long),
            "decoder_input_ids": target_ids.squeeze().contiguous().to(self.device, dtype=torch.long),
            "attention_mask": source["attention_mask"].squeeze().to(self.device, dtype=torch.long),
            "labels": lm_labels.to(self.device)
                }

