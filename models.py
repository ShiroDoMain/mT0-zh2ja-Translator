import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import Config
from data import Data


class ModelLoader:
    def __init__(self, config):
        self.accelerator = None
        self.config = Config(config)
        self.device = self.config.device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model).to(self.device)
        if self.config.load_epoch is not None:
            self.load()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model,
                                                       max_length=self.config.model_max_length).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.lr)
        self.scheduler = None

    def load_data(self):
        data_path = self.config.data_path
        train_data = Data(os.path.join(data_path, f"train.{self.config.source_lang}"),
                          os.path.join(data_path, f"train.{self.config.target_lang}"),
                          self.tokenizer,
                          self.config.source_max_length,
                          self.config.target_max_length,
                          self.config.prefix,
                          self.device)
        val_data = Data(os.path.join(data_path, f"val.{self.config.source_lang}"),
                        os.path.join(data_path, f"val.{self.config.target_lang}"),
                        self.tokenizer,
                        self.config.source_max_length,
                        self.config.target_max_length,
                        self.config.prefix,
                        self.device)

        train = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True,
                           num_workers=self.config.num_workers)
        val = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=True,
                         num_workers=self.config.num_workers)
        return train, val

    def save(self, epoch):
        save_path = os.path.join(self.config.save_path, f"epoch_{epoch}")
        if self.config.use_accelerate:
            self.accelerator.wait_for_everyone()
            unwrapped = self.accelerator.unwrap_model(self.model)
            self.accelerator.save_state(save_path)
            self.accelerator.save(unwrapped.state_dict(), save_path+"_.pt")
        else:
            torch.save(self.model.state_dict(), os.path.join(self.config.save_path, f"model_{epoch}.pt"))

    def load(self):
        print(f"load model from epoch {self.config.load_epoch}")
        self.model.load_state_dict(
            torch.load(os.path.join(self.config.save_path, f"model_{self.config.load_epoch}.pt"))
        ).to(self.device)

    def init_accelerate(self, accelerator, data):
        self.accelerator = accelerator
        self.model, self.optimizer, self.scheduler, data = accelerator.prepare(
            self.model, self.optimizer, self.scheduler, data
        )
        return data

    def step(self):
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def zero_grad(self):
        self.optimizer.step()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
