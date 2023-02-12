from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MT5ForConditionalGeneration, MT5Tokenizer
from data import Data
from config import Config
import torch
from torch.utils.data import DataLoader
import os
from torch.optim import Adam


class ModelLoader:
    def __init__(self, config):
        self.config = Config(config)
        self.model = MT5ForConditionalGeneration.from_pretrained(self.config.model)
        if self.config.load_epoch is not None:
            self.load()
        self.model.to(self.config.device)
        self.tokenizer = MT5Tokenizer.from_pretrained(self.config.model,
                                                      max_length=self.config.model_max_length)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.lr)

    def load_data(self):
        data_path = self.config.data_path
        train_data = Data(os.path.join(data_path, f"train.{self.config.source_lang}"),
                          os.path.join(data_path, f"train.{self.config.target_lang}"),
                          self.tokenizer,
                          self.config.source_max_length,
                          self.config.target_max_length,
                          self.config.prefix,
                          self.config.device)
        val_data = Data(os.path.join(data_path, f"val.{self.config.source_lang}"),
                        os.path.join(data_path, f"val.{self.config.target_lang}"),
                        self.tokenizer,
                        self.config.source_max_length,
                        self.config.target_max_length,
                        self.config.prefix,
                        self.config.device)

        if self.config.use_colossalai:
            from colossalai.utils import get_dataloader
            train = get_dataloader(train_data,
                                   batch_size=self.config.batch_size,
                                   shuffle=True,
                                   num_workers=self.config.num_workers)
            val = get_dataloader(val_data,
                                 batch_size=self.config.batch_size,
                                 shuffle=True,
                                 num_workers=self.config.num_workers)
        else:
            train = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True,
                               num_workers=self.config.num_workers)
            val = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=True,
                             num_workers=self.config.num_workers)
        return train, val

    def save(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.config.save_path, f"model_{epoch}.pt"))

    def load(self):
        print(f"load model from epoch {self.config.load_epoch}")
        self.model.load_state_dict(
            torch.load(os.path.join(self.config.save_path, f"model_{self.config.load_epoch}.pt"))
        )

    def init_colossalai(self, train_data):
        import colossalai
        colossalai.get_default_parser()
        colossalai.launch_from_torch(config='./colossalai_config.py')
        self.model, _data, *_ = colossalai.initialize(model=self.model,
                                                      optimizer=self.optimizer,
                                                      train_dataloader=train_data)
        return _data

    def step(self):
        if self.config.use_colossalai:
            self.model.step()
        else:
            self.optimizer.step()

    def zero_grad(self):
        if self.config.use_colossalai:
            self.model.step()
        else:
            self.optimizer.step()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
