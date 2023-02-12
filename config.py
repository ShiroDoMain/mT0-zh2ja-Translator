import json
import os


class Config:
    model: str
    epochs: int
    prefix: str
    batch_size: int
    source_lang:str
    use_colossalai: bool
    use_fp16: bool
    num_workers: int
    save_interval: int
    target_lang: str
    data_path: str
    save_path: str
    load_epoch: int
    lr: float
    source_max_length: int
    target_max_length: int
    model_max_length: int
    device: str

    def __init__(self, config_path):
        config = json.load(open(config_path))
        for k, v in config.items():
            setattr(self, k, v)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
