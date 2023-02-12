import jieba
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import os
from data import Data
from tqdm import tqdm

os.environ["http_proxy"] = "http://localhost:7890"
SOURCE_MAX_LEN = 512
TARGET_MAX_LEN = 512
PREFIX = "zh2ja"


tokenizer = T5TokenizerFast.from_pretrained("t5-base", model_max_length=1024)
model = T5ForConditionalGeneration.from_pretrained("t5-base")


optimizer = Adam(model.parameters(), lr=1e-5)
data = Data("data/zh.txt", "data/ja.txt", tokenizer, source_max_length=SOURCE_MAX_LEN, target_max_length=TARGET_MAX_LEN, prefix=PREFIX)
dataloader = DataLoader(data, batch_size=8, shuffle=True)
for idx, batch in tqdm(dataloader):
    out = model(**batch)
    loss = out.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
