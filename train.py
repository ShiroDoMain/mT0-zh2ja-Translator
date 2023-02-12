import os

import torch
from matplotlib import pyplot
from torch.optim import Adam
from tqdm import tqdm

from models import ModelLoader


def train(model, epoch, train_data, fp16):
    model.model.train()
    progress = tqdm(train_data, desc=f"Train {epoch}:", total=len(train_data))
    train_loss = 0
    step = 0
    for idx, batch in enumerate(progress):
        model.zero_grad()
        if fp16:
            with autocast():
                output = model(**batch)
                loss = output.loss
                scale.scale(loss).backward()
                scale.step(model)
                scale.update()
        else:
            output = model(**batch)
            loss = output.loss
            loss.backward()
            model.step()
        step += 1
        train_loss += loss.item()
        progress.set_postfix_str(f"setp:{idx + 0} | loss: {loss.item()} | mean loss: {train_loss/step}")
    return train_loss / step


def validate(model, val_data, epoch):
    model.model.eval()
    progress = tqdm(val_data, desc=f"Validate {epoch}:")
    val_loss = 0
    step = 0
    with torch.no_grad():
        for idx, batch in enumerate(progress):
            output = model(**batch)
            loss = output.loss
            progress.set_postfix_str(f"loss: {loss.item()}")
            step += 1
            val_loss += loss.item()
    return val_loss / step


if __name__ == '__main__':
    model = ModelLoader("config/config.json")
    train_data, val_data = model.load_data()
    print(f"The model has {sum(p.numel() for p in model.model.parameters() if p.requires_grad):,} trainable parameters")

    if model.config.use_fp16:
        from torch.cuda.amp import autocast, GradScaler

        scale = GradScaler()

    if model.config.use_colossalai:
        train_data = model.init_colossalai(train_data)

    best = float("inf")
    train_loss_history = []
    val_loss_history = []
    for epoch in range(model.config.load_epoch if model.config.load_epoch and model.config.load_epoch != "best" else 0,
                       model.config.epochs):
        train_loss = train(model, epoch, train_data,  model.config.use_fp16)
        print(f"Epoch {epoch}, train loss: {train_loss}")
        val_loss = validate(model, val_data, epoch)
        print(f"Epoch: {epoch}, validate loss: {val_loss}")
        if train_loss < best:
            model.save("best")
            best = train_loss
        if (epoch + 1) % model.config.save_interval == 0:
            model.save(epoch + 1)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
    pyplot.plot(train_loss_history, label="Train")
    pyplot.plot(val_loss_history, label="Valid")
