import gc
import threading

import torch
from matplotlib import pyplot
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import ModelLoader


def b2mb(x):
    return int(x / 2 ** 20)


class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)


def train(model, epoch, train_data):
    model.model.train()
    progress = tqdm(train_data, desc=f"Train {epoch}:", total=len(train_data))
    train_loss = 0
    step = 0
    for idx, batch in enumerate(progress):
        model.zero_grad()
        if model.config.use_accelerate:
            with TorchTracemalloc() as _:
                output = model(**batch)
                loss = output.loss
                accelerator.backward(loss)
                if model.config.gradient_accumulation_steps \
                        and (idx + 1) % model.config.gradient_accumulation_steps == 0:
                    model.step()
        else:
            output = model(**batch)
            loss = output.loss
            loss.backward()
            if model.config.gradient_accumulation_steps \
                    and (idx + 1) % model.config.gradient_accumulation_steps == 0:
                model.step()
        step += 1
        train_loss += loss.item()
        progress.set_postfix_str(f"setp:{idx + 0} | loss: {loss.item()} | mean loss: {train_loss / step}")

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
    model.scheduler = get_linear_schedule_with_warmup(
        model.optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_data) * model.config.epochs),
    )

    if model.config.use_accelerate:
        from accelerate import Accelerator
        import psutil

        accelerator = Accelerator()
        train_data = model.init_accelerate(accelerator, train_data)

    print(f"The model has {sum(p.numel() for p in model.model.parameters() if p.requires_grad):,} trainable parameters")

    best = float("inf")
    train_loss_history = []
    val_loss_history = []
    for epoch in range(model.config.load_epoch if model.config.load_epoch and model.config.load_epoch != "best" else 0,
                       model.config.epochs):
        train_loss = train(model, epoch, train_data)
        print(f"Epoch {epoch}, train loss: {train_loss}")
        val_loss = validate(model, val_data, epoch)
        print(f"Epoch: {epoch}, validate loss: {val_loss}")
        if train_loss < best:
            model.save("best")
            best = train_loss
        if (epoch + 1) % model.config.save_interval == 0:
            model.save(epoch + 1)
        model.save("latest")
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
    pyplot.plot(train_loss_history, label="Train")
    pyplot.plot(val_loss_history, label="Valid")
