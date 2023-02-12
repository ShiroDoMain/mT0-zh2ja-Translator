import torch

from models import ModelLoader


def generate(model):
    model.model.eval()
    with torch.no_grad():
        while 1:
            text = model.config.prefix + input(f"source({model.config.prefix}):")
            source = model.tokenizer.encode(text,
                                            # return_length=False,
                                            return_tensors="pt",
                                            truncation=True
                                            # padding="max_length",
                                            # max_length=model.config.source_max_length
                                            )
            response = model.model.generate(source)

            predict = model.tokenizer.decode(response[0], skip_special_tokens=True)
            # predict = [model.tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=True) for i in
            print("predict:", predict)


if __name__ == '__main__':
    model = ModelLoader("config/config.json")
    generate(model)
