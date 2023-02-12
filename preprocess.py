import re
from argparse import ArgumentParser

from tqdm import tqdm

from text import text_func, symbols, make_vocab

args = ArgumentParser()
args.add_argument("-d", "--dataset", type=str, required=True, help="dataset path")
args.add_argument("-s", "--save", type=str, default=None, help="save path")
args.add_argument("-m", "--min_freq", type=int, default=2, help="filter the least frequent words")
args.add_argument("-l", "--lang", type=str, required=True, help="language type: en,zh,tw,ja,de...")
args.add_argument("-v", "--vocab", type=bool, default=False, help="create vocab")
args.add_argument("--train_path", type=str, default=None, help="train data save path")
args.add_argument("--test_path", type=str, default=None, help="test data save path")
args.add_argument("--val_path", type=str, default=None, help="val data save path")
args.add_argument("--test_len", type=int, default=1_000, help="test data length")
args.add_argument("--val_len", type=int, default=2_000, help="val data length")


def create_vocab(file_path, save_path, lang, min_freq):
    assert lang in text_func, f"unsupported language {lang}"
    lines = [line.strip() for line in open(file_path, encoding="utf-8").readlines()]
    count = make_vocab(lines, min_freq, lang)
    with open(save_path or "/".join(file_path.split("/")[:-1]) + f"/vocab.{lang}", "w", encoding="utf-8") as f:
        f.write("[SOS]\n[PAD]\n[EOS]\n[UNK]\n")
        f.write("\n".join(symbols) + "\n")
        for voc in count:
            f.write(f"{voc}\n")


def text_filter(text):
    text = text_func[args.lang](text)
    return text if text.endswith("\n") else text + "\n"


def dataset_split(file_path, train_path, test_path, val_path, lang, test_len, val_len):
    lines = [line for line in open(file_path, encoding="utf-8").readlines()]
    with open(train_path or "/".join(file_path.split("/")[:-1]) + f"/train.{lang}", "w", encoding="utf-8") as train, \
            open(test_path or "/".join(file_path.split("/")[:-1]) + f"/test.{lang}", "w", encoding="utf-8") as test, \
            open(val_path or "/".join(file_path.split("/")[:-1]) + f"/val.{lang}", "w", encoding="utf-8") as val:
        data_length = len(lines)
        progress = tqdm(enumerate(lines), desc=f"process {lang}", total=data_length)
        for offset, line in progress:
            if data_length - offset > test_len + val_len:
                train.write(text_filter(line.lower()))
            elif data_length - offset > test_len:
                val.write(text_filter(line.lower()))
            else:
                test.write(text_filter(line.lower()))


if __name__ == '__main__':
    args = args.parse_args()
    if args.vocab:
        create_vocab(args.dataset, args.save, args.lang, args.min_freq)
    dataset_split(args.dataset, args.train_path, args.test_path, args.val_path, args.lang, args.test_len, args.val_len)
