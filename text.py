import jieba
import nagisa
from typing import List, Dict
from collections import Counter
import re
from tqdm import tqdm


symbol_half = r"""!"#$%&()*+,./:;<=>?@[\]^_`{|}~"""
symbol_full = r"""！@#￥%…&*（）——+{}：“”„‘’；【】《》，。？/·~"""
_filter_symbols = r"""♪"""
_symbol_ja = r"""「」"""
symbols = symbol_full + symbol_half + _symbol_ja + _filter_symbols


def segment_en(line: str):
    return line.replace("'er", " are").replace("'s", " is").replace("'m", " am").replace("'t", "dont ")


def japanese_text(line):
    return " ".join(nagisa.wakati(line, lower=True))


def chinese_text(line):
    return " ".join(jieba.lcut(line)).lower()


def make_vocab(line_list: List[str], min_freq, lang) -> Dict[str, int]:
    word_list = []
    process_func = text_func.get(lang)
    for line in tqdm(line_list, desc="create vocabulary"):
        line = re.sub(rf"[{symbols}]", "", line)
        if process_func is not None:
            line = process_func(line)
        word_list += line.lower().split()
    return {word: freq for word, freq in Counter(word_list).items() if freq >= min_freq}


text_func = {
    "en": make_vocab,
    "de": make_vocab,
    "zh": chinese_text,
    "ja": japanese_text
}
