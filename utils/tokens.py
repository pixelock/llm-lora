# coding: utf-8


def cal_text_tokens(text: str, tokenizer, truncation=True, max_length=512, with_bos=False, with_eos=False) -> int:
    cnt_exclude = 0
    if with_bos:
        cnt_exclude += 1
    if with_eos:
        cnt_exclude += 1

    return tokenizer(text, truncation=truncation, max_length=max_length) - cnt_exclude


def cal_text_tokens_alpaca(text: str, tokenizer):
    return cal_text_tokens(text, tokenizer, )
