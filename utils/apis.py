# coding: utf-8

import openai

API_KEY = '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'
openai.api_key = API_KEY


def request_gpt_3_5(content, model='gpt-3.5-turbo'):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # here we use `gpt-3.5-turbo` model, while Stanford-Alpaca uses `text-davinci-003`
        messages=[
            {"role": "user", "content": content},
        ],
    )
    answer = response["choices"][0]["message"]["content"]
    return answer


if __name__ == '__main__':
    a = request_gpt_3_5('你知道一个好的店铺名称要怎么起吗？')
    b = 1
