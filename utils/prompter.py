# coding: utf-8

import os
import json
from typing import Optional


class Prompter(object):
    __slots__ = ('_verbose', 'template')

    def __init__(self, template_name: str = 'alpaca', verbose: bool = False):
        self._verbose = verbose
        file_name = os.path.join('templates', f'{template_name}.json')
        if not os.path.exists(file_name):
            raise FileNotFoundError(f'can not find file: {file_name}')

        with open(file_name, 'r', encoding='utf-8') as f:
            self.template = json.load(f)

        if self._verbose:
            print(f"using prompt template {template_name}: {self.template['description']}")

    def generate_prompt(self, instruction: str, input_: Optional[str] = None, label: Optional[str] = None) -> str:
        if input_:
            res = self.template['prompt_input'].format(instruction=instruction, input=input)
        else:
            res = self.template['prompt_no_input'].format(instruction=instruction)

        if label:
            res = f'{res}{label}'

        if self._verbose:
            print(res)

        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template['response_split'])[1].strip()
