import os
from openprompt.utils.logging import logger

from openprompt.data_utils.data_utils import InputExample, InputFeatures
from typing import *

from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template
from openprompt.prompts import ManualTemplate, ManualVerbalizer

import torch
from torch import nn


class SoftManualTemplate(ManualTemplate):
    registered_inputflag_names = ["soft_token_ids", "loss_ids", "shortenable_ids"]

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[str] = None,
                 mask_token: str = '<mask>',
                 soft_token: str = '<soft>',
                 placeholder_mapping: dict = {'<text_a>': 'text_a', '<text_b>': 'text_b'},
                 ):
        super().__init__(tokenizer=tokenizer,
                         mask_token=mask_token,
                         placeholder_mapping=placeholder_mapping)
        self.raw_embedding = model.get_input_embeddings()
        self.embedding_size = self.raw_embedding.weight.shape[-1]
        self.soft_token = soft_token
        self.text = text

    def get_default_soft_token_ids(self) -> List[int]:
        r"""get the soft token indices for the template
        e.g. when self.text is ['<text_a>', '<train>It', '<train>is', '<mask>', '.'],
        output is [0, 1, 2, 0, 0]
        """
        idx = []
        num_soft_token = 0
        for token in self.text:
            if token.startswith(self.soft_token):
                num_soft_token += 1
                idx.append(num_soft_token)
            else:
                idx.append(0)
        return idx

    def on_text_set(self):
        """
        when template text was set, generate parameters needed for soft-prompt
        """
        self.num_soft_token = sum([token.startswith(self.soft_token) for token in self.text])
        self.generate_parameters()

    def generate_parameters(self) -> None:
        """
        generate parameters needed for soft tokens embedding in soft-prompt
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        specific_len = 15
        self.soft_embedding = nn.Embedding(1 + self.num_soft_token + specific_len * 7, self.embedding_size)  # 公共prompt
        self.soft_embedding2 = nn.Embedding(1 + self.num_soft_token + specific_len * 7,self.embedding_size)  # 特定域prompt

        count = 0
        for token in self.text:
            if token.startswith(self.soft_token):
                count += 1
                orig = token.split(self.soft_token)[1]
                if orig == "":
                    # raise ValueError("hard prompt not given")
                    continue
                token_ids = self.tokenizer(" " + orig, add_special_tokens=False)[
                    "input_ids"]  # TODO no prefix space option
                if len(token_ids) > 1:
                    logger.warning("""soft prompt's hard prompt {} tokenize to more than one tokens: {}
                        By default we use the first token""".format(orig,
                                                                    self.tokenizer.convert_ids_to_tokens(token_ids)))
                self.soft_embedding.weight.data[count, :] = self.raw_embedding.weight.data[token_ids[0],
                                                            :].clone().detach().requires_grad_(True)  # TODO check this

    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:

        # 5是特定提示长度
        specific_len = 15
        spe_len = 2
        if batch['AT']:
            # 把域特定prompt给mask掉
            batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + 2,
                                                                self.num_soft_token + 2).cuda(), 0)
            batch['attention_mask'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + 2,
                                                                self.num_soft_token + 2).cuda(), 0)
        else:

            n = batch['essay_set'][0] - 1
            batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len,
                                                                self.num_soft_token - specific_len + spe_len+1).cuda(),
                                                self.num_soft_token - specific_len - 1 + spe_len + specific_len * n)  # self.num_soft_token-5-1+2+5*n
            batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+1,
                                                                self.num_soft_token - specific_len + spe_len+2).cuda(),
                                                self.num_soft_token - specific_len - 1 + spe_len+1 + specific_len * n)
            batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+2,
                                                                self.num_soft_token - specific_len + spe_len+3).cuda(),
                                                self.num_soft_token - specific_len - 1 + spe_len+2 + specific_len * n)
            batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+3,
                                                                self.num_soft_token - specific_len + spe_len+4).cuda(),
                                                self.num_soft_token - specific_len - 1 + spe_len+3 + specific_len * n)
            batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+4,
                                                                self.num_soft_token - specific_len + spe_len+5).cuda(),
                                                self.num_soft_token - specific_len - 1 + spe_len+4 + specific_len * n)
            batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+5,
                                                                self.num_soft_token - specific_len + spe_len+6).cuda(),
                                                self.num_soft_token - specific_len - 1 + spe_len+5 + specific_len * n)  # self.num_soft_token-5-1+2+5*n
            # 10
            batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+6,
                                                                self.num_soft_token - specific_len + spe_len+7).cuda(),
                                                self.num_soft_token - specific_len - 1 + spe_len+6 + specific_len * n)
            batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+7,
                                                                self.num_soft_token - specific_len + spe_len+8).cuda(),
                                                self.num_soft_token - specific_len - 1 + spe_len+7 + specific_len * n)
            batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+8,
                                                                self.num_soft_token - specific_len + spe_len+9).cuda(),
                                                self.num_soft_token - specific_len - 1 + spe_len+8 + specific_len * n)
            batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+9,
                                                                self.num_soft_token - specific_len + spe_len+10).cuda(),
                                                self.num_soft_token - specific_len - 1 + spe_len+9 + specific_len * n)
            # 15
            batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+10,
                                                                self.num_soft_token - specific_len + spe_len+11).cuda(),
                                                self.num_soft_token - specific_len - 1 + spe_len+10 + specific_len * n)  # self.num_soft_token-5-1+2+5*n
            batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+11,
                                                                self.num_soft_token - specific_len + spe_len+12).cuda(),
                                                self.num_soft_token - specific_len - 1 + spe_len+11 + specific_len * n)
            batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+12,
                                                                self.num_soft_token - specific_len + spe_len+13).cuda(),
                                                self.num_soft_token - specific_len - 1 + spe_len+12 + specific_len * n)
            batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+13,
                                                                self.num_soft_token - specific_len + spe_len+14).cuda(),
                                                self.num_soft_token - specific_len - 1 + spe_len+13 + specific_len * n)
            batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+14,
                                                                self.num_soft_token - specific_len + spe_len+15).cuda(),
                                                self.num_soft_token - specific_len - 1 + spe_len+14 + specific_len * n)
            # 20
            # batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+15,
            #                                                     self.num_soft_token - specific_len + spe_len+16).cuda(),
            #                                     self.num_soft_token - specific_len - 1 + spe_len+15 + specific_len * n)  # self.num_soft_token-5-1+2+5*n
            # batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+16,
            #                                                     self.num_soft_token - specific_len + spe_len+17).cuda(),
            #                                     self.num_soft_token - specific_len - 1 + spe_len+16 + specific_len * n)
            # batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+17,
            #                                                     self.num_soft_token - specific_len + spe_len+18).cuda(),
            #                                     self.num_soft_token - specific_len - 1 + spe_len+17 + specific_len * n)
            # batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+18,
            #                                                     self.num_soft_token - specific_len + spe_len+19).cuda(),
            #                                     self.num_soft_token - specific_len - 1 + spe_len+18 + specific_len * n)
            # batch['soft_token_ids'].index_fill_(1, torch.arange(self.num_soft_token - specific_len + spe_len+19,
            #                                                     self.num_soft_token - specific_len + spe_len+20).cuda(),
            #                                     self.num_soft_token - specific_len - 1 + spe_len+19 + specific_len * n)

        # raw_embedding是词库中每个词的embedding
        raw_embeds = self.raw_embedding(batch['input_ids'])
        soft_embeds = self.soft_embedding(batch['soft_token_ids'])

        if batch['AT'] == False:
            soft_embeds = soft_embeds.detach()  # 冻结公共prompt
            soft_embeds2 = self.soft_embedding2(batch['soft_token_ids'])  # 域特定prompt
        # soft_token_ids大于0的位置保留soft_embeds,否则保存raw_embeds (soft_token_ids最后两个是mask和结束符)
        inputs_embeds = torch.where((batch['soft_token_ids'] > 0).unsqueeze(-1), soft_embeds, raw_embeds)
        if batch['AT'] == False:
            # 公共prompt后面的prompt，替换成域特定的prompt
            inputs_embeds = torch.where((batch['soft_token_ids'] > self.num_soft_token - specific_len).unsqueeze(-1), soft_embeds2,
                                        inputs_embeds)

        # special_soft = self.soft_embedding.weight.data[1 + self.num_soft_token + 5 * 6:1 + self.num_soft_token + 5 * 7]
        # inputs_embeds.index_fill_(1,torch.arange(self.num_soft_token-5+2,self.num_soft_token-5+7).cuda(), special_soft)
        # batch['soft_token_ids'][:,42:].cpu().tolist()
        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        return batch
