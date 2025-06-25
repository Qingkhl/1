from inspect import Parameter
import json
from os import stat

import transformers.activations
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from yacs.config import CfgNode
from openprompt.data_utils import InputFeatures
import re
from openprompt import Verbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from openprompt.utils.logging import logger
import copy
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput, MaskedLMOutput

from transformers.models.t5 import T5ForConditionalGeneration

class SoftVerbalizer(Verbalizer):
    r"""
    The implementation of the verbalizer in `WARP <https://aclanthology.org/2021.acl-long.381/>`_

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        label_words (:obj:`Union[List[str], List[List[str]], Dict[List[str]]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
    """
    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizer],
                 model: Optional[PreTrainedModel],
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                ):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler

        head_name = [n for n,c in model.named_children()][-1]
        logger.info(f"The LM head named {head_name} was retrieved.")
        self.head = copy.deepcopy(getattr(model, head_name))
        # for name, param in self.head.named_parameters():
        #     param.requires_grad = False
        max_loop = 5

        # 总体评分分类
        # self.dense = torch.nn.Linear(768+51+35, 100, bias=True)
        self.dense = torch.nn.Linear(768, 100, bias=True)
        self.transform_act_fn = transformers.activations.GELUActivation()
        self.LayerNorm = torch.nn.LayerNorm(100, eps=1e-12)
        self.decoder = nn.Linear(100, 4, bias=False)

        # 回归(总体评分和trait)
        self.attribute_predict = nn.ModuleList([Attribute_predict() for _ in range(1)])
        self.attribute_predict2 = nn.ModuleList([Attribute_predict2() for _ in range(1)])
        self.self_att = SelfAttention(100, 1)

        # nn.init.normal_(self.decoder.weight)
        if label_words is not None: # use label words as an initialization
            self.label_words = label_words
        # print(sum(p.numel() for p in self.dense.parameters() if p.requires_grad))  # 打印模型参数量


    @property
    def group_parameters_1(self,):
        r"""Include the parameters of head's layer but not the last layer
        In soft verbalizer, note that some heads may contain modules
        other than the final projection layer. The parameters of these part should be
        optimized (or freezed) together with the plm.
        """
        if isinstance(self.head, torch.nn.Linear):
            return []
        else:
            return [p for n, p in self.head.named_parameters() if self.head_last_layer_full_name not in n]

    @property
    def group_parameters_2(self,):
        r"""Include the last layer's parameters
        """
        if isinstance(self.head, torch.nn.Linear):
            return [p for n, p in self.head.named_parameters()]
        else:
            return [p for n, p in self.head.named_parameters() if self.head_last_layer_full_name in n]

    def on_label_words_set(self):
        self.label_words = self.add_prefix(self.label_words, self.prefix)
        self.generate_parameters()

    @staticmethod
    def add_prefix(label_words, prefix):
        r"""Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.

        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]  #wrapped it to a list of list of label words.

        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith("<!>"):
                    new_label_words_per_label.append(word.split("<!>")[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words

    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token.
        """
        words_ids = []
        for word in self.label_words:
            if isinstance(word, list):
                logger.warning("Label word for a class is a list, only use the first word.")
            word = word[0]
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if len(word_ids) > 1:
                logger.warning("Word {} is split into multiple tokens: {}. \
                    If this is not what you expect, try using another word for this verbalizer" \
                    .format(word, self.tokenizer.convert_ids_to_tokens(word_ids)))
            words_ids.append(word_ids)

        max_len  = max([len(ids) for ids in words_ids])
        words_ids_mask = [[1]*len(ids) + [0]*(max_len-len(ids)) for ids in words_ids]
        words_ids = [ids+[0]*(max_len-len(ids)) for ids in words_ids]

        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.label_words_mask = nn.Parameter(words_ids_mask, requires_grad=False)

        init_data = self.original_head_last_layer[self.label_words_ids,:]*self.label_words_mask.to(self.original_head_last_layer.dtype).unsqueeze(-1)
        init_data = init_data.sum(dim=1)/self.label_words_mask.sum(dim=-1,keepdim=True)

        if isinstance(self.head, torch.nn.Linear):
            self.head.weight.data = init_data
            self.head.weight.data.requires_grad=True
        else:
            '''
            getattr(self.head, self.head_last_layer_full_name).weight.data = init_data
            getattr(self.head, self.head_last_layer_full_name).weight.data.requires_grad=True # To be sure
            '''
            self.head_last_layer.weight.data = init_data
            self.head_last_layer.weight.data.requires_grad=True

    def process_hiddens(self, hidden_states: torch.Tensor,batch=None, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:
        """
        # label_logits = self.cls(self.head(hiddens))
        # label_logits = self.head(hidden_states)
        feats_craft = batch['feats_craft'].float().cuda()
        item_features = batch['item_features'].float().cuda()
        # hidden_states = torch.cat([hidden_states,feats_craft,item_features],-1) # 无手工特征

        attributes_pre1 = torch.cat([torch.unsqueeze(att_pre(hidden_states), 1)
                                     for att_pre in self.attribute_predict],dim=1) # trait做注意力前先各自经过一个全连接层

        hidden_states1 = self.dense(hidden_states) # 总体评分(分类)做注意力前经过一个全连接层
        hidden_states2 = self.transform_act_fn(hidden_states1) # 激活函数
        hidden_states3 = torch.unsqueeze(self.LayerNorm(hidden_states2), 1)

        # 分类的不做注意力
        hidden_states3 = torch.squeeze(hidden_states3, 1)
        # attributes_pre = self.self_att(attributes_pre1)

        attributes_pre = torch.cat([self.attribute_predict2[i](torch.squeeze(attributes_pre1[:,i,:],1))
                                    for i in range(1)],dim=1)

        label_logits = self.decoder(hidden_states3) # 分类
        attributes_pre = torch.sigmoid(attributes_pre)
        return label_logits,attributes_pre

    def process_outputs(self, outputs: torch.Tensor, batch: Union[Dict, InputFeatures], **kwargs):
        return self.process_hiddens(outputs,batch)

    def gather_outputs(self, outputs: ModelOutput):
        if isinstance(outputs, Seq2SeqLMOutput):
            ret = outputs.decoder_hidden_states[-1]
        elif isinstance(outputs, MaskedLMOutput) or isinstance(outputs, CausalLMOutputWithCrossAttentions):
            ret = outputs.hidden_states[-1]
        else:
            try:
                ret = outputs.hidden_states[-1]
            except AttributeError:
                raise NotImplementedError(f"Gather outputs method for outputs' type {type(outputs)} not implemented")

        return ret


class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim # 1
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert input_dim % num_heads == 0, "Input dimension must be divisible by number of heads"

        # 定义查询、键、值的线性变换层
        self.query = nn.Linear(self.input_dim, self.input_dim)
        self.key = nn.Linear(self.input_dim, self.input_dim)
        self.value = nn.Linear(self.input_dim, self.input_dim)
        self.layer_norm = torch.nn.LayerNorm(self.input_dim, eps=1e-12)

        # 最后的线性变换层
        self.out = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, x):
        # 将输入张量进行线性变换
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # Q = x
        # K = x
        # V = x
        # 计算注意力分数
        energy = torch.bmm(Q, K.permute(0, 2, 1))  # (batch_size, seq_len, seq_len)

        # 对注意力分数进行 softmax 操作
        attention = F.softmax(energy, dim=-1)

        # 计算注意力输出
        attention_output = torch.bmm(attention, V)  # (batch_size, seq_len, input_dim)

        # 最后的线性变换
        output = attention_output # self.out(attention_output)

        return output


class Attribute_predict(nn.Module):
    def __init__(self):
        super(Attribute_predict, self).__init__()
        # self.fc1 = nn.Linear(768+51+35, 100)
        self.fc1 = nn.Linear(768, 100)
        self.fc1.weight.data.normal_(0, 0.01)
        self.layer_norm = torch.nn.LayerNorm(100, eps=1e-12)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Linear(100, 10) # 没用
        self.fc3 = nn.Linear(10, 1) # 没用
        # self.drop = nn.Dropout(0.2)
        self.att_net = nn.Sequential(
            # self.drop,
            self.fc1,
            # nn.ReLU(True),
            self.fc2,
            # nn.ReLU(True),
            self.fc3
        )

    def forward(self, feature):
        out = self.relu(self.fc1(feature))
        out = self.layer_norm(out)
        # attribute_out = torch.sigmoid(out)
        return out

class Attribute_predict2(nn.Module):
    def __init__(self):
        super(Attribute_predict2, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(100, eps=1e-12)
        self.fc1 = nn.Linear(100, 1)
        self.fc1.weight.data.normal_(0, 0.001)
        self.fc2 = nn.Linear(100,10)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc3 = nn.Linear(100, 1)
        # self.drop = nn.Dropout(0.2)
        self.att_net = nn.Sequential(
            self.layer_norm,
            self.fc1,
            # nn.ReLU(True),
            self.fc2,
            # nn.ReLU(True),
            self.fc3
        )

    def forward(self, feature):
        out = self.layer_norm(feature)
        out = self.fc1(out)
        # out = self.fc2(out)
        # out = self.fc3(out)
        # attribute_out = torch.sigmoid(out)
        return out
