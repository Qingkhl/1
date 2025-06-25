import math

from transformers import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import *
from openprompt.data_utils import InputExample, InputFeatures
from torch.utils.data._utils.collate import default_collate
from tqdm.std import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.prompt_base import Template, Verbalizer
from openprompt.utils import round_list, signature
from torch.utils.data import DataLoader
from openprompt.plms import get_tokenizer_wrapper

class PromptDataLoader(object):
    r"""
    PromptDataLoader wraps the orginal dataset. The input data is firstly wrapped with the
    prompt's template, and then is tokenized by a wrapperd-tokenizer. 
    
    Args:
        dataset (:obj:`Dataset` or :obj:`List`): Either a DatasetObject or a list containing the input examples.
        template (:obj:`Template`): A derived class of of :obj:`Template`
        tokenizer (:obj:`PretrainedTokenizer`): The pretrained tokenizer.
        max_seq_length (:obj:`str`, optional): The max sequence length of the input ids. It's used to trucate sentences.
        batch_size (:obj:`int`, optional): The batch_size of data loader
        teacher_forcing (:obj:`bool`, optional): Whether to fill the mask with target text. Set to true in training generation model.
        decoder_max_length (:obj:`bool`, optional): the decoder maximum length of an encoder-decoder model.
        predict_eos_token (:obj:`bool`, optional): Whether to predict the <eos> token. Suggest to set to true in generation.
        truncate_method (:obj:`bool`, optional): the truncate method to use. select from `head`, `tail`, `balanced`.
        kwargs  :Other kwargs that might be passed into a tokenizer wrapper. 
    """
    def __init__(self, 
                 dataset: Union[Dataset, List],
                 template: Template,
                 tokenizer: PreTrainedTokenizer,
                 max_seq_length: Optional[str] = 512,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = False,
                 teacher_forcing: Optional[bool] = False,
                 decoder_max_length: Optional[int] = -1,
                 predict_eos_token: Optional[bool] = False,
                 truncate_method: Optional[str] = "tail",
                 **kwargs,
                ):

        assert hasattr(dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {dataset}"
        assert hasattr(dataset, "__len__"), f"The dataset must have __len__ method. dataset is {dataset}"
        self.raw_dataset = dataset
        
        self.wrapped_dataset = []
        self.tensor_dataset = []
        self.template = template
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.teacher_forcing = teacher_forcing

        tokenizer_wrapper_class = get_tokenizer_wrapper(tokenizer)
        tokenizer_wrapper_init_keys = signature(tokenizer_wrapper_class.__init__).args
        prepare_kwargs = {
                        "max_seq_length":max_seq_length,
                        "truncate_method":truncate_method,
                        "decoder_max_length":decoder_max_length,
                        "predict_eos_token":predict_eos_token,
                        "tokenizer": tokenizer,
                        **kwargs,
                        }
        to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}
        

        self.tokenizer_wrapper = get_tokenizer_wrapper(tokenizer)(**to_pass_kwargs)
        
        # check the satisfiability of each component
        assert hasattr(self.template, 'wrap_one_example'), "Your prompt has no function variable \
                                                         named wrap_one_example"
        
        # processs
        self.wrap()
        self.tokenize()

        def prompt_collate_fct(batch: List[Union[Dict, InputFeatures]]):
            r'''
            This function is used to collate the current prompt.

            Args:
                batch (:obj:`List[Union[Dict, InputFeatures]]`): A batch of the current data.

            Returns:
                :obj:`InputFeatures`: Return the :py:class:`~openprompt.data_utils.data_utils.InputFeatures of the current batch of data.
            '''

            elem = batch[0]
            return_dict = {key: default_collate([d[key] for d in batch]) for key in elem}
            return InputFeatures(**return_dict)

        self.dataloader = DataLoader(self.tensor_dataset, 
                                     batch_size = self.batch_size,
                                     shuffle = self.shuffle,
                                     collate_fn = prompt_collate_fct
                                    )
    
    
    def wrap(self):
        r"""A simple interface to pass the examples to prompt, and wrap the text with template.
        """
        if isinstance(self.raw_dataset, Dataset) or isinstance(self.raw_dataset, List): # TODO change to iterable 
            assert len(self.raw_dataset) > 0, 'The dataset to be wrapped is empty.'
            # for idx, example in tqdm(enumerate(self.raw_dataset),desc='Wrapping'):
            for idx, example in enumerate(self.raw_dataset):
                wrapped_example = self.template.wrap_one_example(example)
                self.wrapped_dataset.append(wrapped_example)
        else:
            raise NotImplementedError
    
    def tokenize(self) -> None:
        r"""Pass the wraped text into a prompt-specialized tokenizer, 
           the true PretrainedTokenizer inside the tokenizer is flexible, e.g. AlBert, Bert, T5,...
        """
        for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset),desc='tokenizing'):
        # for idx, wrapped_example in enumerate(self.wrapped_dataset):
            inputfeatures = InputFeatures(**self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing), **wrapped_example[1]).to_tensor()
            self.tensor_dataset.append(inputfeatures)


    def __getitem__(self, idx):
        r"""simulate the ``torch.utils.data.Dataset``'s behavior.
        """
        return self.tensor_dataset[idx]
    
    def __len__(self):
        return  len(self.dataloader)

    def __iter__(self,):
        return self.dataloader.__iter__()


class ScheduledOptim():

    def __init__(self, optimizer, lr, decay_step=2000,
                 decay_rate=0.9, steps=0):
        self.init_lr = lr
        self.steps = steps
        self._optimizer = optimizer
        self.decay_rate = decay_rate
        self.decay_step = decay_step

    def step(self):
        '''Step with the inner optimizer'''
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.steps += 1
        if self.steps >= self.decay_step:
            lr = self.init_lr * math.pow(self.decay_rate,
                                         int(self.steps / self.decay_step))
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr
        else:
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = self.init_lr



class PromptModel(nn.Module):
    r'''``PromptModel`` is the encapsulation of ``Template`` and the ``pre-trained model``, 
    with OpenPrompt, these modules could be flexibly combined. And this class is the base class of ``PromptForClassification`` and ``PromptForGeneration``

    Args:
        model (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        template (:obj:`Template`): The ``Template`` object to warp the input data.
    '''
    def __init__(self,
                 model: PreTrainedModel, 
                 template: Template,
                 # template2: Template,
                 # template3: Template,
                 ):
        super().__init__()
        self.model = model
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.template = template
        # self.template2 = template2
        # self.template3 = template3
        # get model's forward function's keywords
        self.forward_keys = signature(self.model.forward).args
        
    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r""" 
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position. 

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """

        # 处理完之后token变成embedding
        batch = self.template.process_batch(batch)
        # torch.cat([input_batch["inputs_embeds"],input_batch["inputs_embeds"]],1)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        outputs = self.model(**input_batch)

        return outputs
    
    def prepare_model_inputs(self, batch: Union[Dict, InputFeatures]) -> Dict:
        r"""Will be used in generation
        """
        batch = self.template.process_batch(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        return input_batch


class PromptForClassification(nn.Module):
    r'''``PromptModel`` with a classification head on top. The classification head will map
    the logits in all position of the sequence (return value of a PromptModel) into the
    logits of the labels, using a verbalizer. 

    Args:
        model (:obj:`PretrainedModel`): A pre-traiend model you decide to use for classification, e.g. BERT.
        template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``ManualTemplate``.
        verbalizer (:obj:`Verbalizer`): A ``Verbalizer`` object you use to project the lables to label words for classification, e.g. ``ManualVerbalizer``.
    '''
    def __init__(self,
                 model: PreTrainedModel, 
                 template: Template,
                 verbalizer: Verbalizer,
                 ):
        super().__init__()
        self.model = model
        # 冻结预训练模型
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.template = template


        self.prompt_model = PromptModel(model, template)
        self.verbalizer = verbalizer


    @property
    def device(self,):
        r"""
        Register the device parameter.
        """
        return self.model.device

    def extract_logits(self,
                       logits: torch.Tensor,
                       batch: Union[Dict, InputFeatures]):
        r"""Get logits of all <mask> token
        Project the logits of shape
        (``batch_size``, ``max_seq_length``, ``vocab_size``)
        into logits of shape (if num_mask_token > 1)
        (``batch_size``, ``num_mask_token``, ``vocab_size``)
        or into logits of shape (if ``num_mask_token`` = 1)
        (``batch_size``, ``vocab_size``).

        Args:
            logits (:obj:`torch.Tensor`): The original logits of the whole sequence.
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch
        
        Returns:
            :obj:`torch.Tensor`: The extracted logits of ``<mask>`` tokens.
            
        """
        logits = logits[torch.where(batch['loss_ids']>0)]
        logits = logits.view(batch['loss_ids'].shape[0], -1, logits.shape[1])
        if logits.shape[1] == 1:
            logits = logits.view(logits.shape[0], logits.shape[2])
        return logits

    def extract_hiddens(self, hiddens, batch):

        hiddens = hiddens[torch.where(batch['loss_ids']>0)]
        hiddens = hiddens.view(batch['loss_ids'].shape[0], -1, hiddens.shape[1])
        if hiddens.shape[1] == 1:
            hiddens = hiddens.view(hiddens.shape[0], hiddens.shape[2])
        return hiddens

    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r""" keys in batch: 
        """
        for name, param in self.model.named_parameters():
            continue
        outputs = self.prompt_model(batch)
        hidden_states_last = outputs.hidden_states[-1] # 最后一个隐藏层的输出
        # 用cls
        # hiddens = outputs.pooler_output
        # logits = None

        # 下面是用mask
        logits = outputs.logits
        logits = self.extract_logits(logits, batch) # (batch_size, max_seq_length, vocab_size)==>(batch_size， num_mask_token， vocab_size)
        hiddens = self.extract_hiddens(hidden_states_last, batch)# (batch_size, max_seq_length, hidden_size)==>(batch_size， num_mask_token， hs)



        label_words_logits, attributes_pre = self.verbalizer.process_outputs(outputs=hiddens, batch=batch) # hidden映射到类别
        if 'label' in batch:
            pass # TODO add caculate label loss here
        # print('ok')
        return label_words_logits, hiddens, logits, attributes_pre
    
    def predict(self):
        pass
    
    def forward_without_verbalize(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        outputs = self.prompt_model(batch)
        logits = outputs.logits
        logits = self.extract_logits(logits, batch)
        return logits

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer
    
    def state_dict(self):
        r""" Save the model using template and verbalizer's save methods.
        Args:
            path (:obj:`str`): the full path of the checkpoint.
            save_plm (:obj:`bool`): whether saving the pretrained language model.
            kwargs: other information, such as the achieved metric value. 
        """
        _state_dict = {}
        _state_dict['plm'] = self.model.state_dict()
        _state_dict['template'] = self.template.state_dict()
        _state_dict['verbalizer'] = self.verbalizer.state_dict()
        _state_dict['prompt_model'] = self.prompt_model.state_dict()
        # _state_dict['verbalizer'] = None
        return _state_dict
    
    def load_state_dict(self, state_dict):
        if 'plm' in state_dict:
            self.model.load_state_dict(state_dict['plm'])
        self.template.load_state_dict(state_dict['template'])
        self.verbalizer.load_state_dict(state_dict['verbalizer'])
        self.prompt_model.load_state_dict(state_dict['prompt_model'])

# 弃用
class DomainClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.0)
        self.out_proj = nn.Linear(768, 1) # 2 domains
        self.optimizer = AdamW(self.parameters(), lr=config.plm.optimize.lr)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class DomainDiscriminators(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.domain_num = len(config.dataset.domains) - 1

        self.domain_st = nn.ModuleList([Discriminators(config) for _ in
                                       range(self.domain_num)])


    def forward(self, features, gamma, global_step=None, total_step=None, first=True):
        if global_step is not None and total_step is not None:
            progress = float(global_step) / float(total_step)
            lmda = 2 / (1 + math.exp(-5 * progress)) - 1
        else:
            lmda = 1.
        domdis_losses = []

        domain_t = torch.ones(features[0].size(0), requires_grad=False).type(torch.FloatTensor).to(
            features[0].device)
        domain_f = torch.zeros(features[0].size(0), requires_grad=False).type(torch.FloatTensor).to(
            features[0].device)

        for i in range(self.domain_num):
            source_data = features[i:(i + 1)]  # 源域
            source_data = torch.cat(source_data, dim=0)
            target_data = features[self.domain_num:]  # 目标域 取最后一个
            target_data = torch.cat(target_data, dim=0)
            # logits_t = self.domain_st[i](target_data.detach()).squeeze(-1)  # True正确判断来自目标域的概率
            # logits_f = self.domain_st[i](source_data.detach()).squeeze(-1)  # False错误判断来自目标域的概率
            logits_t = self.domain_st[i](target_data, gamma).squeeze(-1)  # True正确判断来自目标域的概率
            logits_f = self.domain_st[i](source_data, gamma).squeeze(-1)  # False错误判断来自目标域的概率
            # logits_t = nn.functional.sigmoid(logits_t)
            # logits_f = nn.functional.sigmoid(logits_f)
            domain_discriminator_loss = (F.binary_cross_entropy_with_logits(logits_t, domain_t) +
                                         F.binary_cross_entropy_with_logits(logits_f, domain_f)) * 0.5

            domdis_losses.append(domain_discriminator_loss * lmda)

        return torch.stack(domdis_losses).mean()

class DomainKL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.domain_num = len(config.dataset.domains) - 1
    def forward(self, features, same=True):
        # print(self.domain_num)
        # print(len(features))
        kl_div_list = []
        if same:
            for i in range(self.domain_num):
                for j in range(self.domain_num):
                    if i != j:
                        p = F.softmax(features[i], dim=-1)
                        q = F.softmax(features[j], dim=-1)
                        kl_div = F.kl_div(torch.mean(p, dim=0).log(), torch.mean(q, dim=0),reduction='batchmean')
                        kl_div_list.append(kl_div)
        else:
            # 同一领域的不同类别的KL散度
            for i in range(self.domain_num):
                for cls in range(4):
                    for cls2 in range(4):
                        if cls != cls2:
                            p = F.softmax(features[cls][i], dim=-1)
                            q = F.softmax(features[cls2][i], dim=-1)
                            kl_div = F.kl_div(torch.mean(p, dim=0).log(), torch.mean(q, dim=0), reduction='batchmean')
                            kl_div_list.append(kl_div)
        res = torch.stack(kl_div_list)
        print(features[cls][i])
        print(features[cls2][i])
        return res.mean()

class GRL(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, coeff):
        ctx.coeff = coeff
        ctx.save_for_backward(coeff)
        ctx.save_for_backward(input)
        # 实现前向计算逻辑
        return input.view_as(input)

    @staticmethod
    def backward(ctx, gradOutput):
        # input, = ctx.saved_tensors
        # alpha = 10
        # low = 0.0
        # high = 1.0
        # max_iter = 328
        # ctx.iter_num += 1
        # iter_num = torch.tensor([ctx.iter_num])  # 使用 context 中保存的 iter_num 值
        # ctx.mark_non_differentiable(iter_num)   #  标记 iter_num 为非可导
        # coeff = 2.0 * (high - low) / (1.0 + torch.exp(torch.true_divide(-alpha * iter_num,max_iter))) - (high - low) + low
        # p = float(ctx.iter_num + max_iter) / (max_iter)
        # coeff = torch.true_divide(2.0 * (high - low),(1.0 + torch.exp(torch.true_divide(-alpha * (ctx.iter_num + max_iter),max_iter))))- (high - low) + low
        # 实现后向计算逻辑
        # print(gradOutput)

        return gradOutput.neg() * ctx.coeff, None

class Discriminators(nn.Module):
    def __init__(self,config):
        super(Discriminators, self).__init__()
        self.fc1 = nn.Linear(768, 768)

        # print("--------fc1模型训练的参数量---------")
        # print(sum(p.numel() for p in self.fc1.parameters() if p.requires_grad))  # 打印模型参数量
        # self.fc1.weight.data.normal_(0, 0.01)
        # self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(768, 1) # 二分类
        # self.fc2.weight.data.normal_(0, 0.3)
        # self.fc2.bias.data.fill_(0.0)
        self.iter_num = 0
        self.alpha = -10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 328*20
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(True),
            self.fc2
        )
        # 加入GRL后理想的分类器损失是先下降后增加的趋势，最后网络无法对其分类。
        self.grl_layer = GRL()
        # self.optimizer = AdamW(self.parameters(), lr=0.001)

    def forward(self, feature, gamma):

        if self.training: self.iter_num = self.iter_num+1
        # 相等的时候coeff是0.46
        # coeff = torch.true_divide(2.0, (1.0 + torch.exp(torch.true_divide(self.alpha * (self.iter_num),self.max_iter))))- 1.0

        adversarial_out = self.ad_net(self.grl_layer.apply(feature, gamma))

        #self.grl_layer.apply(feature):表示对feature应用了自定义autograd函数的前向计算。apply方法会自动触发forward方法的执行。
        return adversarial_out