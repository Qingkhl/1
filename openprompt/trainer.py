import math
import os
import sys

import numpy as np
from torch import softmax
from torch.optim import Adam
import torch.nn.functional as F
sys.path.append(".")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import dataloader

from openprompt.utils.utils import load_checkpoint, save_checkpoint
from typing import Callable, OrderedDict, Union
from torch.nn.parallel.data_parallel import DataParallel
from openprompt.pipeline_base import PromptForClassification, DomainDiscriminators, ScheduledOptim
from tqdm import tqdm
import torch
from openprompt import PromptDataLoader
from openprompt.prompts import *
from openprompt.utils.logging import logger
from openprompt.utils.metrics import classification_metrics, generation_metric
from transformers import AdamW, get_linear_schedule_with_warmup


class BaseRunner(object):
    r"""A base runner for training without training tricks.
    Applying training tricks such as ensemble of template or verbalizer, 
    or self-training can use other runner class. 
    This class is specially implemented for classification.
    For generation task, though it can be integrated in this class
    via `task` option, we keep it as another class for simplicity.

    Args:
        prompt_model (:obj:`Union[DataParallel, PromptForClassification]`): One ``PromptModel`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
        config (:obj:`CfgNode`): A configuration object.
        loss_function (:obj:`Callable`, optional): The loss function in the training process.
    """
    def __init__(self, 
                 prompt_model: Union[DataParallel, PromptForClassification],
                 domain_adv = None,
                 train_dataloader = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None,
                 config: CfgNode = None,
                 ):
        self.prompt_model = prompt_model
        self.inner_model = prompt_model.module if isinstance(prompt_model, DataParallel) else prompt_model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.domain_adv = domain_adv
        self.config_optimize()

    
    def config_loss_function(self,):
        raise NotImplementedError
    
    def config_optimize(self,):
        raise NotImplementedError
 
    def evaluate(self, dataloader, split, post_evaluate_hook=None):
        raise NotImplementedError

    def train_epoch(self, epoch):
        raise NotImplementedError
    
    def prompt_initialize(self):
        r"""Some initialization works
        """
        pass

    def plot_tsne1(self,datas1, preds1, datas, preds, filename):
        # TSNE对两个数据集进行降维
        tsne1 = TSNE(n_components=2, learning_rate=100, init='pca', random_state=0, perplexity=10)
        tsne_features1 = tsne1.fit_transform(datas1)

        tsne = TSNE(n_components=2, learning_rate=100, init='pca', random_state=0, perplexity=10)
        tsne_features = tsne.fit_transform(datas)

        fig, ax = plt.subplots()  # 创建图形和轴
        plt.xticks([])
        plt.yticks([])

        # 绘制第一个降维后的数据集，使用绿色
        ax.scatter(tsne_features1[:, 0], tsne_features1[:, 1], s=10, c='red', label='源主题')

        # 绘制第二个降维后的数据集，使用红色
        ax.scatter(tsne_features[:, 0], tsne_features[:, 1], s=10, c='green', label='目标主题')

        # 可选：添加图例
        # ax.legend()
        # 添加图例，并指定放置位置为右上角
        # ax.legend(loc='upper right')

        # 保存图像到文件
        plt.savefig(filename)

    def run(self, start_epoch: int=0, max_score: float=0.0):

        if start_epoch == 0:
            self.prompt_initialize()
            max_score = None
            max_testscore = None
            max_att = None
        for epoch in range(start_epoch, self.config.train.num_epochs):
            # filename2 = '../' + os.path.join('img', str(self.config.dataset.target_domain)) + 'epoch' + str(epoch) + 'clsAtt.svg'

            total_loss,all_scores,all_features_target = self.train_epoch(epoch)
            scores, att_dic, all_scores0, all_features_target2 = self.evaluate(self.train_dataloader[0], "Valid") # 100%

            test_scores,test_att_dic,all_scores0, all_features_target2=self.evaluate(self.test_dataloader, "Test")

            # print("Valid performance:{}".format(att_dic['score']))
            print("Test performance:{}".format(test_att_dic['score']))
            model_state_dict = self.inner_model.state_dict()
            if self.config.plm.optimize.freeze_para:
                model_state_dict.pop('plm')
            state_dict = {
                "epoch": epoch+1,
                "state_dict": self.inner_model.state_dict(),
                "optimizer": [opt.state_dict() if isinstance(opt, torch.optim.Optimizer) else None for opt in self.optimizers] ,
                "scheduler": [sch.state_dict() if isinstance(sch, torch.optim.lr_scheduler._LRScheduler) else None for sch in self.schedulers],
                "scores": att_dic['score'],
                "max_score": max_score
            }
            cur_score = att_dic['score']
            # cur_score = sum(att_dic.values())/len(att_dic)
            is_best = ((cur_score - max_score) >= 0) == \
                self.config.checkpoint.higher_better if max_score is not None else True
            if is_best:
                max_score = cur_score
                max_testscore = test_att_dic['score']
                max_att = att_dic
            print("best performance:{}".format(max_testscore))
            # self.plot_tsne1(all_features_target.cpu().detach().numpy(), all_scores.cpu().detach().numpy(),
            #                 all_features_target2.cpu().detach().numpy(), all_scores0.cpu().detach().numpy(), filename2)
            save_checkpoint(state_dict = state_dict, 
                            is_best=(is_best and self.config.checkpoint.save_best),
                            save_path=self.config.logging.path, domain=self.config.dataset.target_domain)
        print("best performance:{}".format(max_testscore))
        state_dict = load_checkpoint(load_path=self.config.logging.path,
                        load_best = True,
                        map_location="cpu",
                        domain=self.config.dataset.target_domain,            # cpu to prevent CUDA out of memory.
                        )
        self.inner_model.load_state_dict(state_dict['state_dict'])
        # self.inner_model.to("cuda:{}".format(self.config.environment.local_rank))
        self.inner_model.to('cuda')

        print("best performance:{}".format(max_testscore))
        # print("best att performance:{}".format(max_att))
        # print("best test performance:{}".format(max_testscore))

        # self.evaluate(self.test_dataloader, "Test")

    def resume(self, ):
        logger.info("Resume Training ...")
        try:
            state_dict = load_checkpoint(load_path=self.config.logging.path,
                    load_best = True,
                    map_location="cpu",
                    domain=self.config.dataset.target_domain,# cpu to prevent CUDA out of memory.
                    )
        except FileNotFoundError:
            logger.warning("No checkpoint found in {}, start from scratch.".format(self.config.logging.path))
            self.run()
            return 
        
        # load state to model
        self.inner_model.load_state_dict(state_dict['state_dict'])
        # self.inner_model.to("cuda:{}".format(self.config.environment.local_rank))
        self.inner_model.to('cuda')
        # load state to optimizers
        for optimizer, op_state in zip(self.optimizers, state_dict['optimizer']):
            if isinstance(optimizer, torch.optim.Optimizer):
                optimizer.load_state_dict(op_state)
        for scheduler, sc_state in zip(self.schedulers, state_dict['scheduler']):
            if isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
                scheduler.load_state_dict(sc_state)
        # run
        self.run(start_epoch=state_dict['epoch'], max_score=state_dict['max_score'])
        
    def test(self, ):
        logger.info("Resume Training and direct test...")
        try:
            state_dict = load_checkpoint(load_path=self.config.logging.path,
                    load_best = True,
                    map_location="cpu",# cpu to prevent CUDA out of memory.
                    domain=self.config.dataset.target_domain,
                    )
        except FileNotFoundError:
            logger.error("No checkpoint found in {}, can't test.".format(self.config.logging.path))
            exit()

        # load state to model
        self.inner_model.load_state_dict(state_dict['state_dict'])
        # self.inner_model.to("cuda:{}".format(self.config.environment.local_rank))
        self.inner_model.to('cuda')
        self.evaluate(self.valid_dataloader, "Valid")




class ClassificationRunner(BaseRunner):
    r"""A runner for simple training without training tricks.
    Applying training tricks such as ensemble of template or verbalizer,
    or self-training can use other runner class.
    This class is specially implemented for classification.
    For generation task, though it can be integrated in this class
    via `task` option, we keep it as another class for simplicity.

    Args:
        prompt_model (:obj:`Union[DataParallel, PromptForClassification]`): One ``PromptModel`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
        config (:obj:`CfgNode`): A configuration object.
        loss_function (:obj:`Callable`, optional): The loss function in the training process.
    """
    def __init__(self, 
                 prompt_model: Union[DataParallel, PromptForClassification],
                 domain_adv = None,
                 domain_kl = None,
                 train_dataloader = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None,
                 config: CfgNode = None,
                 loss_function: Optional[Callable] = None,
                 ):
        super().__init__(prompt_model=prompt_model,
                         domain_adv=domain_adv,
                         train_dataloader=train_dataloader,
                         valid_dataloader=valid_dataloader,
                         test_dataloader=test_dataloader,
                         config=config)

        self.domain_adv = domain_adv
        self.domain_kl = domain_kl

        if loss_function is None:
            self.config_loss_function()
        else:
            self.loss_function = loss_function
    
    def config_loss_function(self,):
        r"""config the loss function if it's not passed.
        """
        if self.config.classification.loss_function == "cross_entropy":
            self.loss_function = torch.nn.CrossEntropyLoss()
        elif self.config.classification.loss_function == "nll_loss":
            self.loss_function = torch.nn.NLLLoss()
        else:
            raise NotImplementedError
    
    def config_optimize(self,):
        r"""config the optimizer and scheduler for 1. model 2. template 3. verbalizer
        
        """
        if isinstance(self.train_dataloader, list):
            self.train_steps_per_epoch = len(self.train_dataloader[0]) // self.config.train.gradient_accumulation_steps
        else:
            self.train_steps_per_epoch = len(self.train_dataloader) // self.config.train.gradient_accumulation_steps
        num_training_steps = self.train_steps_per_epoch * self.config.train.num_epochs

        # self.model_optimizer = self.inner_model.optimizer
        self.model_optimizer = ScheduledOptim(
         torch.optim.Adam(
             [
               {"params": filter(lambda p: p.requires_grad, self.inner_model.parameters())},
              {"params": filter(lambda p: p.requires_grad, self.domain_adv.parameters())},
             ],
            lr=0.01,
            betas=(0.9, 0.98),
            eps=1e-09
            ), 0.01)

        self.model_scheduler = None
        class Dummy:
            pass

        ## template_config 
        template_config = self.config[self.config.template]
        if hasattr(template_config, "optimize") and template_config.optimize is not None:
            if not hasattr(self.inner_model.template, "optimize"):
                # using default gradient descent optimizer.
                self.template_optimizer = AdamW(self.inner_model.template.parameters(), lr = template_config.optimize.lr)
                if hasattr(template_config.optimize, "scheduler") and template_config.optimize.scheduler is not None:
                    self.template_scheduler = get_linear_schedule_with_warmup(
                        self.template_optimizer, 
                        num_warmup_steps = template_config.optimize.scheduler.num_warmup_steps, 
                        num_training_steps = num_training_steps
                    )
                else:
                    self.template_scheduler = None
            else:
                self.template_optimizer = Dummy()
                # resemble a pytorch optimizer for unified training.
                setattr(self.template_optimizer, "step", self.inner_model.template.optimize)
                setattr(self.template_optimizer, "zero_grad", lambda:None)
                self.template_scheduler = None
        else:
            self.template_optimizer = None
            self.template_scheduler = None
            

        ## verbalizer_optimizer
        verbalizer_config = self.config[self.config.verbalizer]
        if hasattr(verbalizer_config, "optimize") and verbalizer_config.optimize is not None:
            if not hasattr(self.inner_model.verbalizer, "optimize"):
                # using default gradient descent optimizer.
                self.verbalizer_optimizer = AdamW(self.inner_model.verbalizer.parameters(), lr = verbalizer_config.optimize.lr)
                if hasattr(verbalizer_config.optimize, "scheduler") and verbalizer_config.optimize.scheduler is not None:
                    self.verbalizer_scheduler = get_linear_schedule_with_warmup(
                        self.verbalizer_optimizer, 
                        num_warmup_steps = verbalizer_config.optimize.scheduler.num_warmup_steps, 
                        num_training_steps = num_training_steps
                    )
                else:
                    self.verbalizer_scheduler = None
            else:
                self.verbalizer_optimizer = Dummy()
                # resemble a pytorch optimizer for unified training.
                setattr(self.verbalizer_optimizer, "step", self.inner_model.verbalizer.optimize)
                setattr(self.verbalizer_optimizer, "zero_grad", lambda:None)
                self.verbalizer_scheduler = None
        else:
            self.verbalizer_optimizer = None
            self.verbalizer_scheduler = None

        self.optimizers = [self.model_optimizer, self.template_optimizer, self.verbalizer_optimizer]
        self.schedulers = [self.model_scheduler, self.template_scheduler, self.verbalizer_scheduler]

    def confusion_matrix(self,rater_a, rater_b, min_rating=None, max_rating=None):
        assert (len(rater_a) == len(rater_b))
        if min_rating is None:
            min_rating = min(rater_a + rater_b)
        if max_rating is None:
            max_rating = max(rater_a + rater_b)
        num_ratings = int(max_rating - min_rating + 1)
        conf_mat = [[0 for i in range(num_ratings)]
                    for j in range(num_ratings)]
        for a, b in zip(rater_a, rater_b):
            conf_mat[a - min_rating][b - min_rating] += 1
        return conf_mat

    def histogram(self,ratings, min_rating=None, max_rating=None):
        if min_rating is None:
            min_rating = min(ratings)
        if max_rating is None:
            max_rating = max(ratings)
        num_ratings = int(max_rating - min_rating + 1)
        # num_ratings=6
        hist_ratings = [0 for x in range(num_ratings)]
        # hist_ratings=[0, 0, 0, 0, 0, 0]
        for r in ratings:
            # print(r)--->第一次循环取得是rater_a的第一个值：9
            hist_ratings[r - min_rating] += 1
        # print(hist_ratings)--->[0, 0, 6, 4, 0, 0]
        # hist_ratings[2]处为6表示预测结果为8的数据有6条(一共10条数据)
        return hist_ratings

    def quadratic_weighted_kappa(self,rater_a, rater_b, min_rating=None, max_rating=None):
        assert (len(rater_a) == len(rater_b))
        if min_rating is None:
            min_rating = min(min(rater_a), min(rater_b))
        if max_rating is None:
            max_rating = max(max(rater_a), max(rater_b))
        # print(min_rating)----0
        # print(max_rating)-----3
        # 如果采用上面的预测值和真实值例子，min_rating=6，max_rating=11.此时矩阵为11-6+1=6,6行6列矩阵---》因为取值范围是6,7,8,9,10,11这6个种类
        conf_mat = self.confusion_matrix(rater_a, rater_b,
                                    min_rating, max_rating)

        num_ratings = len(conf_mat)

        num_scored_items = float(len(rater_a))

        hist_rater_a = self.histogram(rater_a, min_rating, max_rating)

        hist_rater_b = self.histogram(rater_b, min_rating, max_rating)

        numerator = 0.0
        denominator = 0.0
        if num_ratings != 1:
            for i in range(num_ratings):
                for j in range(num_ratings):
                    expected_count = (hist_rater_a[i] * hist_rater_b[j]
                                      / num_scored_items)
                    d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
                    numerator += d * conf_mat[i][j] / num_scored_items
                    denominator += d * expected_count / num_scored_items
        else:
            numerator = 0.0
            denominator = 1.0
        return 1.0 - numerator / denominator

    def get_score_vector_positions(self,):
        return {
            'score': 0,
            # 'content': 1,
            # 'organization': 2,
            # 'word_choice': 3,
            # 'sentence_fluency': 4,
            # 'conventions': 5,
            # 'prompt_adherence': 6,
            # 'language': 7,
            # 'narrativity': 8
        }

    def get_score_vector_positions2(self,):
        return {
            'score': 0,
            # 'content': 1,
            # 'organization': 2,
            # 'word_choice': 3,
            # 'sentence_fluency': 4,
            # 'conventions': 5,
            # 'prompt_adherence': 6,
            # 'language': 7,
            # 'narrativity': 8
        }
    def get_min_max_scores(self,):
        return {
            1: {'score': (2, 12), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
                'sentence_fluency': (1, 6), 'conventions': (1, 6)},
            2: {'score': (1, 6), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
                'sentence_fluency': (1, 6), 'conventions': (1, 6)},
            3: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3),
                'narrativity': (0, 3)},
            4: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3),
                'narrativity': (0, 3)},
            5: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4),
                'narrativity': (0, 4)},
            6: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4),
                'narrativity': (0, 4)},
            7: {'score': (0, 30), 'content': (0, 6), 'organization': (0, 6), 'conventions': (0, 6)},
            8: {'score': (0, 60), 'content': (2, 12), 'organization': (2, 12), 'word_choice': (2, 12),
                'sentence_fluency': (2, 12), 'conventions': (2, 12)}}

    def get_min_max_scores2(self,):
        return {
            1: {'score': (2, 12)},
            2: {'score': (1, 6)},
            3: {'score': (0, 3)},
            4: {'score': (0, 3)},
            5: {'score': (0, 4)},
            6: {'score': (0, 4)},
            7: {'score': (0, 30)},
            8: {'score': (0, 60)}}

    def separate_and_rescale_attributes_for_scoring(self, scores, set_id):
        score_vector_positions = self.get_score_vector_positions2()
        min_max_scores = self.get_min_max_scores2()
        individual_att_scores_dict = {}
        score_set_comb = list(scores)
        for att_scores in score_set_comb:
            for relevant_attribute in min_max_scores[set_id].keys():
                min_score = min_max_scores[set_id][relevant_attribute][0] # 得分最小值
                max_score = min_max_scores[set_id][relevant_attribute][1] # 得分最大值
                att_position = score_vector_positions[relevant_attribute] # 属性得分的位置
                att_score = att_scores[att_position] # 属性得分
                rescaled_score = att_score * (max_score - min_score) + min_score # 放缩后的得分
                try:
                    individual_att_scores_dict[relevant_attribute].append(np.around(rescaled_score.numpy()).astype(int))
                except KeyError:
                    individual_att_scores_dict[relevant_attribute] = [np.around(rescaled_score.numpy()).astype(int)]
        return individual_att_scores_dict

    def separate_and_rescale_attributes_for_scoring2(self, scores, set_id):
        score_vector_positions = self.get_score_vector_positions2()
        min_max_scores = self.get_min_max_scores2()
        individual_att_scores_dict = {}
        score_set_comb = list(scores)
        for att_scores in score_set_comb:
            for relevant_attribute in min_max_scores[set_id].keys():
                att_position = score_vector_positions[relevant_attribute] # 属性得分的位置
                att_score = att_scores[att_position] # 属性得分
                rescaled_score = att_score # 放缩后的得分
                try:
                    individual_att_scores_dict[relevant_attribute].append(np.around(rescaled_score.numpy()).astype(int))
                except KeyError:
                    individual_att_scores_dict[relevant_attribute] = [np.around(rescaled_score.numpy()).astype(int)]
        return individual_att_scores_dict

    def get_scaled_down_scores(self, scores, prompts):
        score_positions = self.get_score_vector_positions()
        min_max_scores = self.get_min_max_scores2()
        score_prompts = zip(scores, prompts)
        scaled_score_list = []
        for score_vector, prompt in score_prompts:
            rescaled_score_vector = [-1] * len(score_positions)
            for ind, att_val in enumerate(score_vector):
                if att_val != -1:
                    attribute_name = list(score_positions.keys())[list(score_positions.values()).index(ind)]
                    min_val = min_max_scores[int(prompt)][attribute_name][0]
                    max_val = min_max_scores[int(prompt)][attribute_name][1]
                    scaled_score = (att_val - min_val) / (max_val - min_val)
                    rescaled_score_vector[ind] = scaled_score
            scaled_score_list.append(torch.tensor(rescaled_score_vector))
        assert len(scaled_score_list) == len(scores)
        for scores in scaled_score_list:
            assert min(scores) >= -1
            assert max(scores) <= 1
        return scaled_score_list

    def evaluate(self, dataloader, split, post_evaluate_hook=None):
        num = 0
        qwk = 0
        qwk_att2 = {}
        self.prompt_model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=split):
                num = num+1
                # batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
                batch = batch.to('cuda').to_dict()
                label = batch['label'].long().to('cuda')
                label = label.cpu()
                batch.pop('label')
                batch['AT'] = False
                logits,total_score_feat,_,attributes_pre = self.prompt_model(batch)
                test_pred_dict = self.separate_and_rescale_attributes_for_scoring(attributes_pre.cpu(), self.config.dataset.target_domain)
                test_att_lable_dict = self.separate_and_rescale_attributes_for_scoring2(batch['attributes'].cpu(), self.config.dataset.target_domain)
                pred = torch.argmax(logits, dim=-1) # 概率最高的类别
                qwk_att = {key: self.quadratic_weighted_kappa(test_pred_dict[key], test_att_lable_dict[key]) for key in test_pred_dict.keys()}
                scores = batch['attributes'].cpu()[:, 0]
                if num == 1:
                    all_scores0 = scores
                    all_features_target2 = total_score_feat.cpu()
                else:
                    all_scores0 = torch.cat([all_scores0, scores], 0)
                    all_features_target2 = torch.cat([all_features_target2, total_score_feat.cpu()], 0)

                if num==1:
                    qwk_att2 = qwk_att
                else:
                    for key, value in qwk_att.items():
                        # 将第一个字典中的值加到第二个字典中对应键的值上
                        qwk_att2[key] += value
                qwk += self.quadratic_weighted_kappa(pred.cpu(), label)
        for key, value in qwk_att2.items():
            qwk_att2[key] = value/num
        # print("CLS Performance: {}".format(qwk/num))
        # for att in qwk_att2.keys():
        #     print('{} QWK: {}'.format(att, round(qwk_att2[att], 5)))
        # logger.info("Performance: {}".format(qwk/num))
        return qwk/num,qwk_att2,all_scores0,all_features_target2

    def Entropy(self, input_):
        bs = input_.size(0)
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    def masked_loss_function_torch(self, y_true, y_pred, mask_value=-1):
        mask = (y_true != mask_value).float().cuda()
        mse_loss = F.mse_loss(y_true.cuda() * mask, y_pred * mask)
        return mse_loss

    def train_epoch(self, epoch):
        self.prompt_model.train()
        self.prompt_model.zero_grad()
        total_loss = 0.0
        total_dloss = 0.0
        sum_loss = 0.0
        sum_dloss = 0.0
        num = 0
        if not isinstance(self.train_dataloader, list):
            pbar = tqdm(self.train_dataloader, desc="Train epoch {}".format(epoch))
        else:
            pbar = tqdm(zip(*self.train_dataloader), desc="Train epoch {}".format(epoch))
        NC = False
        NA = True
        # ！！！！！！！！！！！ memory_bank初始化
        if NA:
            mem_fea = torch.rand(1445, 768).cuda() # 8域的train数据长度为579, 5是1445
            mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
            mem_cls = torch.ones(1445, 4).cuda() / 4 # (data_len,4)
            mem_fea2 = torch.rand(1445, 768).cuda()  # 8域的train数据长度为579, 5是1445
            mem_fea2 = mem_fea2 / torch.norm(mem_fea2, p=2, dim=1, keepdim=True)
            mem_cls2 = torch.ones(1445, 4).cuda() / 4  # (data_len,4)
        if NC:
            mem_fea = torch.rand(4, 768).cuda()
            mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
        # ！！！！！！！！！！！！

        now_iter = 328 * epoch
        domains = len(self.config.dataset.domains)-1
        loss_domain = 0
        qwk = 0
        for step, batches in enumerate(pbar):
            # batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
            now_iter = now_iter + 1
            logits_list, hidden_list, labels, atts_pre_list,atts_lable_list,prompt_ids = [], [], [],[],[],[]

            for optimizer in self.optimizers:
                if optimizer is not None:
                    optimizer.zero_grad()

            for _, batch in enumerate(batches):
                batch = batch.to('cuda').to_dict()
                batch['AT'] = True
                logits, hiddens, logits_all,attributes_pre = self.prompt_model(batch) # (4,4) (4,768)
                hidden_list.append(hiddens)
                # if step ==0: print(hiddens)
                # if step % 200 == 0: self.predict_voc(logits_all, batch)
                logits_list.append(logits)
                atts_pre_list.append(attributes_pre)
                labels.append(batch['label'])
                atts_lable_list.append(batch['attributes'])
                prompt_ids.append(batch['essay_set'])

            logits_target = logits
            logits_list = logits_list[:-1]
            atts_pre_list = atts_pre_list[:-1]
            labels = labels[:-1]
            atts_lable_list = atts_lable_list[:-1]
            prompt_ids = prompt_ids[:-1]

            logits = torch.cat(logits_list, dim=0)
            labels = torch.cat(labels, dim=0)
            atts_lable_list = torch.cat(atts_lable_list, dim=0)
            atts_pre_list = torch.cat(atts_pre_list, dim=0)
            prompt_ids = torch.cat(prompt_ids, dim=0)

            pred = torch.argmax(logits, dim=-1)  # 概率最高的类别
            qwk += self.quadratic_weighted_kappa(pred, labels)
            if step % 20 == 1:
                print("average qwk {} lr:{}".format(qwk/2 / (step + 1), self.optimizers[0]._optimizer.param_groups[0]['lr']))

            loss_task = self.loss_function(logits, labels)
            atts_lable_list = self.get_scaled_down_scores(atts_lable_list.cpu(), prompt_ids.cpu())
            atts_lable_list = torch.stack(atts_lable_list)
            loss_att = self.masked_loss_function_torch(atts_lable_list,atts_pre_list) # lable没有缩小到0-1之间
            loss = loss_task + 10*loss_att
            # print(atts_pre_list)
            pbar.set_postfix({'loss_task=': loss})
            pbar.set_postfix({'loss_att=': loss_att})
            # 对抗
            if self.config.dataset.adv:
                # gamma = 2 / (1 + math.exp(-10 * (now_iter) / 20*328)) - 1 # 用这个除号会越界导致结果一直为1.0
                gamma = torch.true_divide(2.0, (1.0 + torch.exp(torch.true_divide(-10 * (now_iter), 328 * 30)))) - 1.0
                gamma2 = torch.true_divide(2.0, (1.0 + torch.exp(torch.true_divide(-10 * (now_iter+328*50), 328 * 30)))) - 1.0
                gamma2 = gamma2.cuda()
                loss_domain = torch.mean(self.domain_adv(hidden_list, gamma2.unsqueeze(-1))) # 输入的是8*（4，768），但由于并行4个卡，每个卡分到8*(1,768)
                # if step % 40 == 0: print(gamma)
                pbar.set_postfix({'loss_domain=': loss_domain})
                loss += gamma * loss_domain
                sum_dloss = loss_domain.item()

            # 打伪标签!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # im防止一直打同一个类别的伪标签
            softmax_out = torch.nn.Softmax(dim=1)(logits_target)
            loss_emin = torch.mean(self.Entropy(softmax_out))
            msoftmax_out = softmax_out.mean(dim=0)
            loss_gemin = torch.sum(-msoftmax_out * torch.log(msoftmax_out + 0.00001))
            loss_emin -= loss_gemin
            loss_im = loss_emin
            if NA:
                dis = -torch.mm(hidden_list[domains].detach(), mem_fea.t())
                idx = batches[domains].index.tolist()
                for di in range(dis.size(0)):
                    dis[di, idx[di]] = torch.max(dis)
                _, p1 = torch.sort(dis, dim=1)
                w = torch.zeros(hidden_list[domains].size(0), mem_fea.size(0)).cuda()
                for wi in range(w.size(0)):
                    for wj in range(5):
                        w[wi][p1[wi, wj]] = 1 / 5
                weight_, pred = torch.max(w.mm(mem_cls), 1) # 最大值和下标
                # max_probs, label_p = torch.max(pseudo_label, dim=-1)
                # mask = weight_.ge(0.6).float()
                # class_loss_u_G = (F.cross_entropy(soft_class_token_u, domain_u_label,reduction="none") * mask).sum() / mask.sum()
                loss_ = torch.nn.CrossEntropyLoss(reduction='none')(logits_target, pred) # 这里的损失有很多0，pred每次都是1，1，1，1，logits下标为1的地方也确实最大
                classifier_loss = torch.sum(weight_ * loss_) / (torch.sum(weight_).item()) ## classifier_loss慢慢变为0

                eff = now_iter / (328*30)
                loss = loss + 0.2 * eff * classifier_loss
                loss = loss + eff * loss_im
                if step % 20 == 0: print(pred)
                if step % 20 == 0: print(weight_)
                pbar.set_postfix({'loss_pseudo=': classifier_loss})

            # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！问题：mem_cls有很多0.25

            sum_loss += loss.item()

            loss.backward()
            for optimizer in self.optimizers:
                if optimizer is not None:
                    optimizer.step()

            # ！！！！！！！！！！！！！！！！！！！！！
            # ！！！！！！！！！！！！！！！！！！！！！
            if NA:
                self.prompt_model.eval()
                with torch.no_grad():
                    target_batch=batches[domains].to('cuda').to_dict()
                    target_batch['AT'] = True
                    outputs_target, features_target,_,_ = self.prompt_model(target_batch)
                    features_target = features_target / torch.norm(features_target, p=2, dim=1, keepdim=True)
                    softmax_out = torch.nn.Softmax(dim=1)(outputs_target)
                    outputs_target = softmax_out
                mem_fea[idx] = 0.1 * mem_fea[idx] + 0.9 * features_target.clone()
                mem_cls[idx] = 0.1 * mem_cls[idx] + 0.9 * outputs_target.clone()
            if NC:
                self.prompt_model.eval()
                with torch.no_grad():
                    outputs_target, features_target, _,_ = self.prompt_model(batches[domains].to('cuda').to_dict())
                    softmax_t = torch.nn.Softmax(dim=1)(outputs_target)
                    _, pred_t = torch.max(softmax_t, 1)
                    onehot_t = torch.eye(4)[pred_t].cuda()
                    center_t = torch.mm(features_target.t(), onehot_t) / (onehot_t.sum(dim=0) + 1e-8)
                mem_fea = (1.0 - 0.1) * mem_fea + 0.1 * center_t.t().clone()
            # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            total_dloss += sum_dloss
            total_loss += sum_loss
            sum_loss = 0.
            sum_dloss = 0.

            # ----------------------------------------------------------------------------------------分割线
            # ----------------------------------------------------------------------------------------分割线
            # ----------------------------------------------------------------------------------------分割线
            # batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
            logits_list, hidden_list, labels, atts_pre_list,atts_lable_list,prompt_ids = [], [], [],[],[],[]

            for optimizer in self.optimizers:
                if optimizer is not None:
                    optimizer.zero_grad()

            for j, batch in enumerate(batches):
                batch = batch.to('cuda').to_dict()
                batch['AT'] = False
                logits, hiddens, logits_all,attributes_pre = self.prompt_model(batch)  # (4,4) (4,768)
                hidden_list.append(hiddens)
                # if step ==0: print(hiddens)
                # if step % 200 == 0: self.predict_voc(logits_all, batch)

                logits_list.append(logits)
                atts_pre_list.append(attributes_pre)
                labels.append(batch['label'])
                atts_lable_list.append(batch['attributes'])
                prompt_ids.append(batch['essay_set'])
                if step == 0:
                    all_scores = batch['attributes'].cpu()[:,0]
                    all_features_target = hiddens.cpu().detach()
                else:
                    if j == 7: continue
                    all_scores = torch.cat([all_scores, batch['attributes'].cpu()[:,0]], 0)
                    all_features_target = torch.cat([all_features_target, hiddens.cpu().detach()], 0)

            logits_target = logits
            logits_list = logits_list[:-1]
            atts_pre_list = atts_pre_list[:-1]
            labels = labels[:-1]
            atts_lable_list = atts_lable_list[:-1]
            prompt_ids = prompt_ids[:-1]

            logits = torch.cat(logits_list, dim=0)
            labels = torch.cat(labels, dim=0)
            atts_lable_list = torch.cat(atts_lable_list, dim=0)
            atts_pre_list = torch.cat(atts_pre_list, dim=0)
            prompt_ids = torch.cat(prompt_ids, dim=0)

            pred = torch.argmax(logits, dim=-1)  # 概率最高的类别
            qwk += self.quadratic_weighted_kappa(pred, labels)
            if step % 20 == 1:
                print("average qwk {} lr:{}".format(qwk / (step + 1),
                                                    self.optimizers[0]._optimizer.param_groups[0]['lr']))

            loss_task = self.loss_function(logits, labels)
            atts_lable_list = self.get_scaled_down_scores(atts_lable_list.cpu(), prompt_ids.cpu())
            atts_lable_list = torch.stack(atts_lable_list)
            loss_att = self.masked_loss_function_torch(atts_lable_list, atts_pre_list)
            loss = loss_task + 10*loss_att

            pbar.set_postfix({'loss_task=': loss_task})
            pbar.set_postfix({'loss_att=': loss_att})


            # 打伪标签!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # im防止一直打同一个类别的伪标签
            softmax_out = torch.nn.Softmax(dim=1)(logits_target)
            loss_emin = torch.mean(self.Entropy(softmax_out))
            msoftmax_out = softmax_out.mean(dim=0)
            loss_gemin = torch.sum(-msoftmax_out * torch.log(msoftmax_out + 0.00001))
            loss_emin -= loss_gemin
            loss_im = loss_emin
            if NA:
                dis = -torch.mm(hidden_list[domains].detach(), mem_fea2.t())
                idx = batches[domains].index.tolist()
                for di in range(dis.size(0)):
                    dis[di, idx[di]] = torch.max(dis)
                _, p1 = torch.sort(dis, dim=1)
                w = torch.zeros(hidden_list[domains].size(0), mem_fea2.size(0)).cuda()
                for wi in range(w.size(0)):
                    for wj in range(5):
                        w[wi][p1[wi, wj]] = 1 / 5
                weight_, pred = torch.max(w.mm(mem_cls2), 1)
                # max_probs, label_p = torch.max(pseudo_label, dim=-1)
                # mask = weight_.ge(0.6).float()
                # class_loss_u_G = (F.cross_entropy(soft_class_token_u, domain_u_label,reduction="none") * mask).sum() / mask.sum()
                loss_ = torch.nn.CrossEntropyLoss(reduction='none')(logits_target,pred)  # 这里的损失有很多0，pred每次都是1，1，1，1，logits下标为1的地方也确实最大

                classifier_loss = torch.sum(weight_ * loss_) / (torch.sum(weight_).item()) ## classifier_loss慢慢变为0

                eff = now_iter / (328 * 30)
                loss = loss + 0.2 * eff * classifier_loss
                loss = loss + eff * loss_im
                if step % 20 == 0: print(pred)
                if step % 20 == 0: print(weight_)
                if step % 20 == 0: print(batch['label'])
                pbar.set_postfix({'loss_pseudo=': classifier_loss})
            if NC:
                mem_fea_norm = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
                dis = torch.mm(hidden_list[domains].detach(), mem_fea_norm.t())
                _, pred = torch.max(dis, dim=1)
                classifier_loss = torch.nn.CrossEntropyLoss()(logits_target, pred)
                eff = now_iter / (328 * 30)
                total_loss += eff * classifier_loss
                pbar.set_postfix({'loss_pseudo=': classifier_loss})
            # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！问题：mem_cls有很多0.25

            sum_loss += loss.item()

            loss.backward()
            for optimizer in self.optimizers:
                if optimizer is not None:
                    optimizer.step()

            # ！！！！！！！！！！！！！！！！！！！！！
            # ！！！！！！！！！！！！！！！！！！！！！
            if NA:
                self.prompt_model.eval()
                with torch.no_grad():
                    target_batch = batches[domains].to('cuda').to_dict()
                    target_batch['AT'] = True
                    outputs_target, features_target, _, _ = self.prompt_model(target_batch)
                    features_target = features_target / torch.norm(features_target, p=2, dim=1, keepdim=True)
                    softmax_out = torch.nn.Softmax(dim=1)(outputs_target)
                    outputs_target = softmax_out
                mem_fea2[idx] = 0.1 * mem_fea2[idx] + 0.9 * features_target.clone()
                mem_cls2[idx] = 0.1 * mem_cls2[idx] + 0.9 * outputs_target.clone()
            if NC:
                self.prompt_model.eval()
                with torch.no_grad():
                    outputs_target, features_target, _, _ = self.prompt_model(batches[domains].to('cuda').to_dict())
                    softmax_t = torch.nn.Softmax(dim=1)(outputs_target)
                    _, pred_t = torch.max(softmax_t, 1)
                    onehot_t = torch.eye(4)[pred_t].cuda()
                    center_t = torch.mm(features_target.t(), onehot_t) / (onehot_t.sum(dim=0) + 1e-8)
                mem_fea = (1.0 - 0.1) * mem_fea + 0.1 * center_t.t().clone()
            # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            total_dloss += sum_dloss
            total_loss += sum_loss
            sum_loss = 0.
            sum_dloss = 0.
        print("Epoch {}, avg_loss: {:.4f}, total_loss: {:.4f},avg_domain_loss:{:.4f}".format(epoch, total_loss / self.train_steps_per_epoch, total_loss, total_dloss / self.train_steps_per_epoch))
        return total_loss,all_scores,all_features_target


    def prompt_initialize(self):
        verbalizer_config = self.config[self.config.verbalizer]
        template_config = self.config[self.config.template]
        if not hasattr(self.inner_model.verbalizer, "optimize_to_initialize" ) and \
            not hasattr(self.inner_model.template, "optimize_to_initialize" ):
            return None
        if hasattr(verbalizer_config, "init_using_split"):
            using_split = verbalizer_config.init_using_split
        elif hasattr(template_config, "init_using_split"):
            using_split = template_config.init_using_split
        else:
            using_split = "valid"

        if using_split == "train":
            dataloader = self.train_dataloader
        elif using_split == "valid":
            dataloader = self.valid_dataloader
        else:
            raise NotImplementedError

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Init_using_{}".format(using_split)):
                # batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
                batch = batch.to('cuda').to_dict()
                logits,_ = self.prompt_model(batch)
            if hasattr(self.inner_model.verbalizer, "optimize_to_initialize" ):
                self.inner_model.verbalizer.optimize_to_initialize()
            if hasattr(self.inner_model.template, "optimize_to_initialize" ):
                self.inner_model.template.optimize_to_initialize()

