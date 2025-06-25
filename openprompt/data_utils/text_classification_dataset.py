
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for loading data for all TextClassification tasks.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn import preprocessing

from datasets.AES.split_class import get_model_friendly_scores
from openprompt.data_utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor

def get_score_vector_positions():
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

class AESProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["0", "1", "2", "3"]

    def get_readability_features(self,readability_path):
        with open(readability_path, 'rb') as fp:
            readability_features = pickle.load(fp)
        return readability_features

    # 读取CSV文件数据，并将其以pandas的DataFrame形式返回，方便后续进一步分析和处理数据。
    def get_linguistic_features(self, linguistic_features_path):
        features_df = pd.read_csv(linguistic_features_path)
        return features_df

    # 对输入的特征数据 features_df 进行最小最大标准化，使得特定列的数值被缩放到0到1之间，忽略了不需要标准化处理的列，最后返回一个包含了标准化数据的新DataFrame。
    def get_normalized_features(self, features_df):
        # 首先，函数定义了一个不需要标准化的列名称列表 column_names_not_to_normalize，包括 item_id、prompt_id 和 score。
        column_names_not_to_normalize = ['item_id', 'prompt_id', 'score']
        # 然后，函数创建了一个要标准化的列名称列表 column_names_to_normalize，这个列表初识时包含DataFrame features_df 中的所有列名称。
        column_names_to_normalize = list(features_df.columns.values)
        # 循环移除不需要标准化的列名称，即 column_names_not_to_normalize 中的列。
        for col in column_names_not_to_normalize:
            column_names_to_normalize.remove(col)
        # 创建一个新的列名称列表 final_columns,以 item_id 开始,之后是需要进行标准化的列名称。
        final_columns = ['item_id'] + column_names_to_normalize
        # 初始化变量 normalized_features_df 为 None，用于之后存储最终的标准化特征。
        normalized_features_df = None
        # 循环从1到8（假设代表不同的prompt_id），针对每个 prompt_ 执行以下操作：
        for prompt_ in range(1, 9):
            # 通过条件过滤获取特定 prompt_id 的数据子集 prompt_id_df。
            is_prompt_id = features_df['prompt_id'] == prompt_
            prompt_id_df = features_df[is_prompt_id].copy()
            # 从子集中提取要标准化的列，用 .values 获取其数值。
            x = prompt_id_df[column_names_to_normalize].values
            # 创建 MinMaxScaler 实例用于进行最小最大标准化。
            min_max_scaler = preprocessing.MinMaxScaler()
            # 使用 fit_transform 对数值进行标准化处理，并得到标准化后的数据 normalized_pd1。
            normalized_pd1 = min_max_scaler.fit_transform(x)
            # 将标准化后的数据转换成DataFrame df_temp，列名和标准化处理之前的列相同，索引使用过滤后子集的索引。
            df_temp = pd.DataFrame(normalized_pd1, columns=column_names_to_normalize, index=prompt_id_df.index)
            # 将原始子集 prompt_id_df 中要标准化的列替换为标准化后的数据。
            prompt_id_df[column_names_to_normalize] = df_temp
            # 创建 final_df 保留 final_columns 定义的列。
            final_df = prompt_id_df[final_columns]
            # 对于每个 prompt_ 处理后的 final_df，
            # 如果 normalized_features_df 不是 None，则将其与之前的数据合并；
            # 如果是第一次，则直接赋值 final_df 给 normalized_features_df。
            if normalized_features_df is not None:
                normalized_features_df = pd.concat([normalized_features_df, final_df], ignore_index=True)
            else:
                normalized_features_df = final_df
        # 最后返回 normalized_features_df，包含了所有标准化后的特征及其他非标准化列。
        return normalized_features_df

    # data_dir:数据目录,split:类别
    def get_examples(self, data_dir, split, domain):
        examples = []
        datas, labels, essay_ids, new_datas = [], [], [], []
        idx = -1
        # 直接读取手工特征然后添加到对应的文章中
        readability_features = self.get_readability_features('../datasets/AES/allreadability.pickle')
        linguistic_features = self.get_linguistic_features('../datasets/AES/hand_crafted_v3.csv')
        normalized_linguistic_features = self.get_normalized_features(linguistic_features)
        # with open(os.path.join(data_dir, '{}'.format(split)), mode='r', encoding='utf-8-sig') as input_file:
        with open(os.path.join(data_dir, '{}'.format(split+'.pk')), mode='rb') as input_file:
            train_essays_list = pickle.load(input_file)
            for essay in train_essays_list:
                idx = idx + 1
                essay_set = int(essay['prompt_id']) # 主题==domain
                essay_id = int(essay['essay_id']) # 文章编号
                content = essay['content_text'].strip() # 文章内容
                score = int(essay['score']) # 最终得分
                scores_and_positions = get_score_vector_positions()  # 字典:{'score': 0, 'content': 1, 'organization': 2, 'word_choice': 3, 'sentence_fluency': 4, 'conventions': 5, 'prompt_adherence': 6, 'language': 7, 'narrativity': 8}
                y_vector = [-1] * 9
                for score2 in scores_and_positions.keys():
                    if score2 in essay.keys():
                        y_vector[scores_and_positions[score2]] = int(essay[score2])

                # out_data['data_y'].append(y_vector)

                if essay_set == domain:
                    # 可读性特征
                    item_index = np.where(readability_features[:, :1] == essay_id)
                    item_row_index = item_index[0][0]
                    item_features = readability_features[item_row_index][1:]
                    # out_data['readability_x'].append(item_features)
                    # 手工特征
                    feats_df = normalized_linguistic_features[normalized_linguistic_features.loc[:, 'item_id'] == essay_id]
                    feats_craft = np.array(feats_df.values.tolist()[0][1:])
                    # out_data['features_x'].append(feats_list)
                    # 手工特征end
                    category = get_model_friendly_scores(score, domain)
                    # guid(每句话的id,可选) text_a（输入的文本数据）、text_b （分析两句的关系会用，不必有）、label （分类中标签的ID）、
                    example = InputExample(guid=essay_id, text_a=content, label=category, index = idx,
                                           essay_set = essay_set,feats_craft=feats_craft,item_features=item_features,attributes=y_vector) # 封装
                    examples.append(example)

        return examples

    def get_examples2(self, config, split="train"):
        examples = []
        datas, labels, essay_ids, new_datas = [], [], [], []
        with open(os.path.join(config.dataset.path, '{}.txt'.format(split)), mode='r', encoding='utf-8-sig') as input_file:

            for line in input_file:
                tokens = line.strip().split('\t')
                essay_set = int(tokens[1])  # 主题==domain
                content = tokens[2].strip()  # 文章内容
                score = float(tokens[6])  # 最终得分
                if split=="train":
                    if essay_set != config.dataset.target_domain:
                        category = get_model_friendly_scores(score, essay_set)
                        # guid(每句话的id,可选) text_a（输入的文本数据）、text_b （分析两句的关系会用，不必有）、label （分类中标签的ID）、
                        example = InputExample(guid=tokens[0], text_a=content, label=category)  # 纯文本信息和label
                        examples.append(example)
                elif split == "dev":
                    if essay_set == config.dataset.target_domain:
                        category = get_model_friendly_scores(score, essay_set)
                        # guid(每句话的id,可选) text_a（输入的文本数据）、text_b （分析两句的关系会用，不必有）、label （分类中标签的ID）、
                        example = InputExample(guid=tokens[0], text_a=content, label=category)  # 纯文本信息和label
                        examples.append(example)
                else:
                    if essay_set == config.dataset.target_domain:
                        category = get_model_friendly_scores(score, essay_set)
                        # guid(每句话的id,可选) text_a（输入的文本数据）、text_b （分析两句的关系会用，不必有）、label （分类中标签的ID）、
                        example = InputExample(guid=tokens[0], text_a=content, label=category)  # 纯文本信息和label
                        examples.append(example)
        return examples

PROCESSORS = {
    "aes": AESProcessor
}
