from other.word2vec.config import Config
import torch, collections, random, math, json


class PTB:
    def __init__(self, cfg):
        # 配置文件
        self.cfg = cfg

        # 读取运行模式下的数据集
        with open(cfg.dataset_path[cfg.run_mode], 'r') as file:
            # 读取每一行的数据
            lines = file.readlines()
            # 将每个单词分开并保存到每行列表中
            self.raw_dataset = [sentence.split() for sentence in lines]

        # 建立词语索引
        # 统计单词出现的次数；只保留出现至少5次的单词
        counter = collections.Counter([word for sentence in self.raw_dataset for word in sentence])
        self.counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
        # 将词映射到整数索引
        self.idx_to_token = [token for token, _ in self.counter.items()]
        self.token_to_idx = {token: index for index, token in enumerate(self.idx_to_token)}

        # 将原始数据集中的单词映射为索引;被标记的单词数量（数据集大小）
        self.dataset = [[self.token_to_idx[token] for token in sentence if token in self.token_to_idx]
                        for sentence in self.raw_dataset]
        self.num_tokens = sum([len(sentence) for sentence in self.dataset])

        # 二次采样
        self.sub_dataset = [[idx for idx in sentence if not self._discard(idx)]
                            for sentence in self.dataset]

        # 提取中心词和背景词
        self.centers, self.contexts = self.get_center_and_context(self.sub_dataset,
                                                                  cfg.max_window_size)

        # 负采样
        sampling_weights = [self.counter[w] ** 0.75 for w in self.idx_to_token]
        all_negatives = self.get_negatives(self.contexts, sampling_weights, 5)
        print(all_negatives)

    def _discard(self, idx):
        """
        二次采样，每个索引单词都以一定概率被丢弃.越高频的词被抛弃的概率越大
        :param idx: 单词索引
        :return:
        """
        return random.uniform(0, 1) < 1 - \
               math.sqrt(1e-4/self.counter[self.idx_to_token[idx]]*self.num_tokens)

    def compare(self, token):
        """
        对比数据集和二次采样后的数据集中高频词的数量
        :param token: 标记单词
        :return:
        """
        print('original word:{}, data2 counts:{}, sub data2 counts:{}'.format(
            token, sum([st.count(self.token_to_idx[token]) for st in self.dataset]),
            sum([st.count(self.token_to_idx[token]) for st in self.sub_dataset])
        ))

    def get_center_and_context(self, dataset, max_window_size):
        """
        提取中心词和背景词（与中心词距离不超过背景窗口大小的词）
        :param dataset: 数据集（单词索引）
        :param max_window_size: 每次在整数1和最大背景窗口之间随机均匀采样一个整数作为背景窗口大小
        :return:
        """
        # 定义中心词和背景词列表
        centers, contexts = [], []

        # 遍历整个数据集
        for sentence in dataset:
            # 每个句子必须包含连个单词才能组成‘中心词-背景词’
            if len(sentence) < 2:
                continue

            # 存储中心词
            centers += sentence

            # 遍历句子，汇总每个中心词对应窗口大小的背景词
            for index in range(len(sentence)):
                # 获取背景窗口大小
                window_size = random.randint(1, max_window_size)
                # 获取每个中心词对应的窗口大小背景词的索引区域,特殊的考虑开头和结尾的索引
                indices = list(range(max(0, index-window_size),
                                     min(len(sentence), index+1+window_size)))
                # 去除中心词
                indices.remove(index)
                # 存储背景词
                contexts.append([sentence[idx] for idx in indices])

        return centers, contexts

    def get_negatives(self, all_contexts, sampling_weights, k):
        """
        负采样
        :param all_contexts: 所有的内容
        :param sampling_weights: 采样权重
        :param K: 采样数量
        :return:
        """
        print('进行负采样....')
        # 定义所有负样本，负样本候选框
        all_negatives, neg_candidates, i = [], [], 0
        population = list(range(len(sampling_weights)))

        # 遍历所有背景词
        for contexts in all_contexts:
            j = 1
            # 负样本
            negatives = []
            # 采样负样本
            while len(negatives) < len(contexts)*k:
                if i == len(neg_candidates):
                    i, neg_candidates = 0, random.choices(
                        population, sampling_weights, k=int(1e5))
                neg, i = neg_candidates[i], i+1
                j = j+1
                print('\r {}'.format(j), end='  ')
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
            all_negatives.append(negatives)
            print(3)

        return all_negatives

    def show_center_and_context(self):
        """
        显示中心词和背景词
        :return:
        """
        i = 0
        for st in self.sub_dataset:
            length = len(st)
            print('句子：{}\n'.format([self.idx_to_token[idx] for idx in st]))
            print('中心词：{}\n'.format([self.idx_to_token[self.centers[idx]]
                                     for idx in range(length)]))
            print('背景词：{}\n'.format([[self.idx_to_token[context]
                                      for context in self.contexts[i]]
                                     for i in range(length)]))
            i += 1
            if i > 2:
                break


if __name__ == '__main__':
    cfg = Config()
    ptb = PTB(cfg)
    # ptb.show_center_and_context()
    # ptb.compare('join')
    # sampling_weights = [ptb.counter[w]**0.75 for w in ptb.idx_to_token]
    # all_negatives = ptb.get_negatives(ptb.contexts, sampling_weights, 5)