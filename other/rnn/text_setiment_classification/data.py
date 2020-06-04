import tarfile, os, random, tqdm, collections, torch
import torchtext.vocab as vocab
from torch.utils.data import TensorDataset


class ProcessACLData:
    def __init__(self):
        # 文件路径
        self.file_name = 'aclImdb_v1.tar.gz'
        self.root_dir = '/home/team/xiaonan/data2/aclImdb/aclImdb/'
        self.file_path = os.path.join(self.root_dir, self.file_name)

        # 训练和测试数据目录
        self.train_dir = self.root_dir + 'train/'
        self.test_dir = self.root_dir + 'test/'

        # 提取数据
        if not os.path.exists(self.root_dir):
            print('从压缩包解压...')
            with tarfile.open(self.file_path, 'r') as file:
                    file.extractall(self.root_dir)

    def read_imdb(self, type_dir):
        """
        读取不同类型的数据
        :param type_dir: 数据目录
        :return:
        """
        # 定义保存数据变量
        data = []

        # 正负两样本
        for label in ['pos', 'neg']:
            # 文件目录
            file_dir = os.path.join(type_dir, label)
            # 遍历该目录下的所有文件
            for file in tqdm.tqdm(os.listdir(file_dir)):
                # 打开文件
                with open(os.path.join(file_dir, file), 'rb') as f:
                    # 获取评论
                    review = f.read().decode('utf-8').replace('\n', '').lower()
                    # 将评论和标签附加到整体数据中
                    data.append([review, 1 if label == 'pos' else 0])

        # 将数据打乱
        random.shuffle(data)

        return data

    def get_tokenized_imdb(self, data):
        """
        基于空格对评论数据进行分词
        :param data: 训练或者测试数据
        :return:
        """
        # 将每条评论中的单词基于空格分离出来，并将单词转化成小写
        def token_word(review):
            return [word.lower() for word in review.split(' ')]
        return [token_word(review) for review, _ in data]

    def get_vocab_imdb(self):
        """
        根据分词后的训练数据来创建词典，并过滤掉出现少于5的单词
        :param data2:分词后的词典
        :return:
        """

        # 获得所有数据
        train_data = self.read_imdb(self.train_dir)
        test_data = self.read_imdb(self.test_dir)
        data = train_data + test_data

        # 分词数据
        data = self.get_tokenized_imdb(data)

        # 获得不重复单词的集合
        counter = collections.Counter([tk for st in data for tk in st])

        return vocab.Vocab(counter, min_freq=5)

    def preprocess_data(self, data, vocab, max_length):
        """
        由于每条评论长度不一致所以不能直接组合成小批量，所以通过截断或者补0将每条评论长度固定成500
        :param data:包含标签的数据
        :param vocab:词典
        :param max_length:每条评论最大固定长度
        :return:
        """
        # 对每条评论按照固定长度进行补0或者截断
        def pad(x):
            return x[:max_length] if len(x)>max_length else x + [0]*(max_length-len(x))

        # 分词
        token_word = self.get_tokenized_imdb(data)
        # 长度固定后的特征数据和标签数据
        features = torch.tensor([pad([vocab.stoi[word] for word in words])for words in token_word])
        labels = torch.tensor([score for _, score in data])

        return features, labels

    def get_dataset(self, type_dir, vocab, max_length=500):
        """
        获得预处理后的特征和标签
        :param type_dir: 数据类型目录
        :param vocab: 数据词典（包括训练集与测试集）
        :param max_length: 每条评论最大长度
        :return:
        """
        # 读取数据
        data = self.read_imdb(type_dir)

        # 获得处理后的数据集（特征数据和标签）
        dataset = TensorDataset(*self.preprocess_data(data, vocab, max_length))
        return dataset

if __name__ == '__main__':
    data = ProcessACLData()
    vocab = data.get_vocab_imdb()
    print(len(vocab.itos))

