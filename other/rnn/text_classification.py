import os
import tarfile
from tqdm import tqdm
import random
import collections
import torchtext.vocab as Vocab
import torch.utils.data as data
import torch
from other.rnn.rnn import BiRNN
from torch import nn

# 解压数据集
file_path = '/home/xiaonan/Dataset/movie_review'
file_name = os.path.join(file_path, 'aclImdb_v1.tar.gz')
if not os.path.exists(os.path.join(file_path, 'aclImdb')):
    print('从压缩包解压....')
    with tarfile.open(file_name, 'r') as file:
        file.extractall(file_path)

data_root = os.path.join(file_path, 'aclImdb')

# 读取数据集
def read_imdb(folder='train', data_root=None):
    data = []
    for label in ['pos', 'neg']:
        # 获取训练集或者测试集下对应标签数据
        folder_name = os.path.join(data_root, folder, label)
        # 读取标签下的数据
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
        random.shuffle(data)

        return data


# 对每条数据进行分词
def get_tokenized_imdb(data):
    """
    data2:列表，[string, label]
    """
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]


# 创建词典，过滤掉出现次数少于５的词
def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    counter =  collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter, min_freq=5)


def preprocess_imdb(data, vocab):
    # 将每条评论通过截断或者补０，使得长度变成５００
    max_l = 500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x+[0]*(max_l-len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels

# 获取数据集
train_data, test_data = read_imdb('train', data_root), read_imdb('test', data_root)
vocab = get_vocab_imdb(train_data)
batch_size = 64
train_set = data.TensorDataset(*preprocess_imdb(train_data, vocab))
test_set = data.TensorDataset(*preprocess_imdb(test_data, vocab))
train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
test_loader = data.DataLoader(test_set, batch_size)

# 创建网络
embed_size , num_hiddens, num_layers = 100, 100, 2
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)

glove_vocab = Vocab.GloVe(name='6B', dim=100, cache=os.path.join(data_root, 'glove'))

def load_pretrained_embedding(words, pretrained_vocab):
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])
    oov_count = 0  # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 0
    if oov_count > 0:
        print("There are %d oov words.")
    return embed

net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, glove_vocab))
net.embedding.weight.requires_grad = False

lr, num_epochs = 0.01, 5
# 要过滤掉不计算梯度的embedding参数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()






