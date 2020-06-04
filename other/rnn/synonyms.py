import torch
import torchtext.vocab as vocab

# 获取预训练的词嵌入
keys_name = vocab.pretrained_aliases.keys()
glove_keys = [key for key in keys_name if 'glove' in key]
print(glove_keys)

# 加载预训练的Glove词向量
glove = vocab.GloVe(name='6B', dim=50)
print('共{}词汇量'.format(len(glove.itos)))
print(glove.itos[333], glove.stoi['little'])


# 计算近义词
def get_similar_tokens(query_token, k, embed):
    # 计算最近邻
    topk, cos = knn(embed.vectors, embed.vectors[embed.stoi[query_token]], k+1)
    for i, c in zip(topk[1:], cos[1:]):
        print('cosine sim=%.3f:%s'%(c, embed.itos[i]))


def get_analogy(token_a, token_b, token_c, embed):
    # 获取三个类比单词的词向量
    vecs = [embed.vectors[embed.stoi[i]] for i in [token_a, token_b, token_c]]

    x = vecs[2]+vecs[1]-vecs[0]

    topk, cos = knn(embed.vectors, x, 1)

    return embed.itos[topk[0]]


# 定义k最近邻算法
def knn(w, x, k):
    # 计算余弦相似性，添加1e-9为了保持数值稳定性
    cos = torch.matmul(w, x.view((-1, )))/(
        (torch.sum(w*w, dim=1)+1e-9).sqrt()*(torch.sum(x*x).sqrt())
    )
    print(cos.size())
    # 根据计算查询的词向量和所有词向量的余弦相似度，获得最相似的k个词向量
    value, index = torch.topk(cos, k)
    #将索引值张量转化成numpy
    index = index.cpu().numpy()

    return index, [cos[i] for i in index]


if __name__ == '__main__':
    get_similar_tokens('baby', 6, glove)
    str = get_analogy('man', 'woman', 'son', glove)
    print(str)



