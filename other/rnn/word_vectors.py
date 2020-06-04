import numpy as np
import matplotlib.pyplot as plt

def word_vector_svd():
    """
    对矩阵Ｘ(基于窗口共现得到的矩阵)使用奇异值分解得到词向量
    :return:
    """
    # 获得线性代数包
    la = np.linalg

    # 定义词库words和共现矩阵ｘ
    words = ['I', 'like', 'enjoy', 'deep', 'learning', 'NLP', 'flying', '.']
    x = np.array([[0, 2, 1, 0, 0, 0, 0, 0],
                  [2, 0, 0, 1, 0, 1, 0, 1],
                  [1, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1],
                  [0, 1, 0, 0, 0, 0, 0, 1],
                  [0, 0, 1, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1, 1, 1, 0]])

    # 对x使用奇异值分解
    u, s, vh = la.svd(x, full_matrices=False)
    print(u)

    # 显示词向量
    for i in range(len(words)):
        plt.text(u[i, 0], u[i, 1], words[i])
    plt.show()

if __name__ == '__main__':
    word_vector_svd()