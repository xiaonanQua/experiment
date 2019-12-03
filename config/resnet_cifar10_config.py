from config.config import Config


class ResNetConf(Config):
    def __init__(self):
        # 继承父类构造函数
        super(ResNetConf, self).__init__()
        # 图像宽度、高度、通道
        self.image_width = 70
        self.image_height = 70
        self.image_channels = 3
        # 类别数量
        self.num_classes = 10
        # 实验的超参数配置
        self.epochs = 50
        self.batch_size = 128
        self.learning_rate = 0.001  # 原始是0.01
        self.learning_rate_decay = 0.95
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.keep_prob = 0.5

        # 使用模型，名字必须和models/__init__.py中的名字一致
        self.model = 'ResNet'
        # 模型检查点地址
        self.load_model_path = None


if __name__ == '__main__':
    pass




