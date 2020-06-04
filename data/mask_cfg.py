import glob

class MaskConf:

    def __init__(self):

        # 根目录；数据路径；
        self.root = '/home/team/xiaonan/dataset/mask/'
        self.data_path = {
            'sample': self.root,
            'train': self.root,
            'test': self.root
        }

        # 获取图像和注释路径
        self.image_path = glob.glob(self.data_path['sample']+'*.jpg')
        self.ann_path = glob.glob(self.data_path['sample']+'*.xml')

        # 标签名称
        self.label_dict = {
            'mask': 0,
            'head': 1,
            'back': 2,
            'mid_mask': 3
        }
        self.label_list = ['mask', 'head', 'back', 'mid_mask']

        # 制作图像名称和路径的字典对，即{‘*.jpg’:'/**/**/*.jpg'}
        self.image_path_dict = self.data_dict(self.image_path)

        # 是否使用difficult
        self.use_difficult = True

    def data_dict(self, data):
        """
        制作数据字典，如图像路径列表，将图像文件名称和其路径对应到字典中
        Args:
            data: 数据列表

        Returns:数据字典

        """
        print(data)
        data_dic = dict()
        for idx, path in enumerate(data):
            data_name = str(path.split('/')[-1].lower())
            data_dic[data_name] = path
            print('\r 制作数字字典：【{}|{}】'.format(idx+1, len(data)), end='  ')

        return data_dic

