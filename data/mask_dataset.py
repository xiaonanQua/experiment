import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import glob


class MaskDataset:
    def __init__(self):
        # 根目录；数据路径；
        self.root = '/home/team/xiaonan/dataset/mask/'
        self.data_path = {
            'sample': self.root,
            'train': self.root,
            'test': self.root
        }

        # 获取图像和注释路径
        self.image_path = glob.glob(self.data_path['sample'] + '*.jpg')
        self.ann_path = glob.glob(self.data_path['sample'] + '*.xml')

        # 标签名称
        self.label_dict = {
            'mask': 0,
            'head': 1,
            'back': 2,
            'mid_mask': 3
        }
        self.label_names = ['mask', 'head', 'back', 'mid_mask']

        # 制作图像名称和路径的字典对，即{‘*.jpg’:'/**/**/*.jpg'}
        self.image_path_dict = self.data_dict(self.image_path)

        # 是否使用difficult
        self.use_difficult = True

        # 数据集大小
        self.data_size = len(self.ann_path)

        # 边界框名称
        self.bbox_name = ['ymin', 'xmin', 'ymax', 'xmax']

    def get_example(self, index):
        # 解析单个注释文件
        anno = ET.parse(self.ann_path[index])

        # 定义边界框、标签列表、困难列表
        bbox_list = list()
        label_list = list()
        difficult_list = list()

        # 遍历‘目标’标签
        for attr in anno.findall('object'):
            # 当不使用困难划分时，并是困难时，则跳过以下操作。
            if not self.use_difficult and int(attr.find('difficult').text) == 1:
                print('1')
                continue

            # 获取标签名称(去空格、变成小写)
            label_ = attr.find('name').text.lower().strip()
            label_list.append(self.label_dict[label_])

            # 获取边界框;减去1以使像素索引从0开始
            bbox_ = attr.find('bndbox')
            bbox_list.append([int(bbox_.find(bbox_tag).text) - 1
                              for bbox_tag in self.bbox_name])

            # 获取困难值
            difficult_list.append(int(attr.find('difficult').text))

        # 将标签、边界框、困难列表堆叠成numpy数组
        label = np.stack(label_list).astype(np.int32)
        bbox = np.stack(bbox_list).astype(np.float32)
        # 当' use difficult==False '时，' difficult '中的所有元素都为False。
        difficult = np.array(difficult_list, dtype=np.bool).astype(np.uint8)

        # 加载图像数据
        image_path = self.image_path_dict[anno.find('filename').text.lower().strip()]
        image = self.read_image(image_path)

        return image, bbox, label, difficult

    def __len__(self):
        return self.data_size

    __getitem__ = get_example

    def read_image(self, image_file):
        """
        读取图像数据
        Args:
            image_file: 图像的路径

        Returns:

        """
        # 读取图像
        image = Image.open(image_file)

        # 将图像转化成RGB模式和numpy格式
        try:
            img = image.convert('RGB')
            img = np.asarray(img, dtype=np.float32)
        finally:
            if hasattr(image, 'close'):
                image.close()

        if img.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            return img[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W)
            return img.transpose((2, 0, 1))

    def data_dict(self, data):
        """
        制作数据字典，如图像路径列表，将图像文件名称和其路径对应到字典中
        Args:
            data: 数据列表

        Returns:数据字典

        """
        data_dic = dict()
        for idx, path in enumerate(data):
            data_name = str(path.split('/')[-1].lower())
            data_dic[data_name] = path
            print('\r 制作数字字典：【{}|{}】'.format(idx+1, len(data)), end='  ')

        return data_dic


if __name__ == '__main__':
    dataset = MaskDataset()





