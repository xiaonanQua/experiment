import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
from config.vg_config import VGConf
from torch.utils.data import DataLoader


class VGDataset:
    def __init__(self, cfg, valid=False, use_difficult=False):
        self.cfg = cfg

        # 是否使用difficult
        self.use_difficult = use_difficult

        # 数据集注释路径
        if 'train' in cfg.run_mode:
            if valid:
                # 验证集
                self.ann_path = cfg.data_list[1]
            else:
                # 划分的训练集
                self.ann_path = cfg.data_list[0]
        else:
            # 测试集
            self.ann_path = cfg.data_list[2]

        # 数据集大小
        self.data_size = len(self.ann_path)

        # 边界框名称
        self.bbox_name = ['ymin', 'xmin', 'ymax', 'xmax']
        # cfg.save_log('data2 size:{}'.format(self.data_size))
        print('data2 size:{}'.format(self.data_size))

    def get_example(self, index):
        # 加载图像数据,图像宽度、高度
        image = self.cfg.read_image(self.ann_path[index][0])
        deepth, height, width = image.shape

        # 解析单个注释文件
        anno = self.ann_path[index][1]
        tree = ET.parse(anno)
        objs = tree.findall('object')
        # 注释文件目标数
        num_object = len(objs)

        # 定义边界框、标签列表、困难列表、属性列表
        bboxes = np.zeros((num_object, 4), dtype=np.int32)
        labels = np.zeros(num_object, dtype=np.int32)
        label_name = []
        # 在数据中属性最大的数量
        attributes = np.zeros((num_object, self.cfg.num_attributes),
                              dtype=np.float32)
        overlaps = np.zeros((num_object, self.cfg.num_classes),
                            dtype=np.float32)

        # 加载边界框
        # obj_dict = {}
        ix = 0
        for obj in objs:
            obj_name = obj.find('name').text.lower().strip()
            # 若目标类别存在于类别字典中，则提取出目标信息
            if obj_name in self.cfg.obj_vocab:
                label_name.append(obj_name)
                bbox = obj.find('bndbox')
                x1 = max(0, float(bbox.find('xmin').text))
                y1 = max(0, float(bbox.find('ymin').text))
                x2 = min(width - 1, float(bbox.find('xmax').text))
                y2 = min(height - 1, float(bbox.find('ymax').text))
                # If bboxes are not positive, just give whole image coords (there are a few examples)
                if x2 < x1 or y2 < y1:
                    print('Failed bbox in %s, object %s' % (anno, obj_name))
                    x1 = 0
                    y1 = 0
                    x2 = width - 1
                    y2 = height - 1

                # 根据类别名称查找类别id
                cls = self.cfg.obj_vocab_idx[obj_name]
                # obj_dict[obj.find('object_id').text] = ix

                # 查找所有属性
                atts = obj.findall('attribute')
                n = 0
                for att in atts:
                    att = att.text.lower().strip()
                    if att in self.cfg.attr_vocab_idx:
                        attributes[ix, self.cfg.attr_vocab_idx[att]] = 1.0

                bboxes[ix, :] = [y1, x1, y2, x2]
                # bboxes[ix, :] = [int(bbox.find(idx).text) for idx in self.bbox_name]
                labels[ix] = cls
                overlaps[ix, cls] = 1.0
                ix += 1

        # 切割类别和属性，防止多余的类别
        gt_classes = labels[:ix]
        gt_attributes = attributes[:ix, :]

        # 对overlap和attribute等稀疏矩阵进行压缩
        # print(bboxes.shape, labels.shape, overlaps.shape, gt_attributes.shape)
        overlaps = scipy.sparse.csr_matrix(overlaps)
        # gt_attributes = scipy.sparse.csr_matrix(gt_attributes)

        return image, bboxes, gt_classes, overlaps

    def __len__(self):
        return self.data_size

    __getitem__ = get_example


if __name__ == '__main__':
    cfg = VGConf()
    dataset = VGDataset(cfg)
    dataloader = DataLoader(dataset, batch_size=3)

    for index, data in enumerate(dataloader, start=0):
        image, bbox, label, attr, overlap = data
        print(bbox)

    # import numpy as np
    # import scipy as sp
    #
    # csr_mat = sp.sparse.csr_matrix((4, 3), dtype=np.int8).toarray()
    # data2 = np.ones((3,4))
    # print(sp.sparse.csr_matrix(data2))






