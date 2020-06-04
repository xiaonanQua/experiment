from data2.vg_dataset import VGDataset
import data2.preprocess as prepro
# from demo.parse_xml import XmlToImage
from config.vg_config import VGConf
cfg = VGConf()


class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, input_data):
        # par = XmlToImage()
        # 获取数据及图像形状
        img, bbox, label, attr, label_name = input_data
        _, H, W = img.shape

        # 预处理图像
        img = prepro.image_resize(img, self.min_size, self.max_size)

        # resize前后的尺度比例
        _, o_H, o_W = img.shape
        scale = o_H/H

        # resize边界框的大小
        bbox = prepro.bbox_resize(bbox, (H, W), (o_H, o_W))

        # 水平翻转(图像and边界框)
        img, param = prepro.random_flip(img, x_random=True,y_random=False, return_param=True)
        bbox = prepro.bbox_flip(bbox, (o_H, o_W), x_flip=param['x_flip'])
        # par(image_feature=img, objects=[(bbox[idx], label_name[idx]) for idx in range(len(bbox))],
        #     is_save=True, num_show=10)

        return img, bbox, label, attr, scale


class Dataset:
    def __init__(self, cfg, valid=False, use_difficult=False):
        self.cfg = cfg
        self.valid = valid
        self.use_difficult = use_difficult

        if valid:
            self.db = VGDataset(cfg, valid=valid, use_difficult=use_difficult)
        else:
            self.db = VGDataset(cfg)
        self.transform = Transform(cfg.min_size, cfg.max_size)

    def __getitem__(self, index):
        if self.valid:
            # 获得数据
            ori_img, bbox, label, attr, overlaps, label_name = self.db.get_example(index)
            img = prepro.image_resize(ori_img)
            return img, ori_img.shape[1:], bbox, label, overlaps
        else:
            # 获得数据
            img, bbox, label, attr, overlaps, label_name = self.db.get_example(index)

            # 对数据进行处理
            img, bbox, label, attr, scale = self.transform((img, bbox, label, attr,
                                                        label_name))

            return img.copy(), bbox.copy(), label.copy(),attr.copy(), scale

    def __len__(self):
        return len(self.db)


if __name__ == '__main__':
    cfg = VGConf()

    dataset = Dataset(cfg)
    dataset.__getitem__(4)
