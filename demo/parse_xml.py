import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class XmlToImage(object):
    def __init__(self, xml_path=None):
        self.xml_path = xml_path

        # 解析xml文件
        objs = None
        if xml_path is not None:
            objs = self.parse_xml()
        self.objects = objs

    def parse_xml(self):
        # 定义文件中目标集, 边界框格式
        objs = list()
        bbox_format = ['ymin', 'xmin', 'ymax', 'xmax']

        # 解析xml文件
        data = ET.parse(self.xml_path)

        # 遍历所有目标
        for obj in data.findall('object'):
            label_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            bbox_ = [int(bbox.find(bbox_idx).text)-1 for bbox_idx in bbox_format]
            objs.append((bbox_, label_name))

        return objs

    def __call__(self, image_path=None, image_feature=None, objects=None,
                 is_save=False, num_show=100):
        # 使用xml中的边界框或者自定义边界框
        if objects is None:
            objects = self.objects
        elif self.objects is None and objects is None:
            raise ValueError('Objects can not be None!')

        # 读取图像，numpy格式，图像形状HWC
        if image_path is not None:
            img = np.array(Image.open(image_path), dtype=np.uint8)

        # 读取图像特征
        if image_feature is not None:
            channel = image_feature.shape[0]
            if channel==1 or channel==3:
                image_feature = image_feature.transpose((1, 2, 0))
            img = image_feature

        # 将边界框绘制到图像上
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        # fig, ax = plt.subplot(121)
        ax.imshow(img)
        print(img.shape)

        # 添加边界框及其类别名
        for idx, obj in enumerate(objects):
            # 获得目标数量
            obj, label_name = obj

            # 随机颜色
            color = (np.random.rand(), np.random.rand(), np.random.rand())

            # 限定绘制目标数量
            if idx+1 > num_show:
                break

            # 添加边界框
            ax.add_patch(patches.Rectangle((obj[1], obj[0]),
                                          obj[3]-obj[1], obj[2]-obj[0], fill=True,
                                          linewidth=2, edgecolor=color,
                                          facecolor='none'))
            # 对每个边界框添加标签名
            ax.text(obj[1], obj[0], '{}'.format(label_name),
                    bbox=dict(facecolor=color, alpha=0.4), fontsize=14, color='white')

        plt.show()
        # 是否保存
        if is_save:
            fig.savefig('bbox.png')


if __name__ == '__main__':
    file_path = '../demo/demo.xml'
    img_path = '../demo/demo.jpg'
    parse = XmlToImage(file_path)
    parse(image_path=img_path, is_save=False, num_show=4)
    # print(parse)
