import skimage.transform as transform
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch, random
# from demo.parse_xml import XmlToImage


def image_resize(img, min_size=600, max_size=1000):
    """
    Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
        After the scaling, if the length of the longer edge is longer than
        :param min_size:
        :obj:`self.max_size`, the image is scaled to fit the longer edge
        to :obj:`self.max_size`.

        After resizing the image, the image is subtracted by a mean image value
        :obj:`self.mean`.

        Args:
            img (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray: A preprocessed image.
    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255
    img = transform.resize(img, (C, H * scale, W * scale),
                           mode='reflect', anti_aliasing=False)
    # img = pytorch_normalize(img)
    # img = caffe_normalize(img)
    return img


def bbox_resize(bbox, in_size, out_size):
    """
    Resize bounding boxes according to image resize.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.
    """

    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]

    bbox[:, 0] = bbox[:, 0] * y_scale
    bbox[:, 2] = bbox[:, 2] * y_scale
    bbox[:, 1] = bbox[:, 1] * x_scale
    bbox[:, 3] = bbox[:, 3] * x_scale
    return bbox


def random_flip(img, x_random=False, y_random=False,
                return_param=False, copy=False):
    """Randomly flip an image in vertical or horizontal direction.

       Args:
           img (~numpy.ndarray): An array that gets flipped. This is in
               CHW format.
           y_random (bool): Randomly flip in vertical direction.
           x_random (bool): Randomly flip in horizontal direction.
           return_param (bool): Returns information of flip.
           copy (bool): If False, a view of :obj:`img` will be returned.

       Returns:
           ~numpy.ndarray or (~numpy.ndarray, dict):

           If :obj:`return_param = False`,
           returns an array :obj:`out_img` that is the result of flipping.

           If :obj:`return_param = True`,
           returns a tuple whose elements are :obj:`out_img, param`.
           :obj:`param` is a dictionary of intermediate parameters whose
           contents are listed below with key, value-type and the description
           of the value.

           * **y_flip** (*bool*): Whether the image was flipped in the\
               vertical direction or not.
           * **x_flip** (*bool*): Whether the image was flipped in the\
               horizontal direction or not.

       """
    # 水平或者垂直翻转
    x_flip, y_flip = False, False

    # 随机选择是否翻转（水平或者垂直）
    if x_random:
        x_flip = random.choice([True, False])
    if y_random:
        y_flip = random.choice([True, False])

    # 对图像进行水平或者垂直翻转
    if x_flip:
        img = img[:, :, ::-1]
    if y_flip:
        img = img[:, ::-1, :]

    # 是否复制翻转后的图像
    if copy:
        img = img.copy()

    # 返回翻转的方向
    if return_param:
        return img, {'x_flip': x_flip, 'y_flip': y_flip}
    else:
        return img


def bbox_flip(bbox, img_shape, x_flip=False, y_flip = False):

    # 图像的高和宽
    H, W = img_shape
    bbox = bbox.copy()
    # 水平翻转，将边界框中宽度坐标（x）进行更新
    if x_flip:
        bbox[:, 1] = W - bbox[:, 1]
        bbox[:, 3] = W - bbox[:, 3]
    if y_flip:
        bbox[:, 0] = H - bbox[:, 0]
        bbox[:, 2] = H - bbox[:, 2]

    return bbox


def pytorch_normalize(img):
    """
    进行正则化，加载的模型是pytorch预训练的
    :param img: numpy格式的数字图像
    :return: 正则化后的数据
    """
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    进行正则化，加载的模型是caffe预训练的
    :param img:格式是numpy
    :return:
    """
    img = img[[2, 1, 0], :, :]
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def inverse_normalize(img, caffe_pretrain=False):
    if caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

if __name__ == "__main__":
    img_path = '../demo/demo.jpg'
    xml_path = '../demo/demo.xml'
    data = XmlToImage()
    objs = bbo