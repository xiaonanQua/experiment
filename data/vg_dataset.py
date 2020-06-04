import os, torch
from data.data_vg import VisualGenome
from torch.utils.data import Dataset, DataLoader

class VGDataset(Dataset):
    def __init__(self, cfg):

        # config
        self.cfg = cfg

        # reading 'attributes' data, 'attributes' class, object class
        self.attr_data = cfg.load_json_data(cfg.ann_path['attributes'])
        self.attr_class, self.attr_to_idx = cfg.read_vocab(cfg.vocab_path['attributes'])
        self.obj_class, self.obj_to_idx = cfg.read_vocab(cfg.vocab_path['objects'])

        # splitting dataset list
        self.train_list, self.val_list, self.test_list = \
            cfg.split_dataset(cfg.split_type[0], len(self.attr_data))

        # acquire all image path dict
        self.id_to_img_path = cfg.idx_to_image(cfg.image_list)

        # data size
        self.data_size = len(self.train_list if cfg.run in ['train']
                             else self.test_list)

        print('dataset size:{}'.format(self.data_size))

    def __getitem__(self, index):
        # acquire data by index
        data = self.attr_data[index]

        # reading image,return numpy type
        image = self.cfg.read_image(self.id_to_img_path[str(data['image_id'])])

        # reading bounding box, labels, attributes
        bbox, label, attr = self.cfg.get_bbox_label_attr(data['attributes'],
                                                         self.attr_to_idx,
                                                         self.obj_to_idx
                                                         )

        return image, bbox, label, attr

    def __len__(self):
        return self.data_size


if __name__ == '__main__':
    cfg = VisualGenome()
    vg = VGDataset(cfg)

    dataloader = DataLoader(vg, batch_size=2)

    for data in dataloader:
        image, bbox, label, attr = data
        print(image.size(), bbox.size(), label.size(), attr.size())



