import os, glob, json, torch, collections
from PIL import Image
import numpy as np


class VisualGenome:
    def __init__(self):
        # Dataset root folder
        self.root_path = '/home/team/xiaonan/dataset/visual-gnome-2/'

        # Image folder path:VG_100K,VG_100K_2
        self.image_path = {
            'vg_100k': self.root_path+'VG_100K/',
            'vg_100k_2': self.root_path+'VG_100K_2/'
        }

        # Annotation json file
        self.ann_path = {
            'attributes': self.root_path+'attributes.json',
            'objects': self.root_path+'objects.json',
            'relationships': self.root_path+'relationships.json',
            'image_data': self.root_path+'image_data.json',
            'question_answers': self.root_path+'question_answers.json',
            'region_descriptions': self.root_path+'region_descriptions.json',
            'synsets':self.root_path+'synsets.json',
            'region_graphs':self.root_path+'region_graphs.json',
            'scene_graphs':self.root_path+'scene_graphs.json',
            'qa_to_region_mapping':self.root_path+'qa_to_region_mapping.json'
        }

        # attribute、objects、relations vocab path
        self.vocab_path = {
            'attributes': self.root_path+'vocab-data/attributes_vocab.json',
            'objects': self.root_path+'vocab-data/objects_vocab.json',
            'relations': self.root_path+'vocab-data/relations_vocab.json'
        }

        # Images path:VG_100k,VG_100k_2
        self.image_list1 = glob.glob(self.image_path['vg_100k']+'*.jpg')
        self.image_list2 = glob.glob(self.image_path['vg_100k_2']+'*.jpg')
        self.image_list = self.image_list1+self.image_list2

        # run mode
        self.run = 'train'
        # Split data into train、valid、test set
        self.split_type = ['train+valid+test', 'train+test']

        # Corresponding image path with image id
        # self.id_to_image_path = self.idx_to_image(self.image_list)

        # Read vocab array and vocab into index dict
        # self.attr_vocab, self.attr_to_idx = self.read_vocab(self.vocab_path['attributes'])

    def idx_to_image(self, path_list):
        """
        Corresponding image path with image id
        Args:
            path_list: image path list, eg,['.../image_id.jpg',..]

        Returns:dict,{'image_id':'image_path', ...}

        """
        # dict data
        image_dict = {}

        # Iterating through all path list
        for idx, path in enumerate(path_list):
            # Extract image id from path
            image_id = str(int(path.split('/')[-1].split('.')[0]))
            # corresponding image with id
            image_dict[image_id] = path

            # show result
            print('\rimage id to image path:[{}|{}]'.format(idx+1, len(path_list)),
                  end='   ')

        return image_dict

    def load_json_data(self, file_path):
        """
        Loading json file data
        Args:
            file_path: json file path

        Returns:data

        """
        data = json.load(open(file_path, 'r'))
        return data

    def make_vocab_json(self):
        """
        Making vocab data and save json format
        """
        print('make vocab data...')

        # define vocab data and vocab dict variable
        attr_vocab = []
        attr_dict = {}
        obj_vocab = []
        obj_dict = {}
        real_vocab = []
        real_dict = {}

        # making 'attributes' data
        if not os.path.exists(self.vocab_path['attributes']):
            attr_data = json.load(open(self.ann_path['attributes']))
            attr_set = list()
            obj_set = list()
            count = 0
            for attr in attr_data:
                for obj in attr['attributes']:
                    # print(obj)
                    try:
                        # attr_set.append(' '.join(obj['attributes']))
                        [attr_set.append(str(attr)) for attr in obj['attributes']]
                    except KeyError:
                        count+=1
                    obj_set.append(''.join(obj['names']))
            print(len(attr_set), len(obj_set), count)
            attr_counter = collections.Counter(attr_set)
            obj_counter = collections.Counter(obj_set)
            print(len(attr_counter), len(obj_counter))
            attr_data = ['no_attribute']
            for index, attr in attr_data:
                pass

    def read_vocab(self, file_path):
        """
        Reading vocab data from [attributes, relations, objects] txt file
        Args:
            file_path: [attributes, relations, objects] txt file path

        Returns:Vocab Array:['vocab1',...];Vocab to index dict:{'vocab1':index}

        """
        # define vocab data array and vocab into index dict variable
        vocab = []
        vocab_to_idx = {}

        # initial vocab array and dict
        # acquire file name by split function
        file_name = str(file_path.split('/')[-1].split('.')[0])
        vocab = ['__backgrounds__'.format(file_name) if file_name in ['objects']
                 else '__no_{}__'.format(file_name)]
        vocab_to_idx[vocab[0]] = 0

        # reading data from txt file
        with open(file_path, 'r') as file:
            # counting
            count = 1
            # reading by mutli lines
            for data_line in file.readlines():
                name_list = [name.lower().strip() for name in data_line.split(',')]
                vocab.append(name_list[0])

                for name in name_list:
                    vocab_to_idx[name] = count

                count += 1

        return vocab, vocab_to_idx

    def split_dataset(self, split_type, data_length):
        """
        Splitting data to train、val、test set on special ratio,eg,train:test[80:20],
        [train:val]:test:[[80:20]:20]
        Args:
            split_type: 划分类型
            data_length: 数据长度

        Returns:

        """
        train_set = []
        val_set = []
        test_set = []

        split_type = split_type.split('+')
        if len(split_type) == 3:
            test_set = range(int(data_length*0.8), data_length)
            val_set = range(int(int(data_length*0.8)*0.8), int(data_length*0.8))
            train_set = range(int(int(data_length*0.8)*0.8))
        else:
            train_set = range(int(data_length*0.8))
            test_set = range(int(data_length*0.8), data_length)

        return train_set, val_set, test_set

    def read_image(self, image_path, dtype=np.float32):
        """
        Reading image data by image path
        Args:
            image_path: image absolute path
            dtype: data type

        Returns:

        """
        image = Image.open(image_path)
        try:
            image.convert('RGB')
            # convert tensor into numpy
            image = np.asarray(image, dtype=dtype)
        finally:
            if hasattr(image, 'close'):
                image.close()

        # change channel position,(H,W,C)-->(C,H,W)
        image.transpose((2, 0, 1))

        return image

    def get_bbox_label_attr(self, data, attr_to_idx, obj_to_idx):
        """
        Acquire bounding box 、object name and attribute name by ‘attributes’ data
        Args:
            data: the array of bounding box、object name、attribute name
            attr_to_idx: dict type;the attribute name corresponds to the index

        Returns:

        """
        # define bbox、label、attr variable
        bbox_list = list()
        label_list = list()
        attr_list = list()

        # iterate all data
        for idx, object in enumerate(data):
            print(object)

            # process bounding box
            bbox_list.append([int(object['x']), int(object['y']),
                              int(object['w']), int(object['h'])])

            # process object(label) name
            label_list.append(obj_to_idx[str(object['names'][0])])

            # process attribute name
            attr_list.append([int(attr_to_idx[str(attr)]) for attr in object['attributes']])

            # show process
            print('\r process [{}|{}] objects'.format(idx+1, len(data)), end='  ')

        # stack list to numpy data
        bbox = np.stack(bbox_list).astype(np.uint16)
        label = np.stack(label_list).astype(np.int32)
        attri = np.stack(attr_list).astype(np.int32)

        return bbox, label, attri


if __name__ == '__main__':
    # print(' '.join(['1', '2','3']))
    vg = VisualGenome()
    vg.make_vocab_json()
    # print(json.load(open(vg.ann_path['image_data'], 'r')))
    # print(len(vg.image_list), len(vg.image_list1), len(vg.image_list2))
    # print(vg.image_list[0])
    # print(vg.id_to_image_path)
    # data = vg.load_json_data(vg.ann_path['attributes'])
    # print(data[0], data[1], data[1001], data[10000])
    # print(len(vg.attr_vocab))
    # x, y, z = vg.split_dataset(vg.split_type[0], 108077)
    # print(x[-1], y[0],y[-1], z[0], z[-1])