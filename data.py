from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import multiprocessing

import numpy as np
import tensorflow as tf
import tflib as tl


_N_CPU = multiprocessing.cpu_count()


class Celeba(tl.Dataset):

    att_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
                'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
                'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
                'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
                'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
                'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
                'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
                'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
                'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
                'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
                'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
                'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
                'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

    def __init__(self,
                 data_dir,
                 atts,
                 img_resize,
                 batch_size,
                 prefetch_batch=_N_CPU + 1,
                 drop_remainder=True,
                 num_threads=_N_CPU,
                 shuffle=True,
                 shuffle_buffer_size=None,
                 repeat=-1,
                 sess=None,
                 split='train',
                 crop=True):
        super(Celeba, self).__init__()

        list_file = os.path.join(data_dir, 'list_attr_celeba.txt')
        if crop:
            img_dir_jpg = os.path.join(data_dir, 'img_align_celeba')
            img_dir_png = os.path.join(data_dir, 'img_align_celeba_png')
        else:
            img_dir_jpg = os.path.join(data_dir, 'img_crop_celeba')
            img_dir_png = os.path.join(data_dir, 'img_crop_celeba_png')

        names = np.loadtxt(list_file, skiprows=2, usecols=[0], dtype=np.str)
        if os.path.exists(img_dir_png):
            img_paths = [os.path.join(img_dir_png, name.replace('jpg', 'png')) for name in names]
        elif os.path.exists(img_dir_jpg):
            img_paths = [os.path.join(img_dir_jpg, name) for name in names]

        att_id = [Celeba.att_dict[att] + 1 for att in atts]
        labels = np.loadtxt(list_file, skiprows=2, usecols=att_id, dtype=np.int64)

        if img_resize == 64:
            # crop as how VAE/GAN do
            offset_h = 40
            offset_w = 15
            img_size = 148
        else:
            offset_h = 26
            offset_w = 3
            img_size = 170

        def _map_func(img, label):
            if crop:
                img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, img_size, img_size)
            # img = tf.image.resize_images(img, [img_resize, img_resize]) / 127.5 - 1
            # or
            img = tf.image.resize_images(img, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            label = (label + 1) // 2
            return img, label

        if split == 'test':
            drop_remainder = False
            shuffle = False
            repeat = 1
            img_paths = img_paths[182637:]
            labels = labels[182637:]
        elif split == 'val':
            img_paths = img_paths[182000:182637]
            labels = labels[182000:182637]
        else:
            img_paths = img_paths[:182000]
            labels = labels[:182000]

        dataset = tl.disk_image_batch_dataset(img_paths=img_paths,
                                              labels=labels,
                                              batch_size=batch_size,
                                              prefetch_batch=prefetch_batch,
                                              drop_remainder=drop_remainder,
                                              map_func=_map_func,
                                              num_threads=num_threads,
                                              shuffle=shuffle,
                                              shuffle_buffer_size=shuffle_buffer_size,
                                              repeat=repeat)
        self._bulid(dataset, sess)

        self._img_num = len(img_paths)

    def __len__(self):
        return self._img_num


if __name__ == '__main__':
    import imlib as im
    atts = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
    data = Celeba('./data', atts, 128, 32, split='val')
    batch = data.get_next()
    print(len(data))
    print(batch[1][1], batch[1].dtype)
    print(batch[0].min(), batch[1].max(), batch[0].dtype)
    im.imshow(batch[0][1])
    im.show()
