import os
from os.path import join
from torchvision.transforms import Compose
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
import torch
import time
from data_utils import gtransforms
from data_utils.data_parser import WebmDataset
from numpy.random import choice as ch
import random
import json


class VideoFolder(torch.utils.data.Dataset):
    bbox_folder_path = '/home/ly/CAction/sth_else/bbox_jsons'

    def __init__(self,
                 root,
                 file_input,
                 file_labels,
                 frames_duration,
                 args=None,
                 multi_crop_test=False,
                 sample_rate=2,
                 is_test=False,
                 is_val=False,
                 num_boxes=10,
                 model=None,
                 if_augment=True):
        """
        :param root: data root path
        :param file_input: inputs path
        :param file_labels: labels path
        :param frames_duration: number of frames
        :param multi_crop_test:
        :param sample_rate: FPS
        :param is_test: is_test flag
        :param k_split: number of splits of clips from the video
        :param sample_split: how many frames sub-sample from each clip
        :param sample_split: how many frames sub-sample from each clip
        :param sample_split: how many frames sub-sample from each clip
        """
        self.in_duration = frames_duration
        self.coord_nr_frames = self.in_duration // 2
        self.multi_crop_test = multi_crop_test
        self.sample_rate = sample_rate
        self.if_augment = if_augment
        self.is_val = is_val
        self.data_root = root
        self.dataset_object = WebmDataset(file_input, file_labels, root, is_test=is_test)
        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.model = model
        self.num_boxes = num_boxes

        with open(
                "/home/ly/CAction/cdn_com/data/dataset_splits/compositional/object_All_new.json",
                encoding="utf-8") as f:
            self.data = json.load(f)

        with open(
                "/home/ly/CAction/cdn_com/data/dataset_splits/compositional/train.json",
                encoding="utf-8") as f:
            self.train = json.load(f)

        with open(
                "/home/ly/CAction/cdn_com/data/dataset_splits/compositional/object_All_rewrite.json",
                encoding="utf-8") as f:
            self.train_all_object = json.load(f)

        # Prepare data for the data loader
        self.args = args
        self.prepare_data()
        self.pre_resize_shape = (256, 340)

        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        # Transformations
        if not self.is_val:
            self.transforms = [
                gtransforms.GroupResize((224, 224)),
            ]
        elif self.multi_crop_test:
            self.transforms = [
                gtransforms.GroupResize((256, 256)),
                gtransforms.GroupRandomCrop((256, 256)),
            ]
        else:
            self.transforms = [
                gtransforms.GroupResize((224, 224))
            ]
        self.transforms += [
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(self.img_mean, self.img_std),
        ]
        self.transforms = Compose(self.transforms)

        if self.if_augment:
            if not self.is_val:  # train, multi scale cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(output_size=224,
                                                                   scales=[1, .875, .75])
            else:  # val, only center cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(output_size=224,
                                                                   scales=[1],
                                                                   max_distort=0,
                                                                   center_crop_only=True)
        else:
            self.random_crop = None

    def load_one_video_json(self, folder_id):
        with open(os.path.join(VideoFolder.bbox_folder_path, folder_id + '.json'), 'r', encoding='utf-8') as f:
            video_data = json.load(f)
        return video_data

    def prepare_data(self):
        """
        This function creates 3 lists: vid_names, labels and frame_cnts
        This process will take up a long time, I want to save these objects into a file and read it
        :return:
        """
        print("Loading label strings")
        self.label_strs = ['_'.join(class_name.split(' ')) for class_name in
                           self.classes]

        if not self.is_val:
            with open('/home/ly/CAction/KCMM/data/misc/train_vid_name.json',
                      'r') as f:
                vid_names = json.load(f)
                print('train vid len:{}'.format(len(vid_names)))
            with open('/home/ly/CAction/KCMM/data/misc/train_labels.json',
                      'r') as f:
                labels = json.load(f)
            with open(
                    '/home/ly/CAction/KCMM/data/misc/train_frame_cnts.json',
                    'r') as f:
                frame_cnts = json.load(f)
        else:
            with open('/home/ly/CAction/KCMM/data/misc/val_vid_name.json',
                      'r') as f:
                vid_names = json.load(f)
                print('test vid len:{}'.format(len(vid_names)))
            with open('/home/ly/CAction/KCMM/data/misc/val_labels.json',
                      'r') as f:
                labels = json.load(f)
            with open('/home/ly/CAction/KCMM/data/misc/val_frame_cnts.json',
                      'r') as f:
                frame_cnts = json.load(f)

        # debug
        # self.vid_names = vid_names[:100]
        # self.labels = labels[:100]
        # self.frame_cnts = frame_cnts[:100]

        self.vid_names = vid_names
        self.labels = labels
        self.frame_cnts = frame_cnts

    def load_frame(self, vid_name, frame_idx):
        """
        Load frame
        :param vid_name: video name
        :param frame_idx: index
        :return:
        """
        return Image.open(
            join(self.data_root, 'frames', vid_name, '%04d.jpg' % (frame_idx + 1))).convert('RGB')

    def _sample_indices(self, nr_video_frames):
        average_duration = nr_video_frames * 1.0 / self.coord_nr_frames
        if average_duration > 0:
            offsets = np.multiply(list(range(self.coord_nr_frames)), average_duration) \
                      + np.random.uniform(0, average_duration, size=self.coord_nr_frames)
            offsets = np.floor(offsets)
        elif nr_video_frames > self.coord_nr_frames:
            offsets = np.sort(np.random.randint(nr_video_frames, size=self.coord_nr_frames))
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def _get_val_indices(self, nr_video_frames):
        if nr_video_frames > self.coord_nr_frames:
            tick = nr_video_frames * 1.0 / self.coord_nr_frames
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.coord_nr_frames)])
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def test_sample_single(self, index):
        """
        Choose and Load frames per video
        :param index:
        :return:
        """
        n_frame = self.frame_cnts[index] - 1
        d = self.in_duration * self.sample_rate
        if n_frame > d:
            if not self.is_val:
                # random sample
                offset = np.random.randint(0, n_frame - d)
            else:
                # center crop
                offset = (n_frame - d) // 2
            frame_list = list(range(offset, offset + d, self.sample_rate))
        else:
            # Temporal Augmentation
            if not self.is_val:  # train
                if n_frame - 2 < self.in_duration:
                    # less frames than needed
                    pos = np.linspace(0, n_frame - 2, self.in_duration)
                else:  # take one
                    pos = np.sort(np.random.choice(list(range(n_frame - 2)), self.in_duration, replace=False))
            else:
                pos = np.linspace(0, n_frame - 2, self.in_duration)
            frame_list = [round(p) for p in pos]

        frame_list = [int(x) for x in frame_list]

        if not self.is_val:  # train
            coord_frame_list = self._sample_indices(n_frame)
        else:  # val
            coord_frame_list = self._get_val_indices(n_frame)

        assert len(coord_frame_list) == len(frame_list) // 2

        folder_id = str(int(self.vid_names[index]))
        video_data = self.load_one_video_json(folder_id)

        # union the objects of two frames
        object_set = set()
        for frame_id in coord_frame_list:
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']  # standard category: [0001, 0002]
                object_set.add(standard_category)
        object_set = sorted(list(object_set))

        # NOTE: IMPORTANT!!! To accelerate the data loader, here it only reads one image
        #  (they're of the same scale in a video) to get its height and width
        #  It must be modified for models using any appearance features.
        frames = []
        for fidx in coord_frame_list:
            frames.append(self.load_frame(self.vid_names[index], fidx))
            break  # only one image
        height, width = frames[0].height, frames[0].width

        frames = [img.resize((self.pre_resize_shape[1], self.pre_resize_shape[0]), Image.BILINEAR) for img in
                  frames]  # just one frame in List:frames

        if self.random_crop is not None:
            frames, (offset_h, offset_w, crop_h, crop_w) = self.random_crop(frames)
        else:
            offset_h, offset_w, (crop_h, crop_w) = 0, 0, self.pre_resize_shape
        if self.model not in ['coord', 'coord_latent', 'coord_latent_nl', 'coord_latent_concat']:
            frames = []
            for fidx in frame_list:
                frames.append(self.load_frame(self.vid_names[index], fidx))

        else:
            # Now for accelerating just pretend we have had frames
            frames = frames * self.in_duration

        scale_resize_w, scale_resize_h = self.pre_resize_shape[1] / float(width), self.pre_resize_shape[0] / float(
            height)
        scale_crop_w, scale_crop_h = 224 / float(crop_w), 224 / float(crop_h)

        box_tensors = torch.zeros((self.coord_nr_frames, self.num_boxes, 4), dtype=torch.float32)  # (cx, cy, w, h)
        box_categories = torch.zeros((self.coord_nr_frames, self.num_boxes))
        for frame_index, frame_id in enumerate(coord_frame_list):
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']
                global_box_id = object_set.index(standard_category)

                box_coord = box_data['box2d']
                x0, y0, x1, y1 = box_coord['x1'], box_coord['y1'], box_coord['x2'], box_coord['y2']

                # scaling due to initial resize
                x0, x1 = x0 * scale_resize_w, x1 * scale_resize_w
                y0, y1 = y0 * scale_resize_h, y1 * scale_resize_h

                # shift
                x0, x1 = x0 - offset_w, x1 - offset_w
                y0, y1 = y0 - offset_h, y1 - offset_h

                x0, x1 = np.clip([x0, x1], a_min=0, a_max=crop_w - 1)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=crop_h - 1)

                # scaling due to crop
                x0, x1 = x0 * scale_crop_w, x1 * scale_crop_w
                y0, y1 = y0 * scale_crop_h, y1 * scale_crop_h

                # precaution
                x0, x1 = np.clip([x0, x1], a_min=0, a_max=223)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=223)

                # (cx, cy, w, h)
                gt_box = np.array([(x0 + x1) / 2., (y0 + y1) / 2., x1 - x0, y1 - y0], dtype=np.float32)

                # normalize gt_box into [0, 1]
                gt_box /= 224

                # load box into tensor
                try:
                    box_tensors[frame_index, global_box_id] = torch.tensor(gt_box).float()
                    box_categories[frame_index, global_box_id] = 1 if box_data['standard_category'] == 'hand' else 2
                except IndexError:
                    pass

                x0, y0, x1, y1 = list(map(int, [x0, y0, x1, y1]))
        return frames, box_tensors, box_categories

    def train_sample_single(self, index):
        """
        Choose and Load frames per video
        :param index:
        :return:
        """

        n_frame = self.frame_cnts[index] - 1

        d = self.in_duration * self.sample_rate

        # print("data:{}".format(data))

        if n_frame > d:
            if not self.is_val:
                # random sample
                offset = np.random.randint(0, n_frame - d)
            else:
                # center crop
                offset = (n_frame - d) // 2
            frame_list = list(range(offset, offset + d, self.sample_rate))
        else:
            # Temporal Augmentation
            if not self.is_val:  # train
                if n_frame - 2 < self.in_duration:
                    # less frames than needed
                    pos = np.linspace(0, n_frame - 2, self.in_duration)
                else:  # take one
                    pos = np.sort(np.random.choice(list(range(n_frame - 2)), self.in_duration, replace=False))
            else:
                pos = np.linspace(0, n_frame - 2, self.in_duration)
            frame_list = [round(p) for p in pos]

        frame_list = [int(x) for x in frame_list]

        if not self.is_val:  # train
            coord_frame_list = self._sample_indices(n_frame)
        else:  # val
            coord_frame_list = self._get_val_indices(n_frame)

        assert len(coord_frame_list) == len(frame_list) // 2

        folder_id = str(int(self.vid_names[index]))
        video_data = self.load_one_video_json(folder_id)

        # union the objects of two frames
        object_set = set()
        for frame_id in coord_frame_list:
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']  # standard category: [0001, 0002]
                object_set.add(standard_category)
        object_set = sorted(list(object_set))

        # NOTE: IMPORTANT!!! To accelerate the data loader, here it only reads one image
        #  (they're of the same scale in a video) to get its height and width
        #  It must be modified for models using any appearance features.
        frames = []
        for fidx in coord_frame_list:
            frames.append(self.load_frame(self.vid_names[index], fidx))
            break  # only one image
        height, width = frames[0].height, frames[0].width

        frames = [img.resize((self.pre_resize_shape[1], self.pre_resize_shape[0]), Image.BILINEAR) for img in
                  frames]  # just one frame in List:frames

        if self.random_crop is not None:
            frames, (offset_h, offset_w, crop_h, crop_w) = self.random_crop(frames)
        else:
            offset_h, offset_w, (crop_h, crop_w) = 0, 0, self.pre_resize_shape
        if self.model not in ['coord', 'coord_latent', 'coord_latent_nl', 'coord_latent_concat']:
            frames = []
            for fidx in frame_list:
                frames.append(self.load_frame(self.vid_names[index], fidx))

        else:
            # Now for accelerating just pretend we have had frames
            frames = frames * self.in_duration

        scale_resize_w, scale_resize_h = self.pre_resize_shape[1] / float(width), self.pre_resize_shape[0] / float(
            height)
        scale_crop_w, scale_crop_h = 224 / float(crop_w), 224 / float(crop_h)

        box_tensors = torch.zeros((self.coord_nr_frames, self.num_boxes, 4), dtype=torch.float32)  # (8, 4, 4)
        box_categories = torch.zeros((self.coord_nr_frames, self.num_boxes))

        gt_placeholders_id = []
        gt_placeholders = []
        for frame_index, frame_id in enumerate(coord_frame_list):
            try:
                frame_data = video_data[
                    frame_id]  # frame_data:{'name': '551/03.jpg', 'labels': [{'box2d': {'x1': 8.0, 'y1': 10.0, 'x2': 150.0, 'y2': 235.0}, 'standard_category': 'hand'}, {'box2d': {'x1': 110.0, 'y1': 45.0, 'x2': 196.0, 'y2': 146.0}, 'standard_category': '0000'}, {'box2d': {'x1': 129.0, 'y1': 97.0, 'x2': 224.0, 'y2': 185.0}, 'standard_category': '0001'}]}
                # print("frame_data:{}".format(frame_data))
            except:
                frame_data = {'labels': []}

            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']
                gt_placeholders = frame_data['gt_placeholders']

                global_box_id = object_set.index(standard_category)

                box_coord = box_data['box2d']
                x0, y0, x1, y1 = box_coord['x1'], box_coord['y1'], box_coord['x2'], box_coord['y2']

                # scaling due to initial resize
                x0, x1 = x0 * scale_resize_w, x1 * scale_resize_w
                y0, y1 = y0 * scale_resize_h, y1 * scale_resize_h

                # shift
                x0, x1 = x0 - offset_w, x1 - offset_w
                y0, y1 = y0 - offset_h, y1 - offset_h

                x0, x1 = np.clip([x0, x1], a_min=0, a_max=crop_w - 1)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=crop_h - 1)

                # scaling due to crop
                x0, x1 = x0 * scale_crop_w, x1 * scale_crop_w
                y0, y1 = y0 * scale_crop_h, y1 * scale_crop_h

                # precaution
                x0, x1 = np.clip([x0, x1], a_min=0, a_max=223)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=223)

                # (cx, cy, w, h)
                gt_box = np.array([(x0 + x1) / 2., (y0 + y1) / 2., x1 - x0, y1 - y0], dtype=np.float32)

                # normalize gt_box into [0, 1]
                gt_box /= 224

                # load box into tensor
                try:
                    box_tensors[frame_index, global_box_id] = torch.tensor(gt_box).float()
                    # box_categories[frame_index, global_box_id] = 1 if box_data['standard_category'] == 'hand' else 2
                    if box_data['standard_category'] == '0000':
                        box_categories[frame_index, global_box_id] = 2  # 第一个物体
                    elif box_data['standard_category'] == '0001':
                        box_categories[frame_index, global_box_id] = 3  # 第二个物体
                    elif box_data['standard_category'] == 'hand':
                        box_categories[frame_index, global_box_id] = 1  # 手

                except IndexError:
                    pass

                x0, y0, x1, y1 = list(map(int, [x0, y0, x1, y1]))
        train_json = self.train
        video_id = folder_id
        gt_placeholders_label_new_id = []
        placeholders_label = []
        for i in range(len(train_json)):
            video_dict = train_json[i]
            if video_id == train_json[i]['id']:
                placeholders_label = video_dict['new_placeholdes']

        for object_str in placeholders_label:
            if object_str in list(self.train_all_object.keys()):
                gt_placeholders_label_new_id.append(self.train_all_object[object_str])
            else:
                print(object_str)

        verb_label = self.labels[index]
        for object in gt_placeholders:
            if object in list(self.data.keys()):
                gt_placeholders_id.append(self.data[object])

        if len(gt_placeholders_id) == 0:
            gt_placeholders_id.append(10000)
            gt_placeholders_id.append(10000)
            gt_placeholders_id.append(10000)

        elif len(gt_placeholders_id) == 1:
            gt_placeholders_id.append(10000)
            gt_placeholders_id.append(10000)

        elif len(gt_placeholders_id) == 2:
            gt_placeholders_id.append(10000)

        elif len(gt_placeholders_id) == 4:

            gt_placeholders_id.pop(-1)

        gt_placeholders_id = torch.tensor(gt_placeholders_id, dtype=torch.long)

        if len(gt_placeholders_label_new_id) == 0:
            gt_placeholders_label_new_id.append(10000)
            gt_placeholders_label_new_id.append(10000)
            gt_placeholders_label_new_id.append(10000)

        elif len(gt_placeholders_label_new_id) == 1:
            gt_placeholders_label_new_id.append(10000)
            gt_placeholders_label_new_id.append(10000)

        elif len(gt_placeholders_label_new_id) == 2:
            gt_placeholders_label_new_id.append(10000)
        gt_placeholders_label_new_id = torch.tensor(gt_placeholders_label_new_id, dtype=torch.long)

        return frames, box_tensors, box_categories, gt_placeholders_id, verb_label, gt_placeholders_label_new_id

    def __getitem__(self, index):
        '''
        box_tensors: [nr_frames, num_boxes, 4]
        box_categories: [nr_frames, num_boxes], value is 0(none), 1 (hand), 2 (object)
        frames: what about the frames shape?
        '''
        if self.is_val:
            frames, box_tensors, box_categories = self.test_sample_single(index)
            frames = self.transforms(frames)  # original size is (t, c, h, w)
            global_img_tensors = frames.permute(1, 0, 2, 3)  # (c, t, h, w)
            return global_img_tensors, box_tensors, box_categories, self.classes_dict[self.labels[index]], 0, 0, 0
        else:
            frames, box_tensors, box_categories, gt_placeholders_id, verb_label, gt_placeholders_label_new_id = self.train_sample_single(
                index)
            frames = self.transforms(frames)
            global_img_tensors = frames.permute(1, 0, 2, 3)  # (c, t, h, w)

            return global_img_tensors, box_tensors, box_categories, \
                self.classes_dict[self.labels[index]], gt_placeholders_id, verb_label, gt_placeholders_label_new_id

    def __len__(self):
        return len(self.vid_names)

    def unnormalize(self, img, divisor=255):
        """
        The inverse operation of normalization
        Both the input & the output are in the format of BxCxHxW
        """
        for c in range(len(self.img_mean)):
            img[:, c, :, :].mul_(self.img_std[c]).add_(self.img_mean[c])

        return img / divisor

    def img2np(self, img):
        """
        Convert image in torch tensors of BxCxTxHxW [float32] to a numpy array of BxHxWxC [0-255, uint8]
        Take the first frame along temporal dimension
        if C == 1, that dimension is removed
        """
        img = self.unnormalize(img[:, :, 0, :, :], divisor=1).to(torch.uint8).permute(0, 2, 3, 1)
        if img.shape[3] == 1:
            img = img.squeeze(3)
        return img.cpu().numpy()
