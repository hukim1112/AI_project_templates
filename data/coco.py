#!/usr/bin/python3
# coding=utf-8
import os, json
import numpy as np
import cv2
import sys
from random import shuffle, choice
import tensorflow as tf
from functools import partial
from pycocotools.coco import COCO
from utils import box_utils, augmentation

def set_id_to_label(label_set):
    id_to_label = {}
    for idx, label in enumerate(label_set):
        id_to_label[idx+1] = label
    return id_to_label

class Path_finder():
    def __init__(self, dataset_dir, annotation_path):
        with open(annotation_path, 'r') as file:
            source_info = json.load(file)['source']
        self.dataset_dir = dataset_dir
        self.img_dir = {}
        for src in source_info:
            _path = os.path.join(dataset_dir, 'path_finder', src['name']+'.json')
            with open(_path) as file:
                image_dir = json.load(file)['image_dir']
            self.img_dir[src['id']] = image_dir
    def find_path(self, source_id, filename):
        return os.path.join(self.dataset_dir, self.img_dir[source_id], filename)

class Dataset():
    def __init__(self, dataset_root, input_shape, label_set, anchors):
        self.dataset_dir = dataset_root
        self.id_to_label = set_id_to_label(label_set)
        self.input_shape = input_shape
        self.default_boxes = anchors

    def __len__(self):
        return len(self.ids)

    def _get_image(self, image_id, coco):
        image_info = coco.loadImgs(image_id)[0]
        filename = image_info['file_name']
        source = image_info['source']
        original_size = (int(image_info['height']), int(image_info['width']))
        image_path = self.path_finder.find_path(source, filename)
        image = cv2.imread(image_path)
        if image is None:
            filename = os.path.splitext(filename)[0] + '.JPG'
            image_path = self.path_finder.find_path(source, filename)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Can't not find correct filepath", filename)
            print(filename)
        image = image[:,:,::-1]
        return filename, image, original_size

    def _get_annotation(self, image_id, coco):
        ann_ids = coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids, iscrowd=None)
        boxes = []
        for ann_id in ann_ids:
            x, y, w, h = coco.loadAnns(ann_id)[0]['bbox']
            xmin = x
            ymin = y
            xmax = (x+w)
            ymax = (y+h)
            box = [xmin, ymin, xmax, ymax]
            boxes.append(box)
        labels = [ self.annoID_to_modelID[coco.loadAnns(ann_id)[0]['category_id']] for ann_id in ann_ids]
        return boxes, labels

    def generate(self, ids, coco, is_augment, is_box):
        for image_id in ids:
            filename, image, original_size = self._get_image(image_id, coco)
            boxes, labels = self._get_annotation(image_id, coco)
            if len(labels) == 0:
                continue
            #assert image.shape[:2] == original_size[:2]
            height, width = original_size
            #normalize bounding box coord (range in 0~1.0), [xmin, ymin, xmax, ymax]
            boxes = list(map(lambda box : (box[0]/width, box[1]/height, box[2]/width, box[3]/height), boxes))
            boxes = np.array(boxes, np.float32); labels = np.array(labels, np.float32);
            if is_augment:
                image, boxes, labels = augmentation.augment(image, boxes, labels)
            boxes = np.array(boxes, np.float32)
            gt_confs, gt_locs = box_utils.compute_target(self.default_boxes, boxes, labels)
            image = cv2.resize(image, self.input_shape[:2], interpolation = cv2.INTER_AREA)

            if is_box:
                yield filename, image, labels, boxes
            else:
                yield filename, image, gt_confs, gt_locs

    def load_data_generator(self, annotation_path, is_augment=True, is_box=False, num_examples=-1):
        """
            num_examples : The number of examples to be used.
            It's used if you want to make model overfit a few examples
        """
        coco = COCO(annotation_path)
        self.path_finder = Path_finder(self.dataset_dir, annotation_path)
        ids = coco.getImgIds()
        if num_examples>0:
            ids =ids[:num_examples]
        shuffle(ids)
        # Sometimes, we use some categories instead of using every category.
        # Then category id of annotation is different from category id of model.
        # The dict "annoID_to_model_ID" converts category id to model id.
        self.cat_ids = coco.getCatIds(self.id_to_label.values())
        self.annoID_to_modelID = {}
        for model_id, cat_id in enumerate(self.cat_ids):
            self.annoID_to_modelID[cat_id] = model_id+1
        # pre-argumenting self.generate function.
        gen = partial(self.generate, ids, coco, is_augment, is_box)
        # generate data pipeline with from_generator in TensorFlow dataset APIs
        dataset = tf.data.Dataset.from_generator(gen,
            (tf.string, tf.float32, tf.int32, tf.float32))
        return dataset, len(ids)
