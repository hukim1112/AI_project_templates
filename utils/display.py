import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
import cv2
import tensorflow as tf

class ImageVisualizer(object):
    """ Class for visualizing image

    Attributes:
        idx_to_name: list to convert integer to string label
        class_colors: colors for drawing boxes and labels
        save_dir: directory to store images
    """

    def __init__(self, idx_to_name, class_colors=None, save_dir=None):
        self.idx_to_name = idx_to_name
        if class_colors is None or len(class_colors) != len(self.idx_to_name):
            self.class_colors = [[0, 255, 0]] * len(self.idx_to_name)
        else:
            self.class_colors = class_colors

        if save_dir is None:
            self.save_dir = './'
        else:
            self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)


    def draw_label_and_box(self, image, cls_name, box, font_args):
        """ 
        Args:
            image: numpy array (width, height, 3)
            cls_name : str, means the name of box's object.
            box: numpy array (4) == (min_x, min_y, max_x, max_y)
        """

        # label = [l_0, l_1, l_2 .. l_n], A list of integers, means each class id.
        # box = (min_x, min_y, max_x, max_y)
        cv2.rectangle(image, tuple( (int(box[0]), int(box[1])) ), tuple( (int(box[2]), int(box[3])) ),
            font_args['box_color'], font_args['box_thickness'] )

        cv2.putText(image, label, tuple( (int(box[0]), int(box[1])) ),
            font_args['font'], font_args['font_scale'],
            font_args['font_color'], font_args['line_type'])
        return

    def annotate_image(self, image, boxes, labels):
        """ Method to draw boxes and labels on a image.

        Args:
            image: numpy array (width, height, 3)
            boxes: numpy array (num_boxes, 4)
            labels: numpy array (num_boxes)
        """
        font_args = {'font':cv2.FONT_HERSHEY_SIMPLEX, 'font_scale':0.5, 
                'font_color':(0,0,0), 'line_type':2,
                'box_thickness':2, 'box_color':(0, 0, 255),
                'pred_box_color' :(0, 255, 0)}

        img = image.copy()
        for i, box in enumerate(boxes):
            idx = labels[i] - 1
            cls_name = self.idx_to_name[idx]
            # top_left = (box[0], box[1])
            # bot_right = (box[2], box[3])
            self.draw_label_and_box(img, cls_name, box, font_args)
        return img

    def save_image(self, name, image):
        save_path = os.path.join(self.save_dir, name)
        cv2.imwrite(save_path, image)


class VideoVisualizer(object):
    """ Class for visualizing image

    Attributes:
        idx_to_name: list to convert integer to string label
        class_colors: colors for drawing boxes and labels
        save_dir: directory to store images
    """

    def __init__(self, idx_to_name, class_colors=None, save_dir=None):
        self.idx_to_name = idx_to_name
        if class_colors is None or len(class_colors) != len(self.idx_to_name):
            self.class_colors = [[0, 255, 0]] * len(self.idx_to_name)
        else:
            self.class_colors = class_colors

        if save_dir is None:
            self.save_dir = './'
        else:
            self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def draw_label_and_box(self, img, label, box, font_args):
        cv2.rectangle(img, tuple( (int(box[0]), int(box[1])) ), tuple( (int(box[2]), int(box[3])) ),
            font_args['box_color'], font_args['box_thickness'] )

        cv2.putText(img, label, tuple( (int(box[0]), int(box[1])) ),
            font_args['font'], font_args['font_scale'],
            font_args['font_color'], font_args['line_type'])
        return

    def annotate_frame(self, frame, boxes, labels):
        """ Method to draw boxes and labels
            then save to dir

        Args:
            img: numpy array (width, height, 3)
            boxes: numpy array (num_boxes, 4)
            labels: numpy array (num_boxes)
        """
        font_args = {'font':cv2.FONT_HERSHEY_SIMPLEX, 'font_scale':0.5, 
                'font_color':(0,0,0), 'line_type':2,
                'box_thickness':2, 'box_color':(0, 0, 255),
                'pred_box_color' :(0, 255, 0)}

        img = frame.copy()
        for i, box in enumerate(boxes):
            idx = labels[i] - 1
            cls_name = self.idx_to_name[idx]
            # top_left = (box[0], box[1])
            # bot_right = (box[2], box[3])

            self.draw_label_and_box(img, cls_name, box, font_args)
        return img
