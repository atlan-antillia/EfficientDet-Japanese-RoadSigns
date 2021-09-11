r"""Convert raw Microsoft COCO dataset to TFRecord for object_detection.
Attention Please!!!

1)For easy use of this script, Your coco dataset directory struture should like this :
    +Your coco dataset root
        +train2017
        +val2017
        +annotations
            -instances_train2017.json
            -instances_val2017.json
2)To use this script, you should download python coco tools from "http://mscoco.org/dataset/#download" and make it.
After make, copy the pycocotools directory to the directory of this "create_coco_tf_record.py"
or add the pycocotools path to  PYTHONPATH of ~/.bashrc file.

Example usage:
    python create_coco_tf_record.py --data_dir=/path/to/your/coco/root/directory \
        --set=train \
        --output_path=/where/you/want/to/save/pascal.record
        --shuffle_imgs=True
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pycocotools.coco import COCO
from PIL import Image
from random import shuffle
import os, sys
import numpy as np
import tensorflow as tf
import logging

import dataset_util
from absl import app
from absl import flags
from absl import logging
import pprint

flags.DEFINE_string('data_dir', '', 'Root directory to raw Microsoft COCO dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set or validation set')
flags.DEFINE_string('output_filepath', '', 'Path to output TFRecord')
flags.DEFINE_bool('shuffle_imgs',True,'whether to shuffle images of coco')
FLAGS = flags.FLAGS


def load_coco_dection_dataset(imgs_dir, annotations_filepath, shuffle_img = True ):
    """Load data from dataset by pycocotools. This tools can be download from "http://mscoco.org/dataset/#download"
    Args:
        imgs_dir: directories of coco images
        annotations_filepath: file path of coco annotations file
        shuffle_img: wheter to shuffle images order
    Return:
        coco_data: list of dictionary format information of each image
    """
    coco = COCO(annotations_filepath)
   
    img_ids = coco.getImgIds() # totally 82783 images
    cat_ids = coco.getCatIds() # totally 90 catagories, however, the number of categories is not continuous, \
                               # [0,12,26,29,30,45,66,68,69,71,83] are missing, this is the problem of coco dataset.
    print("----- len img_ids {}".format(len(img_ids)))
    if shuffle_img:
        shuffle(img_ids)

    coco_data = []

    nb_imgs = len(img_ids)
    for index, img_id in enumerate(img_ids):
        #if index % 100 == 0:
        #    print("Readling images: %d / %d "%(index, nb_imgs))
        img_info = {}
        bboxes = []
        labels = []

        img_detail = coco.loadImgs(img_id)[0]
        pic_height = img_detail['height']
        pic_width = img_detail['width']

        ann_ids = coco.getAnnIds(imgIds=img_id,catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            bboxes_data = ann['bbox']
            bboxes_data = [bboxes_data[0]/float(pic_width), bboxes_data[1]/float(pic_height),\
                                  bboxes_data[2]/float(pic_width), bboxes_data[3]/float(pic_height)]
                         # the format of coco bounding boxs is [Xmin, Ymin, width, height]
            bboxes.append(bboxes_data)
            labels.append(ann['category_id'])


        img_path = os.path.join(imgs_dir, img_detail['file_name'])
        #print("--- img_path {}".format(img_path))
        img_bytes = 0
        try:
          img_bytes = tf.io.gfile.GFile(img_path,'rb').read()
          #if img_bytes >0:
          img_info['pixel_data'] = img_bytes
          img_info['height'] = pic_height
          img_info['width'] = pic_width
          img_info['bboxes'] = bboxes
          img_info['labels'] = labels
          #print("==== img_boxes {}".format(img_info['bboxes']))
        except Exception as ex:
          print( ex)
        coco_data.append(img_info)
    return coco_data



def dict_to_coco_example(img_data):
    """Convert python dictionary formath data of one image to tf.Example proto.
    Args:
        img_data: infomation of one image, inclue bounding box, labels of bounding box,\
            height, width, encoded pixel data.
    Returns:
        example: The converted tf.Example
    """
    bboxes = img_data['bboxes']
    #print("---- image_data labels {}".format(img_data['labels'][0]) )
    #print("--- {}".format(dataset_util.bytes_list_feature(img_data['labels'])) )
    """
    classes_text = []
    dic = { 1: "car", 2: "track", 3: "crane", 4:"dump", 5: "mixer", 6:"trailer"}
    for label in img_data['labels']:
      
       print("---- label {}".format(label))
       name = dic[label]
       print("---- name {}".format(name))
       classes_text.append(name.encode('utf8'))
    print("--- classes_text {}".format(classes_text))
    """
    xmin, xmax, ymin, ymax = [], [], [], []

    for bbox in bboxes:
        xmin.append(bbox[0])
        xmax.append(bbox[0] + bbox[2])
        ymin.append(bbox[1])
        ymax.append(bbox[1] + bbox[3])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(img_data['height']),
        'image/width': dataset_util.int64_feature(img_data['width']),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        #'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(img_data['labels']),
        'image/encoded': dataset_util.bytes_feature(img_data['pixel_data']),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf-8')),
    }))

    return example

"""
./train/
    train.json
./valid/
    valid.json

./train/
    japanese_roadsigns.tfrecord
./valid/
    japanese_roadsigns.tfrecord

"""

def main(set="train"):
    if set == "train":
        imgs_dir = "./train/"

        annotations_filepath = "./train/train.json"
        print("Convert coco train file to tf record")
    elif set == "valid":
        imgs_dir = "./valid/"
        annotations_filepath = "./valid/valid.json"
        print("Convert coco val file to tf record")
    else:
        raise ValueError("you must either convert train data or val data")
    # load total coco data
    coco_data = load_coco_dection_dataset(imgs_dir,annotations_filepath,shuffle_img=False)
    total_dataset = len(coco_data)
    
    print("== total dataset {}".format(total_dataset))
    # write coco data to tf record
    tfrecord = "./" + set + "/japanese_roadsigns.tfrecord"
    print("==== tfrecord {}".format(tfrecord))
    with tf.io.TFRecordWriter(tfrecord) as tfrecord_writer:
        
        for img_data in coco_data:
            #print( "img_data {}".format(img_data))
            #if index % 100 == 0:
            #    print("Converting images: %d / %d" % (index, total_imgs))
            if img_data:
              example = dict_to_coco_example(img_data)
              tfrecord_writer.write(example.SerializeToString())


##
##
## python create_coco_tf_record.py --data_dir=./train --set=train --output_path=c:/tfrecord_japanese_roadsigns/train --shuffle_imgs=True
if __name__ == "__main__":

  set = "train"
  if len(sys.argv) > 1:
    set = sys.argv[1]
  print("===== set{}".format(set))
  try:
    if set== "train" or set == "valid":
      main(set)
    else:
      print("Invalid argment {}".format(set))
  except Exception as ex:
    print("Exception {}".format(ex))