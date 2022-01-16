import pandas as pd
from pathlib import Path
from math import ceil

import json
import sys
import contextlib2
import io
import IPython
import json
import numpy as np
import os
import pathlib
import pandas as pd
import sys
import tensorflow as tf
import time
from sklearn.model_selection import StratifiedKFold

from PIL import Image, ImageDraw

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

from object_detection.utils import dataset_util
from object_detection.dataset_tools import tf_record_creation_util


IMAGE_HEIGHT=720
IMAGE_WIDTH=1280

def parse_box_dict(box_dict):
  x_mins=[]
  y_mins=[]
  y_maxs=[]
  x_maxs=[]

  box_dict=json.loads(box_dict.replace("'", '"'))
  for v in (box_dict):
    x_mins.append(v['x']/IMAGE_WIDTH)
    x_maxs.append((v['x']+v['width'])/IMAGE_WIDTH)
    y_mins.append(v['y']/IMAGE_HEIGHT)
    y_maxs.append((v['y']+v['height'])/IMAGE_HEIGHT)

  return (y_mins, x_mins, y_maxs, x_maxs)

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )

def create_example(row):

  box_dict=row['annotations']
  y_mins,x_mins, ymaxs, x_maxs=parse_box_dict(box_dict)

  image_path=path_to_data/'train_images'/f"video_{row['video_id']}"/f"{row['video_frame']}.jpg"
  im = tf.io.decode_jpeg(tf.io.read_file(str(image_path)))
  y_mins, x_mins, y_maxs, x_maxs=parse_box_dict(box_dict)
  n_classes=len(y_mins)
  classes=[('COTS').encode()]*n_classes
  labels=[1]*n_classes

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(IMAGE_HEIGHT),
      'image/width': dataset_util.int64_feature(IMAGE_WIDTH),
      'image/detections_number':dataset_util.int64_feature(n_classes),
      'image/encoded': image_feature(im),
      'image/path': dataset_util.bytes_feature(str(image_path).encode()),
      'image/sequence_id':dataset_util.int64_feature(row['sequence']),
      'image/video_id':dataset_util.int64_feature(row['video_id']),
      'image/video_frame':dataset_util.int64_feature(row['video_frame']),
      'image/sequence_frame':dataset_util.int64_feature(row['sequence_frame']),
      'image/object/bbox/xmin': dataset_util.float_list_feature(x_mins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(x_maxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(y_mins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(y_maxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes),
      'image/object/class/label': dataset_util.int64_list_feature(labels)}))
  
  return tf_example

def convert_to_tfrecord(data_df, tfrecords_dir, num_shards = 5):
  """Convert the object detection dataset to TFRecord as required by the TF ODT API."""

  if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)
  
  fold_number={f:i for i,f in enumerate(data_df['fold'].unique())}
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, tfrecords_dir, num_shards)
  
    print(fold_number)

    for i,(index, row) in enumerate(data_df.iterrows()):
      if i % 100 == 0:
        print('Processed {0} images.'.format(i))
      tf_example = create_example(row)
      output_shard_index = int(fold_number[row['fold']])
      output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
  
  print('Completed processing {0} images.'.format(len(data_df)))

if __name__=="__main__":

    INPUT_DIR = sys.argv[1]
    OUTPUT_DIR=sys.argv[2]

    path_to_data=Path(INPUT_DIR)
    tfrecords_dir=Path(OUTPUT_DIR)

    n_folds=8

    df=pd.read_csv(path_to_data/'train.csv')

    df = df[df.annotations!='[]']
    df = df.reset_index(drop=True)

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=df.video_id)): 
        df.loc[v_, 'fold'] = f

    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)

    test_folds=[1,4]

    convert_to_tfrecord(df[~df['fold'].isin(test_folds)], 
                        tfrecords_dir=tfrecords_dir/'train', 
                        num_shards = 8)

    convert_to_tfrecord(df[df['fold'].isin(test_folds)], 
                        tfrecords_dir=tfrecords_dir/'test', 
                        num_shards = 2)