from logging import WARNING
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

from string import Template

from paths_creation import *


config_file_template = """
# Faster R-CNN with Resnet-152 (v1),
# Initialized from Imagenet classification checkpoint
#
# Train on GPU-8
#
# Achieves 37.3 mAP on COCO17 val

model {
  faster_rcnn {
    num_classes: 1
    image_resizer {
      identity_resizer {
      }
    }
    feature_extractor {
      type: 'faster_rcnn_resnet152_keras'
    }
    first_stage_anchor_generator {
      grid_anchor_generator {
        scales: [0.25, 0.5, 1.0, 2.0]
        aspect_ratios: [0.5, 1.0, 2.0]
        height_stride: 16
        width_stride: 16
      }
    }
    first_stage_box_predictor_conv_hyperparams {
      op: CONV
      regularizer {
        l2_regularizer {
          weight: 0.0
        }
      }
      initializer {
        truncated_normal_initializer {
          stddev: 0.01
        }
      }
    }
    first_stage_nms_score_threshold: 0.0
    first_stage_nms_iou_threshold: 0.7
    first_stage_max_proposals: 300
    first_stage_localization_loss_weight: 2.0
    first_stage_objectness_loss_weight: 1.0
    initial_crop_size: 14
    maxpool_kernel_size: 2
    maxpool_stride: 2
    second_stage_box_predictor {
      mask_rcnn_box_predictor {
        use_dropout: false
        dropout_keep_probability: 0.7
        fc_hyperparams {
          op: FC
          regularizer {
            l2_regularizer {
              weight: 0.0
            }
          }
          initializer {
            variance_scaling_initializer {
              factor: 1.0
              uniform: true
              mode: FAN_AVG
            }
          }
        }
        share_box_across_classes: true
      }
    }
    second_stage_post_processing {
      batch_non_max_suppression {
        score_threshold: 0.00
        iou_threshold: 0.6
        max_detections_per_class: 100
        max_total_detections: 100
      }
      score_converter: SOFTMAX
    }
    second_stage_localization_loss_weight: 2.0
    second_stage_classification_loss_weight: 1.0
  }
}

train_config: {
  batch_size: 3
  num_steps: $training_steps
  max_number_of_boxes: 100
  startup_delay_steps: 0
  unpad_groundtruth_tensors: false
  optimizer {
    momentum_optimizer: {
      learning_rate: {
        cosine_decay_learning_rate {
          learning_rate_base: $training_lr
          total_steps: $training_steps
          warmup_learning_rate: $warmup_lr
          warmup_steps: $warmup_steps
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
  gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint_version: V2
  fine_tune_checkpoint: $chckpnt_path
  fine_tune_checkpoint_type: "detection"
  data_augmentation_options {
      random_horizontal_flip {
    }
  }
}

train_input_reader: {
  label_map_path: $label_path
  tf_record_input_reader {
    input_path: "COTS/workspace/images/train-?????-of-00008"
  }
}

eval_config: {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
  batch_size: 1;
}

eval_input_reader: {
  label_map_path: $label_path
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "COTS/workspace/images/test-?????-of-00002"
  }
}
"""

TRAINING_STEPS = training_parameters["TRAINING_STEPS"]
WARMUP_STEPS = training_parameters["WARM_UP_STEPS"]
TRAINING_LR=training_parameters["TRAINING_LR"]
WARMUP_LR=training_parameters["WARMUP_LR"]
chkpnt_dir=os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')

pipeline = Template(config_file_template).substitute(
    training_steps=TRAINING_STEPS, 
    warmup_steps=WARMUP_STEPS,
    training_lr=TRAINING_LR,
    warmup_lr=WARMUP_LR,
    label_path='"'+paths['ANNOTATION_PATH']+'/label_map.pbtxt'+'"',
    chckpnt_path='"'+chkpnt_dir+'"')

with open(files['PIPELINE_CONFIG'], 'w') as f:
    f.write(pipeline)