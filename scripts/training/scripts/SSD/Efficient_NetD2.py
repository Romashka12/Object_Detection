from logging import WARNING
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

from string import Template

from paths_creation import *

from string import Template

config_file_template = """
# SSD with EfficientNet-b0 + BiFPN feature extractor,
# shared box predictor and focal loss (a.k.a EfficientDet-d0).
# See EfficientDet, Tan et al, https://arxiv.org/abs/1911.09070
# See Lin et al, https://arxiv.org/abs/1708.02002
# Initialized from an EfficientDet-D0 checkpoint.
#
# Train on GPU

model {
  ssd {
    num_classes: 1
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 1280
        max_dimension: 1280
        pad_to_max_dimension: true
      }
    }
    feature_extractor {
      type: "ssd_efficientnet-b2_bifpn_keras"
      conv_hyperparams {
        regularizer {
          l2_regularizer {
            weight: 3.9999998989515007e-05
          }
        }
        initializer {
          truncated_normal_initializer {
            mean: 0.0
            stddev: 0.029999999329447746
          }
        }
        activation: SWISH
        batch_norm {
          decay: 0.9900000095367432
          scale: true
          epsilon: 0.0010000000474974513
        }
        force_use_bias: true
      }
      bifpn {
        min_level: 3
        max_level: 7
        num_iterations: 5
        num_filters: 112
      }
    }
    matcher {
      argmax_matcher {
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
        use_matmul_gather: true
      }
    }
    similarity_calculator {
      iou_similarity {
      }
    }
    box_predictor {
      weight_shared_convolutional_box_predictor {
        conv_hyperparams {
          regularizer {
            l2_regularizer {
              weight: 3.9999998989515007e-05
            }
          }
          initializer {
            random_normal_initializer {
              mean: 0.0
              stddev: 0.01
            }
          }
          activation: SWISH
          batch_norm {
            decay: 0.9900000095367432
            scale: true
            epsilon: 0.0010000000474974513
          }
          force_use_bias: true
        }
        depth: 112
        num_layers_before_predictor: 3
        kernel_size: 3
        class_prediction_bias_init: -4.599999904632568
        use_depthwise: true
      }
    }
    anchor_generator {
      multiscale_anchor_generator {
        min_level: 3
        max_level: 7
        anchor_scale: 4.0
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        scales_per_octave: 3
      }
    }
    post_processing {
      batch_non_max_suppression {
        score_threshold: 9.99999993922529e-09
        iou_threshold: 0.5
        max_detections_per_class: 100
        max_total_detections: 100
      }
      score_converter: SIGMOID
    }
    normalize_loss_by_num_matches: true
    loss {
      localization_loss {
        weighted_smooth_l1 {
        }
      }
      classification_loss {
        weighted_sigmoid_focal {
          alpha: 0.25
          gamma: 1.5
        }
      }
      classification_weight: 1.0
      localization_weight: 1.0
    }
    encode_background_as_zeros: true
    normalize_loc_loss_by_codesize: true
    inplace_batchnorm_update: true
    freeze_batchnorm: false
    add_background_class: false
  }
}
train_config {
  batch_size: 1
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_scale_crop_and_pad_to_square {
      output_size: 1280
      scale_min: 0.5
      scale_max: 2.0
    }
  }
  sync_replicas: false
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
  fine_tune_checkpoint: $chckpnt_path

  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_scale_crop_and_pad_to_square {
      output_size: 1280
      scale_min: 0.5
      scale_max: 2.0
    }
  }

  num_steps: $training_steps
  startup_delay_steps: 0.0
  replicas_to_aggregate: 1
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
  fine_tune_checkpoint_type: "detection"
  use_bfloat16: false
  fine_tune_checkpoint_version: V2
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
  batch_size: 2;
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
chkpnt_dir=os.path.join(paths['PRETRAINED_MODEL_PATH'], parameters["PRETRAINED_MODEL_NAME"], 'checkpoint', 'ckpt-0')

pipeline = Template(config_file_template).substitute(
    training_steps=TRAINING_STEPS, 
    warmup_steps=WARMUP_STEPS,
    training_lr=TRAINING_LR,
    warmup_lr=WARMUP_LR,
    label_path='"'+paths['ANNOTATION_PATH']+'/label_map.pbtxt'+'"',
    chckpnt_path='"'+chkpnt_dir+'"')

with open(files['PIPELINE_CONFIG'], 'w') as f:
    f.write(pipeline)