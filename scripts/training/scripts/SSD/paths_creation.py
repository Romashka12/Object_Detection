import os

training_parameters={"TRAINING_STEPS":40000,
            "WARM_UP_STEPS":1000,
            "WARMUP_LR":1e-4,
            "TRAINING_LR":1e-3}

parameters= {"CUSTOM_MODEL_NAME":'SSD_EfficientD2',
             "PRETRAINED_MODEL_NAME":'efficientdet_d2_coco17_tpu-32',
             "INITIAL_DATA_DIR":"/content/drive/MyDrive/deep_learning/Object_detection/Coral_Reef/data/tensorflow-great-barrier-reef.zip",
             'TF_RECORD_SCRIPT_NAME':'generate_tfrecord.py',
             'PRETRAINED_MODEL_URL':'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz',
             'GDRIVE_SCRIPTS':"/content/drive/MyDrive/deep_learning/Object_detection/Coral_Reef/High_level/SSD_scripts",
             'LABEL_MAP_NAME':'label_map.pbtxt',
             'DRIVE_SAVE_PATH':"/content/drive/MyDrive/deep_learning/Object_detection/Coral_Reef"
             }

paths = {
    'WORKSPACE_PATH': os.path.join('COTS', 'workspace'),
    'RAW_DATA_PATH': os.path.join('COTS', 'raw_data'),
    'SCRIPTS_PATH': os.path.join('COTS','scripts'),
    'APIMODEL_PATH': os.path.join('COTS','models'),
    'ANNOTATION_PATH': os.path.join('COTS', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('COTS', 'workspace','images'),
    'MODEL_PATH': os.path.join('COTS', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('COTS', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('COTS', 'workspace','models',parameters["CUSTOM_MODEL_NAME"]), 
    'OUTPUT_PATH': os.path.join('COTS', 'workspace','models',parameters["CUSTOM_MODEL_NAME"], 'export'), 
    'TFJS_PATH':os.path.join('COTS', 'workspace','models',parameters["CUSTOM_MODEL_NAME"], 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('COTS', 'workspace','models',parameters["CUSTOM_MODEL_NAME"], 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('COTS','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('COTS', 'workspace','models', parameters["CUSTOM_MODEL_NAME"], 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], parameters["TF_RECORD_SCRIPT_NAME"]), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], parameters["LABEL_MAP_NAME"])
}

if __name__=="__main__":
    
    if not os.path.exists('COTS'):
        os.mkdir('COTS')

    for path in paths.values():
        if not os.path.exists(path):
            os.mkdir(path)