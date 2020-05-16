# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves...

Some of the potential reasons for handling custom layers are...

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were Model Size, Model Inference, Model Accuracy

### Model Size

| Model | Before Conversion | After Conversion |
| ------ | ------ | ------ |
| SSD Resnet 50 | 358MB | 200MB |
| SSD MobileNet v1 | 74.7MB | 26.5MB |
| SSD MobileNet v2 | 183MB | 65MB |

### Inference Time

| Model | Before Conversion | After Conversion |
| ------ | ------ | ------ |
| SSD Resnet 50 | 76ms | 1.6s |
| SSD MobileNet v1 | 30ms | 50ms |
| SSD MobileNet v2 | 31ms | 55ms |

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

Afer researunstatfied ching 3 model and atlast used a model from Intel OpenVino Model Zoo due to poor performance. The problem with converted model is its accuracy, I have also included model size as a secondary metric. I have stated the models I experimented with. For more information take a look at Model Research.

In investigating potential people counter models, I tried each of the following three models:

- Model 1: SSD Resnet 50
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments
```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: SSD MobileNet v1
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments
```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: SSD MobileNet v2
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments
```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
