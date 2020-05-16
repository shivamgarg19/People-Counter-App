# Project Write-Up

The people counter application to create a smart video IoT solution using Intel® hardware and software tools. The app will detect people in a designated area, providing the number of people in the frame, average duration of people in frame, and total count.

## Explaining Custom Layers

OpenVino toolkit use model optimizer to reduce the size of the model. Model optimizer searches for the list of known layers for each layer in the model. The inference engine loads the layers from the model IR into the specified device plugin, which will search a list of known layer implementations for the device. If your model architecure contains layer or layers that are not in the list of known layers for the device, the Inference Engine considers the layer to be unsupported and reports an error. To use the model having the unsupported layer we need to use the custom layer feature of OpenVino Toolkit.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations were Model Size, Model Inference, Model Accuracy

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

This project can used in many areas, here are the few i think that can have huge impact.

* Queue Management System
* Security Management 
* Space Management

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows.

* Poor lighting can huge impact on device accuracy, but this can avoided by having good Night Vision Hardware.
* Distorted input from camera due to change in focal length, image size will affect the model accuracy because it may fail to detect person. An approach to solve this would be to use some augmented images while training models and specifying the threshold skews.

## Model Research

Afer researunstatfied ching 3 model and atlast used a model from Intel OpenVino Model Zoo due to poor performance. The problem with converted model is its accuracy, I have also included model size as a secondary metric. I have stated the models I experimented with. For more information take a look at Model Research.

In investigating potential people counter models, I tried each of the following three models:

- Model 1: SSD Resnet 50
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments
```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```
  - The model was insufficient for the app because the inference time(1.6s) is too slow for any practical use.
  
- Model 2: SSD MobileNet v1
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments
```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```
  - The model was insufficient for the app because the accuracy of the model is poor.

- Model 3: SSD MobileNet v2
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments
```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```
  - The model was insufficient for the app because the accuracy of the model is not good (This model has best of all three)
  
  ## Final Model
  
  After Researching 3 model for this project, none of the model seems promosing that's why i decided to to use intel pretrained model -> [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)

This pretrained model work best is all aspect(Size, Inference time, Accuracy) compared to all 3 models.

