
# Code Description


## Introduction to the Competition
Global Artificial Intelligence Innovation Competition (GAIIC) is organised by the Chinese Society for Artificial Intelligence and Hangzhou Municipal People's Government. The competition is based on an international perspective, focuses on cutting-edge technology and application innovation, promotes academic exchanges, talent cultivation, technological development, and cross-border application and fusion in the field of AI, and creates a platform for the exchange of talents and an industrial ecosystem for AI.


The competition has been successfully held for three times so far, and has attracted nearly 20,000 teams from universities, famous enterprises and research institutions at home and abroad to participate in the competition, which has gradually grown into one of the influential activities in the field of artificial intelligence.


The past competitions have brought together big names in the academic field, and the guiding guests of the competition include Dai Qionghai, Counsellor of the State Council, Chairman of CAAI, Academician of the Chinese Academy of Engineering, Dean of the School of Information Technology of Tsinghua University, and CAAI Fellow; Chen Jie, Vice Chairman of CAAI, Academician of the Chinese Academy of Engineering and CAAI Fellow; Wang Endong, Vice Chairman of CAAI, Academician of the Chinese Academy of Engineering, Chief Scientist of Wave Group and CAAI Fellow; and Wang Jie, Vice Chairman of CAAI. Fellow Wang Endong, CAAI Vice President, Academician of Chinese Academy of Engineering, Director of National Agricultural Informatisation Engineering and Technology Research Centre, CAAI Fellow Zhao Chunjiang, CAAI Supervisor, Academician of Chinese Academy of Engineering, Professor of Hunan University, CAAI Fellow Wang Yaonan, Foreign Academician of the European Academy of Sciences, Executive Vice President of the Institute of Artificial Intelligence of Tsinghua University, CAAI Fellow, Sun Maosong, CAAI Vice President, Foreign Academician of the European Academy of Sciences, Dean of the School of Artificial Intelligence of Nanjing University, CAAI Fellow, Zhou Zhihua.

![image](https://github.com/user-attachments/assets/86417b1c-8c97-4da0-a6b1-c9521ff93715)

## Project Background
- 2024GAIIC - Dual Spectrum Object Detection from Drone Perspectives. We won the championship in a competition with over 1200 teams and more than 8200 submissions
## Directory Structure
```
project
    code - We use mmdet as the training framework and use co-detr as the base model
        configs - Contains training and inference configurations
        mmdet - Contains various model framework codes
        tools - Contains code to transform initial weights into a dual-stream model
        dist_train.sh - Distributed training launch script
        test.py - Inference script
        train.py - Training script
    data
        pretrain_model - Download the official ViT initialization weights here
        contest_data - Official data
        Three pth weights - Used for testing the best performance and performing WBF inference
    make_merge_test - Used to generate fused registered RGB images
    index.py - Best performance inference script
    init.sh - Environment setup script
    test.sh - Run inference
    train.sh - Run training
    wbf.py - Model fusion code
===== Four JSON files will be generated during the process, and the final JSON will be placed in data/result/result.json  
===== Note that this version requires an internet connection to run, as some models may have dependencies that need to be downloaded. 

```


## Training Configuration
- Hardware Configuration: 8 x A100/A800 GPUs (80GB VRAM)
- Training Duration: 2 days in total (Model 1: 12h, Model 2: 12h, Model 3: 24h); for detailed configuration, refer to train.sh.

## Environment Setup
- Detailed instructions in init.sh comments

## Data
- We only used the official contest_data dataset for training, with a total of 17k training images, 1.4k validation images, and 1k test images.

## Pretrained Models
- We only used the official contest_data dataset for training, with a total of 17k training images, 1.4k validation images, and 1k test images.，[下载地址](https://drive.google.com/drive/folders/1-vAVIHHJ6Gyw0E6mGdbjdZ1hjkEdo3Rt)

## Algorithm

### Overview
-For network selection, we used Co-DETR, which is close to SOTA on the COCO dataset. Compared to the DETR detector, it adds multiple parallel auxiliary heads and various label assignment methods, increasing the number of positive samples matched with GT and improving the encoder learning ability in end-to-end detectors. The backbone network uses an improved ViT-L structure, and a multi-scale output is added in the neck part, achieving better accuracy than Swin Transformer.
-For visible and infrared dual-spectral fusion, we adopted a feature-level fusion method with higher potential. The dual-path structure with the same backbone is used to extract features from visible and infrared input images, and feature fusion is performed at the neck input side. After fusion, the tensor dimensions are kept the same as the original object detection network and then sent to a single path neck and head for object classification and regression.
-The model training size uses 640 inputs, consistent with the original image size. Larger input sizes do not retain more valid information and do not bring stable benefits to the final result.
## Dataset Description     
```
contest_data
      |--val
            |--rgb      # Visible spectrum validation data
                  |--000001.jpg
                  ...
            |--tir      # Infrared spectrum validation data
                  |--000001.jpg
                  ...
            val.json  # Validation data annotation file
      |--train
            |--rgb      # Visible spectrum training data
                  |--000001.jpg
                  ...
            |--tir      # Infrared spectrum training data
                  |--000001.jpg
                  ...
            train.json  # Training data annotation file
      |--test
            |--rgb      # Visible spectrum test data
                  |--000001.jpg
                  ...
            |--tir      # Infrared spectrum test data
                  |--000001.jpg
                  ...
            test.json  # Test data annotation file, generated during inference

```
## Training Process
- Training Process
- bash train.sh

## Testing Process
- bash test.sh

## Results Description
- During training, four JSON files will be generated, corresponding to Model 1, Model 2, Model 3, and WBF fusion results;
- The final result will be placed in data/result/result.json.
