
# 代码说明

## 项目背景
- 2024GAIIC-无人机视角下的双光目标检测，我们获得此次挑战的冠军，[比赛链接](https://www.heywhale.com/org/2024gaiic/competition/area/65f7abcf019d8282037f3924/content/4)

## 目录结构
```
project
    code 我们使用mmdet作为训练框架，使用里面的co-detr作为基础模型
        condigs 里面是是训练配置和推理配置
        mmdet 里面是模型各个框架代码
        tools 里面的代码时将初始权重变成双流模型
        dist_train.sh 分布式训练启动脚本
        test.py 推理脚本
        train.py 训练脚本
    data
        pretrain_model 这里需要下载vit官方初始化权重
        contest_data 官方数据
        三个pth权重，用于测试最好的成绩，进行wbf推理
    make_merge_test 用于生成融合配准的rgb图像
    index.py 最好成绩推理脚本
    init.sh 环境构建脚本
    test.sh 运行推理
    train.sh 运行训练
    wbf.py 模型融合代码
==========运行过程中会产生4个json文件，然后最终json放在data/result/result.json文件下
==========注意，这个版本需要在有网络情况下才能运行，部分模型可能有依赖需要下载，
```


## 训练配置
- 硬件配置：8卡A100/A800（80G显存）
- 训练花费时长：共2天（模型1：12h, 模型2：12h，模型3：24h）；相关配置文件详见train.sh。

## 环境配置
- 详见init.sh注释说明

## 数据
- 我们训练仅使用了官方提供的contest_data数据集，共有1.7w张训练图片，1.4k张验证图片，官方测试集为1000张图片。

## 预训练模型
- 仅使用了Co-DETR作者开源的ViT预训练权重，下载地址：https://drive.google.com/drive/folders/1-vAVIHHJ6Gyw0E6mGdbjdZ1hjkEdo3Rt

## 算法

### 整体思路介绍
- 网络选型方面，使用了在COCO数据集上接近SOTA的模型Co-DETR，相比DETR检测器增加了多个并行辅助头部以及多种标签分配方式，增加了与gt匹配的正样本数量，提高了编码器在端到端检测器中的学习能力；主干网络使用了改进后的ViT-L结构，在neck部分增加了多尺度输出，取得了优于Swin Transformer的精度；
- 在可见光和红外光双光融合的部分，采用了上限更高的特征级融合的方法，使用包含相同backbone的双路结构分别提取可见光和红外光输入图像的特征，并在neck输入侧进行特征融合，融合后保持和原始目标检测网络相同的tensor维度，再送入单路的neck和head进行目标的分类和回归；
- 模型训练尺寸使用640输入，和原图尺寸一致，更大的输入尺寸并没有保留更多的有效信息，对于最终结果也不会带来稳定的收益；


## 数据集说明     
```
contest_data
      |--val
            |--rgb      # 可见光模态测试数据
                  |--000001.jpg
                  ...
            |--tir      # 红外光模态测试数据
                  |--000001.jpg
                  ...
            val.json  # 验证数据标注文件
      |--train
            |--rgb      # 可见光模态训练数据
                  |--000001.jpg
                  ...
            |--tir      # 红外光模态训练数据
                  |--000001.jpg
                  ...
            train.json  # 训练数据标注文件
      |--test
            |--rgb      # 可见光模态测试数据
                  |--000001.jpg
                  ...
            |--tir      # 红外光模态测试数据
                  |--000001.jpg
                  ...
            test.json  # 测试数据标注文件, 推理阶段会自动生成
```
## 训练流程
- 将contest_data数据放入到project/data目录下；
- bash train.sh

## 测试流程
- bash test.sh

## 结果说明
- 训练过程中，会生成4个json文件，分别是模型1、模型2、模型3、wbf融合结果；
- 最终结果会放在data/result/result.json文件下；
