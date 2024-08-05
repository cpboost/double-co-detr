cd ./code
python3 tools/modify_pth_key_in_mmdet.py      # 转换常规预训练模型为双流模型
python3 train.py ./configs/codino_vit_twostream_640_autoaugv1_train1.py #训练第一权重
python3 train.py ./configs/codino_vit_twostream_640_autoaugv2_train2.py         
python3 train.py ./configs/codino_vit_twostream_1280_autoaugv1_train3.py


############ 多卡训练如下
# gpu_nums=8
# bash dist_train.sh ./configs/codino_vit_twostream_640_autoaugv1_train1.py ${gpu_nums}
# bash dist_train.sh ./configs/codino_vit_twostream_640_autoaugv2_train2.py ${gpu_nums}
# bash dist_train.sh ./configs/codino_vit_twostream_1280_autoaugv1_train3.py ${gpu_nums}