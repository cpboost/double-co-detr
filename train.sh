cd ./code
python3 tools/modify_pth_key_in_mmdet.py      # Convert a conventional pre-trained model into a dual-stream model
python3 train.py ./configs/codino_vit_twostream_640_autoaugv1_train1.py   #train step 1
python3 train.py ./configs/codino_vit_twostream_640_autoaugv2_train2.py    #train step 2     
python3 train.py ./configs/codino_vit_twostream_1280_autoaugv1_train3.py    #train step 3


############ Multi-GPU training is as follows:
# gpu_nums=8
# bash dist_train.sh ./configs/codino_vit_twostream_640_autoaugv1_train1.py ${gpu_nums}
# bash dist_train.sh ./configs/codino_vit_twostream_640_autoaugv2_train2.py ${gpu_nums}
# bash dist_train.sh ./configs/codino_vit_twostream_1280_autoaugv1_train3.py ${gpu_nums}
