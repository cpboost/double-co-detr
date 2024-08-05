import os
import os
import cv2
import json
run_py = os.path.abspath(__file__)
dirname = os.path.dirname(run_py) #! model_dir
import shutil
import glob


from tqdm import tqdm
import numpy as np
from collections import defaultdict
from wbf import *  

print(dirname)
def make_test_json(input_dir):
    data_dir = input_dir
    image_file_dir = os.path.join(data_dir, 'rgb')
    annotations_info = {'images': [], 'annotations': [], 'categories': []}
    categories_map = {"car": 1, "truck": 2, "bus":3, "van":4, "freight_car": 5}
    for key in categories_map:
        categoriy_info = {"id":categories_map[key], "name":key}
        annotations_info['categories'].append(categoriy_info)

    file_names = sorted([image_file_name.split('.')[0]
                for image_file_name in os.listdir(image_file_dir)])
    for i, file_name in enumerate(file_names):
        # print(i)
        image_file_name = file_name + '.jpg'

        image_file_path = os.path.join(image_file_dir, image_file_name)
        image_info = dict()
        image = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        image_info = {'file_name': image_file_name, 'id': i+1,
                    'height': height, 'width': width}
        annotations_info['images'].append(image_info)
    end_path = dirname +'/code/test.json'
    print('============',end_path)
    with  open(end_path, 'w')  as f:
        json.dump(annotations_info, f, indent=4)


def run_wbf(bboxes, confs, labels, iou_thr=0.50, skip_box_thr=0.0001, weights=None):
    scores = [conf for conf in confs]
    bboxes, scores, labels = weighted_boxes_fusion(
        bboxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type='avg')
        # bboxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type='max')
    return bboxes, scores, labels
def make_wbf_json(len_jpg):
    result_paths = [
        f'{dirname}/1.json.bbox.json',
        f'{dirname}/2.json.bbox.json',
        f'{dirname}/3.json.bbox.json',
    ]
    img_w = 640
    iou_thr = 0.8         # 0.492单光和0.502双光融合：0.519，目前最佳; 0.467单光融合后0.518 （iou=0.8, 然后是avg的方式，max只有0.513）
    # iou_thr = 0.9           # 0.492单光和0.502双光融合：0.516
    # iou_thr = 0.7         # 0.492单光和0.502双光融合：0.494
    # score_thr = 0.02   # clw note TODO
    score_thr = 0.000001
    norm_h, norm_w = 512, 640         # 防止wbf有些框x超出1
    # flip_flags = [False, True]
    flip_flags = [False, False, False]
    save_path = f'{dirname}/4.json.bbox.json'



    model_nums = len(result_paths)
    results_all_model = []

    for result_path in result_paths:
        result = json.load(open(result_path))

        imgid_to_results_dict = defaultdict(list)
        for i in range(1, len_jpg+1):   # 测试集共1000张图，id从1到1000
            imgid_to_results_dict[i] = []

        for item in tqdm(result):
            image_id = item['image_id']
            imgid_to_results_dict[image_id].append(item)

        results_all_model.append(imgid_to_results_dict)
    results_wbf_all = []

    test_sample_nums = len_jpg
    for sample_id in tqdm(range(1, test_sample_nums+1)):
        bboxes_all_model = []
        confs_all_model = []
        labels_all_model = []
        for model_idx, results_single_model in enumerate(results_all_model):

            results = results_single_model[sample_id]
            bboxes_single_model = []
            confs_single_model = []
            labels_single_model = []
            for dt in results:

                conf = dt['score']
                category_id = dt['category_id']
                if conf < score_thr:
                    continue

                if flip_flags[model_idx] == True:
                    dt['bbox'][0] = img_w - (dt['bbox'][0] + dt['bbox'][2])

                bboxes_single_model.append([dt['bbox'][0]/norm_w, dt['bbox'][1]/norm_h, (dt['bbox'][0]+dt['bbox'][2])/norm_w, (dt['bbox'][1]+dt['bbox'][3])/norm_h])  # x1,y1,x2,y2
                confs_single_model.append(conf)
                labels_single_model.append(category_id)

            bboxes_all_model.append(bboxes_single_model)
            confs_all_model.append(confs_single_model)
            labels_all_model.append(labels_single_model)
        bboxes_new, scores_new, labels_new = run_wbf(bboxes_all_model, confs_all_model, labels_all_model, iou_thr=iou_thr, skip_box_thr=score_thr, weights=None)
        bboxes_new[:, 0::2] *= norm_w
        bboxes_new[:, 1::2] *= norm_h
        for bbox_wbf, score_wbf, label_wbf in zip(bboxes_new, scores_new, labels_new):
            x1, y1, x2, y2 = bbox_wbf
            w = x2 - x1
            h = y2 - y1
            res_dict = {
                'image_id': sample_id,
                'bbox': [x1, y1, w, h],
                'score': score_wbf,
                'category_id': int(label_wbf)
            }
            results_wbf_all.append(res_dict)


    id_set = set()
    for res in results_wbf_all:
        id_set.add(res['image_id'])
    with open(save_path, 'w') as f:
        json.dump(results_wbf_all, f)

def invoke(input_dir, output_path):


    make_test_json(input_dir)
    path1=dirname+'/make_merge_test'
    path2 = dirname+'/code'
    print("=====================222======================")
    os.system(f'cd {path1} && python test.py --data_path {input_dir}')
    input_dir2 = os.path.join(input_dir, "rgb-2")
    print("=====================333======================")
    output_path2=dirname+'/1.json'
    output_path3=dirname+'/2.json'
    output_path4=dirname+'/3.json'
    os.system(f'cd {path2} && python test.py   ./configs/co_dino_5scale_vit_large_gaiic2024_two_stream_custom_v3.py ../data/best_model_epoch_9_only_keep_statedict.pth --data_path {input_dir2} --format-only --options "jsonfile_prefix={output_path2}"  ')
    print("=====================444======================")
    os.system(f'cd {path2} && python test.py   ./configs/co_dino_5scale_vit_large_gaiic2024_two_stream_custom_v3.py ../data/best_model_best_bbox_mAP_epoch_11_autov2_640_only_keep_statedict.pth --data_path {input_dir2} --format-only --options "jsonfile_prefix={output_path3}"  ')

    os.system(f'cd {path2} && python test.py   ./configs/1.py ../data/best_model_best_bbox_mAP_epoch_10_only_keep_statedict.pth --data_path {input_dir2} --format-only --options "jsonfile_prefix={output_path4}"  ')
    source_file =dirname +'/4.json.bbox.json'

    
    print("================wbf=======================")
    jpg_files = glob.glob(os.path.join(input_dir2, '*.jpg'))
    len_jpg = len(jpg_files)
    print(f'======================={len_jpg}===================')
    make_wbf_json(len_jpg)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copyfile(source_file, output_path)


if __name__ == '__main__':
    invoke(f'{dirname}/data/contest_data/test', f'{dirname}/data/result/result.json')