import mmcv
from mmdet.apis import init_detector, inference_detector
import time
import json
import argparse

import numpy as np

import glob
import os

def parse_args():
    parser = argparse.ArgumentParser(description='create submission file')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='the checkpoint file to resume from')
    parser.add_argument('--save-dir', help='the dir to save images')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    config_file = args.config
    checkpoint_file = args.checkpoint
    save_folder = args.save_dir
    public_test = "data/za_traffic_2020/traffic_public_test/images/*.png"
    score_thr = 0.3

    mmcv.mkdir_or_exist(save_folder)

    class_names = ["1. No entry", "2. No parking / waiting", \
                "3. No turning", "4. Max Speed", \
                "5. Other prohibition signs", "6. Warning", "7. Mandatory"]

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    submit_list = []

    for idx_image, test_file_path in enumerate(glob.glob(public_test)):
        img_file = os.path.basename(test_file_path)
        out_img_path = os.path.join(save_folder, img_file)
        img_name, _ = img_file.split('.')
        print ("#"*50)
        print ("BEGIN - image index: {}, image name: {}".format(idx_image, img_name))
        img_im_temp = mmcv.imread(test_file_path)
        bbox_result = inference_detector(model, test_file_path)
        model.show_result(img_im_temp, bbox_result, score_thr=0.3, show=False, out_file=out_img_path)
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
            temp_ = dict()
            temp_["image_id"] = int(img_name)
            bbox_elem0, bbox_elem1, bbox_elem2, bbox_elem3, score = bbox
            bbox_width = bbox_elem2 - bbox_elem0
            bbox_height = bbox_elem3 - bbox_elem1
            
            temp_["category_id"] = int(label) + 1
            temp_["bbox"] = [float(item) for item in [round(bbox_elem0, 2), round(bbox_elem1, 2), round(bbox_width, 2), round(bbox_height, 2)]]
            temp_["score"] = float(score)
            print (bbox_elem0, bbox_elem1, bbox_elem2, bbox_elem3)
            print ((round(bbox_elem0, 2)), round(bbox_elem1, 2), round(bbox_width, 2), round(bbox_height, 2))
            print ('{:.2f} {:.2f} {:.2f} {:.2f}'.format(bbox_elem0, bbox_elem1, bbox_width, bbox_height))
            print (label)

            submit_list.append(temp_)
        print ("END - image index: {}, image name: {}".format(idx_image, img_name))
        print ("#"*50)
    out_file = open("./submit_{}.json".format(time.time()), "w")  
    json.dump(submit_list, out_file, indent = 6)  
    out_file.close()

if __name__ == '__main__':
    main()