import torch
import os
import numpy as np

import cv2
from PIL import Image
import argparse
from tqdm import tqdm

def extract_bboxes(mask):
    y_coord, x_coord = mask.nonzero()
    ymin, xmin = int(y_coord.min()), int(x_coord.min())
    ymax, xmax = int(y_coord.max()), int(x_coord.max())
    xmin = 0 if xmin < 0 else xmin
    ymin = 0 if ymin < 0 else ymin
    ymax = mask.shape[-2] - 1 if ymax >= mask.shape[-2] else ymax
    xmax = mask.shape[-1] - 1 if xmax >= mask.shape[-1] else xmax
    return [xmin, ymin, xmax, ymax]
    
    

def merge_regions(num_labels, stats, centroids, mask):
    if num_labels <= 1:
        return num_labels, stats, None
    region_mask = mask.copy()
    region_mapid = {}
    
    for i in range(num_labels):
        x1, y1, w1, h1, area1 = stats[i]
        x2 = x1 + w1
        y2 = y1 + h1
        
        if np.sum(region_mask==i+1) < 5:
            region_mask[region_mask==i+1] = 0
            region_mapid[i+1] = 0
        else:
            region_mapid[i+1] = i + 1
    
    for i in range(num_labels):
        tidi = region_mapid[i+1]
        if tidi == 0:
            continue
        xi1, yi1, xi2, yi2 = extract_bboxes((region_mask==tidi).astype(np.uint8))
        box_areai = (xi2-xi1+1)*(yi2-yi1+1)
        mask_areai = np.sum(region_mask==tidi)
        for j in range(i+1, num_labels):
            tidj = region_mapid[j+1]
            if tidj == 0:
                continue
            if tidi == tidj:
                continue
            xj1, yj1, xj2, yj2 = extract_bboxes((region_mask==tidj).astype(np.uint8))
            box_areaj = (xj2-xj1+1)*(yj2-yj1+1)
            mask_areaj = np.sum(region_mask==tidj)

            x1_merge = min(xi1, xj1)
            x2_merge = max(xi2, xj2)
            y1_merge = min(yi1, yj1)
            y2_merge = max(yi2, yj2)
            area_merge = (x2_merge - x1_merge) * (y2_merge - y1_merge)
            
            
            if box_areai + box_areaj > area_merge * 0.5 and (mask_areai+mask_areaj)/area_merge > min(0.5, min(mask_areai/box_areai, mask_areaj/box_areaj)-0.1):
                
                region_mask[region_mask==tidj] = tidi
                region_mapid[tidj] = tidi
                
                region_mapid[tidj] = tidi
                
                xi1, yi1, xi2, yi2 = extract_bboxes((region_mask==tidi).astype(np.uint8))
                box_areai = (xi2-xi1+1)*(yi2-yi1+1)
                mask_areai = np.sum(region_mask==tidi)
                for kid in region_mapid.keys():
                    if region_mapid[kid] == tidj:
                        region_mapid[kid] = tidi
    
    merge_regionid = np.unique(region_mask)
    merge_regionid = list(merge_regionid)
    if 0 in merge_regionid:
        merge_regionid.remove(0)
    
    
    num_labels_merge = len(merge_regionid)
    stats_merge = []
    mask_merge = []
    for mid in merge_regionid:
        if mid == 0:
            continue
        m = region_mask == mid
        mask_merge.append(m)
        boxes = extract_bboxes(m.astype(np.uint8))
        x1, y1, x2, y2 = boxes
        stats_merge.append([x1, y1, x2-x1+1, y2-y1+1, (x2-x1)*(y2-y1)])
    return num_labels_merge, stats_merge, mask_merge#region_mapid


def process(filename):
    if filename.endswith('.png'):
        label_path = os.path.join(args.input_dir, filename)
        cls_labels = np.asarray(Image.open(label_path), dtype=np.uint8)
    elif filename.endswith('.npy'):
        cam_dict = np.load(os.path.join(args.input_dir, filename), allow_pickle=True).item()
        cams = cam_dict[args.cam_type]
        bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), 1)
        cams = np.concatenate((bg_score, cams), axis=0)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels].astype(np.uint8)
    else:
        print('not support data type')
        exit()
    
    cateids = np.unique(cls_labels)
    
    for cateid in cateids:
        if cateid > 0:
            cate_mask = cls_labels == cateid
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cate_mask.astype(np.uint8), connectivity=8)
        
            num_labels_merge, stats_merge, mask_merge = merge_regions(num_labels-1, stats[1:], centroids[1:], labels)
            
            if mask_merge is not None:
                for ins_idx, m in enumerate(mask_merge):
                    save_name = '{}-{}-{}.png'.format(filename.split('.')[0], cateid, ins_idx+1)
                    cv2.imwrite(os.path.join(args.output_dir, save_name), 255*m.astype(np.uint8))
            else:
                for ins_idx in range(1, num_labels_merge+1):
                    m = labels == ins_idx
                    save_name = '{}-{}-{}.png'.format(filename.split('.')[0], cateid, ins_idx)
                    cv2.imwrite(os.path.join(args.output_dir, save_name), 255*m.astype(np.uint8))
            
     
        
def main():
    for file in tqdm(file_list):
        process(file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./cam_out", type=str)
    parser.add_argument("--cam_type", default="attn_highres", type=str)
    parser.add_argument("--output_dir", default="./ins_out", type=str)
    args = parser.parse_args()
    
    file_list = sorted(os.listdir(args.input_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    main()
    
    