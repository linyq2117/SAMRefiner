import torch
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from torch.nn import functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import FastGeodis

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device) 
    return image.permute(2, 0, 1).contiguous()

def gaussian_2d(shape, gamma_x=1, gamma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * gamma_x * gamma_x) + y * y / (2 * gamma_y * gamma_y)))
    #h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h




def get_mask_embed(mask, img_embed):
    """Extract mask embedding

    :param mask: H x W, binary mask in numpy
    :param img_embed: 1 x c x h x w, image embedding tensor [256, 64, 64]
    :return:
        query_embed: 1 x c, mask embedding tensor
        mask_resize: 1 x 1 x h x w, the mask that is used to extract the embedding
    """
    orig_H, orig_W = mask.shape[:2]
    embed_H, embed_W = img_embed.shape[-2:]
    if orig_H >= orig_W:
        resize_W = int(embed_H * orig_W / orig_H)
        resize_H = embed_H
    else:
        resize_H = int(embed_W * orig_H / orig_W)
        resize_W = embed_W
    mask_resize = F.interpolate(mask[None, None].float(),#.to(img_embed.device),
                size=(resize_H, resize_W), mode='nearest')
    query_embed = (img_embed[:, :, :resize_H, :resize_W] * mask_resize).sum(dim=(-2, -1)) / mask_resize.sum()
    return query_embed, mask_resize




def extract_bboxes_expand(image_embeddings, mask, margin=0, img_path=None):
    """Compute bounding boxes from masks.
    mask: [num_instances,height, width,]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    ori_h, ori_w = mask.shape[-2:]
    if margin > 0 and ori_h > 0 and ori_w > 0:
        embed_H, embed_W = image_embeddings.shape[-2:]
        if ori_h >= ori_w:
            resize_W = int(embed_H * ori_w / ori_h)
            resize_H = embed_H
        else:
            resize_H = int(embed_W * ori_h / ori_w)
            resize_W = embed_W
        image_embeddings_resize = image_embeddings[:, :, :resize_H, :resize_W]
        image_embeddings_resize = F.interpolate(image_embeddings_resize, size=(ori_h, ori_w), mode='bilinear')
        image_embeddings_resize = image_embeddings_resize.permute(0, 2, 3, 1)
        image_embeddings_resize = image_embeddings_resize / image_embeddings_resize.norm(dim=-1, keepdim=True)
    
    boxes = []
    box_masks = []
    areas = []
    expand_list = []
    for i in range(mask.shape[0]):
        m = mask[i, :, :]
        # Bounding box.
        coord = torch.nonzero(m)
        y_coord, x_coord = coord[:, 0], coord[:, 1]
        try:
            y1, x1 = int(y_coord.min()), int(x_coord.min())
            y2, x2 = int(y_coord.max()), int(x_coord.max())
        except:
            y1, x1 = 0, 0
            y2, x2 = 0, 0
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        y2 = min(mask.shape[-2] - 1, y2)
        x2 = min(mask.shape[-1] - 1, x2)
        
        box_h = y2 - y1
        box_w = x2 - x1
        final_x1, final_x2, final_y1, final_y2 = x1, x2, y1, y2
        changed = False
        
        if box_h > 0 and box_w > 0 and margin > 0 and ori_h > 0 and ori_w > 0:
            steph = min(box_h * 0.1, 10)
            stepw = min(box_w * 0.1, 10)
            
        
            query_embed, mask_resize = get_mask_embed(m, image_embeddings)
            query_embed = query_embed / query_embed.norm(dim=-1, keepdim=True)
            sim = image_embeddings_resize @ query_embed.transpose(0, 1)
            sim = sim.squeeze()
            sim = sim > 0.5
            assert len(sim.shape) == 2, (sim.shape, m.shape, x1, x2, y1, y2, image_embeddings_resize.shape, ori_h, ori_w, query_embed.shape, img_path, mask.shape)
            
            temp_x1 = int(x1-stepw)
            
            if temp_x1 > 0 and temp_x1 < x1:
                context_area = (y2-y1) * (x1-temp_x1)
                sim_context = sim[y1:y2, temp_x1:x1]
                pos_area = sim_context.sum()
                if pos_area / context_area > margin:
                    final_x1 = temp_x1
                    changed = True
                    
            temp_x2 = int(x2+stepw)
            if temp_x2 < ori_w and temp_x2 > x2:
                context_area = (y2-y1) * (temp_x2-x2)
                sim_context = sim[y1:y2, x2:temp_x2]
                pos_area = sim_context.sum()
                if pos_area / context_area > margin:
                    final_x2 = temp_x2
                    changed = True
                    
            temp_y1 = int(y1-steph)
            if temp_y1 > 0 and temp_y1 < y1:
                context_area = (y1-temp_y1) * (x2-x1)
                sim_context = sim[temp_y1:y1, x1:x2]
                pos_area = sim_context.sum()
                if pos_area / context_area > margin:
                    final_y1 = temp_y1
                    changed = True
                    
            temp_y2 = int(y2+steph)
            if temp_y2 < ori_h and temp_y2 > y2:
                context_area = (temp_y2-y2) * (x2-x1)
                sim_context = sim[y2:temp_y2, x1:x2]
                pos_area = sim_context.sum()
                if pos_area / context_area > margin:
                    final_y2 = temp_y2
                    changed = True
                
        if changed:
            expand_list.append(1)
        else:
            expand_list.append(0)
            
        x1, x2, y1, y2 = final_x1, final_x2, final_y1, final_y2
        boxes.append(torch.tensor([x1, y1, x2, y2]))
        box_mask = torch.zeros((m.shape[0], m.shape[1])).to(image_embeddings.device)
        
        box_mask[y1:y2, x1:x2] = 1
        box_masks.append(box_mask)
        areas.append(1.0*(x2-x1)*(y2-y1))
    boxes = torch.stack(boxes, dim=0).reshape(-1, 4).to(image_embeddings.device)
    box_masks = torch.stack(box_masks,dim=0).to(image_embeddings.device)
    areas = torch.tensor(areas).reshape(-1).to(image_embeddings.device)
    expand_list = torch.tensor(expand_list).reshape(-1).to(image_embeddings.device)
    return boxes, box_masks, areas, expand_list




def extract_points(pred_masks, add_neg=True, use_mask=True, gamma=1.0):
    
    point_coords = []
    point_labels = []
    gaus_dt = []
    
    image_pt = torch.ones(pred_masks.shape[-2:]).float().unsqueeze_(0).unsqueeze_(0).to(pred_masks.device)
    v = 1e10
    # lamb = 0.0 (Euclidean) or 1.0 (Geodesic) or (0.0, 1.0) (mixture)
    lamb = 0.0
    iterations = 2
    for idx, pred_mask in enumerate(pred_masks):
        h, w = pred_mask.shape
       
        
        #pos point
        pred_mask_dt = FastGeodis.generalised_geodesic2d(
            image_pt, pred_mask.unsqueeze(0).unsqueeze(0), v, lamb, iterations
        )
        pred_mask_dt = pred_mask_dt.squeeze()
        
        
        pred_max_dist = pred_mask_dt.max()
        coords_y, coords_x = torch.where(pred_mask_dt == pred_max_dist)  # coords is [y, x]
        point_coords.append([coords_x[0], coords_y[0]])
        point_labels.append(1)
        
        coord = torch.nonzero(pred_mask)
        y_coord, x_coord = coord[:, 0], coord[:, 1]
        try:
            ymin, xmin = int(y_coord.min()), int(x_coord.min())
            ymax, xmax = int(y_coord.max()), int(x_coord.max())
        except:
            ymin, xmin = 0, 0
            ymax, xmax = 0, 0

        
        box_mask = torch.zeros_like(pred_mask).to(pred_masks.device)
        box_mask[ymin:ymax, xmin:xmax] = 1
        
        if add_neg:
            pred_mask_rev = pred_mask.clone().detach()
            
            assert pred_mask_rev.dtype == torch.uint8, "unsuitable data type {}".format(pred_mask_rev.dtype)
            assert pred_mask_rev.device == pred_mask.device
            pred_mask_rev[pred_mask_rev > 0] = 255
            pred_mask_rev = (~pred_mask_rev) / 255
            assert pred_mask_rev.max() <= 1
            
            pred_mask_dt_rev = FastGeodis.generalised_geodesic2d(
                image_pt, pred_mask_rev.unsqueeze(0).unsqueeze(0), v, lamb, iterations
            )
            pred_mask_dt_rev = pred_mask_dt_rev.squeeze()

            pred_mask_dt_rev[box_mask == 0] = 0

            pred_max_dist_rev = pred_mask_dt_rev.max()
            coords_y_neg, coords_x_neg = torch.where(pred_mask_dt_rev == pred_max_dist_rev)  # coords is [y, x]
            
            point_coords.append([coords_x_neg[0], coords_y_neg[0]])
            point_labels.append(0)
        
        if use_mask:
            pred_mask_dt_copy = pred_mask_dt.clone().detach()

            
            boxh, boxw = ymax-ymin, xmax-xmin
            mask_area = pred_mask.sum() / gamma
            mask_area = max(mask_area, 1)
            pred_max_dist = pred_mask_dt.max()
            pred_mask_dt0 = pred_mask_dt - pred_max_dist
            pred_mask_dt0 = torch.exp(-pred_mask_dt0*pred_mask_dt0/mask_area)
            pred_mask_dt0[pred_mask_dt_copy==0] = 0
            gaus_dt.append(pred_mask_dt0)
    
    point_coords = torch.tensor(point_coords).reshape(len(pred_masks),-1,2).to(pred_masks.device)
    point_labels = torch.tensor(point_labels).reshape(len(pred_masks),-1).to(pred_masks.device)
    if use_mask:
        gaus_dt = torch.stack(gaus_dt, dim=0).to(pred_masks.device)
    return point_coords, point_labels, gaus_dt




def extract_mask(pred_masks, gaus_dt, target_size, is01, strength=15, device=0, expand_list=0):
    pred_masks = pred_masks.float().unsqueeze(1)
    gaus_dt = gaus_dt.float().unsqueeze(1)

    if is01:
        pred_masks[pred_masks==0] = -1
        pred_masks[pred_masks==1] = 1
        padvalue = -1
    else:
        padvalue = -100
    pred_masks = F.interpolate(
            pred_masks, target_size, mode="bilinear", align_corners=False,
        )

    gaus_dt = F.interpolate(
                gaus_dt, target_size, mode="bilinear", align_corners=False,
            )

    h, w = pred_masks.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    pred_masks = F.pad(pred_masks, (0, padw, 0, padh), 'constant', padvalue)
    pred_masks = F.interpolate(
            pred_masks, (256,256),mode="bilinear", align_corners=False,
        )

    gaus_dt = F.pad(gaus_dt, (0, padw, 0, padh), 'constant', 0)
    gaus_dt = F.interpolate(
                gaus_dt, (256,256),mode="bilinear", align_corners=False,
            )

    if is01:
        for i in range(len(pred_masks)):
            if expand_list[i] == 0:
                pred_masks[pred_masks<=0] = -1*strength
                pred_masks[pred_masks>0] = strength
            else:
                pred_masks[pred_masks<=0] = -1
                pred_masks[pred_masks>0] = 1

        gaus_dt[gaus_dt<=0] = 1
        pred_masks = pred_masks * gaus_dt
        
    return pred_masks