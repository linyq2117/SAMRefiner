import numpy as np
import torch
import os
from collections import defaultdict
import cv2
from tqdm import tqdm
from segment_anything.utils.transforms import ResizeLongestSide
from utils import prepare_image, extract_bboxes_expand, extract_points, extract_mask


def sam_input_prepare(image, pred_masks, image_embeddings=None, resize_transform=None, use_point=True, use_box=True, use_mask=True, add_neg=True, margin=0.0, gamma=1.0, strength=15):
    ori_size = pred_masks.shape[-2:]
    input_dict = {
         'image': image,
         'original_size': ori_size,
         }
    
    target_size = image.shape[1:]
    expand_list = torch.zeros((len(pred_masks))).to(image.device)
    if use_box:
        bboxes, box_masks, areas, expand_list = extract_bboxes_expand(image_embeddings, pred_masks, margin=margin)
        input_dict['boxes'] = resize_transform.apply_boxes_torch(bboxes, ori_size)
    
    point_coords, point_labels, gaus_dt = extract_points(pred_masks, add_neg=add_neg, use_mask=use_mask, gamma=gamma)
    if use_point:
        input_dict['point_coords'] = resize_transform.apply_coords_torch(point_coords, ori_size)
        input_dict['point_labels'] = point_labels
        
    
        
    if use_mask:
        input_dict['mask_inputs'] = extract_mask(pred_masks, gaus_dt, target_size, is01=True, strength=strength, device=image.device, expand_list=expand_list)
   
    return input_dict,point_coords
    

def sam_refiner(image_path, 
                coarse_masks,
                sam,
                resize_transform=None,
                use_point=True,
                use_box=True,
                use_mask=True,
                add_neg=True,
                iters=5,
                margin=0.0,
                gamma=4.0,
                strength=30,
                use_samhq=False,
                ddp=False,
                is_train=False):
    """
    SAMRefiner refines coarse masks from an image by generating noise-tolerant prompts for SAM.

    Arguments:
      image_path (str): The image path for the target image.
      coarse_masks (list(array) or array): The coarse masks to be refined.
      sam (Sam): The Sam model.
      resize_transform (list(float)): The resize_transform used in sam. Default: ResizeLongestSide.
      use_point (bool): Whether to use point prompts. Default: True
      use_box (bool): Whether to use box prompts. Default: True
      use_mask (bool): Whether to use mask prompts. Default: True
      add_neg (bool): Whether to use the negative point prompts. Default: True
      iters (int): The number of iterative refinement. Default: 5
      margin (float): The parameter used to control whether to enlarge the box. Default: 0 (not enlarge)
      gamma (float): The parameter used to control the span of Gaussian distribution in mask prompt. Default: 4.0
      gamma (float): The parameter used to control the amplitude of Gaussian distribution in mask prompt. Default: 30
      use_samhq (bool): Whether to use samhq model. Default: False
    """
    
    if isinstance(coarse_masks, list):
        coarse_masks = np.stack(coarse_masks, axis=0)
        
    if len(coarse_masks.shape) == 2:
        coarse_masks = coarse_masks[None: ,]
    coarse_masks = torch.tensor(coarse_masks, dtype=torch.uint8).to(sam.device)
        
    assert len(coarse_masks.shape) == 3, "coarse mask dim must be (n, h, w), but got {}".format(coarse_masks.shape)

    if resize_transform is None:
        resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ori_size = image.shape[:2]
    image = [prepare_image(image, resize_transform, sam.device)]
    
    with torch.no_grad():
        if ddp:
            input_images = torch.stack([sam.module.preprocess(x) for x in image], dim=0)
            if not use_samhq:
                image_embeddings = sam.module.image_encoder(input_images) # torch.Size([1, 256, 64, 64])
            else:
                image_embeddings, interm_embeddings = sam.module.image_encoder(input_images)
                interm_embeddings = interm_embeddings[0] # early layer
        else:
            input_images = torch.stack([sam.preprocess(x) for x in image], dim=0)
            if not use_samhq:
                image_embeddings = sam.image_encoder(input_images) # torch.Size([1, 256, 64, 64])
            else:
                image_embeddings, interm_embeddings = sam.image_encoder(input_images)
                interm_embeddings = interm_embeddings[0] # early layer
        
    for i in range(iters):
        if i == 0:
            pred_mask_list = coarse_masks
        else:
            pred_mask_list = sam_masks_list.to(torch.uint8)
        
        input_dict, point_coords = sam_input_prepare(image[0],
                                                     pred_mask_list,
                                                     image_embeddings,
                                                     resize_transform,
                                                     use_point=use_point,
                                                     use_box=use_box,
                                                     use_mask=use_mask,
                                                     add_neg=add_neg,
                                                     margin=margin,
                                                     gamma=gamma,
                                                     strength=strength)
        
        sam_input = [input_dict]
        
        if not is_train:
            with torch.no_grad():
                if ddp:
                    if not use_samhq:
                        sam_output = sam.module.forward_with_image_embeddings(image_embeddings, sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
                    else:
                        sam_output = sam.module.forward_with_image_embeddings(image_embeddings, interm_embeddings,sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
                else:
                    if not use_samhq:
                        sam_output = sam.forward_with_image_embeddings(image_embeddings, sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
                    else:
                        sam_output = sam.forward_with_image_embeddings(image_embeddings, interm_embeddings,sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
        else:
            if ddp:
                sam_output = sam.module.forward_with_image_embeddings(image_embeddings, sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
            else:
                sam_output = sam.forward_with_image_embeddings(image_embeddings, sam_input, multimask_output=True)[0] #dict_keys(['masks', 'iou_predictions', 'low_res_logits'])

        sam_masks = sam_output['masks']
        sam_masks3 = sam_masks.clone().detach()
        sam_ious = sam_output['iou_predictions']
        sam_masks_logits = sam_output["low_res_logits"]

        if is_train:
            return sam_masks, sam_ious, sam_masks3
        sam_masks_list = []
        sam_masks_logits_list = []
        cnt = 0
        for sm, si in zip(sam_masks, sam_ious):
            max_idx = torch.argmax(si)
            sam_masks_list.append(sm[max_idx])
            sam_masks_logits_list.append(sam_masks_logits[cnt][max_idx])
            cnt += 1

        sam_masks = torch.stack(sam_masks_list, dim=0)
        sam_masks_logits = torch.stack(sam_masks_logits_list, dim=0)

        sam_masks_list = sam_masks > 0
        
    refined_masks = sam_masks_list.cpu().numpy().astype(np.uint8)
    assert len(refined_masks) == len(coarse_masks)
    return refined_masks, sam_ious, sam_masks3