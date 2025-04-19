import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import cv2
from segment_anything import sam_model_registry
from sam_refiner import sam_refiner
from torch.utils.data import Dataset, DataLoader

class Davis585Dataset(Dataset):
    def __init__(self, dataset_path,
                 init_mask_mode=None, **kwargs):
        super(Davis585Dataset, self).__init__(**kwargs)
        self.dataset_path = dataset_path + '/'
        self.sample_dict = self.generate_sample_dict(self.dataset_path)
        self.dataset_samples = list(self.sample_dict.keys())
        self.init_mask_mode = init_mask_mode

    def generate_sample_dict(self, dataset_path):
        # sample_dict = {'mask_path': ['image_path','stm_init_path','sp_init_path']}
        sample_dict = {}
        sequence_names = os.listdir(dataset_path)
        for sequence_name in sequence_names:
            sequence_dir = self.dataset_path + sequence_name + '/'
            gt_names = os.listdir(sequence_dir)
            gt_names = [i for i in gt_names if '.png' in i and 'init' not in i]
            for gt_name in gt_names:
                mask_path = sequence_dir + gt_name
                image_name = gt_name.split('_')[-1].replace('.png', '.jpg')
                image_path = sequence_dir + image_name
                stm_init_name = 'init_stm_' + gt_name
                stm_init_path = sequence_dir + stm_init_name
                sp_init_name = 'init_sp_' + gt_name
                sp_init_path = sequence_dir + sp_init_name
                sample_dict[mask_path] = [image_path, stm_init_path, sp_init_path]
        return sample_dict

    def __getitem__(self, index):
        mask_path = self.dataset_samples[index]
        image_path, stm_init_path, sp_init_path = self.sample_dict[mask_path]
        return image_path, stm_init_path, sp_init_path, mask_path

    def __len__(self):
        return len(self.dataset_samples)

def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union


def get_iu(seg, gt):
    intersection = np.count_nonzero(seg & gt)
    union = np.count_nonzero(seg | gt)

    return intersection, union

# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode



def evaluation(model, dataset):
    model.eval()
    iou_list = []
    iou_list_samrefine = []
    i_list = []
    u_list = []
    i_list_samrefine = []
    u_list_samrefine = []

    biou_list = []
    biou_list_samrefine = []
    bi_list = []
    bu_list = []
    bi_list_samrefine = []
    bu_list_samrefine = []
    correct_pred = 0
    for index, mask_path in enumerate(tqdm(dataset.dataset_samples)):
        image_path = dataset.sample_dict[mask_path][0]
        stm_init_path = dataset.sample_dict[mask_path][1]
        sp_init_path = dataset.sample_dict[mask_path][2]

        instances_mask = cv2.imread(mask_path)[:, :, 0] > 128
        stm_init_mask = cv2.imread(stm_init_path)[:, :, 0] > 128
        sp_init_mask = cv2.imread(sp_init_path)[:, :, 0] > 128

        if dataset.init_mask_mode == 'sp':
            init_mask = sp_init_mask
        elif dataset.init_mask_mode == 'stm':
            init_mask = stm_init_mask
        elif dataset.init_mask_mode == 'zero':
            init_mask = None

        gt_mask = instances_mask
        i_, u_ = get_iu(gt_mask, init_mask)
        iou_list.append(i_ / u_)
        i_list.append(i_)
        u_list.append(u_)

        bi_, bu_ = get_iu(mask_to_boundary(gt_mask.astype(np.uint8)), mask_to_boundary(init_mask.astype(np.uint8)))
        biou_list.append(bi_ / bu_)
        bi_list.append(bi_)
        bu_list.append(bu_)

        refined_masks, sam_ious, sam_masks3 = sam_refiner(image_path,
                                                      [init_mask],
                                                      model,
                                                      iters=1,
                                                      is_train=False,
                                                      use_point=True,
                                                      use_box=True,
                                                      use_mask=True,
                                                      add_neg=True,
                                                      )


        sam_ious_gt = torch.tensor([get_iou(gt_mask, sm.detach().cpu().numpy() > 0) for sm in sam_masks3[0]]).reshape(
            refined_masks.shape[0], -1)
        # print(sam_ious_gt.shape, sam_ious.shape)
        # print(sam_ious, sam_ious_gt)
        pred_maxid = torch.argmax(sam_ious[0])
        gt_maxid = torch.argmax(sam_ious_gt[0])
        if pred_maxid == gt_maxid:
            correct_pred += 1

        i_, u_ = get_iu(gt_mask, refined_masks[0])

        iou_list_samrefine.append(i_ / u_)
        i_list_samrefine.append(i_)
        u_list_samrefine.append(u_)

        bi_, bu_ = get_iu(mask_to_boundary(gt_mask.astype(np.uint8)),
                          mask_to_boundary(refined_masks[0].astype(np.uint8)))
        biou_list_samrefine.append(bi_ / bu_)
        bi_list_samrefine.append(bi_)
        bu_list_samrefine.append(bu_)

    print(len(iou_list))
    print(">>>IoU before refinement:{}".format(np.sum(i_list) / np.sum(u_list)))
    print(">>>IoU after refinement:{}".format(np.sum(i_list_samrefine) / np.sum(u_list_samrefine)))
    print(">>>Boundary IoU before refinement:{}".format(np.sum(bi_list) / np.sum(bu_list)))
    print(">>>Boundary IoU after refinement:{}".format(np.sum(bi_list_samrefine) / np.sum(bu_list_samrefine)))
    print(">>>Top1 acc:{}".format(correct_pred / len(iou_list)))
    return


def main():
    print(">>> Performing evaluation before IoU adaption...")
    evaluation(sam, dataset)

    sam.train()
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    loss_fn = torch.nn.MarginRankingLoss(margin=0.02, reduction='sum')
    optimizer = torch.optim.SGD(sam.mask_decoder.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 100], gamma=0.1)
    for name, param in sam.named_parameters():
        if 'lora' in name:  # or 'iou_token' in name:#'iou_prediction_head.layers.2' in name:
            param.requires_grad = True
            print('>>>updating {}'.format(name))
        else:
            param.requires_grad = False

    train_iters = 0
    for epoch in range(args.train_epoch):
        acc_loss = 0
        for image_path, stm_init_path, sp_init_path, mask_path in tqdm(loader):
            if dataset.init_mask_mode == 'sp':
                init_mask  = cv2.imread(sp_init_path[0])[:,:,0] > 128
            elif dataset.init_mask_mode == 'stm':
                init_mask  = cv2.imread(stm_init_path[0])[:,:,0] > 128
            elif dataset.init_mask_mode == 'zero':
                init_mask = None

            sam_masks, sam_ious, sam_masks3 = sam_refiner(image_path[0],
                                                          [init_mask],
                                                          sam,
                                                          iters=1,
                                                          is_train=True,
                                                          use_point=False,
                                                          use_box=True,
                                                          use_mask=False,
                                                          add_neg=False,
                                                          )

            sam_ious_gt = torch.tensor([get_iou(init_mask, sm.cpu().numpy() > 0) for sm in sam_masks[0]]).reshape(
                sam_masks.shape[0], -1)
            # print(sam_ious, sam_ious_gt)
            sam_ious_gt = sam_ious_gt.to(sam_ious.device)
            pos_ind = torch.argmax(sam_ious_gt[0])
            # print(pos_ind)
            neg_ind = []
            for i_ in range(3):
                if i_ != pos_ind:
                    neg_ind.append(i_)

            loss1 = loss_fn(sam_ious[:, pos_ind], sam_ious[:, neg_ind[0]],
                            torch.ones_like(sam_ious[:, pos_ind]).to(sam_ious.device))
            loss2 = loss_fn(sam_ious[:, pos_ind], sam_ious[:, neg_ind[1]],
                            torch.ones_like(sam_ious[:, pos_ind]).to(sam_ious.device))

            # print(loss1, loss2)
            loss = loss1 + loss2

            loss = loss / 5.0
            acc_loss += loss

            train_iters += 1

            if train_iters % 5 == 0:
                optimizer.zero_grad()
                acc_loss.backward()
                optimizer.step()
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                print('LR: {:.6f}, LOSS:{:.4f}'.format(current_lr, acc_loss.item()))
                acc_loss = 0


        save_path = args.sam_checkpoint.replace('.pth', '_iou_adaption.pth')
        print(">>Saving model to {}".format(save_path))
        torch.save(sam.state_dict(), save_path)

        print(">>> Performing evaluation after IoU adaption......")
        evaluation(sam, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--dataset_path', type=str, default='./DAVIS585/data')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--train_epoch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    print(">>>Loading model from {}".format(args.sam_checkpoint))

    dataset = Davis585Dataset(args.dataset_path, init_mask_mode='sp')

    main()