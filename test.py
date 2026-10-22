# coding:utf-8

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from train import ConvNeXtV2Backbone
from model.CD_Model import CD_Model
from model.Dino_Clip import DINOiser
from CDDataset import CDDataset

from hydra import initialize, compose


#########################################################
#########################################################

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size',type=int,default=16)


parser.add_argument('--data_root',type=str,default='assets/data')          


parser.add_argument('--config',default='clip_dinoiser.yaml')
parser.add_argument('--gpu',type=str,default="1")


parser.add_argument('--checkpoint',
                    type=str,
                    default=
                    "outputs_WHU-CD/iou_best_model.pth")      # best model : iou_best_model.pth  




args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu



#########################################################
#########################################################

def setup_seed(seed=3407):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True



#########################################################
# model
#########################################################

def build_model(cfg, device):


    ####################################
    # ConvNeXtV2 Backbone
    ####################################

    backbone = ConvNeXtV2Backbone(
        pretrained=True,
        weight_path="pretrained/pytorch_model.bin"
    )


    ####################################
    # Change Detection Network
    ####################################

    net = CD_Model(
        backbone=backbone,
        feature_dim=256,
        inp_A_channels=3,
        inp_B_channels=3,
        out_channels=1
    ).to(device)



    ####################################
    # DINO semantic branch
    ####################################

    model_ex = DINOiser(
        clip_backbone=cfg.model,
        vit_arch="vit_base",
        vit_patch_size=16,
        enc_type_feats="v",
        feats_idx=-3,
        gamma=0.2,
        delta=0.99,
        in_dim=256
    ).to(device)


    model_ex.load_dino()


    model_ex.eval()

    for p in model_ex.parameters():
        p.requires_grad = False



    return net, model_ex



#########################################################
#########################################################

def load_checkpoint(model, path, device):

    print("="*60)
    print("Loading checkpoint:")
    print(path)


    checkpoint = torch.load(
        path,
        map_location=device
    )


    state_dict = checkpoint["model"]


    # DataParallel
    new_state_dict = {}

    for k,v in state_dict.items():

        if k.startswith("module."):

            new_state_dict[k[7:]]=v

        else:

            new_state_dict[k]=v


    model.load_state_dict(
        new_state_dict,
        strict=True
    )


    print(
        "Best Epoch:",
        checkpoint.get("epoch")
    )

    print(
        "Best IoU:",
        checkpoint.get("best_iou")
    )

    print("="*60)


def calc_metrics(pred, gt):

    """
    pred:
        binary prediction (H,W)

    gt:
        ground truth (H,W)
    """

    pred = pred.long()
    gt = gt.long()


    TP = torch.sum(
        (pred == 1) & (gt == 1)
    ).item()

    TN = torch.sum(
        (pred == 0) & (gt == 0)
    ).item()

    FP = torch.sum(
        (pred == 1) & (gt == 0)
    ).item()

    FN = torch.sum(
        (pred == 0) & (gt == 1)
    ).item()


    return TP, TN, FP, FN


#########################################################
#########################################################

def create_error_map(pred, gt):

    """
    pred:
        (H,W)

    gt:
        (H,W)

    out:
        RGB error map

    black:
        TN

    white:
        TP

    green:
        FP

    red:
        FN
    """

    if pred.dim() > 2:
        pred = pred.squeeze()

    if gt.dim() > 2:
        gt = gt.squeeze()


    pred = pred.long()
    gt = gt.long()


    h,w = pred.shape


    result = torch.zeros(
        3,
        h,
        w
    )


    TP = (pred == 1) & (gt == 1)

    FP = (pred == 1) & (gt == 0)

    FN = (pred == 0) & (gt == 1)


    # TP 
    result[:, TP] = 1.0


    # FP 
    result[0, FP] = 0
    result[1, FP] = 1
    result[2, FP] = 0


    # FN 
    result[0, FN] = 1
    result[1, FN] = 0
    result[2, FN] = 0


    return result



#########################################################
#########################################################

@torch.no_grad()
def test():

    device="cuda" if torch.cuda.is_available() else "cpu"


    ####################################
    # hydra config
    ####################################

    initialize(
        config_path="configs",
        version_base=None
    )

    cfg = compose(
        config_name=args.config
    )



    ####################################
    # Dataset
    ####################################

    test_dataset = CDDataset(
        root=args.data_root,
        split="val"
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    print(
        "Val samples:",
        len(test_dataset)
    )



    ####################################
    # Model
    ####################################

    net, model_ex = build_model(
        cfg,
        device
    )


    load_checkpoint(
        net,
        args.checkpoint,
        device
    )


    net.eval()



    ####################################
    ####################################

    dataset_name = os.path.basename(
        os.path.normpath(args.data_root)
    )

    save_dir = os.path.join(
        "dataset",
        "dataset_name"             
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    print("Save prediction results to:")
    print(save_dir)


    # ==============================
    # Overall Metrics
    # ==============================

    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0

    ####################################
    # inference
    ####################################

    for imgA, imgB, mask, filename in tqdm(test_loader):

        imgA = imgA.to(device)
        imgB = imgB.to(device)
        mask = mask.to(device)

        semA = model_ex(imgA)

        semB = model_ex(imgB)

        pred, trans, c_feat = net(
            imgA,
            imgB,
            semA,
            semB
        )

        pred = torch.sigmoid(pred)

        pred = (pred > 0.5).float()

        for i in range(pred.shape[0]):

            pred_i = pred[i, 0].cpu()
            gt_i = mask[i].cpu()

            if gt_i.dim() > 2:
                gt_i = gt_i.squeeze()

            # ==================================
            # ==================================

            TP, TN, FP, FN = calc_metrics(
                pred_i,
                gt_i
            )

            total_TP += TP
            total_TN += TN
            total_FP += FP
            total_FN += FN

            # ==================================
            # ==================================

            error_map = create_error_map(
                pred_i,
                gt_i
            )

            # ==================================
            # ==================================

            original_name = filename[i]

            original_name = (
                    os.path.splitext(original_name)[0]
                    + ".png"
            )

            save_path = os.path.join(
                save_dir,
                original_name
            )

            save_image(
                error_map,
                save_path
            )

            # print(
            #     f"Saved: {save_path}"
            # )



    print("="*50)

    print(
        "Testing finished!"
    )

    # ==============================
    # Calculate final metrics
    # ==============================

    eps = 1e-6

    precision = total_TP / (
            total_TP + total_FP + eps
    )

    recall = total_TP / (
            total_TP + total_FN + eps
    )

    f1 = 2 * precision * recall / (
            precision + recall + eps
    )

    iou = total_TP / (
            total_TP + total_FP + total_FN + eps
    )

    oa = (
                 total_TP + total_TN
         ) / (
                 total_TP +
                 total_TN +
                 total_FP +
                 total_FN +
                 eps
         )

    far = total_FP / (
            total_FP + total_TN + eps
    )

    print("=" * 60)
    print("Final Test Metrics")
    print("=" * 60)

    print(f"TP : {total_TP}")

    print(f"TN : {total_TN}")

    print(f"FP : {total_FP}")

    print(f"FN : {total_FN}")

    print("-" * 60)

    print(f"IoU       : {iou:.4f}")

    print(f"F1        : {f1:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"OA        : {oa:.4f}")

    print(f"FAR       : {far:.4f}")

    print("=" * 60)

    print(
        "Saved:",
        save_dir
    )



#########################################################

if __name__=="__main__":

    setup_seed()

    test()