import contextlib
import logging
import os
import sys
from wscxr.utils import create_logger
import numpy as np
import torch

import wscxr.backbones
import wscxr.common
import wscxr.metrics
import wscxr.feat_extractor
import wscxr.sampler
import wscxr.utils
from wscxr.wscxr import WSCXR
from datetime import datetime
from datasets.dataset import MedDataset,MedAbnormalDataset
import tqdm
import warnings
import PIL
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn.functional as F
import argparse
from easydict import EasyDict
import yaml


warnings.filterwarnings('ignore')


def normalization(segmentations,avgpool_size = 64):

    segmentations_ = torch.tensor(segmentations[:, None, ...]).cuda()  # N x 1 x H x W
    segmentations_ =  F.avg_pool2d(segmentations_, (avgpool_size,avgpool_size), stride=1).cpu().numpy()

    min_scores = (
        segmentations_.reshape(-1).min(axis=-1).reshape(1)
    )

    max_scores = (
        segmentations_.reshape(-1).max(axis=-1).reshape(1)
    )

    segmentations = (segmentations - min_scores) / (max_scores - min_scores)
    segmentations=np.clip(segmentations,a_min=0,a_max=1)

    return segmentations


def main(args):

    dataloader_dict =dataset(
                                os.path.join(args.config.data_path,args.dataset_name),
                                args.config.batch_size,
                                args.config.imagesize,
                                args.config.num_workers,
                                args.config.max_normal,
                                )(args.seed)

    device = wscxr.utils.set_torch_device(args.gpu)

    imagesize = dataloader_dict["test"].dataset.imagesize

    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    wscxr.utils.fix_seeds(args.seed, device)

    with device_context:

        WSCXR = create_wscxr_instance(
            args.config.backbone_name,
            args.config.layers_to_extract_from,
            args.config.pretrain_embed_dimension,
            args.config.target_embed_dimension,
            args.config.patchsize,
            args.config.meta_epochs,
            args.config.gan_epochs,
            args.config.dsc_layers,
            args.config.dsc_hidden,
            args.config.dsc_margin,
            args.config.dsc_lr,

        )(imagesize, device)

        discriminator_path=os.path.join(args.config.results_path,args.dataset_name,args.config.dsc_save_path)
        print("load discriminator path: {}".format(discriminator_path))
        WSCXR.discriminator.load_state_dict(torch.load(discriminator_path))

        if WSCXR.backbone.seed is not None:
            wscxr.utils.fix_seeds(WSCXR.backbone.seed, device)

        torch.cuda.empty_cache()

        scores, segmentations, labels_gt, image_paths = WSCXR.predict(dataloader_dict["test"])

        scores = np.array(scores)
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores)).tolist()


        results = wscxr.metrics.compute_imagewise_retrieval_metrics(scores, labels_gt)
        print("auroc: {}, acc: {}, f1: {}".format(results['auroc'], results['acc'], results['f1']))

        segmentations = normalization(np.array(segmentations))

        save_images_root=os.path.join(args.config.results_path,args.dataset_name,"seg_images")
        os.makedirs(save_images_root,exist_ok=True)

        for image_path, segmentation ,label in tqdm.tqdm(
                zip(image_paths,  segmentations,labels_gt),
                total=len(image_paths),
                desc="Generating Segmentation Images...",
                leave=False,
        ):
            _, image_name = os.path.split(image_path)

            image = PIL.Image.open(image_path).convert("RGB")
            image = image.resize((args.config.imagesize,args.config.imagesize))
            image = np.array(image).astype(np.uint8)

            heat = show_cam_on_image(image / 255, segmentation, use_rgb=True)

            label_= "normal" if label==0 else "abnormal"

            PIL.Image.fromarray(heat).save(os.path.join(save_images_root,label_+"_"+image_name))



def create_sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return wscxr.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return wscxr.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return wscxr.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return  get_sampler


def dataset(
    data_path,
    batch_size,
    imagesize,
    num_workers,
    max_normal,
):

    def get_dataloaders(seed):


        test_dataset = MedDataset(
            data_path,
            imagesize=imagesize,
            split='test',
            normal_only=False,
            seed=seed,
        )

        print("test:{}".format(len(test_dataset)))

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_dataloader.name = '{}_test'.format(args.dataset_name)

        dataloader_dict = {
            "test": test_dataloader
        }

        return dataloader_dict

    return get_dataloaders


def create_wscxr_instance(
    backbone_name,
    layers_to_extract_from,

    pretrain_embed_dimension,
    target_embed_dimension,

    patchsize,
    meta_epochs,
    gan_epochs,

    dsc_layers,
    dsc_hidden,
    dsc_margin,
    dsc_lr,

):

    def get_wscxr(input_shape, device):
        backbone_seed = None
        backbone = wscxr.backbones.load(backbone_name)
        backbone.name, backbone.seed = backbone_name, backbone_seed

        wscxr_inst = WSCXR(device)

        wscxr_inst.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,

                patchsize=patchsize,
                meta_epochs=meta_epochs,
                gan_epochs=gan_epochs,

                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                dsc_lr=dsc_lr,
            )

        return wscxr_inst

    return get_wscxr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSCXR")

    parser.add_argument("--gpu", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset_name", default='zhanglab', type=str,
                        choices=['zhanglab', 'chexpert12'])

    parser.add_argument("--faiss_on_gpu", type=bool, default=False)
    parser.add_argument("--faiss_num_workers", type=int, default=8)

    args = parser.parse_args()

    with open(os.path.join("config", "{}.yaml".format(args.dataset_name))) as f:
        args.config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    main(args)