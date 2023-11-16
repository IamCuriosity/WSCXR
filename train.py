import argparse
import contextlib
import os
import pprint
import warnings
from datetime import datetime

import torch
import yaml
from easydict import EasyDict

import wscxr.backbones
import wscxr.common
import wscxr.metrics
import wscxr.feat_extractor
import wscxr.sampler
import wscxr.utils
from datasets.dataset import MedDataset, MedAbnormalDataset
from wscxr.wscxr import WSCXR
from wscxr.utils import create_logger

warnings.filterwarnings('ignore')


def get_current_time():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return current_time


def main(args):

    run_save_path = os.path.join(args.config.results_path,args.dataset_name)
    os.makedirs(run_save_path,exist_ok=True)

    current_time = get_current_time()

    logger = wscxr.utils.create_logger("logger", os.path.join(run_save_path, "log_{}.log".format(current_time)))

    logger.info("args: {}".format(pprint.pformat(args)))

    dataloader_dict =dataset(   os.path.join(args.config.data_path,args.dataset_name),
                                args.config.batch_size,
                                args.config.imagesize,
                                args.config.num_workers,
                                args.config.max_normal,
                                )(args.seed)

    device = wscxr.utils.set_torch_device(args.gpu)

    imagesize = dataloader_dict["train_normal"].dataset.imagesize

    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    wscxr.utils.fix_seeds(args.seed, device)

    with device_context:

        torch.cuda.empty_cache()
        sampler = create_sampler('approx_greedy_coreset',args.config.normal_percentage)(
            device,
        )

        WSCXR = create_wscxr_instance(
                     args.config.backbone_name,
                     args.config.layers_to_extract_from,
                     args.config.pretrain_embed_dimension,
                     args.config.target_embed_dimension,
                     args.config.preprocessing,
                     args.config.aggregation,
                     args.config.patchsize,
                     args.config.patchscore,
                     args.config.patchoverlap,
                     args.config.anomaly_scorer_num_nn,
                     args.config.patchsize_aggregate,
                     args.faiss_on_gpu,
                     args.faiss_num_workers)(imagesize, sampler, device)

        if WSCXR.backbone.seed is not None:
            wscxr.utils.fix_seeds(WSCXR.backbone.seed, device)

        torch.cuda.empty_cache()

        normal_features = WSCXR.fit(dataloader_dict["train_normal"])
        WSCXR.featuresampler = wscxr.sampler.IdentitySampler()
        abnormal_features = WSCXR.fit(dataloader_dict["train_abnormal"])

        WSCXR.anomaly_scorer.fit([normal_features])
        anomaly_scores = WSCXR.anomaly_scorer.predict([abnormal_features])[0]

        _,index=torch.topk(torch.from_numpy(anomaly_scores),
                           int(anomaly_scores.shape[0]*args.config.abnormal_percentage),
                           largest=True)


        abnormal_features=abnormal_features[index.numpy()]

        logger.info("abnormal_prototype shape:({},{}), normal_prototype shape:({},{})".
                    format(abnormal_features.shape[0],abnormal_features.shape[1],
                           normal_features.shape[0],normal_features.shape[1],))


        WSCXR = create_samplenet_instance(
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
            abnormal_features,
            logger,
            os.path.join(run_save_path ,args.config.dsc_save_path),
        )(imagesize, device)

        if WSCXR.backbone.seed is not None:
            wscxr.utils.fix_seeds(WSCXR.backbone.seed, device)

        torch.cuda.empty_cache()

        i_auroc, p_auroc, pro_auroc = WSCXR.train_(dataloader_dict["train_normal"], dataloader_dict["test"])
        logger.info("best i_auroc: {}, best i_acc: {}, i_f1: {}".format(i_auroc, p_auroc, pro_auroc))


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

        train_normal_dataset = MedDataset(
            data_path,
            imagesize=imagesize,
            split='train',
            seed=seed,
            normal_only=True,
            max_normal=max_normal,
        )

        train_abnormal_dataset = MedAbnormalDataset(
            data_path,
            args.config.abnormal_k,
            imagesize=imagesize,
            split='train',
            seed=seed,
        )

        test_dataset = MedDataset(
            data_path,
            imagesize=imagesize,
            split='test',
            normal_only=False,
            seed=seed,
        )
        print("train_normal:{}, train_abnormal:{}, test:{}".
              format(len(train_normal_dataset),len(train_abnormal_dataset),len(test_dataset)))

        train_normal_dataloader = torch.utils.data.DataLoader(
            train_normal_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        train_abnormal_dataloader = torch.utils.data.DataLoader(
            train_abnormal_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        train_normal_dataloader.name = '{}_train_normal'.format(args.dataset_name)
        train_abnormal_dataloader.name = '{}_train_abnormal'.format(args.dataset_name)
        test_dataloader.name = '{}_test'.format(args.dataset_name)

        dataloader_dict = {
            "train_normal": train_normal_dataloader,
            "train_abnormal": train_abnormal_dataloader,
            "test": test_dataloader
        }

        return dataloader_dict

    return get_dataloaders



def create_wscxr_instance(
    backbone_name,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
):

    def get_wscxr(input_shape, sampler, device):
        backbone_seed = None

        backbone = wscxr.backbones.load(backbone_name)
        backbone.name, backbone.seed = backbone_name, backbone_seed

        nn_method = wscxr.common.FaissNN(faiss_on_gpu, faiss_num_workers)

        wscxr_instance = wscxr.wscxr.WSCXR(device)
        wscxr_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
            )

        return wscxr_instance

    return  get_wscxr


def create_samplenet_instance(
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

    prototypes,
    logger,

    dsc_save_path,
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

                logger=logger,
                dsc_save_path=dsc_save_path,
                prototypes=prototypes,

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

    parser.add_argument("--dataset_name", default='zhanglab', type=str,choices=['zhanglab','chexpert12'])

    parser.add_argument("--faiss_on_gpu", type=bool, default=False)
    parser.add_argument("--faiss_num_workers", type=int, default=8)

    args = parser.parse_args()

    with open(os.path.join("config", "{}.yaml".format(args.dataset_name))) as f:
        args.config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    main(args)